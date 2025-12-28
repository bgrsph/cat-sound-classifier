"""
🐱 Cat Mood Classifier - Gradio Web App

Record your cat's meow and detect its mood!
"""

import gradio as gr
import torch
import librosa
import numpy as np
import pickle
from pathlib import Path

# Import models
from models import CatMeowCNN, TransferCNN

# ========== Configuration ==========
MODEL_PATH = Path("results/tuned_model.pt")
TUNING_RESULTS_PATH = Path("results/tuning_results.pkl")
METADATA_PATH = Path("data/interim/metadata.pkl")

# Audio preprocessing (must match preprocess.ipynb)
SAMPLE_RATE = 22050
DURATION = 4  # seconds
N_SAMPLES = SAMPLE_RATE * DURATION
N_MELS = 128
HOP_LENGTH = 512


# ========== Load Model & Metadata ==========
def load_model():
    """Load tuned model and label mapping."""
    # Load metadata
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    
    idx_to_label = metadata["idx_to_label"]
    n_classes = len(idx_to_label)
    
    # Load tuning results to get best model type
    with open(TUNING_RESULTS_PATH, "rb") as f:
        tuning_results = pickle.load(f)
    
    best_params = tuning_results["best_params"]
    
    # Create correct model type
    if best_params["model"] == "TransferCNN":
        model = TransferCNN(
            n_classes=n_classes,
            backbone=best_params.get("backbone", "resnet18"),
            freeze_backbone=best_params.get("freeze_backbone", True),
            dropout=best_params.get("dropout", 0.5),
        )
        print(f"Using TransferCNN with {best_params.get('backbone')} backbone")
    else:
        model = CatMeowCNN(n_classes=n_classes, dropout=best_params.get("dropout", 0.5))
        print("Using CatMeowCNN")
    
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location="cpu"))
    model.eval()
    
    return model, idx_to_label


# Initialize on startup
print("Loading model...")
model, idx_to_label = load_model()
print(f"Model loaded! Classes: {list(idx_to_label.values())}")


# ========== Preprocessing ==========
def preprocess_audio(audio_tuple):
    """
    Convert raw audio to mel spectrogram.
    
    Args:
        audio_tuple: (sample_rate, waveform) from Gradio
        
    Returns:
        mel_db: log mel spectrogram (128, 173)
    """
    sr, waveform = audio_tuple
    
    print(f"\n📥 [Step 2] Preprocessing audio...")
    print(f"   → Input: sample_rate={sr}Hz, shape={waveform.shape}, dtype={waveform.dtype}")
    
    # Convert to float32 and normalize
    print(f"   → Converting to float32 and normalizing...")
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform = waveform.astype(np.float32) / 2147483648.0
    elif waveform.dtype == np.float64:
        waveform = waveform.astype(np.float32)
    else:
        waveform = waveform.astype(np.float32)
    print(f"   ✓ Normalized to float32")
    
    # Convert stereo to mono (Gradio sends (samples, channels) for stereo)
    print(f"   → Converting to mono...")
    if len(waveform.shape) > 1:
        if waveform.shape[1] == 2:  # (samples, 2)
            waveform = waveform.mean(axis=1)
        elif waveform.shape[0] == 2:  # (2, samples)
            waveform = waveform.mean(axis=0)
    
    # Flatten just in case
    waveform = waveform.flatten()
    print(f"   ✓ Mono audio: {len(waveform)} samples ({len(waveform)/sr:.2f}s)")
    
    # Resample to target sample rate
    if sr != SAMPLE_RATE:
        print(f"   → Resampling from {sr}Hz to {SAMPLE_RATE}Hz...")
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)
        print(f"   ✓ Resampled: {len(waveform)} samples")
    else:
        print(f"   ✓ Sample rate already {SAMPLE_RATE}Hz, no resampling needed")
    
    # Pad or truncate to fixed length
    print(f"   → Adjusting to fixed length ({DURATION}s = {N_SAMPLES} samples)...")
    if len(waveform) < N_SAMPLES:
        pad_amount = N_SAMPLES - len(waveform)
        waveform = np.pad(waveform, (0, pad_amount))
        print(f"   ✓ Padded with {pad_amount} zeros")
    else:
        trimmed = len(waveform) - N_SAMPLES
        waveform = waveform[:N_SAMPLES]
        print(f"   ✓ Truncated {trimmed} samples")
    
    # Extract mel spectrogram
    print(f"   → Extracting mel spectrogram ({N_MELS} mel bands)...")
    mel = librosa.feature.melspectrogram(
        y=waveform, 
        sr=SAMPLE_RATE, 
        n_mels=N_MELS, 
        hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    print(f"   ✓ Mel spectrogram created: shape={mel_db.shape}")
    
    return mel_db


# ========== Prediction ==========
def predict(audio):
    """
    Predict cat mood from audio.
    
    Args:
        audio: (sample_rate, waveform) from Gradio microphone
        
    Returns:
        dict: {label: probability} for Gradio Label component
    """
    print("\n" + "="*60)
    print("🎤 [Step 1] SUBMIT BUTTON CLICKED!")
    print("="*60)
    
    if audio is None:
        print("❌ Error: No audio received!")
        return {"No audio recorded": 1.0}
    
    try:
        sr, waveform = audio
        duration = len(waveform) / sr
        print(f"✓ Audio received: {duration:.2f} seconds at {sr}Hz")
        
        # Preprocess
        mel = preprocess_audio(audio)
        
        # Convert to tensor
        print(f"\n🧠 [Step 3] Running model inference...")
        print(f"   → Converting to PyTorch tensor...")
        x = torch.FloatTensor(mel).unsqueeze(0)  # (1, 128, 173)
        print(f"   ✓ Tensor shape: {x.shape}")
        
        # Predict
        print(f"   → Passing through model...")
        with torch.no_grad():
            logits = model(x)
            print(f"   ✓ Raw logits: {logits.shape}")
            
            print(f"   → Applying softmax to get probabilities...")
            probs = torch.softmax(logits, dim=1)[0]
            print(f"   ✓ Probabilities computed")
        
        # Create result dict
        result = {idx_to_label[i]: float(probs[i]) for i in range(len(probs))}
        
        # Find top predictions
        sorted_results = sorted(result.items(), key=lambda x: -x[1])
        
        print(f"\n🏆 [Step 4] PREDICTION COMPLETE!")
        print(f"   Top 5 predictions:")
        for i, (mood, prob) in enumerate(sorted_results[:5], 1):
            emoji = MOOD_EMOJIS.get(mood, "🐱")
            bar = "█" * int(prob * 20)
            print(f"   {i}. {emoji} {mood:12} {bar} {prob:.1%}")
        
        print("="*60 + "\n")
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        return {"Error - check console": 1.0}


# ========== Gradio Interface ==========
# Cat mood emoji mapping
MOOD_EMOJIS = {
    "Angry": "😾",
    "Defense": "🙀", 
    "Fighting": "👊",
    "Happy": "😺",
    "HuntingMind": "🐾",
    "Mating": "💕",
    "MotherCall": "🐱",
    "Paining": "😿",
    "Resting": "😸",
    "Warning": "⚠️",
}

# Translations
TRANSLATIONS = {
    "EN": {
        "title": "🐱 Moods from Meows",
        "subtitle": "**Record your cat's meow to detect its mood!**",
        "instructions": """
- 📁 Don't want to record? Upload any cat sound file by clicking the upload icon.
- 🎙️ To record, first allow microphone access in your browser.
- 🔴 Click **Record** to start recording.
- 🐱 Record your cat's meow. A few seconds is enough!
- ⏹️ Click 🔴 **Stop** to finish recording.
- 🔮 Click **"Detect Mood"** and wait for the AI to analyze your cat's mood.
""",
        "audio_label": "🎤 Record or Upload Cat Sound",
        "output_label": "🐱 Detected Mood",
        "submit_btn": "🔮 Detect Mood",
        "clear_btn": "🗑️ Clear",
        "language_label": "🌐 Language",
        "moods": {
            "Angry": "Angry",
            "Defense": "Defensive",
            "Fighting": "Fighting",
            "Happy": "Happy",
            "HuntingMind": "Hunting",
            "Mating": "Mating Call",
            "MotherCall": "Mother Call",
            "Paining": "In Pain",
            "Resting": "Resting",
            "Warning": "Warning",
        }
    },
    "TR": {
        "title": "🐱 Miyavdan Haller",
        "subtitle": "**Kedinizin miyavlamasını kaydedin ve ruh halini öğrenin!**",
        "instructions": """
- 📁 Kayıt yapmak istemezseniz, yükleme ikonuna tıklayarak istediğiniz bir kedi sesi dosyasını yükleyebilirsiniz.
- 🎙️ Kayıt yapmak için öncelikle tarayıcınızın mikrofon kullanma isteğini kabul edin.
- 🔴 **Record**'a basarak kaydı başlatın.
- 🐱 Kedinizin miyavlamasını kaydedin. Birkaç saniye dahi yeterli olacaktır.
- ⏹️ Durdurmak için 🔴 **Stop**'a tıklayın.
- 🔮 **"Ruh Halini Belirle"** ye basın ve yapay zeka kedinizin halet-i ruhiyesini tahmin etsin!
""",
        "audio_label": "🎤 Kedi Sesi Kaydet veya Yükle",
        "output_label": "🐱 Tespit Edilen Ruh Hali",
        "submit_btn": "🔮 Ruh Halini Belirle",
        "clear_btn": "🗑️ Temizle",
        "language_label": "🌐 Dil",
        "moods": {
            "Angry": "Kızgın",
            "Defense": "Savunmada",
            "Fighting": "Kavgacı",
            "Happy": "Mutlu",
            "HuntingMind": "Avcı Modunda",
            "Mating": "Çiftleşme Çağrısı",
            "MotherCall": "Anne Çağrısı",
            "Paining": "Acı Çekiyor",
            "Resting": "Dinleniyor",
            "Warning": "Uyarı",
        }
    }
}

def predict_with_translation(audio, language):
    """Predict and translate results based on selected language."""
    result = predict(audio)
    
    if result is None or "Error" in str(result) or "No audio" in str(result):
        return result
    
    # Translate mood labels
    translations = TRANSLATIONS[language]["moods"]
    translated_result = {}
    for mood, prob in result.items():
        translated_mood = translations.get(mood, mood)
        emoji = MOOD_EMOJIS.get(mood, "🐱")
        translated_result[f"{emoji} {translated_mood}"] = prob
    
    return translated_result

# Create the interface with Blocks for language switching
# Default language
DEFAULT_LANG = "TR"
DEFAULT_T = TRANSLATIONS[DEFAULT_LANG]

with gr.Blocks(title="Cat Mood Classifier") as demo:
    # State for current language
    current_lang = gr.State(DEFAULT_LANG)
    
    # Header row with title on left, language on right
    with gr.Row():
        with gr.Column(scale=4):
            title_md = gr.Markdown(f"# {DEFAULT_T['title']}")
        with gr.Column(scale=1, min_width=120):
            language_select = gr.Radio(
                choices=["EN", "TR"],
                value=DEFAULT_LANG,
                label="🌐",
                interactive=True,
            )
    
    subtitle_md = gr.Markdown(DEFAULT_T["subtitle"])
    instructions_md = gr.Markdown(DEFAULT_T["instructions"])
    
    # Main interface - vertical layout
    audio_input = gr.Audio(
        sources=["microphone", "upload"],
        type="numpy",
        label=DEFAULT_T["audio_label"],
    )
    
    with gr.Row():
        clear_btn = gr.ClearButton([audio_input], value=DEFAULT_T["clear_btn"])
        submit_btn = gr.Button(DEFAULT_T["submit_btn"], variant="primary")
    
    output_label = gr.Label(
        num_top_classes=5,
        label=DEFAULT_T["output_label"],
    )
    
    # Function to update UI text based on language
    def update_language(lang):
        t = TRANSLATIONS[lang]
        return (
            f"# {t['title']}",
            t["subtitle"],
            t["instructions"],
            gr.update(label=t["audio_label"]),
            gr.update(label=t["output_label"]),
            gr.update(value=t["submit_btn"]),
            gr.update(value=t["clear_btn"]),
        )
    
    # Connect language selector
    language_select.change(
        fn=update_language,
        inputs=[language_select],
        outputs=[title_md, subtitle_md, instructions_md, audio_input, output_label, submit_btn, clear_btn],
    )
    
    # Connect submit button
    submit_btn.click(
        fn=predict_with_translation,
        inputs=[audio_input, language_select],
        outputs=output_label,
    )


# ========== Launch ==========
if __name__ == "__main__":
    print("\n🚀 Starting Cat Mood Classifier...")
    print("   Supports: English (EN) and Turkish (TR)")
    print("   Press Ctrl+C to stop\n")
    demo.launch(
        share=True,  # Creates public URL (great for mobile testing!)
        # server_name="0.0.0.0",  # Uncomment to allow LAN access
        # server_port=7860,
    )

