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

# Import model
from models import CatMeowCNN

# ========== Configuration ==========
MODEL_PATH = Path("results/cat_meow.pt")
METADATA_PATH = Path("data/interim/metadata.pkl")

# Audio preprocessing (must match preprocess.ipynb)
SAMPLE_RATE = 22050
DURATION = 4  # seconds
N_SAMPLES = SAMPLE_RATE * DURATION
N_MELS = 128
HOP_LENGTH = 512


# ========== Load Model & Metadata ==========
def load_model():
    """Load trained model and label mapping."""
    # Load metadata
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    
    idx_to_label = metadata["idx_to_label"]
    n_classes = len(idx_to_label)
    
    # Load model
    model = CatMeowCNN(n_classes=n_classes)
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
        print(f"   → Passing through CatMeowCNN model...")
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

# Custom CSS for better mobile experience
custom_css = """
.gradio-container {
    max-width: 600px !important;
    margin: auto !important;
}
h1 {
    text-align: center;
    font-size: 2.5rem !important;
}
.description {
    text-align: center;
}
"""

# Create the interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(
        sources=["microphone", "upload"],
        type="numpy",
        label="🎤 Record or Upload Cat Sound",
    ),
    outputs=gr.Label(
        num_top_classes=5,
        label="🐱 Detected Mood",
    ),
    title="🐱 Cat Mood Classifier",
    description="""
    **Record your cat's meow to detect its mood!**
    
    Click the microphone button to record, or upload an audio file.
    The model will classify the sound into one of 10 cat moods.
    """,
    examples=[
        # Add example audio files here if you have them
        # ["examples/angry_cat.wav"],
        # ["examples/happy_cat.wav"],
    ],
    flagging_mode="never",
)


# ========== Launch ==========
if __name__ == "__main__":
    demo.launch(
        share=True,  # Creates public URL (great for mobile testing!)
        # server_name="0.0.0.0",  # Uncomment to allow LAN access
        # server_port=7860,
        theme=gr.themes.Soft(),
        css=custom_css,
    )

