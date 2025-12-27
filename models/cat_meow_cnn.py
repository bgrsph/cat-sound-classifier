"""
Simple CNN for Cat Sound Classification

Architecture based on proven cat meow classification models.
Input: Mel spectrogram (1, n_mels, time_frames)
Output: Class probabilities
"""

import torch
import torch.nn as nn


class CatMeowCNN(nn.Module):
    """
    Simple CNN for classifying cat sounds from mel spectrograms.
    
    Architecture:
        - 4 Convolutional blocks (Conv2D + BatchNorm + ReLU + MaxPool)
        - Global Average Pooling
        - Fully connected classifier
    """
    
    def __init__(self, n_classes: int = 10, input_shape: tuple = (128, 173), dropout: float = 0.5):
        """
        Args:
            n_classes: Number of output classes
            input_shape: (n_mels, time_frames) of input spectrogram
            dropout: Dropout rate for classifier layers
        """
        super().__init__()
        
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.dropout = dropout
        
        # Convolutional blocks
        self.conv_blocks = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier with configurable dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),  # Slightly lower for second layer
            nn.Linear(128, n_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_frames)
               or (batch, n_mels, time_frames) - will add channel dim
        
        Returns:
            Logits of shape (batch, n_classes)
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions (argmax of softmax)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities (softmax)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


# Quick test
if __name__ == "__main__":
    model = CatMeowCNN(n_classes=10, input_shape=(128, 173))
    
    # Test with random input
    x = torch.randn(4, 128, 173)  # batch of 4
    output = model(x)
    
    print(f"Model: CatMeowCNN")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

