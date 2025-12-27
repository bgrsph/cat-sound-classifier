"""
Transfer Learning Models for Audio Classification

Uses pretrained ImageNet models on mel spectrograms (treated as images).
"""

import torch
import torch.nn as nn
from torchvision import models


class TransferCNN(nn.Module):
    """
    Transfer learning model using pretrained ImageNet backbone.
    
    Converts single-channel mel spectrograms to 3-channel for pretrained models,
    then fine-tunes the classifier head.
    """
    
    def __init__(
        self, 
        n_classes: int = 10, 
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5,
    ):
        """
        Args:
            n_classes: Number of output classes
            backbone: Which pretrained model to use 
                      Options: "resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b2"
            pretrained: Whether to use pretrained ImageNet weights
            freeze_backbone: If True, freeze backbone weights (only train classifier)
            dropout: Dropout rate for classifier layers
        """
        super().__init__()
        
        self.backbone_name = backbone
        
        # Load pretrained backbone
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            n_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original classifier
            
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            n_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            n_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            n_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b2(weights=weights)
            n_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classifier head with configurable dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),  # Slightly lower for second layer
            nn.Linear(256, n_classes),
        )
        self.dropout = dropout
        
        self.n_features = n_features
    
    def forward(self, x):
        # x shape: (batch, n_mels, n_frames) e.g., (16, 128, 173)
        
        # Add channel dimension: (batch, 1, n_mels, n_frames)
        x = x.unsqueeze(1)
        
        # Convert 1-channel to 3-channel by repeating
        # (batch, 3, n_mels, n_frames)
        x = x.repeat(1, 3, 1, 1)
        
        # Extract features
        features = self.backbone(x)
        
        # Classify
        out = self.classifier(features)
        return out
    
    def unfreeze_backbone(self, unfreeze_layers: int = None):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            unfreeze_layers: Number of layers from the end to unfreeze.
                            If None, unfreeze all layers.
        """
        if unfreeze_layers is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Get all named parameters
            params = list(self.backbone.named_parameters())
            # Unfreeze last N layers
            for name, param in params[-unfreeze_layers:]:
                param.requires_grad = True
    
    def get_trainable_params(self):
        """Return count of trainable vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total


class TransferCNNv2(nn.Module):
    """
    Alternative: Modify first conv layer to accept 1 channel directly.
    This preserves pretrained weights better for the rest of the network.
    """
    
    def __init__(
        self, 
        n_classes: int = 10, 
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.backbone_name = backbone
        
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            n_features = self.backbone.fc.in_features
            
            # Modify first conv to accept 1 channel
            # Average the pretrained weights across input channels
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                # Initialize with mean of RGB weights
                self.backbone.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            
            self.backbone.fc = nn.Identity()
            
        else:
            raise ValueError(f"TransferCNNv2 only supports resnet18 for now, got: {backbone}")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Always keep first conv trainable since we modified it
            for param in self.backbone.conv1.parameters():
                param.requires_grad = True
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )
        
        self.n_features = n_features
    
    def forward(self, x):
        # x shape: (batch, n_mels, n_frames)
        x = x.unsqueeze(1)  # (batch, 1, n_mels, n_frames)
        features = self.backbone(x)
        out = self.classifier(features)
        return out


if __name__ == "__main__":
    # Quick test
    model = TransferCNN(n_classes=10, backbone="resnet18", pretrained=True)
    trainable, total = model.get_trainable_params()
    print(f"TransferCNN (resnet18): {trainable:,} / {total:,} trainable params")
    
    # Test forward pass
    x = torch.randn(4, 128, 173)  # (batch, n_mels, n_frames)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    
    # Test frozen version
    model_frozen = TransferCNN(n_classes=10, backbone="resnet18", freeze_backbone=True)
    trainable, total = model_frozen.get_trainable_params()
    print(f"TransferCNN (frozen): {trainable:,} / {total:,} trainable params")

