"""
Data loading utilities for cat sound classification.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class AudioDataset(Dataset):
    """Dataset with optional transforms."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]


def load_train_data(data_dir: Path):
    """Load preprocessed training data."""
    X = np.load(data_dir / "X_train.npy")
    y = np.load(data_dir / "y_train.npy")
    return X, y


def load_test_data(data_dir: Path):
    """Load preprocessed test data."""
    X = np.load(data_dir / "X_test.npy")
    y = np.load(data_dir / "y_test.npy")
    return X, y


def get_train_val_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 16,
    val_split: float = 0.2,
    random_state: int = 42,
    train_transform=None,
):
    """
    Create train and validation DataLoaders from training data.
    
    Args:
        X: Training features array
        y: Training labels array
        batch_size: Batch size
        val_split: Fraction for validation (from training set)
        random_state: Random seed
        train_transform: Transform to apply to training data
    
    Returns:
        (train_loader, val_loader)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, stratify=y, random_state=random_state
    )
    
    train_dataset = AudioDataset(X_train, y_train, transform=train_transform)
    val_dataset = AudioDataset(X_val, y_val, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def get_test_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 16):
    """
    Create test DataLoader.
    
    Args:
        X: Test features array
        y: Test labels array
        batch_size: Batch size
    
    Returns:
        test_loader
    """
    dataset = AudioDataset(X, y, transform=None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
