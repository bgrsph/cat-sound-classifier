"""
Testing function for cat sound classification.
"""

import torch
import torch.nn as nn
import numpy as np


def test(
    model: nn.Module,
    test_loader,
    device: str = None,
    verbose: bool = True,
):
    """
    Evaluate a model.
    
    Args:
        model: PyTorch model
        test_loader: Test DataLoader
        device: Device (auto-detected if None)
        verbose: Print results
    
    Returns:
        dict with accuracy, predictions, and labels
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    
    if verbose:
        print(f"Test Loss: {total_loss / len(test_loader):.4f}")
        print(f"Test Accuracy: {accuracy:.3f}")
    
    return {
        "accuracy": accuracy,
        "predictions": all_preds,
        "labels": all_labels,
        "loss": total_loss / len(test_loader),
    }
