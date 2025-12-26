"""
Training function for cat sound classification.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = None,
    save_path: str = None,
    patience: int = None,
    verbose: bool = True,
):
    """
    Train a model.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on (auto-detected if None)
        save_path: Path to save best model (optional)
        patience: Early stopping patience (None to disable)
        verbose: Print progress
    
    Returns:
        dict with training history
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == y).sum().item()
            train_total += y.size(0)
        
        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == y).sum().item()
                val_total += y.size(0)
        
        # Record
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f}, {train_acc:.3f} | "
                  f"Val: {val_loss:.4f}, {val_acc:.3f}")
        
        # Save best & early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                if verbose:
                    print(f"  -> Saved (val_acc: {val_acc:.3f})")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if patience and epochs_without_improvement >= patience:
            if verbose:
                print(f"\nEarly stopping: no improvement for {patience} epochs")
            break
    
    if verbose:
        print(f"\nBest val_acc: {best_val_acc:.3f}")
    
    return history


def cross_validate(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 16,
    patience: int = None,
    device: str = None,
    verbose: bool = True,
    **model_kwargs,
):
    """
    K-fold cross-validation.
    
    Args:
        model_class: Model class to instantiate (e.g., CatMeowCNN)
        X: Features array (n_samples, ...)
        y: Labels array (n_samples,)
        n_splits: Number of folds
        epochs: Epochs per fold
        learning_rate: Learning rate
        batch_size: Batch size
        patience: Early stopping patience
        device: Device
        verbose: Print progress
        **model_kwargs: Additional args for model_class
    
    Returns:
        dict with fold results and summary
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{n_splits}")
            print(f"{'='*50}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=batch_size, shuffle=False
        )
        
        # Create fresh model for each fold
        model = model_class(**model_kwargs)
        
        # Train
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            patience=patience,
            verbose=verbose,
        )
        
        best_val_acc = max(history["val_acc"])
        fold_results.append({
            "fold": fold + 1,
            "best_val_acc": best_val_acc,
            "history": history,
        })
        
        if verbose:
            print(f"Fold {fold + 1} best val_acc: {best_val_acc:.3f}")
    
    # Summary
    accuracies = [r["best_val_acc"] for r in fold_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Cross-Validation Results")
        print(f"{'='*50}")
        for r in fold_results:
            print(f"  Fold {r['fold']}: {r['best_val_acc']:.3f}")
        print(f"\n  Mean: {mean_acc:.3f} ± {std_acc:.3f}")
    
    return {
        "fold_results": fold_results,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "accuracies": accuracies,
    }
