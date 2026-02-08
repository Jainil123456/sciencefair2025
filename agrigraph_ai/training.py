"""
Training loop and evaluation for AgriGraph AI.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, Optional
import os

from .model import AgriGraphGCN, EarlyStopping


def split_data(
    data: torch.Tensor,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split data into train/val/test sets.
    
    Args:
        data: Data tensor to split
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_mask, val_mask, test_mask)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    num_nodes = data.shape[0]
    np.random.seed(random_seed)
    indices = np.random.permutation(num_nodes)
    
    train_end = int(num_nodes * train_split)
    val_end = train_end + int(num_nodes * val_split)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    return train_mask, val_mask, test_mask


def train_epoch(
    model: nn.Module,
    data: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_mask: torch.Tensor
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: GCN model
        data: PyTorch Geometric Data object
        optimizer: Optimizer
        criterion: Loss function
        train_mask: Boolean mask for training nodes
        
    Returns:
        Average training loss
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Compute loss only on training nodes
    loss = criterion(out[train_mask], data.y[train_mask].unsqueeze(1))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(
    model: nn.Module,
    data: torch.Tensor,
    criterion: nn.Module,
    mask: torch.Tensor
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model on given mask.
    
    Args:
        model: GCN model
        data: PyTorch Geometric Data object
        criterion: Loss function
        mask: Boolean mask for nodes to evaluate
        
    Returns:
        Tuple of (loss, predictions, targets)
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = criterion(out[mask], data.y[mask].unsqueeze(1))
        
        predictions = out[mask].cpu().numpy().flatten()
        targets = data.y[mask].cpu().numpy()
    
    return loss.item(), predictions, targets


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted values
        targets: True values
        
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def train_model(
    model: nn.Module,
    data: torch.Tensor,
    num_epochs: int = 200,
    learning_rate: float = 0.001,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 0.001,
    random_seed: int = 42,
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, list], Dict[str, float]]:
    """
    Train the GCN model.
    
    Args:
        model: GCN model to train
        data: PyTorch Geometric Data object
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        early_stopping_patience: Early stopping patience
        early_stopping_min_delta: Early stopping minimum delta
        random_seed: Random seed
        verbose: Print training progress
        
    Returns:
        Tuple of (trained_model, training_history, test_metrics)
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Split data
    train_mask, val_mask, test_mask = split_data(
        data.y, train_split, val_split, test_split, random_seed
    )
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, data, optimizer, criterion, train_mask)
        
        # Validate
        val_loss, val_pred, val_target = evaluate(model, data, criterion, val_mask)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Compute metrics
        train_pred, train_target = None, None
        with torch.no_grad():
            model.eval()
            out = model(data.x, data.edge_index)
            train_pred = out[train_mask].cpu().numpy().flatten()
            train_target = data.y[train_mask].cpu().numpy()
        
        train_metrics = compute_metrics(train_pred, train_target)
        val_metrics = compute_metrics(val_pred, val_target)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val R²: {val_metrics['r2']:.4f}")
        
        # Early stopping
        if early_stopping(val_loss):
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    test_loss, test_pred, test_target = evaluate(model, data, criterion, test_mask)
    test_metrics = compute_metrics(test_pred, test_target)
    
    if verbose:
        print("\n" + "="*50)
        print("Test Set Results:")
        print(f"MSE: {test_metrics['mse']:.4f}")
        print(f"MAE: {test_metrics['mae']:.4f}")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        print(f"R²: {test_metrics['r2']:.4f}")
        print("="*50)
    
    return model, history, test_metrics


def save_model(model: nn.Module, filepath: str):
    """Save model to file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load_model(model: nn.Module, filepath: str) -> nn.Module:
    """Load model from file."""
    model.load_state_dict(torch.load(filepath))
    return model

