"""
Visualization of risk zones and predictions over the farm field.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
from typing import Optional, Tuple
import os


def plot_field_map(
    locations: np.ndarray,
    risk_scores: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    field_width: float = 100.0,
    field_height: float = 100.0,
    title: str = "Soil Gas Risk Prediction",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300
) -> plt.Figure:
    """
    Plot field map with risk zones color-coded by risk scores.
    
    Args:
        locations: Array of (x, y) coordinates, shape (num_nodes, 2)
        risk_scores: Predicted risk scores, shape (num_nodes,)
        true_labels: Optional true labels for comparison
        field_width: Width of field
        field_height: Height of field
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        figsize: Figure size
        dpi: DPI for saved figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2 if true_labels is not None else 1, 
                            figsize=figsize, dpi=dpi)
    
    if true_labels is not None:
        ax_pred, ax_true = axes
    else:
        ax_pred = axes
        ax_true = None
    
    # Create colormap for risk scores
    cmap = plt.cm.RdYlGn_r  # Red (high risk) to Green (low risk)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
    # Plot predicted risk
    scatter_pred = ax_pred.scatter(
        locations[:, 0], locations[:, 1],
        c=risk_scores, cmap=cmap, norm=norm,
        s=100, edgecolors='black', linewidths=0.5,
        alpha=0.8
    )
    ax_pred.set_xlim(0, field_width)
    ax_pred.set_ylim(0, field_height)
    ax_pred.set_xlabel('X Position (m)', fontsize=12)
    ax_pred.set_ylabel('Y Position (m)', fontsize=12)
    ax_pred.set_title('Predicted Risk Zones', fontsize=14, fontweight='bold')
    ax_pred.grid(True, alpha=0.3)
    ax_pred.set_aspect('equal')
    
    # Add colorbar
    cbar_pred = plt.colorbar(scatter_pred, ax=ax_pred)
    cbar_pred.set_label('Risk Score', fontsize=11)
    
    # Plot true labels if provided
    if ax_true is not None:
        scatter_true = ax_true.scatter(
            locations[:, 0], locations[:, 1],
            c=true_labels, cmap=cmap, norm=norm,
            s=100, edgecolors='black', linewidths=0.5,
            alpha=0.8
        )
        ax_true.set_xlim(0, field_width)
        ax_true.set_ylim(0, field_height)
        ax_true.set_xlabel('X Position (m)', fontsize=12)
        ax_true.set_ylabel('Y Position (m)', fontsize=12)
        ax_true.set_title('True Risk Zones', fontsize=14, fontweight='bold')
        ax_true.grid(True, alpha=0.3)
        ax_true.set_aspect('equal')
        
        cbar_true = plt.colorbar(scatter_true, ax=ax_true)
        cbar_true.set_label('Risk Score', fontsize=11)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_risk_heatmap(
    locations: np.ndarray,
    risk_scores: np.ndarray,
    field_width: float = 100.0,
    field_height: float = 100.0,
    resolution: int = 100,
    title: str = "Risk Heatmap",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300
) -> plt.Figure:
    """
    Create interpolated heatmap of risk zones.
    
    Args:
        locations: Array of (x, y) coordinates
        risk_scores: Risk scores
        field_width: Width of field
        field_height: Height of field
        resolution: Resolution of heatmap grid
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
        figsize: Figure size
        dpi: DPI for saved figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create interpolation grid
    xi = np.linspace(0, field_width, resolution)
    yi = np.linspace(0, field_height, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate risk scores
    zi = griddata(
        locations, risk_scores,
        (xi_grid, yi_grid),
        method='cubic',
        fill_value=risk_scores.mean()
    )
    
    # Plot heatmap
    cmap = plt.cm.RdYlGn_r
    im = ax.contourf(xi_grid, yi_grid, zi, levels=20, cmap=cmap, alpha=0.8)
    ax.contour(xi_grid, yi_grid, zi, levels=20, colors='black', alpha=0.2, linewidths=0.5)
    
    # Overlay sensor locations
    scatter = ax.scatter(
        locations[:, 0], locations[:, 1],
        c=risk_scores, cmap=cmap, s=50,
        edgecolors='black', linewidths=1,
        zorder=5
    )
    
    ax.set_xlim(0, field_width)
    ax.set_ylim(0, field_height)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Risk Score', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (15, 5),
    dpi: int = 300
) -> plt.Figure:
    """
    Plot training history (loss and metrics).
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        show: Whether to display the plot
        figsize: Figure size
        dpi: DPI for saved figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss (MSE)', fontsize=11)
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot R²
    ax2 = axes[1]
    train_r2 = [m['r2'] for m in history['train_metrics']]
    val_r2 = [m['r2'] for m in history['val_metrics']]
    ax2.plot(epochs, train_r2, 'b-', label='Train R²', linewidth=2)
    ax2.plot(epochs, val_r2, 'r-', label='Val R²', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('R² Score', fontsize=11)
    ax2.set_title('R² Score', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot RMSE
    ax3 = axes[2]
    train_rmse = [m['rmse'] for m in history['train_metrics']]
    val_rmse = [m['rmse'] for m in history['val_metrics']]
    ax3.plot(epochs, train_rmse, 'b-', label='Train RMSE', linewidth=2)
    ax3.plot(epochs, val_rmse, 'r-', label='Val RMSE', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('RMSE', fontsize=11)
    ax3.set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig







