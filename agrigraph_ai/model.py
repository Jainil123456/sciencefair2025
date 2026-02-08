"""
Graph Convolutional Network (GCN) model for soil gas risk prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional


class AgriGraphGCN(nn.Module):
    """
    Graph Convolutional Network for soil gas risk prediction.
    
    Architecture:
    - Input: Node features (NH3, CH4, NO2, CO, x, y)
    - Hidden: Multiple GCN layers with ReLU activation
    - Output: Single risk score per node (regression)
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        output_dim: int = 1
    ):
        """
        Initialize GCN model.
        
        Args:
            input_dim: Number of input features per node
            hidden_dim: Hidden dimension for GCN layers
            num_layers: Number of GCN layers
            dropout: Dropout rate
            output_dim: Output dimension (1 for regression)
        """
        super(AgriGraphGCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.conv1 = GCNConv(input_dim, hidden_dim)
        
        # Hidden layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.conv_out = GCNConv(hidden_dim, output_dim)
        
        # Batch normalization layers (optional, can improve training)
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GCN.
        
        Args:
            x: Node features, shape (num_nodes, input_dim)
            edge_index: Graph edge indices, shape (2, num_edges)
            
        Returns:
            Risk scores, shape (num_nodes, output_dim)
        """
        # Input layer
        x = self.conv1(x, edge_index)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = self.batch_norms[i + 1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation for regression)
        x = self.conv_out(x, edge_index)
        
        # Apply sigmoid to ensure output is in [0, 1] range
        x = torch.sigmoid(x)
        
        return x
    
    def reset_parameters(self):
        """Reset all parameters."""
        self.conv1.reset_parameters()
        for conv in self.conv_layers:
            conv.reset_parameters()
        self.conv_out.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


if __name__ == "__main__":
    # Test model
    model = AgriGraphGCN(input_dim=6, hidden_dim=64, num_layers=3)
    
    # Create dummy graph
    num_nodes = 50
    num_edges = 200
    x = torch.randn(num_nodes, 6)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Forward pass
    output = model(x, edge_index)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")







