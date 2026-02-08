"""
Graph construction from sensor data.
Builds spatial graphs with distance-based or k-NN connectivity.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
from typing import Optional, Tuple


def build_knn_graph(
    locations: np.ndarray,
    k: int = 8,
    use_distance_weights: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build k-nearest neighbors graph from sensor locations.
    
    Args:
        locations: Array of (x, y) coordinates, shape (num_nodes, 2)
        k: Number of nearest neighbors
        use_distance_weights: If True, use distance as edge weights
        
    Returns:
        Tuple of (edge_index, edge_attr)
        - edge_index: (2, num_edges) array of edge connections
        - edge_attr: (num_edges, 1) array of edge weights (if use_distance_weights)
    """
    num_nodes = locations.shape[0]
    
    # Build k-NN graph using sklearn
    knn_graph = kneighbors_graph(
        locations, n_neighbors=k, mode='connectivity', 
        include_self=False, metric='euclidean'
    )
    
    # Convert to edge index format
    knn_graph = knn_graph.tocoo()
    edge_index = np.row_stack([knn_graph.row, knn_graph.col])
    
    # Compute edge weights (distances) if requested
    if use_distance_weights:
        edge_attr = np.zeros((edge_index.shape[1], 1))
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            dist = np.linalg.norm(locations[src] - locations[dst])
            edge_attr[i, 0] = dist
    else:
        edge_attr = np.ones((edge_index.shape[1], 1))
    
    return edge_index, edge_attr


def build_distance_threshold_graph(
    locations: np.ndarray,
    threshold: float = 20.0,
    use_distance_weights: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build graph using distance threshold.
    
    Args:
        locations: Array of (x, y) coordinates, shape (num_nodes, 2)
        threshold: Maximum distance for edge connection
        use_distance_weights: If True, use distance as edge weights
        
    Returns:
        Tuple of (edge_index, edge_attr)
    """
    # Compute distance matrix
    distances = cdist(locations, locations)
    
    # Create edges for distances below threshold
    mask = (distances < threshold) & (distances > 0)  # Exclude self-connections
    rows, cols = np.where(mask)
    
    edge_index = np.row_stack([rows, cols])
    
    # Compute edge weights
    if use_distance_weights:
        edge_attr = distances[mask].reshape(-1, 1)
    else:
        edge_attr = np.ones((edge_index.shape[1], 1))
    
    return edge_index, edge_attr


def build_irrigation_graph(
    locations: np.ndarray,
    irrigation_network: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build graph based on irrigation connectivity.
    
    Args:
        locations: Array of (x, y) coordinates
        irrigation_network: Optional array of irrigation connections
        
    Returns:
        Tuple of (edge_index, edge_attr)
    """
    # Placeholder for irrigation network connectivity
    # If no network provided, return empty graph
    if irrigation_network is None:
        num_nodes = locations.shape[0]
        return np.zeros((2, 0), dtype=int), np.zeros((0, 1))
    
    # Use provided irrigation network
    edge_index = irrigation_network
    edge_attr = np.ones((edge_index.shape[1], 1))
    
    return edge_index, edge_attr


def construct_graph(
    locations: np.ndarray,
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'knn',
    k: int = 8,
    distance_threshold: float = 20.0,
    use_distance_weights: bool = True,
    use_irrigation: bool = False,
    irrigation_network: Optional[np.ndarray] = None
) -> Data:
    """
    Construct PyTorch Geometric Data object from sensor data.
    
    Args:
        locations: Array of (x, y) coordinates, shape (num_nodes, 2)
        features: Node features, shape (num_nodes, num_features)
        labels: Optional risk labels, shape (num_nodes,)
        method: Graph construction method ('knn' or 'distance')
        k: Number of neighbors for k-NN
        distance_threshold: Distance threshold for distance method
        use_distance_weights: Use distance as edge weights
        use_irrigation: Use irrigation connectivity
        irrigation_network: Optional irrigation network connections
        
    Returns:
        PyTorch Geometric Data object
    """
    # Build graph edges
    if use_irrigation and irrigation_network is not None:
        edge_index, edge_attr = build_irrigation_graph(locations, irrigation_network)
    elif method == 'knn':
        edge_index, edge_attr = build_knn_graph(
            locations, k=k, use_distance_weights=use_distance_weights
        )
    elif method == 'distance':
        edge_index, edge_attr = build_distance_threshold_graph(
            locations, threshold=distance_threshold, 
            use_distance_weights=use_distance_weights
        )
    else:
        raise ValueError(f"Unknown graph construction method: {method}")
    
    # Convert to tensors
    edge_index = torch.from_numpy(edge_index).long()
    edge_attr = torch.from_numpy(edge_attr).float()
    node_features = torch.from_numpy(features).float()
    
    # Create Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    # Add labels if provided
    if labels is not None:
        data.y = torch.from_numpy(labels).float()
    
    # Store locations for visualization
    data.locations = torch.from_numpy(locations).float()
    
    return data


if __name__ == "__main__":
    # Test graph construction
    try:
        from .data_generation import generate_synthetic_dataset
    except ImportError:
        import sys
        sys.path.insert(0, '.')
        from data_generation import generate_synthetic_dataset
    
    locations, features, labels = generate_synthetic_dataset(num_nodes=50)
    graph = construct_graph(locations, features, labels, method='knn', k=5)
    
    print(f"Graph nodes: {graph.x.shape[0]}")
    print(f"Graph features: {graph.x.shape[1]}")
    print(f"Graph edges: {graph.edge_index.shape[1]}")
    print(f"Has labels: {graph.y is not None}")

