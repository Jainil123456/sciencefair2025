"""
Synthetic IoT sensor data generation for farm field simulation.
Simulates drone-collected sensor data with spatial correlation.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal


def generate_sensor_locations(
    num_nodes: int,
    field_width: float,
    field_height: float,
    random_seed: int = 42
) -> np.ndarray:
    """
    Generate random sensor locations across the field.
    
    Args:
        num_nodes: Number of sensor locations
        field_width: Width of the field in meters
        field_height: Height of the field in meters
        random_seed: Random seed for reproducibility
        
    Returns:
        Array of shape (num_nodes, 2) with (x, y) coordinates
    """
    np.random.seed(random_seed)
    x_coords = np.random.uniform(0, field_width, num_nodes)
    y_coords = np.random.uniform(0, field_height, num_nodes)
    return np.column_stack([x_coords, y_coords])


def generate_gas_concentrations(
    locations: np.ndarray,
    gas_ranges: dict,
    num_timesteps: int = 1,
    spatial_correlation: float = 0.7,
    random_seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic gas concentrations with spatial correlation.
    
    Args:
        locations: Array of (x, y) coordinates, shape (num_nodes, 2)
        gas_ranges: Dictionary with gas names and (min, max) ranges
        num_timesteps: Number of time steps
        spatial_correlation: Spatial correlation strength (0-1)
        random_seed: Random seed for reproducibility
        
    Returns:
        Array of shape (num_nodes, num_timesteps, num_gases) with gas concentrations
    """
    np.random.seed(random_seed)
    num_nodes = locations.shape[0]
    num_gases = len(gas_ranges)
    gas_names = list(gas_ranges.keys())
    
    # Compute distance matrix for spatial correlation
    distances = cdist(locations, locations)
    max_dist = np.max(distances)
    # Convert distances to correlation matrix
    correlation_matrix = np.exp(-distances / (max_dist * (1 - spatial_correlation)))
    
    # Generate spatially correlated gas concentrations
    gas_data = np.zeros((num_nodes, num_timesteps, num_gases))
    
    for gas_idx, gas_name in enumerate(gas_names):
        min_val, max_val = gas_ranges[gas_name]
        mean_val = (min_val + max_val) / 2
        
        # Generate correlated random values
        for t in range(num_timesteps):
            # Sample from multivariate normal with spatial correlation
            try:
                samples = np.random.multivariate_normal(
                    mean=np.zeros(num_nodes),
                    cov=correlation_matrix
                )
            except np.linalg.LinAlgError as e:
                # Fallback if correlation matrix is not positive definite
                print(f"⚠️  Warning: Correlation matrix not positive definite. Using independent samples.")
                print(f"   Error: {e}")
                samples = np.random.normal(0, 1, num_nodes)
            
            # Normalize and scale to gas range
            samples = (samples - samples.mean()) / (samples.std() + 1e-8)
            concentrations = mean_val + samples * (max_val - min_val) / 4
            concentrations = np.clip(concentrations, min_val, max_val)
            
            gas_data[:, t, gas_idx] = concentrations
    
    return gas_data


def generate_synthetic_risk_labels(
    gas_data: np.ndarray,
    locations: np.ndarray,
    gas_ranges: dict,
    spatial_smoothing: float = 0.5
) -> np.ndarray:
    """
    Generate synthetic risk labels based on gas concentrations and spatial patterns.
    
    Args:
        gas_data: Array of shape (num_nodes, num_timesteps, num_gases)
        locations: Array of (x, y) coordinates
        gas_ranges: Dictionary with gas ranges
        spatial_smoothing: Spatial smoothing factor for risk calculation
        
    Returns:
        Array of shape (num_nodes,) with risk scores (0-1)
    """
    num_nodes = gas_data.shape[0]
    num_timesteps = gas_data.shape[1]
    num_gases = gas_data.shape[2]
    
    # Use latest timestep for risk calculation
    current_gas = gas_data[:, -1, :]  # Shape: (num_nodes, num_gases)
    
    # Normalize each gas to [0, 1] based on its range
    normalized_gas = np.zeros_like(current_gas)
    gas_names = list(gas_ranges.keys())
    
    for gas_idx, gas_name in enumerate(gas_names):
        min_val, max_val = gas_ranges[gas_name]
        normalized_gas[:, gas_idx] = (current_gas[:, gas_idx] - min_val) / (max_val - min_val + 1e-8)
    
    # Weight different gases (NH3 and CH4 are more critical)
    gas_weights = np.array([0.3, 0.3, 0.2, 0.2])  # NH3, CH4, NO2, CO
    
    # Calculate base risk from normalized concentrations
    base_risk = np.dot(normalized_gas, gas_weights)
    
    # Apply spatial smoothing using distance-weighted average
    distances = cdist(locations, locations)
    # Use inverse distance weighting (with small epsilon to avoid division by zero)
    weights = 1.0 / (distances + 1e-6)
    np.fill_diagonal(weights, 0)  # Remove self-connections
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
    
    # Smooth risk scores spatially
    smoothed_risk = (1 - spatial_smoothing) * base_risk + spatial_smoothing * np.dot(weights, base_risk)
    
    # Add some non-linear effects (higher concentrations increase risk exponentially)
    smoothed_risk = np.power(smoothed_risk, 0.8)  # Slight non-linearity
    
    # Normalize to [0, 1]
    risk_scores = np.clip(smoothed_risk, 0.0, 1.0)
    
    return risk_scores


def generate_synthetic_dataset(
    num_nodes: int = 150,
    field_width: float = 100.0,
    field_height: float = 100.0,
    num_timesteps: int = 1,
    gas_ranges: Optional[dict] = None,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate complete synthetic dataset for farm field.
    
    Args:
        num_nodes: Number of sensor locations
        field_width: Width of field in meters
        field_height: Height of field in meters
        num_timesteps: Number of time steps
        gas_ranges: Dictionary with gas ranges (uses default if None)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (locations, features, risk_labels)
        - locations: (num_nodes, 2) array of (x, y) coordinates
        - features: (num_nodes, num_features) array with [NH3, CH4, NO2, CO, x, y]
        - risk_labels: (num_nodes,) array of risk scores
    """
    if gas_ranges is None:
        from .config import Config
        gas_ranges = Config.GAS_RANGES
    
    # Generate sensor locations
    locations = generate_sensor_locations(num_nodes, field_width, field_height, random_seed)
    
    # Generate gas concentrations
    gas_data = generate_gas_concentrations(
        locations, gas_ranges, num_timesteps, 
        spatial_correlation=0.7, random_seed=random_seed
    )
    
    # Generate risk labels
    risk_labels = generate_synthetic_risk_labels(gas_data, locations, gas_ranges)
    
    # Combine features: [NH3, CH4, NO2, CO, x, y]
    # Use latest timestep for features
    current_gas = gas_data[:, -1, :]  # Shape: (num_nodes, num_gases)
    features = np.column_stack([current_gas, locations])
    
    return locations, features, risk_labels


if __name__ == "__main__":
    # Test data generation
    try:
        from .config import Config
        gas_ranges = Config.GAS_RANGES
    except ImportError:
        # Fallback if running directly
        gas_ranges = {
            'NH3': (0.0, 50.0),
            'CH4': (0.0, 100.0),
            'NO2': (0.0, 5.0),
            'CO': (0.0, 10.0)
        }
    
    locations, features, labels = generate_synthetic_dataset(num_nodes=50, gas_ranges=gas_ranges)
    print(f"Generated {len(locations)} sensor locations")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Risk score range: [{labels.min():.3f}, {labels.max():.3f}]")

