"""
Configuration parameters for AgriGraph AI.
Centralized configuration for all components.
"""

class Config:
    """Configuration class for AgriGraph AI system."""
    
    # Data Generation Parameters
    NUM_NODES = 150  # Number of sensor sampling locations
    FIELD_WIDTH = 100.0  # Field width in meters
    FIELD_HEIGHT = 100.0  # Field height in meters
    NUM_TIMESTEPS = 1  # Number of time steps (1 for static, >1 for time-series)
    
    # Gas concentration ranges (in ppm)
    GAS_RANGES = {
        'NH3': (0.0, 50.0),   # Ammonia
        'CH4': (0.0, 100.0),  # Methane
        'NO2': (0.0, 5.0),    # Nitrogen dioxide
        'CO': (0.0, 10.0)     # Carbon monoxide
    }
    
    # Graph Construction Parameters
    K_NEIGHBORS = 8  # Number of nearest neighbors for k-NN graph
    DISTANCE_THRESHOLD = 20.0  # Maximum distance for edge connection (meters)
    USE_DISTANCE_WEIGHTS = True  # Use distance as edge weights
    USE_IRRIGATION_CONNECTIVITY = False  # Use irrigation network (if available)
    
    # Model Architecture Parameters
    INPUT_FEATURES = 6  # NH3, CH4, NO2, CO, x, y (time optional)
    HIDDEN_DIM = 128  # Hidden dimension for GCN layers
    NUM_LAYERS = 3  # Number of GCN layers
    DROPOUT = 0.3  # Dropout rate
    OUTPUT_DIM = 1  # Single risk score output (regression)
    
    # Training Parameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 200
    BATCH_SIZE = 1  # For single graph, batch size is 1
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    EARLY_STOPPING_PATIENCE = 20
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Risk Thresholds for Interpretation
    RISK_THRESHOLDS = {
        'low': 0.3,
        'medium': 0.5,
        'high': 0.7,
        'critical': 0.9
    }
    
    # Visualization Parameters
    FIGURE_SIZE = (12, 10)
    DPI = 300
    OUTPUT_DIR = 'outputs'
    
    # Random seed for reproducibility and data variance control
    # Seed Modes:
    # - 'auto': Generates new seed each run (timestamp-based, different data each time)
    # - 'fixed': Uses seed=42 (same data every run, reproducible)
    # - 'custom': User-provided seed value (reproducible, user-controlled)
    RANDOM_SEED = None  # None means use auto mode by default
    ENABLE_SEED_CONTROL = True  # Allow users to control seed via UI
    DEFAULT_SEED_MODE = 'auto'  # Default to auto mode for data variance







