"""
AgriGraph AI - Graph Neural Network for Soil Gas Risk Prediction
"""

__version__ = "1.0.0"

from .model import AgriGraphGCN
from .data_generation import generate_synthetic_dataset
from .graph_construction import construct_graph
from .training import train_model
from .visualization import plot_field_map, plot_risk_heatmap, plot_training_history
from .interpretation import generate_alerts, alerts_to_json, print_alerts_summary

__all__ = [
    'AgriGraphGCN',
    'generate_synthetic_dataset',
    'construct_graph',
    'train_model',
    'plot_field_map',
    'plot_risk_heatmap',
    'plot_training_history',
    'generate_alerts',
    'alerts_to_json',
    'print_alerts_summary'
]







