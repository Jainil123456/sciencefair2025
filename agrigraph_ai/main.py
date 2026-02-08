"""
Main entry point for AgriGraph AI.
Orchestrates the complete pipeline from data generation to alert production.
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path

from .config import Config
from .data_generation import generate_synthetic_dataset
from .graph_construction import construct_graph
from .model import AgriGraphGCN
from .training import train_model, save_model
from .visualization import plot_field_map, plot_risk_heatmap, plot_training_history
from .interpretation import generate_alerts, alerts_to_json, print_alerts_summary


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='AgriGraph AI - Soil Gas Risk Prediction')
    parser.add_argument('--num-nodes', type=int, default=Config.NUM_NODES,
                       help='Number of sensor locations')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=Config.HIDDEN_DIM,
                       help='Hidden dimension for GCN layers')
    parser.add_argument('--num-layers', type=int, default=Config.NUM_LAYERS,
                       help='Number of GCN layers')
    parser.add_argument('--k-neighbors', type=int, default=Config.K_NEIGHBORS,
                       help='Number of neighbors for k-NN graph')
    parser.add_argument('--output-dir', type=str, default=Config.OUTPUT_DIR,
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=Config.RANDOM_SEED,
                       help='Random seed')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only generate visualizations')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model (for inference only)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("AgriGraph AI - Soil Gas Risk Prediction System")
    print("="*80)
    
    # Step 1: Generate synthetic data
    print("\n[1/7] Generating synthetic sensor data...")
    locations, features, risk_labels = generate_synthetic_dataset(
        num_nodes=args.num_nodes,
        field_width=Config.FIELD_WIDTH,
        field_height=Config.FIELD_HEIGHT,
        num_timesteps=Config.NUM_TIMESTEPS,
        gas_ranges=Config.GAS_RANGES,
        random_seed=args.seed
    )
    print(f"  Generated {len(locations)} sensor locations")
    print(f"  Features shape: {features.shape}")
    print(f"  Risk score range: [{risk_labels.min():.3f}, {risk_labels.max():.3f}]")
    
    # Step 2: Construct graph
    print("\n[2/7] Constructing spatial graph...")
    graph = construct_graph(
        locations, features, risk_labels,
        method='knn',
        k=args.k_neighbors,
        use_distance_weights=Config.USE_DISTANCE_WEIGHTS
    )
    print(f"  Graph nodes: {graph.x.shape[0]}")
    print(f"  Graph edges: {graph.edge_index.shape[1]}")
    print(f"  Average degree: {graph.edge_index.shape[1] / graph.x.shape[0]:.2f}")
    
    # Step 3: Initialize model
    print("\n[3/7] Initializing GCN model...")
    model = AgriGraphGCN(
        input_dim=Config.INPUT_FEATURES,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=Config.DROPOUT,
        output_dim=Config.OUTPUT_DIM
    )
    print(f"  Model architecture: {Config.INPUT_FEATURES} -> {args.hidden_dim} (x{args.num_layers}) -> {Config.OUTPUT_DIM}")
    
    # Step 4: Train model
    if not args.skip_training:
        print("\n[4/7] Training model...")
        trained_model, history, test_metrics = train_model(
            model, graph,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            train_split=Config.TRAIN_SPLIT,
            val_split=Config.VAL_SPLIT,
            test_split=Config.TEST_SPLIT,
            early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
            early_stopping_min_delta=Config.EARLY_STOPPING_MIN_DELTA,
            random_seed=args.seed,
            verbose=True
        )
        
        # Save model
        model_path = output_dir / 'model.pt'
        save_model(trained_model, str(model_path))
        print(f"  Model saved to {model_path}")
    else:
        # Load model if provided
        if args.model_path:
            from .training import load_model
            model = load_model(model, args.model_path)
            print(f"  Model loaded from {args.model_path}")
        trained_model = model
        history = None
        test_metrics = None
    
    # Step 5: Generate predictions
    print("\n[5/7] Generating predictions...")
    trained_model.eval()
    with torch.no_grad():
        predictions = trained_model(graph.x, graph.edge_index)
        predictions = predictions.cpu().numpy().flatten()
    
    print(f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    # Step 6: Visualize results
    print("\n[6/7] Generating visualizations...")
    
    # Field map with predictions
    plot_field_map(
        locations, predictions, true_labels=risk_labels,
        field_width=Config.FIELD_WIDTH,
        field_height=Config.FIELD_HEIGHT,
        title="AgriGraph AI - Risk Prediction Results",
        save_path=str(output_dir / 'risk_prediction_map.png'),
        show=False,
        figsize=Config.FIGURE_SIZE,
        dpi=Config.DPI
    )
    
    # Risk heatmap
    plot_risk_heatmap(
        locations, predictions,
        field_width=Config.FIELD_WIDTH,
        field_height=Config.FIELD_HEIGHT,
        title="Risk Heatmap - Interpolated View",
        save_path=str(output_dir / 'risk_heatmap.png'),
        show=False,
        figsize=Config.FIGURE_SIZE,
        dpi=Config.DPI
    )
    
    # Training history (if available)
    if history is not None:
        plot_training_history(
            history,
            save_path=str(output_dir / 'training_history.png'),
            show=False,
            figsize=(15, 5),
            dpi=Config.DPI
        )
    
    print(f"  Visualizations saved to {output_dir}")
    
    # Step 7: Generate alerts
    print("\n[7/7] Generating risk alerts...")
    
    # Extract gas features (first 4 columns)
    gas_features = features[:, :4]
    gas_names = list(Config.GAS_RANGES.keys())
    
    alerts = generate_alerts(
        locations, predictions, gas_features,
        gas_names, Config.GAS_RANGES, Config.RISK_THRESHOLDS
    )
    
    # Print summary
    print_alerts_summary(alerts, top_n=10)
    
    # Save alerts as JSON
    alerts_json_path = output_dir / 'alerts.json'
    alerts_to_json(alerts, str(alerts_json_path))
    
    # Save high-risk alerts separately
    high_risk_alerts = generate_alerts(
        locations, predictions, gas_features,
        gas_names, Config.GAS_RANGES, Config.RISK_THRESHOLDS,
        filter_by_level='high'
    )
    critical_alerts = generate_alerts(
        locations, predictions, gas_features,
        gas_names, Config.GAS_RANGES, Config.RISK_THRESHOLDS,
        filter_by_level='critical'
    )
    
    if high_risk_alerts or critical_alerts:
        urgent_alerts = high_risk_alerts + critical_alerts
        urgent_alerts.sort(key=lambda x: x.risk_score, reverse=True)
        alerts_to_json(urgent_alerts, str(output_dir / 'urgent_alerts.json'))
        print(f"\n  {len(urgent_alerts)} urgent alerts (high/critical) saved to {output_dir / 'urgent_alerts.json'}")
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print(f"All outputs saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()







