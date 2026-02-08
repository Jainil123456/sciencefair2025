"""
Web application for visualizing AgriGraph AI results.
Serves interactive visualizations on localhost.
"""

import os
import json
import numpy as np
import torch
from flask import Flask, render_template, jsonify, send_from_directory, request
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Set template folder path before creating Flask app
template_dir = Path(__file__).parent.parent / 'templates'
template_dir.mkdir(exist_ok=True)

from .config import Config
from .data_generation import generate_synthetic_dataset
from .graph_construction import construct_graph
from .model import AgriGraphGCN
from .training import train_model, load_model
from .interpretation import generate_alerts

app = Flask(__name__, template_folder=str(template_dir))
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Global variables to store results
results_data = {
    'locations': None,
    'features': None,
    'risk_labels': None,
    'predictions': None,
    'graph': None,
    'model': None,
    'history': None,
    'test_metrics': None,
    'alerts': None
}


def create_field_map_plotly(locations, risk_scores, true_labels=None, field_width=100, field_height=100):
    """Create interactive Plotly field map."""
    fig = go.Figure()
    
    # Convert numpy arrays to lists for JSON serialization
    x_coords = locations[:, 0].tolist()
    y_coords = locations[:, 1].tolist()
    risk_list = risk_scores.tolist() if hasattr(risk_scores, 'tolist') else list(risk_scores)
    
    # Add predicted risk scatter
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(
            size=12,
            color=risk_list,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Risk Score", x=1.02, len=0.7),
            cmin=0,
            cmax=1,
            line=dict(width=0.5, color='rgba(0,0,0,0.3)')
        ),
        text=[f"Location {i}<br>Risk: {score:.3f}" for i, score in enumerate(risk_list)],
        hovertemplate='<b>Location %{text}</b><br>X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Risk: %{marker.color:.3f}<extra></extra>',
        name='Predicted Risk',
        showlegend=True
    ))
    
    # Add true labels if available
    if true_labels is not None:
        true_list = true_labels.tolist() if hasattr(true_labels, 'tolist') else list(true_labels)
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=10,
                color=true_list,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="True Risk", x=1.15, len=0.7),
                cmin=0,
                cmax=1,
                symbol='square',
                line=dict(width=0.5, color='rgba(0,0,0,0.3)')
            ),
            text=[f"Location {i}<br>True Risk: {score:.3f}" for i, score in enumerate(true_list)],
            hovertemplate='<b>Location %{text}</b><br>X: %{x:.1f} m<br>Y: %{y:.1f} m<br>True Risk: %{marker.color:.3f}<extra></extra>',
            name='True Risk',
            visible='legendonly',
            showlegend=True
        ))
    
    fig.update_layout(
        title=dict(
            text='Field Map - Risk Prediction',
            font=dict(size=18, color='#333')
        ),
        xaxis=dict(
            title='X Position (m)',
            range=[-5, field_width + 5],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False
        ),
        yaxis=dict(
            title='Y Position (m)',
            range=[-5, field_height + 5],
            scaleanchor="x",
            scaleratio=1,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False
        ),
        hovermode='closest',
        template='plotly_white',
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        autosize=True
    )
    
    return fig


def create_heatmap_plotly(locations, risk_scores, field_width=100, field_height=100, resolution=50):
    """Create interactive heatmap."""
    from scipy.interpolate import griddata
    
    # Convert to lists
    x_coords = locations[:, 0].tolist()
    y_coords = locations[:, 1].tolist()
    risk_list = risk_scores.tolist() if hasattr(risk_scores, 'tolist') else list(risk_scores)
    
    # Create interpolation grid
    xi = np.linspace(0, field_width, resolution)
    yi = np.linspace(0, field_height, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate risk scores
    zi = griddata(
        locations, risk_scores,
        (xi_grid, yi_grid),
        method='cubic',
        fill_value=np.mean(risk_scores)
    )
    
    fig = go.Figure(data=go.Contour(
        z=zi.tolist(),
        x=xi.tolist(),
        y=yi.tolist(),
        colorscale='RdYlGn_r',
        colorbar=dict(title="Risk Score", x=1.02, len=0.7),
        contours=dict(
            start=0,
            end=1,
            size=0.05,
            showlines=True,
            coloring='heatmap'
        ),
        hovertemplate='<b>Interpolated Risk</b><br>X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Risk: %{z:.3f}<extra></extra>',
        name='Risk Heatmap'
    ))
    
    # Add sensor locations
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(size=8, color='black', symbol='x', line=dict(width=1, color='white')),
        name='Sensor Locations',
        hovertemplate='<b>Sensor Location</b><br>X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>',
        showlegend=True
    ))
    
    fig.update_layout(
        title=dict(
            text='Risk Heatmap - Interpolated View',
            font=dict(size=18, color='#333')
        ),
        xaxis=dict(
            title='X Position (m)',
            range=[-5, field_width + 5],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False
        ),
        yaxis=dict(
            title='Y Position (m)',
            range=[-5, field_height + 5],
            scaleanchor="x",
            scaleratio=1,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False
        ),
        template='plotly_white',
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        autosize=True
    )
    
    return fig


def create_training_history_plotly(history):
    """Create training history plots."""
    if history is None:
        return None
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig = go.Figure()
    
    # Loss plot
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['train_loss'],
        mode='lines',
        name='Train Loss',
        line=dict(color='#3b82f6', width=2.5),
        hovertemplate='Epoch: %{x}<br>Train Loss: %{y:.4f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['val_loss'],
        mode='lines',
        name='Val Loss',
        line=dict(color='#ef4444', width=2.5),
        hovertemplate='Epoch: %{x}<br>Val Loss: %{y:.4f}<extra></extra>'
    ))
    
    # Add R² scores if available
    if history.get('train_metrics') and history.get('val_metrics'):
        train_r2 = [m['r2'] for m in history['train_metrics']]
        val_r2 = [m['r2'] for m in history['val_metrics']]
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=train_r2,
            mode='lines',
            name='Train R²',
            line=dict(color='#10b981', width=2, dash='dash'),
            yaxis='y2',
            hovertemplate='Epoch: %{x}<br>Train R²: %{y:.4f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_r2,
            mode='lines',
            name='Val R²',
            line=dict(color='#f59e0b', width=2, dash='dash'),
            yaxis='y2',
            hovertemplate='Epoch: %{x}<br>Val R²: %{y:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='Training History',
            font=dict(size=18, color='#333')
        ),
        xaxis=dict(
            title='Epoch',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title='Loss (MSE)',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis2=dict(
            title='R² Score',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=60, r=80, t=60, b=60),
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        autosize=True
    )
    
    return fig


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/data')
def get_data():
    """Get all visualization data."""
    if results_data['locations'] is None:
        return jsonify({'error': 'No data available. Please run analysis first.'}), 404
    
    data = {
        'locations': results_data['locations'].tolist(),
        'risk_scores': results_data['predictions'].tolist() if results_data['predictions'] is not None else None,
        'true_labels': results_data['risk_labels'].tolist() if results_data['risk_labels'] is not None else None,
        'field_width': Config.FIELD_WIDTH,
        'field_height': Config.FIELD_HEIGHT,
        'num_nodes': len(results_data['locations']),
        'test_metrics': results_data['test_metrics'],
        'alerts_count': {
            'total': len(results_data['alerts']) if results_data['alerts'] else 0,
            'critical': len([a for a in (results_data['alerts'] or []) if a['risk_level'] == 'critical']),
            'high': len([a for a in (results_data['alerts'] or []) if a['risk_level'] == 'high']),
            'medium': len([a for a in (results_data['alerts'] or []) if a['risk_level'] == 'medium']),
            'low': len([a for a in (results_data['alerts'] or []) if a['risk_level'] == 'low'])
        }
    }
    
    return jsonify(data)


@app.route('/api/field_map')
def get_field_map():
    """Get field map visualization."""
    if results_data['locations'] is None:
        return jsonify({'error': 'No data available'}), 404
    
    fig = create_field_map_plotly(
        results_data['locations'],
        results_data['predictions'],
        results_data['risk_labels'],
        Config.FIELD_WIDTH,
        Config.FIELD_HEIGHT
    )
    
    return jsonify(json.loads(fig.to_json()))


@app.route('/api/heatmap')
def get_heatmap():
    """Get heatmap visualization."""
    if results_data['locations'] is None:
        return jsonify({'error': 'No data available'}), 404
    
    fig = create_heatmap_plotly(
        results_data['locations'],
        results_data['predictions'],
        Config.FIELD_WIDTH,
        Config.FIELD_HEIGHT
    )
    
    return jsonify(json.loads(fig.to_json()))


@app.route('/api/training_history')
def get_training_history():
    """Get training history visualization."""
    if results_data['history'] is None:
        return jsonify({'error': 'No training history available'}), 404
    
    fig = create_training_history_plotly(results_data['history'])
    if fig is None:
        return jsonify({'error': 'Could not create training history'}), 404
    
    return jsonify(json.loads(fig.to_json()))


@app.route('/api/alerts')
def get_alerts():
    """Get risk alerts."""
    if results_data['alerts'] is None:
        return jsonify({'error': 'No alerts available'}), 404
    
    # Convert alerts to JSON-serializable format
    alerts_json = []
    for alert in results_data['alerts']:
        if isinstance(alert, dict):
            alerts_json.append(alert)
        else:
            # If it's a dataclass, convert to dict
            alerts_json.append({
                'location_id': alert.location_id,
                'x': alert.x,
                'y': alert.y,
                'risk_level': alert.risk_level,
                'risk_score': alert.risk_score,
                'primary_gas': alert.primary_gas,
                'primary_gas_concentration': alert.primary_gas_concentration,
                'recommendation': alert.recommendation
            })
    
    return jsonify({'alerts': alerts_json})


@app.route('/api/check_model')
def check_model():
    """Check if a pre-trained model exists."""
    model_path = Path(Config.OUTPUT_DIR) / 'model.pt'
    model_exists = model_path.exists()
    
    return jsonify({
        'model_exists': model_exists,
        'model_path': str(model_path) if model_exists else None
    })


@app.route('/api/run_analysis', methods=['POST'])
def run_analysis():
    """Run the complete analysis pipeline."""
    try:
        data = request.get_json() if request.is_json else {}
        use_pretrained = data.get('use_pretrained', False)
        
        # Set random seeds
        torch.manual_seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
        
        # Step 1: Generate data
        locations, features, risk_labels = generate_synthetic_dataset(
            num_nodes=Config.NUM_NODES,
            field_width=Config.FIELD_WIDTH,
            field_height=Config.FIELD_HEIGHT,
            num_timesteps=Config.NUM_TIMESTEPS,
            gas_ranges=Config.GAS_RANGES,
            random_seed=Config.RANDOM_SEED
        )
        
        # Step 2: Construct graph
        graph = construct_graph(
            locations, features, risk_labels,
            method='knn',
            k=Config.K_NEIGHBORS,
            use_distance_weights=Config.USE_DISTANCE_WEIGHTS
        )
        
        # Step 3: Initialize model
        model = AgriGraphGCN(
            input_dim=Config.INPUT_FEATURES,
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT,
            output_dim=Config.OUTPUT_DIM
        )
        
        # Step 4: Train or load model
        model_path = Path(Config.OUTPUT_DIR) / 'model.pt'
        
        if use_pretrained and model_path.exists():
            # Load pre-trained model
            from .training import load_model
            trained_model = load_model(model, str(model_path))
            history = None
            test_metrics = None
            print("Loaded pre-trained model from", model_path)
        else:
            # Train new model
            trained_model, history, test_metrics = train_model(
                model, graph,
                num_epochs=Config.NUM_EPOCHS,
                learning_rate=Config.LEARNING_RATE,
                train_split=Config.TRAIN_SPLIT,
                val_split=Config.VAL_SPLIT,
                test_split=Config.TEST_SPLIT,
                early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
                early_stopping_min_delta=Config.EARLY_STOPPING_MIN_DELTA,
                random_seed=Config.RANDOM_SEED,
                verbose=False  # Don't print during web request
            )
            
            # Save the newly trained model
            from .training import save_model
            save_model(trained_model, str(model_path))
            print("Trained and saved new model to", model_path)
        
        # Step 5: Generate predictions
        trained_model.eval()
        with torch.no_grad():
            predictions = trained_model(graph.x, graph.edge_index)
            predictions = predictions.cpu().numpy().flatten()
        
        # Step 6: Generate alerts
        gas_features = features[:, :4]
        gas_names = list(Config.GAS_RANGES.keys())
        from .interpretation import generate_alerts as gen_alerts
        alerts = gen_alerts(
            locations, predictions, gas_features,
            gas_names, Config.GAS_RANGES, Config.RISK_THRESHOLDS
        )
        
        # Convert alerts to dict format
        alerts_dict = []
        for alert in alerts:
            alerts_dict.append({
                'location_id': alert.location_id,
                'x': alert.x,
                'y': alert.y,
                'risk_level': alert.risk_level,
                'risk_score': alert.risk_score,
                'primary_gas': alert.primary_gas,
                'primary_gas_concentration': alert.primary_gas_concentration,
                'recommendation': alert.recommendation
            })
        
        # Store results
        results_data['locations'] = locations
        results_data['features'] = features
        results_data['risk_labels'] = risk_labels
        results_data['predictions'] = predictions
        results_data['graph'] = graph
        results_data['model'] = trained_model
        results_data['history'] = history
        results_data['test_metrics'] = test_metrics
        results_data['alerts'] = alerts_dict
        
        return jsonify({
            'success': True,
            'message': 'Analysis completed successfully',
            'num_nodes': len(locations),
            'test_metrics': test_metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def run_web_app(host='127.0.0.1', port=5000, debug=True):
    """Run the Flask web application."""
    import socket
    
    # Try to find an available port if 5000 is taken
    original_port = port
    if port == 5000:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            sock.close()
        except OSError:
            # Port 5000 is in use, try 5001, 5002, etc.
            for p in range(5001, 5010):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((host, p))
                    sock.close()
                    port = p
                    print(f"⚠️  Port {original_port} is in use. Using port {port} instead.")
                    break
                except OSError:
                    continue
            else:
                print(f"❌ Could not find an available port between {original_port} and 5009")
                raise RuntimeError("No available ports found")
    
    print(f"\n{'='*80}")
    print("AgriGraph AI - Web Visualization Server")
    print(f"{'='*80}")
    print(f"\n✅ Starting server on http://{host}:{port}")
    print(f"🌐 Open your browser and navigate to: http://{host}:{port}")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    except Exception as e:
        print(f"\n❌ Error starting Flask server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure no other process is using the port")
        print("2. Check if virtual environment is activated")
        print("3. Verify all dependencies are installed: pip install -r requirements.txt")
        raise


if __name__ == '__main__':
    run_web_app()

