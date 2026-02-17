"""
Web application for visualizing AgriGraph AI results.
Serves interactive visualizations on localhost.
"""

import os
import json
import numpy as np
import torch
from flask import Flask, render_template, jsonify, send_from_directory, request, Response, stream_with_context, session
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import threading
from datetime import datetime
from .seed_manager import seed_manager
import secrets

# Set template and static folder paths before creating Flask app
template_dir = Path(__file__).parent.parent / 'templates'
template_dir.mkdir(exist_ok=True)

static_dir = Path(__file__).parent.parent / 'static'
static_dir.mkdir(exist_ok=True)

from .config import Config
from .data_generation import generate_synthetic_dataset
from .graph_construction import construct_graph
from .model import AgriGraphGCN
from .training import train_model, load_model
from .interpretation import generate_alerts
from .progress_manager import progress_manager

app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))

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


@app.route('/about')
def about():
    """Technical documentation page."""
    return render_template('about.html')


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


@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """Upload and process CSV/Excel sensor data."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        from werkzeug.utils import secure_filename
        import tempfile
        import os

        # Get file extension
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

        if ext not in ['csv', 'xlsx', 'xls']:
            return jsonify({
                'success': False,
                'error': 'Only CSV and Excel files supported. Received: ' + ext
            }), 400

        # Save temporarily
        with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp:
            file.save(tmp.name)

            try:
                # Import loader
                from .csv_loader import load_sensor_data, validate_csv_format

                # Validate format first
                validation = validate_csv_format(tmp.name)
                if not validation['valid']:
                    return jsonify({
                        'success': False,
                        'error': validation['error']
                    }), 400

                # Flexible loader handles any format - no strict column requirements

                # Load data
                locations, features, risk_labels = load_sensor_data(tmp.name, file_type=ext)

                # Store in global results_data
                results_data['locations'] = locations
                results_data['features'] = features
                results_data['risk_labels'] = risk_labels
                results_data['uploaded_data'] = {
                    'filename': filename,
                    'num_nodes': len(locations),
                    'has_labels': risk_labels is not None
                }

                # Reset predictions/alerts since data changed
                results_data['predictions'] = None
                results_data['alerts'] = None

                return jsonify({
                    'success': True,
                    'filename': filename,
                    'num_nodes': len(locations),
                    'num_features': features.shape[1],
                    'has_labels': risk_labels is not None,
                    'columns': validation['columns'],
                    'message': f'Successfully loaded {len(locations)} sensor locations'
                })

            except ValueError as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 400
            finally:
                # Cleanup temp file
                if os.path.exists(tmp.name):
                    os.remove(tmp.name)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }), 500


@app.route('/api/training/progress/<job_id>')
def training_progress_stream(job_id):
    """SSE endpoint for real-time training progress."""
    def generate():
        for update in progress_manager.iter_updates(job_id, timeout=600):
            if update.get('heartbeat'):
                yield 'event: heartbeat\ndata: {}\n\n'
            else:
                yield f'event: progress\ndata: {json.dumps(update)}\n\n'

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/training/cancel/<job_id>', methods=['POST'])
def cancel_training(job_id):
    """Cancel a running training job."""
    progress_manager.cancel_job(job_id)
    return jsonify({'success': True, 'message': 'Cancellation requested'})


@app.route('/api/export/alerts/csv')
def export_alerts_csv():
    """Export alerts as CSV file."""
    if results_data['alerts'] is None or len(results_data['alerts']) == 0:
        return jsonify({'error': 'No alerts available'}), 404

    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        'Location ID', 'X Coordinate', 'Y Coordinate', 'Risk Level', 'Risk Score',
        'Primary Gas', 'Concentration (ppm)', 'Recommendation'
    ])

    # Data rows
    for alert in results_data['alerts']:
        if isinstance(alert, dict):
            writer.writerow([
                alert.get('location_id', ''),
                f"{alert.get('x', 0):.2f}",
                f"{alert.get('y', 0):.2f}",
                alert.get('risk_level', ''),
                f"{alert.get('risk_score', 0):.4f}",
                alert.get('primary_gas', ''),
                f"{alert.get('primary_gas_concentration', 0):.2f}",
                alert.get('recommendation', '')
            ])
        else:
            # If it's a dataclass object
            writer.writerow([
                alert.location_id,
                f"{alert.x:.2f}",
                f"{alert.y:.2f}",
                alert.risk_level,
                f"{alert.risk_score:.4f}",
                alert.primary_gas,
                f"{alert.primary_gas_concentration:.2f}",
                alert.recommendation
            ])

    csv_content = output.getvalue()

    from flask import make_response
    response = make_response(csv_content)
    response.headers['Content-Type'] = 'text/csv; charset=utf-8'
    response.headers['Content-Disposition'] = 'attachment; filename=agrigraph_alerts.csv'

    return response


@app.route('/api/seed/generate', methods=['POST'])
def generate_seed():
    """Generate new random seed."""
    data = request.get_json() or {}
    mode = data.get('mode', 'auto')
    custom_seed = data.get('custom_seed')

    try:
        seed = seed_manager.generate_seed(mode, custom_seed)

        # Store in session
        if 'session_id' not in session:
            session['session_id'] = secrets.token_hex(16)
        seed_manager.store_session_seed(session['session_id'], seed)
        session.modified = True  # Force session save

        return jsonify({
            'success': True,
            'seed': seed,
            'mode': mode
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/seed/current')
def get_current_seed():
    """Get current session's seed."""
    # Ensure session_id exists
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(16)
        session.modified = True

    session_id = session.get('session_id')
    if session_id:
        seed = seed_manager.get_session_seed(session_id)
        if seed is not None:
            return jsonify({
                'success': True,
                'seed': seed,
                'session_id': session_id
            })

    return jsonify({
        'success': False,
        'error': 'No seed set for this session',
        'session_id': session_id
    })


def _run_training_thread(job_id, use_pretrained, locations, features, risk_labels, graph, random_seed):
    """Background thread function for training."""
    try:
        # Step 3: Initialize model - use actual feature count from data
        actual_input_dim = features.shape[1] if features is not None else Config.INPUT_FEATURES
        print(f"[{job_id}] Model input_dim={actual_input_dim} (features shape: {features.shape if features is not None else 'None'})")

        model = AgriGraphGCN(
            input_dim=actual_input_dim,
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
            print(f"[{job_id}] Loaded pre-trained model from {model_path}")

            # For pretrained models, mark as complete immediately
            progress_manager.complete_job(job_id, {
                'message': 'Pre-trained model loaded',
                'test_metrics': test_metrics
            })
        else:
            # Define progress callback
            def progress_callback(epoch, train_loss, val_loss, val_r2, eta_seconds, message):
                # Check for cancellation
                if progress_manager.is_cancelled(job_id):
                    raise KeyboardInterrupt("Training cancelled by user")
                progress_manager.update_progress(
                    job_id, epoch, train_loss, val_loss,
                    metrics={'r2': val_r2, 'eta_seconds': eta_seconds}
                )

            # Define cancel check function
            def cancel_check():
                return progress_manager.is_cancelled(job_id)

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
                random_seed=random_seed,
                verbose=False,
                progress_callback=progress_callback,
                cancel_check=cancel_check
            )

            # Save the newly trained model
            from .training import save_model
            save_model(trained_model, str(model_path))
            print(f"[{job_id}] Trained and saved new model to {model_path}")

            # Mark job as complete
            progress_manager.complete_job(job_id, {
                'message': 'Training completed',
                'test_metrics': test_metrics
            })

        # Step 5: Generate predictions
        trained_model.eval()
        with torch.no_grad():
            predictions = trained_model(graph.x, graph.edge_index)
            predictions = predictions.cpu().numpy().flatten()

        # Step 6: Generate alerts
        # Handle variable number of feature columns
        num_feature_cols = features.shape[1] - 2  # subtract x, y coordinates
        gas_names = list(Config.GAS_RANGES.keys())

        if num_feature_cols >= 4:
            gas_features = features[:, :4]
        else:
            # Pad with zeros if fewer than 4 gas columns
            gas_features = np.zeros((features.shape[0], 4))
            gas_features[:, :num_feature_cols] = features[:, :num_feature_cols]

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

        print(f"[{job_id}] Analysis pipeline completed successfully")

    except KeyboardInterrupt:
        print(f"[{job_id}] Training cancelled by user")
        progress_manager.fail_job(job_id, "Training cancelled by user")
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[{job_id}] ERROR in training thread:")
        print(error_traceback)
        progress_manager.fail_job(job_id, str(e))


@app.route('/api/run_analysis', methods=['POST'])
def run_analysis():
    """Run the complete analysis pipeline asynchronously."""
    try:
        data = request.get_json() if request.is_json else {}
        use_pretrained = data.get('use_pretrained', False)
        seed_mode = data.get('seed_mode', 'auto')
        custom_seed = data.get('custom_seed')

        # Generate seed using seed manager
        random_seed = seed_manager.generate_seed(seed_mode, custom_seed)

        # Store in session
        if 'session_id' not in session:
            session['session_id'] = secrets.token_hex(16)
        seed_manager.store_session_seed(session.get('session_id'), random_seed)
        session.modified = True  # Force session save

        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Create a new job
        job_id = progress_manager.create_job(Config.NUM_EPOCHS)

        # Step 1: Check if uploaded data exists, otherwise generate synthetic
        if results_data.get('locations') is not None and results_data.get('features') is not None:
            # Use uploaded data
            locations = results_data['locations']
            features = results_data['features']
            risk_labels = results_data.get('risk_labels')

            # Generate risk labels if not provided
            if risk_labels is None:
                from .data_generation import generate_synthetic_risk_labels
                gas_ranges = Config.GAS_RANGES
                num_gas_cols = min(features.shape[1] - 2, 4)
                gas_data = features[:, :num_gas_cols].reshape(features.shape[0], 1, num_gas_cols)
                risk_labels = generate_synthetic_risk_labels(gas_data, locations, gas_ranges)
                print(f"[{job_id}] Generated risk labels from uploaded features")

            print(f"[{job_id}] Using uploaded data: {locations.shape[0]} locations, {features.shape[1]} features")
        else:
            # Generate synthetic data
            locations, features, risk_labels = generate_synthetic_dataset(
                num_nodes=Config.NUM_NODES,
                field_width=Config.FIELD_WIDTH,
                field_height=Config.FIELD_HEIGHT,
                num_timesteps=Config.NUM_TIMESTEPS,
                gas_ranges=Config.GAS_RANGES,
                random_seed=random_seed
            )
            print(f"[{job_id}] Generated synthetic data: {locations.shape[0]} locations")

        # Step 2: Construct graph
        graph = construct_graph(
            locations, features, risk_labels,
            method='knn',
            k=Config.K_NEIGHBORS,
            use_distance_weights=Config.USE_DISTANCE_WEIGHTS
        )

        # Start training in background thread
        training_thread = threading.Thread(
            target=_run_training_thread,
            args=(job_id, use_pretrained, locations, features, risk_labels, graph, random_seed),
            daemon=False
        )
        training_thread.start()

        # Return immediately with job_id
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Analysis started. Use /api/training/progress/{job_id} to track progress.',
            'num_nodes': len(locations)
        })

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print("ERROR in run_analysis:")
        print(error_traceback)
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_traceback if False else None  # Set to True to debug
        }), 500


@app.route('/api/graph/structure')
def get_graph_structure():
    """Get GNN graph structure for visualization."""
    if results_data['graph'] is None or results_data['locations'] is None:
        return jsonify({'error': 'No graph available'}), 404

    try:
        graph = results_data['graph']
        locations = results_data['locations']
        predictions = results_data['predictions']

        # Build node list
        nodes = []
        for i in range(len(locations)):
            risk_score = float(predictions[i]) if predictions is not None and i < len(predictions) else 0.0

            # Determine risk level
            if risk_score >= 0.7:
                risk_level = 'critical'
            elif risk_score >= 0.5:
                risk_level = 'high'
            elif risk_score >= 0.3:
                risk_level = 'medium'
            else:
                risk_level = 'low'

            nodes.append({
                'id': int(i),
                'x': float(locations[i, 0]),
                'y': float(locations[i, 1]),
                'risk_score': risk_score,
                'risk_level': risk_level,
                'degree': int(graph.edge_index[0, graph.edge_index[0] == i].shape[0]) if hasattr(graph, 'edge_index') else 0
            })

        # Build edge list
        edges = []
        if hasattr(graph, 'edge_index'):
            edge_index = graph.edge_index
            for j in range(edge_index.shape[1]):
                edges.append([
                    int(edge_index[0, j].item()),
                    int(edge_index[1, j].item())
                ])

        return jsonify({
            'success': True,
            'nodes': nodes,
            'edges': edges,
            'num_nodes': len(nodes),
            'num_edges': len(edges)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/llm_recommendations', methods=['POST'])
def get_llm_recommendations():
    """Generate LLM recommendations from current analysis results."""
    if results_data['alerts'] is None or len(results_data['alerts']) == 0:
        return jsonify({
            'success': False,
            'error': 'No analysis results available. Please run analysis first.'
        }), 404

    try:
        from .llm_recommendations import compare_llm_recommendations

        # Convert alerts to dict format
        alerts_list = []
        for alert in results_data['alerts'][:20]:  # Use top 20 alerts
            if isinstance(alert, dict):
                alerts_list.append(alert)
            else:
                # Handle alert dataclass objects
                alerts_list.append({
                    'location_id': alert.location_id,
                    'x': alert.x,
                    'y': alert.y,
                    'risk_level': alert.risk_level,
                    'risk_score': alert.risk_score,
                    'primary_gas': alert.primary_gas,
                    'primary_gas_concentration': alert.primary_gas_concentration,
                    'recommendation': alert.recommendation
                })

        # Prepare field statistics for LLM context
        field_stats = {
            'num_nodes': len(results_data['locations']) if results_data['locations'] is not None else 0,
            'x_min': float(results_data['locations'][:, 0].min()) if results_data['locations'] is not None else 0,
            'x_max': float(results_data['locations'][:, 0].max()) if results_data['locations'] is not None else 100,
            'y_min': float(results_data['locations'][:, 1].min()) if results_data['locations'] is not None else 0,
            'y_max': float(results_data['locations'][:, 1].max()) if results_data['locations'] is not None else 100,
            'r2_score': results_data['test_metrics'].get('r2', 0) if results_data['test_metrics'] else 0
        }

        print(f"Generating LLM recommendations for {len(alerts_list)} alerts...")

        # Get LLM recommendations (both APIs are called)
        comparison = compare_llm_recommendations(alerts_list, field_stats)

        # Store in results for future reference
        results_data['llm_comparison'] = comparison

        return jsonify({
            'success': True,
            'comparison': comparison
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to generate recommendations: {str(e)}'
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
        app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"\n❌ Error starting Flask server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure no other process is using the port")
        print("2. Check if virtual environment is activated")
        print("3. Verify all dependencies are installed: pip install -r requirements.txt")
        raise


if __name__ == '__main__':
    run_web_app()

