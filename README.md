# AgriGraph AI - Soil Gas Risk Prediction System

🌐 **[Launch Web Dashboard](http://127.0.0.1:5000)** (Run `python3 run_web.py` first)

## Project Description

AgriGraph AI is a Python-based prototype that models farm fields as graphs and applies Graph Neural Networks (GNNs) for soil gas risk prediction. The system simulates IoT sensor data collected by drones, where each sampling location is a node with features including ammonia (NH3), methane (CH4), nitrogen dioxide (NO2), carbon monoxide (CO), spatial coordinates (x, y), and time.

The system uses a Graph Convolutional Network (GCN) implemented with PyTorch and PyTorch Geometric to learn spatial gas diffusion patterns and outputs a risk score for each node indicating potential soil degradation or hazardous gas buildup. The system includes visualization of predicted risk zones and an interpretation layer that converts high-risk predictions into human-readable alerts suitable for farmer-facing mobile applications.

## Architecture

The system is organized into modular components:

- **Data Generation** (`data_generation.py`): Synthetic IoT sensor data simulation with spatial correlation
- **Graph Construction** (`graph_construction.py`): Builds spatial graphs from sensor locations using k-NN or distance-based connectivity
- **Model** (`model.py`): Graph Convolutional Network (GCN) implementation for risk prediction
- **Training** (`training.py`): Training loop with early stopping, evaluation metrics, and model checkpointing
- **Visualization** (`visualization.py`): Risk zone visualization with field maps and heatmaps
- **Interpretation** (`interpretation.py`): Converts risk scores to human-readable alerts with recommendations
- **Configuration** (`config.py`): Centralized configuration parameters
- **Main** (`main.py`): Complete pipeline orchestration

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or navigate to this repository:
```bash
cd sciencefair2025
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
# Or use: pip3 install -r requirements.txt
```

Note: PyTorch Geometric may require additional setup depending on your system. If you encounter issues, refer to the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Usage

### Basic Usage

Run the complete pipeline with default settings:

```bash
python3 -m agrigraph_ai.main
```

**Note:** On macOS and some Linux systems, use `python3` instead of `python`. If `python` works on your system, you can use either command.

This will:
1. Generate synthetic sensor data
2. Construct the spatial graph
3. Train the GCN model
4. Generate predictions
5. Create visualizations
6. Produce risk alerts

All outputs will be saved to the `outputs/` directory.

### Command-Line Options

```bash
python3 -m agrigraph_ai.main --help
```

Common options:
- `--num-nodes N`: Number of sensor locations (default: 150)
- `--epochs N`: Number of training epochs (default: 200)
- `--lr FLOAT`: Learning rate (default: 0.001)
- `--hidden-dim N`: Hidden dimension for GCN layers (default: 128)
- `--k-neighbors N`: Number of neighbors for k-NN graph (default: 8)
- `--output-dir PATH`: Output directory (default: outputs)
- `--seed N`: Random seed (default: 42)

### Example: Custom Configuration

```bash
python3 -m agrigraph_ai.main \
    --num-nodes 200 \
    --epochs 300 \
    --hidden-dim 256 \
    --k-neighbors 10 \
    --output-dir results
```

### Inference Only (Skip Training)

To generate visualizations and alerts using a pre-trained model:

```bash
python3 -m agrigraph_ai.main \
    --skip-training \
    --model-path outputs/model.pt
```

### Web-Based Visualization Dashboard

**🚀 Quick Start (Recommended):**

```bash
# Make sure virtual environment is activated first
source venv/bin/activate

# Option 1: Start server (auto-finds available port)
python3 start_dashboard.py

# Option 2: Kill processes on ports 5000-5009 first, then start
python3 start_dashboard.py --kill-ports

# Option 3: Manually kill ports first
python3 kill_ports.py
python3 start_dashboard.py
```

Or use the helper script (automatically activates venv):

```bash
./start_server.sh
```

This will automatically start the server AND open your browser to the dashboard!

**Note:** 
- If port 5000 is in use (common on macOS), the server will automatically use port 5001, 5002, etc.
- To start on a specific clean port: `python3 start_dashboard.py 5002`
- To kill Python processes on ports first: `python3 kill_ports.py` (or `python3 kill_ports.py --force` for all processes)

**Alternative methods:**

```bash
# Method 1: Simple launcher
python3 run_web.py

# Method 2: Direct module
python3 -m agrigraph_ai.web_app
```

**🌐 Dashboard Links:**
- **Direct URL:** [http://127.0.0.1:5000](http://127.0.0.1:5000)
- **Launch Page:** Open `launch_dashboard.html` in your browser (auto-detects when server is ready)

The web server runs on `http://127.0.0.1:5000`. After starting the server, open this URL in your browser or use the launch page.

#### Using Pre-trained Models

The dashboard supports two modes:

1. **Train New Model** (Default for first run)
   - Trains a fresh model from scratch
   - Takes 2-5 minutes depending on your system
   - Automatically saves the model to `outputs/model.pt` after training
   - Shows training history and metrics

2. **Use Pre-trained Model** (Faster option)
   - Loads a previously trained model from `outputs/model.pt`
   - Takes only seconds (no training required)
   - Requires that you've trained a model at least once before
   - Button appears automatically when a pre-trained model exists

**How to use pre-trained models:**

**Step 1: Train and save a model (one-time setup)**
```bash
# Option A: Via web dashboard
# 1. Start the dashboard: python3 start_dashboard.py
# 2. Click "Train New Model" button
# 3. Wait for training to complete
# 4. Model is automatically saved to outputs/model.pt

# Option B: Via command line
python3 -m agrigraph_ai.main
# This trains and saves the model to outputs/model.pt
```

**Step 2: Load the pre-trained model**
```bash
# Via web dashboard:
# 1. Start the dashboard: python3 start_dashboard.py
# 2. Click "Use Pre-trained Model" button (appears if model.pt exists)
# 3. Results appear in seconds!

# Via command line:
python3 -m agrigraph_ai.main --skip-training --model-path outputs/model.pt
```

**Note:** The pre-trained model is trained on synthetic data. If you generate new data with different patterns, you may want to retrain for better accuracy.

The web dashboard provides:
- Interactive field maps with risk zones
- Heatmap visualizations
- Training history charts (only when training new model)
- Real-time risk alerts
- Statistics and metrics
- Model status indicator showing if pre-trained model is available

## Output Files

The system generates the following outputs in the specified output directory:

- `model.pt`: Trained model checkpoint
- `risk_prediction_map.png`: Field map with predicted and true risk zones
- `risk_heatmap.png`: Interpolated heatmap of risk zones
- `training_history.png`: Training loss and metrics over epochs
- `alerts.json`: All risk alerts in JSON format (suitable for mobile app integration)
- `urgent_alerts.json`: High and critical risk alerts only

## Project Structure

```
sciencefair2025/
├── agrigraph_ai/
│   ├── __init__.py
│   ├── config.py              # Configuration parameters
│   ├── data_generation.py     # Synthetic data generation
│   ├── graph_construction.py  # Graph building
│   ├── model.py               # GCN model definition
│   ├── training.py            # Training loop
│   ├── visualization.py       # Risk visualization
│   ├── interpretation.py      # Alert generation
│   └── main.py                # Main entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── outputs/                   # Generated outputs (created at runtime)
```

## Key Features

- **Spatial Graph Modeling**: Models farm fields as graphs with spatial connectivity
- **Graph Neural Networks**: Uses GCN to learn spatial gas diffusion patterns
- **Synthetic Data Generation**: Realistic simulation of IoT sensor data with spatial correlation
- **Risk Prediction**: Continuous risk scores (0-1) for each sensor location
- **Visualization**: Field maps and heatmaps showing risk zones
- **Human-Readable Alerts**: Converts predictions to actionable recommendations
- **Mobile-Ready Output**: JSON format suitable for mobile app integration

## Configuration

Modify `agrigraph_ai/config.py` to adjust:
- Field dimensions
- Gas concentration ranges
- Graph construction parameters
- Model architecture
- Training hyperparameters
- Risk thresholds

## License

This project is created for Science Fair 2025. Please specify your license here.

