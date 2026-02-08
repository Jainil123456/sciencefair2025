# How to Use Pre-trained Models in AgriGraph AI

This guide explains how to train, save, and load pre-trained models for faster inference.

## Overview

AgriGraph AI supports two modes:
- **Training Mode**: Train a new model from scratch (2-5 minutes)
- **Inference Mode**: Load a pre-trained model for instant predictions (seconds)

## Step-by-Step Guide

### Method 1: Using the Web Dashboard (Recommended)

#### Training a New Model

1. **Start the web server:**
   ```bash
   python3 start_dashboard.py
   ```
   This will open your browser automatically to `http://127.0.0.1:5000`

2. **Train a new model:**
   - Click the **"Train New Model"** button
   - Wait for training to complete (2-5 minutes)
   - The model is automatically saved to `outputs/model.pt`
   - You'll see a status message: "✓ Model trained and saved. You can use it next time!"

3. **View results:**
   - Field map visualization
   - Risk heatmap
   - Training history charts
   - Risk alerts

#### Loading a Pre-trained Model

1. **Start the web server:**
   ```bash
   python3 start_dashboard.py
   ```

2. **Check model status:**
   - The dashboard will show: "✓ Pre-trained model available (faster)"
   - A green **"Use Pre-trained Model"** button will appear

3. **Load the pre-trained model:**
   - Click **"Use Pre-trained Model"** button
   - Results appear in seconds (no training required!)
   - Note: Training history won't be shown (since no training occurred)

### Method 2: Using Command Line

#### Training and Saving a Model

```bash
# Train a new model (saves automatically to outputs/model.pt)
python3 -m agrigraph_ai.main

# With custom parameters
python3 -m agrigraph_ai.main \
    --num-nodes 200 \
    --epochs 300 \
    --hidden-dim 256 \
    --output-dir outputs
```

The model will be saved to `outputs/model.pt` after training completes.

#### Loading a Pre-trained Model

```bash
# Load and use pre-trained model (skips training)
python3 -m agrigraph_ai.main \
    --skip-training \
    --model-path outputs/model.pt
```

This will:
- Generate new synthetic data
- Build the graph
- Load the pre-trained model from `outputs/model.pt`
- Generate predictions instantly
- Create visualizations and alerts

## Model File Location

- **Default location**: `outputs/model.pt`
- **File format**: PyTorch state dictionary (`.pt` file)
- **Size**: Typically 1-5 MB depending on model architecture

## When to Retrain

You should retrain a new model when:
- ✅ You change the model architecture (hidden dimensions, layers)
- ✅ You change the data generation parameters significantly
- ✅ You want to see training progress and metrics
- ✅ The model performance is not satisfactory

You can reuse a pre-trained model when:
- ✅ You're just generating new data with same parameters
- ✅ You want fast results without waiting for training
- ✅ You're demonstrating the system to others
- ✅ You're iterating on visualizations or alerts

## Troubleshooting

### "No pre-trained model found"

**Solution:** Train a model first using "Train New Model" button or:
```bash
python3 -m agrigraph_ai.main
```

### "Model file not found" error

**Solution:** Check that the model file exists:
```bash
ls -lh outputs/model.pt
```

If it doesn't exist, train a new model first.

### Model architecture mismatch

**Error:** If you change model parameters (hidden_dim, num_layers) in `config.py`, the old model won't work.

**Solution:** Delete the old model and train a new one:
```bash
rm outputs/model.pt
python3 -m agrigraph_ai.main
```

### Loading model is slow

**Note:** Loading should be fast (seconds). If it's slow, check:
- Model file size (should be 1-5 MB)
- Disk I/O performance
- System resources

## Advanced Usage

### Custom Model Path

```bash
# Save to custom location
python3 -m agrigraph_ai.main --output-dir my_models

# Load from custom location
python3 -m agrigraph_ai.main \
    --skip-training \
    --model-path my_models/model.pt
```

### Multiple Models

You can save multiple models with different configurations:

```bash
# Train model with 128 hidden dims
python3 -m agrigraph_ai.main --hidden-dim 128 --output-dir models_128
# Saves to: models_128/model.pt

# Train model with 256 hidden dims
python3 -m agrigraph_ai.main --hidden-dim 256 --output-dir models_256
# Saves to: models_256/model.pt

# Use specific model
python3 -m agrigraph_ai.main \
    --skip-training \
    --model-path models_256/model.pt
```

## Summary

| Action | Command/Button | Time | Model Saved |
|--------|---------------|------|-------------|
| Train new model | "Train New Model" button or `python3 -m agrigraph_ai.main` | 2-5 min | ✅ Yes |
| Use pre-trained | "Use Pre-trained Model" button or `--skip-training` | Seconds | ❌ No |
| Check if model exists | Dashboard status or `ls outputs/model.pt` | Instant | - |

## Quick Reference

```bash
# Train and save model
python3 -m agrigraph_ai.main

# Load pre-trained model
python3 -m agrigraph_ai.main --skip-training --model-path outputs/model.pt

# Start web dashboard
python3 start_dashboard.py
```

For web dashboard:
- **"Train New Model"** = Train from scratch (slow but shows training history)
- **"Use Pre-trained Model"** = Load saved model (fast, no training history)

