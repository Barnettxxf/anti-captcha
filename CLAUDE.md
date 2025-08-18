# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Anti-captcha rotation prediction using PyTorch. Predict rotation angles from paired images where center regions have been rotated.

## Data Structure
- `data/images/big_images/` - Large JPGs (320x200)
- `data/images/small_images/` - Small PNGs (130x130) 
- `data/images/labels/` - Rotation angle labels

## Development Commands
```bash
uv sync           # Install dependencies
uv run python main.py
source .venv/bin/activate
uv add package    # Add new packages
```

## Architecture
Two-phase approach:
1. Simple CNN model for feasibility validation
2. Optimized model for accuracy improvement

Required components:
- Image preprocessing pipeline
- CNN rotation predictor
- Angle regression loss
- Training/validation loops

## Key Components

### Models (src/models.py)
- **SimpleCNN**: Basic 4-layer CNN for initial validation
- **OptimizedCNN**: Enhanced architecture with batch normalization and dropout
- **ResNetCNN**: ResNet18-based with custom final layers for transfer learning

### Loss Functions (src/losses.py)
- **CircularLoss**: Handles circular nature of angles using sin/cos components
- **CombinedLoss**: Weighted combination of circular and smooth L1 losses
- **SmoothL1AngleLoss**: L1 loss with circular difference handling

### Data Pipeline (src/data_loader.py)
- **RotationDataset**: Loads paired images (big/small) with rotation labels
- **get_data_loaders**: Creates train/validation splits with configurable batch sizes

### Training (src/trainer.py)
- **Trainer**: Main training class with configurable models, losses, optimizers
- **get_default_config**: Default training configuration
- Supports Adam/AdamW/SGD optimizers, various schedulers

### Evaluation (src/evaluator.py)
- **Evaluator**: Comprehensive evaluation with metrics and visualizations
- Generates error distributions, scatter plots, and detailed reports
- Saves results to CSV and generates visualizations

## Running Experiments
```bash
# Phase 1: Simple model validation
uv run python main.py --phase 1

# Phase 2: Optimized model
uv run python main.py --phase 2

# Phase 3: ResNet model
uv run python main.py --phase 3

# Evaluate trained model
uv run python main.py --evaluate checkpoints/best_model.pth --model-type optimized
```

## Model Evaluation
All trained models are automatically evaluated with:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Accuracy within 5°, 10°, 15° thresholds
- Comprehensive visualizations saved to `./results/`