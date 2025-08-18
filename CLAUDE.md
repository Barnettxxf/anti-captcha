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