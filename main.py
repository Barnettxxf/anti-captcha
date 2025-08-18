import os
import argparse
import torch
from src.trainer import Trainer, get_default_config
from src.evaluator import Evaluator
from src.data_loader import get_data_loaders
from src.models import SimpleCNN, OptimizedCNN, ResNetCNN


def phase1_experiment():
    """Phase 1: Simple model for feasibility validation"""
    print("=== Phase 1: Simple Model Validation ===")
    
    config = get_default_config()
    config.update({
        'model_type': 'simple',
        'loss_type': 'circular',
        'optimizer': 'adam',
        'learning_rate': 1e-3,
        'batch_size': 32,
        'num_epochs': 30
    })
    
    trainer = Trainer(config)
    trainer.train(config['num_epochs'])
    
    print("Phase 1 training completed!")


def phase2_experiment():
    """Phase 2: Optimized model for accuracy improvement"""
    print("=== Phase 2: Optimized Model ===")
    
    config = get_default_config()
    config.update({
        'model_type': 'optimized',
        'loss_type': 'circular',
        'optimizer': 'adamw',
        'learning_rate': 1e-4,
        'batch_size': 64,
        'num_epochs': 100,
        'scheduler': 'cosine'
    })
    
    trainer = Trainer(config)
    trainer.train(config['num_epochs'])
    
    print("Phase 2 training completed!")


def phase3_experiment():
    """Phase 3: ResNet-based model with transfer learning"""
    print("=== Phase 3: ResNet Model ===")
    
    config = get_default_config()
    config.update({
        'model_type': 'resnet',
        'loss_type': 'combined',
        'optimizer': 'adamw',
        'learning_rate': 1e-4,
        'batch_size': 32,
        'num_epochs': 50,
        'scheduler': 'plateau'
    })
    
    trainer = Trainer(config)
    trainer.train(config['num_epochs'])
    
    print("Phase 3 training completed!")


def evaluate_model(model_path, model_type='optimized'):
    """Evaluate a trained model"""
    print(f"=== Evaluating {model_type} model ===")
    
    # Load model
    if model_type == 'simple':
        model = SimpleCNN()
    elif model_type == 'optimized':
        model = OptimizedCNN()
    elif model_type == 'resnet':
        model = ResNetCNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Create test loader
    _, test_loader = get_data_loaders(
        data_dir='/Users/barnettxu/projects/anti-captcha/data/images',
        batch_size=32,
        num_workers=4
    )
    
    # Evaluate
    evaluator = Evaluator(model, device=device)
    test_results = evaluator.test_model(test_loader)
    
    # Generate report
    evaluator.visualize_results(test_results)
    evaluator.generate_report(test_results)
    evaluator.save_results(test_results)
    
    print("Evaluation completed!")


def main():
    parser = argparse.ArgumentParser(description='Anti-Captcha Rotation Prediction')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=1,
                        help='Training phase (1: simple, 2: optimized, 3: resnet)')
    parser.add_argument('--evaluate', type=str, help='Path to model checkpoint for evaluation')
    parser.add_argument('--model-type', type=str, choices=['simple', 'optimized', 'resnet'], 
                        default='optimized', help='Model type for evaluation')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    if args.evaluate:
        evaluate_model(args.evaluate, args.model_type)
    else:
        if args.phase == 1:
            phase1_experiment()
        elif args.phase == 2:
            phase2_experiment()
        elif args.phase == 3:
            phase3_experiment()


if __name__ == "__main__":
    main()