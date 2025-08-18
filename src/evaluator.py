import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from PIL import Image
import pandas as pd


class Evaluator:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate_predictions(self, predictions, targets):
        """Evaluate rotation predictions"""
        errors = np.abs(predictions - targets)
        errors = np.minimum(errors, 360 - errors)
        
        metrics = {
            'mae': np.mean(errors),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'median_ae': np.median(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'std_error': np.std(errors),
            'accuracy_5': np.mean(errors <= 5),
            'accuracy_10': np.mean(errors <= 10),
            'accuracy_15': np.mean(errors <= 15)
        }
        
        return metrics, errors
    
    def test_model(self, test_loader):
        """Test model on test dataset"""
        predictions = []
        targets = []
        image_paths = []
        
        with torch.no_grad():
            for batch in test_loader:
                big_images = batch['big_image'].to(self.device)
                small_images = batch['small_image'].to(self.device)
                angles = batch['angle'].to(self.device)
                
                combined_images = self._combine_images(big_images, small_images)
                outputs = self.model(combined_images).squeeze()
                
                predictions.extend(outputs.cpu().numpy())
                targets.extend(angles.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics, errors = self.evaluate_predictions(predictions, targets)
        
        return {
            'predictions': predictions,
            'targets': targets,
            'errors': errors,
            'metrics': metrics
        }
    
    def visualize_results(self, test_results, save_dir='./results'):
        """Visualize test results"""
        os.makedirs(save_dir, exist_ok=True)
        
        predictions = test_results['predictions']
        targets = test_results['targets']
        errors = test_results['errors']
        metrics = test_results['metrics']
        
        # Error distribution
        plt.figure(figsize=(15, 10))
        
        # Error histogram
        plt.subplot(2, 3, 1)
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(metrics['mae'], color='red', linestyle='--', label=f'MAE: {metrics["mae"]:.2f}°')
        plt.xlabel('Error (degrees)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.legend()
        
        # Scatter plot
        plt.subplot(2, 3, 2)
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([0, 360], [0, 360], 'r--', lw=2)
        plt.xlabel('True Angle')
        plt.ylabel('Predicted Angle')
        plt.title('True vs Predicted Angles')
        
        # Error vs True Angle
        plt.subplot(2, 3, 3)
        plt.scatter(targets, errors, alpha=0.5)
        plt.xlabel('True Angle')
        plt.ylabel('Error (degrees)')
        plt.title('Error vs True Angle')
        
        # Cumulative error distribution
        plt.subplot(2, 3, 4)
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        plt.plot(sorted_errors, cumulative)
        plt.xlabel('Error (degrees)')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Error Distribution')
        plt.grid(True)
        
        # Angle distribution
        plt.subplot(2, 3, 5)
        plt.hist(targets, bins=36, alpha=0.7, label='True')
        plt.hist(predictions, bins=36, alpha=0.7, label='Predicted')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Frequency')
        plt.title('Angle Distribution')
        plt.legend()
        
        # Error heatmap
        plt.subplot(2, 3, 6)
        error_df = pd.DataFrame({
            'true': targets,
            'pred': predictions,
            'error': errors
        })
        
        error_df['true_bin'] = pd.cut(error_df['true'], bins=12)
        error_df['pred_bin'] = pd.cut(error_df['pred'], bins=12)
        
        pivot = error_df.pivot_table(values='error', index='true_bin', columns='pred_bin', aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Reds')
        plt.title('Error Heatmap')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary
        print("=== Evaluation Results ===")
        print(f"MAE: {metrics['mae']:.2f}°")
        print(f"RMSE: {metrics['rmse']:.2f}°")
        print(f"Median AE: {metrics['median_ae']:.2f}°")
        print(f"Max Error: {metrics['max_error']:.2f}°")
        print(f"Accuracy @5°: {metrics['accuracy_5']*100:.1f}%")
        print(f"Accuracy @10°: {metrics['accuracy_10']*100:.1f}%")
        print(f"Accuracy @15°: {metrics['accuracy_15']*100:.1f}%")
    
    def visualize_predictions(self, test_loader, num_samples=16, save_dir='./results'):
        """Visualize individual predictions"""
        os.makedirs(save_dir, exist_ok=True)
        
        samples = []
        with torch.no_grad():
            for batch in test_loader:
                big_images = batch['big_image'][:num_samples].to(self.device)
                small_images = batch['small_image'][:num_samples].to(self.device)
                angles = batch['angle'][:num_samples]
                
                combined_images = self._combine_images(big_images, small_images)
                outputs = self.model(combined_images).squeeze()
                
                for i in range(min(num_samples, len(angles))):
                    samples.append({
                        'big_image': big_images[i].cpu(),
                        'small_image': small_images[i].cpu(),
                        'true_angle': angles[i].item(),
                        'pred_angle': outputs[i].item() if outputs.dim() > 0 else outputs.item()
                    })
                
                if len(samples) >= num_samples:
                    break
        
        # Create visualization
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()
        
        for idx, sample in enumerate(samples[:16]):
            # Create combined image for display
            big_img = sample['big_image'].permute(1, 2, 0).numpy()
            small_img = sample['small_image'].permute(1, 2, 0).numpy()
            
            if small_img.shape[2] == 4:  # RGBA
                alpha = small_img[:, :, 3:4]
                rgb_small = small_img[:, :, :3]
                combined = big_img * (1 - alpha) + rgb_small * alpha
            else:
                combined = big_img
            
            axes[idx].imshow(np.clip(combined, 0, 1))
            axes[idx].set_title(f"True: {sample['true_angle']:.1f}°, Pred: {sample['pred_angle']:.1f}°")
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, test_results, save_dir='./results'):
        """Generate detailed evaluation report"""
        os.makedirs(save_dir, exist_ok=True)
        
        predictions = test_results['predictions']
        targets = test_results['targets']
        errors = test_results['errors']
        metrics = test_results['metrics']
        
        report_path = os.path.join(save_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=== Anti-Captcha Rotation Evaluation Report ===\n\n")
            f.write(f"Total samples: {len(predictions)}\n")
            f.write(f"Mean Absolute Error: {metrics['mae']:.2f}°\n")
            f.write(f"Root Mean Square Error: {metrics['rmse']:.2f}°\n")
            f.write(f"Median Absolute Error: {metrics['median_ae']:.2f}°\n")
            f.write(f"Maximum Error: {metrics['max_error']:.2f}°\n")
            f.write(f"Minimum Error: {metrics['min_error']:.2f}°\n")
            f.write(f"Standard Deviation: {metrics['std_error']:.2f}°\n\n")
            
            f.write("=== Accuracy Metrics ===\n")
            f.write(f"Accuracy within 5°: {metrics['accuracy_5']*100:.1f}%\n")
            f.write(f"Accuracy within 10°: {metrics['accuracy_10']*100:.1f}%\n")
            f.write(f"Accuracy within 15°: {metrics['accuracy_15']*100:.1f}%\n\n")
            
            f.write("=== Error Distribution ===\n")
            f.write(f"25th percentile: {np.percentile(errors, 25):.2f}°\n")
            f.write(f"75th percentile: {np.percentile(errors, 75):.2f}°\n")
            f.write(f"90th percentile: {np.percentile(errors, 90):.2f}°\n")
            f.write(f"95th percentile: {np.percentile(errors, 95):.2f}°\n\n")
            
            f.write("=== Worst Predictions ===\n")
            worst_indices = np.argsort(errors)[-10:][::-1]
            for i, idx in enumerate(worst_indices):
                f.write(f"{i+1}. True: {targets[idx]:.1f}°, Predicted: {predictions[idx]:.1f}°, Error: {errors[idx]:.1f}°\n")
            
            f.write("\n=== Best Predictions ===\n")
            best_indices = np.argsort(errors)[:10]
            for i, idx in enumerate(best_indices):
                f.write(f"{i+1}. True: {targets[idx]:.1f}°, Predicted: {predictions[idx]:.1f}°, Error: {errors[idx]:.1f}°\n")
    
    def _combine_images(self, big_images, small_images):
        """Combine big and small images for model input"""
        if small_images.shape[1] == 4:
            alpha = small_images[:, 3:4, :, :]
            rgb_small = small_images[:, :3, :, :]
            combined = big_images * (1 - alpha) + rgb_small * alpha
        else:
            combined = big_images
        
        return combined
    
    def save_results(self, test_results, save_dir='./results'):
        """Save results to CSV file"""
        os.makedirs(save_dir, exist_ok=True)
        
        df = pd.DataFrame({
            'true_angle': test_results['targets'],
            'predicted_angle': test_results['predictions'],
            'error': test_results['errors']
        })
        
        df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
        print(f"Results saved to {os.path.join(save_dir, 'predictions.csv')}")