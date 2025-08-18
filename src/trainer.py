import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os

from src.data_loader import get_data_loaders
from src.models import SimpleCNN, OptimizedCNN, ResNetCNN
from src.losses import AngleLoss, CircularLoss, CombinedLoss, SmoothL1AngleLoss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_loader, self.val_loader = get_data_loaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        self.model = self._build_model(config['model_type'])
        self.criterion = self._build_loss(config['loss_type'])
        self.optimizer = self._build_optimizer(config['optimizer'], config['learning_rate'])
        self.scheduler = self._build_scheduler(config['scheduler'])
        
        self.writer = SummaryWriter(config['log_dir'])
        self.best_val_loss = float('inf')
        
    def _build_model(self, model_type):
        if model_type == 'simple':
            model = SimpleCNN(input_channels=3)
        elif model_type == 'optimized':
            model = OptimizedCNN(input_channels=3)
        elif model_type == 'resnet':
            model = ResNetCNN(input_channels=3, pretrained=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def _build_loss(self, loss_type):
        if loss_type == 'angle':
            return AngleLoss()
        elif loss_type == 'circular':
            return CircularLoss()
        elif loss_type == 'combined':
            return CombinedLoss()
        elif loss_type == 'smooth_l1':
            return SmoothL1AngleLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _build_optimizer(self, optimizer_type, learning_rate):
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _build_scheduler(self, scheduler_type):
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        else:
            return None
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            big_images = batch['big_image'].to(self.device)
            small_images = batch['small_image'].to(self.device)
            angles = batch['angle'].to(self.device)
            
            combined_images = self._combine_images(big_images, small_images)
            
            self.optimizer.zero_grad()
            outputs = self.model(combined_images).squeeze()
            loss = self.criterion(outputs, angles)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(angles)
            total_samples += len(angles)
            
            pbar.set_postfix({'loss': loss.item()})
            
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), epoch * len(self.train_loader) + batch_idx)
        
        avg_loss = total_loss / total_samples
        return avg_loss
    
    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                big_images = batch['big_image'].to(self.device)
                small_images = batch['small_image'].to(self.device)
                angles = batch['angle'].to(self.device)
                
                combined_images = self._combine_images(big_images, small_images)
                
                outputs = self.model(combined_images).squeeze()
                loss = self.criterion(outputs, angles)
                
                total_loss += loss.item() * len(angles)
                total_samples += len(angles)
                
                pred_values = outputs.cpu().numpy()
                if np.isscalar(pred_values):
                    pred_values = [pred_values]
                all_predictions.extend(pred_values)
                all_targets.extend(angles.cpu().numpy())
        
        avg_loss = total_loss / total_samples
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Calculate circular error
        errors = np.abs(predictions - targets)
        errors = np.minimum(errors, 360 - errors)
        
        metrics = {
            'val_loss': avg_loss,
            'mae': np.mean(errors),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'max_error': np.max(errors)
        }
        
        return metrics
    
    def _combine_images(self, big_images, small_images):
        # Resize both images to target size (128x128) and combine
        import torch.nn.functional as F
        
        # Resize both images to 128x128
        target_size = (128, 128)
        big_resized = F.interpolate(big_images, size=target_size, mode='bilinear', align_corners=False)
        
        if small_images.shape[1] == 4:
            # Handle RGBA small images
            small_resized = F.interpolate(small_images, size=target_size, mode='bilinear', align_corners=False)
            alpha = small_resized[:, 3:4, :, :]
            rgb_small = small_resized[:, :3, :, :]
            combined = big_resized * (1 - alpha) + rgb_small * alpha
        else:
            # Handle RGB small images
            small_resized = F.interpolate(small_images[:, :3, :, :], size=target_size, mode='bilinear', align_corners=False)
            combined = big_resized
        
        return combined
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val MAE: {val_metrics['mae']:.2f}°, Val RMSE: {val_metrics['rmse']:.2f}°")
            
            # Log to tensorboard
            self.writer.add_scalar('Train/Epoch_Loss', train_loss, epoch)
            self.writer.add_scalar('Val/Epoch_Loss', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('Val/MAE', val_metrics['mae'], epoch)
            self.writer.add_scalar('Val/RMSE', val_metrics['rmse'], epoch)
            
            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                torch.save(self.model.state_dict(), os.path.join(self.config['save_dir'], 'best_model.pth'))
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
        
        self.writer.close()
    
    def predict(self, images):
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            predictions = self.model(images)
            return predictions.cpu().numpy()


def get_default_config():
    return {
        'data_dir': '/Users/barnettxu/projects/anti-captcha/data/images',
        'model_type': 'simple',
        'loss_type': 'angle',
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'learning_rate': 1e-3,
        'batch_size': 32,
        'num_workers': 4,
        'num_epochs': 50,
        'log_dir': './logs',
        'save_dir': './checkpoints'
    }