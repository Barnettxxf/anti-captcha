import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import random
from PIL import Image


class AdvancedAugmentation:
    def __init__(self, image_size=(128, 128)):
        self.image_size = image_size
        
    def get_train_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_val_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class RotationAugmentation:
    def __init__(self, max_rotation=30):
        self.max_rotation = max_rotation
    
    def __call__(self, big_image, small_image):
        # Random rotation augmentation
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        
        # Apply rotation to both images
        rotated_big = F.rotate(big_image, angle)
        rotated_small = F.rotate(small_image, angle)
        
        return rotated_big, rotated_small


class NoiseInjection:
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std
    
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_std
        return torch.clamp(tensor + noise, 0, 1)


class AdvancedPreprocessor:
    def __init__(self, image_size=(128, 128)):
        self.image_size = image_size
        self.train_transform = AdvancedAugmentation(image_size).get_train_transforms()
        self.val_transform = AdvancedAugmentation(image_size).get_val_transforms()
        
    def process_pair(self, big_image, small_image, training=True):
        if training:
            # Apply rotation augmentation
            rotation_aug = RotationAugmentation(max_rotation=15)
            big_image, small_image = rotation_aug(big_image, small_image)
            
            # Apply color jitter
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            big_image = color_jitter(big_image)
            
            # Resize
            resize = transforms.Resize(self.image_size)
            big_image = resize(big_image)
            small_image = resize(small_image)
            
            # Convert to tensor
            big_tensor = transforms.ToTensor()(big_image)
            small_tensor = transforms.ToTensor()(small_image)
            
            # Apply normalization
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            big_tensor = normalize(big_tensor)
            
            # Handle small image with alpha channel
            if small_tensor.shape[0] == 4:
                alpha = small_tensor[3:4, :, :]
                rgb_small = small_tensor[:3, :, :]
                combined = big_tensor * (1 - alpha) + rgb_small * alpha
            else:
                combined = big_tensor
                
        else:
            big_tensor = self.val_transform(big_image)
            small_tensor = self.val_transform(small_image)
            
            if small_tensor.shape[0] == 4:
                alpha = small_tensor[3:4, :, :]
                rgb_small = small_tensor[:3, :, :]
                combined = big_tensor * (1 - alpha) + rgb_small * alpha
            else:
                combined = big_tensor
        
        return combined


class MixUp:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x1, x2, y1, y2):
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        return mixed_x, mixed_y