import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np


class RotationTransforms:
    @staticmethod
    def get_train_transforms():
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_val_transforms():
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class Preprocessor:
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size
    
    def process_big_image(self, image):
        image = image.resize(self.target_size)
        return np.array(image)
    
    def process_small_image(self, image):
        image = image.resize(self.target_size)
        return np.array(image)
    
    def combine_images(self, big_image, small_image):
        if big_image.shape[-1] == 3:
            big_image = np.concatenate([big_image, np.ones((*big_image.shape[:2], 1))], axis=-1)
        
        if small_image.shape[-1] == 4:
            alpha = small_image[:, :, 3:4]
            rgb_small = small_image[:, :, :3]
            combined = big_image.copy()
            combined[:, :, :3] = big_image[:, :, :3] * (1 - alpha) + rgb_small * alpha
            return combined[:, :, :3]
        else:
            return big_image[:, :, :3]
    
    def __call__(self, big_image, small_image):
        big_processed = self.process_big_image(big_image)
        small_processed = self.process_small_image(small_image)
        combined = self.combine_images(big_processed, small_processed)
        
        combined = torch.tensor(combined, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return combined