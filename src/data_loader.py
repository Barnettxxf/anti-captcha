import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class RotationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.big_images_dir = os.path.join(data_dir, 'big_images')
        self.small_images_dir = os.path.join(data_dir, 'small_images')
        self.labels_dir = os.path.join(data_dir, 'labels')
        
        self.samples = self._get_samples()
    
    def _get_samples(self):
        samples = []
        
        for filename in os.listdir(self.big_images_dir):
            if filename.endswith('_big.jpeg'):
                id_prefix = filename.replace('_big.jpeg', '')
                
                big_path = os.path.join(self.big_images_dir, f'{id_prefix}_big.jpeg')
                small_path = os.path.join(self.small_images_dir, f'{id_prefix}_small.png')
                label_path = os.path.join(self.labels_dir, f'{id_prefix}_label.txt')
                
                if os.path.exists(big_path) and os.path.exists(small_path) and os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                        # Handle various formats including null bytes
                        content = content.replace('\x00', '').strip()
                        if content:
                            angle = float(content)
                        else:
                            continue
                    
                    samples.append({
                        'big_path': big_path,
                        'small_path': small_path,
                        'angle': angle
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        big_image = Image.open(sample['big_path']).convert('RGB')
        small_image = Image.open(sample['small_path']).convert('RGBA')
        
        angle = sample['angle']
        
        if self.transform:
            big_image = self.transform(big_image)
            small_image = self.transform(small_image)
        else:
            big_image = torch.tensor(np.array(big_image), dtype=torch.float32).permute(2, 0, 1) / 255.0
            small_image = torch.tensor(np.array(small_image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        return {
            'big_image': big_image,
            'small_image': small_image,
            'angle': torch.tensor(angle, dtype=torch.float32)
        }


def get_data_loaders(data_dir, batch_size=32, num_workers=4, train_split=0.8):
    dataset = RotationDataset(data_dir)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader