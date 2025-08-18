import torch
import torch.nn as nn
import torch.nn.functional as F


class AngleLoss(nn.Module):
    def __init__(self):
        super(AngleLoss, self).__init__()
    
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        diff = torch.min(diff, 360 - diff)
        return torch.mean(diff)


class CircularLoss(nn.Module):
    def __init__(self):
        super(CircularLoss, self).__init__()
    
    def forward(self, pred, target):
        pred_rad = torch.deg2rad(pred)
        target_rad = torch.deg2rad(target)
        
        cos_pred = torch.cos(pred_rad)
        sin_pred = torch.sin(pred_rad)
        cos_target = torch.cos(target_rad)
        sin_target = torch.sin(target_rad)
        
        cos_loss = F.mse_loss(cos_pred, cos_target)
        sin_loss = F.mse_loss(sin_pred, sin_target)
        
        return cos_loss + sin_loss


class CombinedLoss(nn.Module):
    def __init__(self, angle_weight=1.0, circular_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.angle_loss = AngleLoss()
        self.circular_loss = CircularLoss()
        self.angle_weight = angle_weight
        self.circular_weight = circular_weight
    
    def forward(self, pred, target):
        angle_loss = self.angle_loss(pred, target)
        circular_loss = self.circular_loss(pred, target)
        
        return self.angle_weight * angle_loss + self.circular_weight * circular_loss


class SmoothL1AngleLoss(nn.Module):
    def __init__(self):
        super(SmoothL1AngleLoss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
    
    def forward(self, pred, target):
        diff = pred - target
        diff = torch.remainder(diff + 180, 360) - 180
        return self.smooth_l1(diff, torch.zeros_like(diff))