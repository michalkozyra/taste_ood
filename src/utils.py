"""
Utility functions for Stein shift detection.
"""

import torch
import torch.nn as nn
from typing import Optional


def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device: The device to use
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def has_maxpool_layers(model: nn.Module) -> bool:
    """
    Check if a model contains MaxPool layers (which are not twice differentiable).
    
    Args:
        model: PyTorch model
    
    Returns:
        bool: True if model contains MaxPool2d or MaxPool1d layers
    """
    for module in model.modules():
        if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)):
            return True
    return False


def is_resnet_model(model: nn.Module) -> bool:
    """
    Check if a model is a ResNet (torchvision ResNet architecture).
    
    Args:
        model: PyTorch model
    
    Returns:
        bool: True if model has ResNet structure (layer4, avgpool, fc)
    """
    return (
        hasattr(model, 'layer4') and 
        hasattr(model, 'avgpool') and 
        hasattr(model, 'fc')
    )
