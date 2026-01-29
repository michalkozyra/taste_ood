"""
Helper script to prepare models for benchmark evaluation.

This script can:
1. Download pre-trained models from torchvision
2. Train models on CIFAR-10/100
3. Save checkpoints in the expected format
"""

import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import train_classifier, train_score_model_annealed
from src.models import UNetScore
from src.utils import get_device


def download_pretrained_imagenet(output_path: str = 'checkpoints/imagenet_resnet50.pth'):
    """Download pre-trained ImageNet ResNet-50."""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    print("Downloading pre-trained ImageNet ResNet-50...")
    # Use newer weights API if available, fallback to pretrained
    try:
        from torchvision.models import ResNet50_Weights
        model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except (ImportError, AttributeError):
        # Fallback to deprecated pretrained parameter
        model = torchvision.models.resnet50(pretrained=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved to {output_path}")
    return model


def train_cifar10_classifier(
    epochs: int = 100,
    output_path: str = 'checkpoints/cifar10_resnet18.pth',
    data_dir: str = './data',
    use_pretrained: bool = True,
):
    """
    Train ResNet-18 on CIFAR-10.
    
    Args:
        epochs: Number of training epochs
        output_path: Path to save checkpoint
        data_dir: Data directory
        use_pretrained: If True, start from ImageNet pre-trained weights
    """
    from scripts.benchmark_ood_evaluation import get_cifar10_dataloaders
    
    print("Training ResNet-18 on CIFAR-10...")
    device = get_device()
    
    # Load data
    train_loader, val_loader, _ = get_cifar10_dataloaders(
        data_dir=data_dir, batch_size=128, num_workers=4
    )
    
    # Create model - optionally start from ImageNet pre-trained
    if use_pretrained:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        print("Starting from ImageNet pre-trained ResNet-18...")
        # Use newer weights API if available, fallback to pretrained
        try:
            from torchvision.models import ResNet18_Weights
            model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            # Fallback to deprecated pretrained parameter
            model = torchvision.models.resnet18(pretrained=True)
        # Replace final layer for CIFAR-10 (10 classes)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
    else:
        model = torchvision.models.resnet18(num_classes=10)
    
    # Train
    model = train_classifier(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=1e-3,
        checkpoint_path=output_path
    )
    
    print(f"Training complete. Model saved to {output_path}")
    return model


def train_cifar10_score_model(
    epochs: int = 50,
    output_path: str = 'checkpoints/score_unet_cifar10.pth',
    data_dir: str = './data',
):
    """Train score model on CIFAR-10."""
    from scripts.benchmark_ood_evaluation import get_cifar10_dataloaders
    
    print("Training UNet score model on CIFAR-10...")
    device = get_device()
    
    # Load data
    train_loader, _, _ = get_cifar10_dataloaders(
        data_dir=data_dir, batch_size=128, num_workers=4
    )
    
    # Create score model (CIFAR-10 is RGB, so 3 channels)
    score_model = UNetScore(in_channels=3)
    
    # Train
    score_model, sigmas = train_score_model_annealed(
        score_model, train_loader.dataset, device,
        epochs=epochs, batch_size=128, lr=2e-4,
        ckpt=output_path
    )
    
    print(f"Training complete. Score model saved to {output_path}")
    return score_model, sigmas


def main():
    parser = argparse.ArgumentParser(description='Prepare models for benchmark evaluation')
    parser.add_argument(
        '--action',
        type=str,
        required=True,
        choices=['download-imagenet', 'train-cifar10-classifier', 'train-cifar10-score'],
        help='Action to perform',
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output path for model checkpoint',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (for training actions)',
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Data directory',
    )
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Do not use pre-trained weights (train from scratch)',
    )
    
    args = parser.parse_args()
    
    if args.action == 'download-imagenet':
        output_path = args.output_path or 'checkpoints/imagenet_resnet50.pth'
        download_pretrained_imagenet(output_path)
    
    elif args.action == 'train-cifar10-classifier':
        output_path = args.output_path or 'checkpoints/cifar10_resnet18.pth'
        epochs = args.epochs or 100
        use_pretrained = not args.no_pretrained
        train_cifar10_classifier(
            epochs=epochs, 
            output_path=output_path, 
            data_dir=args.data_dir,
            use_pretrained=use_pretrained,
        )
    
    elif args.action == 'train-cifar10-score':
        output_path = args.output_path or 'checkpoints/score_unet_cifar10.pth'
        epochs = args.epochs or 50
        train_cifar10_score_model(epochs=epochs, output_path=output_path, data_dir=args.data_dir)


if __name__ == '__main__':
    main()

