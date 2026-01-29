"""
Main OOD Detection Benchmark Evaluation Script.

This script evaluates SteinDetector against baseline methods (MSP, ODIN, Energy, Mahalanobis, KNN)
on standard OOD detection benchmarks (CIFAR-10/100, ImageNet-1K with various OOD test sets).

Usage:
    python scripts/benchmark_ood_evaluation.py \
        --id-dataset cifar10 \
        --ood-dataset tinyimagenet \
        --model-path checkpoints/cifar10_resnet.pth \
        --output-dir results/benchmark_results
"""

import os
import sys
import argparse
import json
import csv
import urllib.request
import tarfile
import ssl
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import SteinDetector, SteinFactoryDetector, ScoreNormDetector, GradFNormDetector
from src.detector.baselines import create_all_baseline_detectors
from src.evaluation import (
    evaluate_all_detectors,
    fit_all_detectors,
    save_results,
    print_results_summary,
)
from src.evaluation.ood_benchmark import _agent_log as _agent_log
from src.utils import get_device

# region agent log
_AGENT_DEBUG_LOG_PATH = "/Users/michalkozyra/Developer/PhD/stein_shift_detection/.cursor/debug.log"
def _agent_maybe_clear_debug_log() -> None:
    """
    Debug-only helper: clear NDJSON debug log at the start of a run.
    Enabled via env var AGENT_CLEAR_DEBUG_LOG=1.
    """
    try:
        if os.environ.get("AGENT_CLEAR_DEBUG_LOG", "0") == "1":
            with open(_AGENT_DEBUG_LOG_PATH, "w") as f:
                f.write("")
    except Exception:
        pass
# endregion


# ============================================================================
# Dataset Loading Functions
# ============================================================================

def convert_to_rgb(img):
    """Convert PIL image to RGB format (for multiprocessing compatibility)."""
    return img.convert('RGB')


def subsample_cifar10c_by_severity(
    images,
    labels,
    corruption_type: str,
    samples_per_severity: int = 1000,
    base_seed: int = 42,
):
    """
    Subsample CIFAR-10-C data by selecting random samples from each severity level.
    
    Args:
        images: Array of shape (50000, H, W, C) - all 5 severities concatenated
        labels: Array of shape (50000,) - labels (repeated 5 times)
        corruption_type: Name of corruption type (for seed offset)
        samples_per_severity: Number of samples to select from each severity (default: 1000)
        base_seed: Base random seed (default: 42)
    
    Returns:
        subsampled_images: Array of shape (samples_per_severity * 5, H, W, C)
        subsampled_labels: Array of shape (samples_per_severity * 5,)
    """
    import numpy as np
    
    # Verify input shape
    if len(images) != 50000:
        raise ValueError(
            f"Expected 50000 samples, got {len(images)}. "
            "CIFAR-10-C should have 5 severities × 10000 samples each."
        )
    
    if len(labels) != 50000:
        raise ValueError(
            f"Expected 50000 labels, got {len(labels)}. "
            "Labels should match images."
        )
    
    # Define all corruption types in order (for consistent offset)
    all_corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]
    
    # Get offset based on corruption type index
    if corruption_type not in all_corruption_types:
        # Fallback: use hash if type not in list
        import hashlib
        offset = int(hashlib.md5(corruption_type.encode()).hexdigest()[:4], 16) % 1000
    else:
        offset = all_corruption_types.index(corruption_type)
    
    # Calculate seed: base_seed + offset
    seed = base_seed + offset
    
    # Split into 5 severity groups (10K samples each)
    severity_groups = [
        (0, 10000),      # Severity 1: indices 0-9999
        (10000, 20000),  # Severity 2: indices 10000-19999
        (20000, 30000),  # Severity 3: indices 20000-29999
        (30000, 40000),  # Severity 4: indices 30000-39999
        (40000, 50000),  # Severity 5: indices 40000-49999
    ]
    
    subsampled_images = []
    subsampled_labels = []
    
    for severity_idx, (start_idx, end_idx) in enumerate(severity_groups):
        # Set seed for this severity (different seed per severity to ensure diversity)
        # Use: base_seed + offset + severity_index
        severity_seed = seed + severity_idx
        np.random.seed(severity_seed)
        
        # Get indices for this severity
        severity_indices = np.arange(start_idx, end_idx)
        
        # Randomly sample from this severity
        if samples_per_severity > len(severity_indices):
            # If requesting more than available, use all
            selected_indices = severity_indices
        else:
            selected_indices = np.random.choice(
                severity_indices, 
                size=samples_per_severity, 
                replace=False
            )
        
        # Extract images and labels
        subsampled_images.append(images[selected_indices])
        subsampled_labels.append(labels[selected_indices])
    
    # Concatenate all severities
    subsampled_images = np.concatenate(subsampled_images, axis=0)
    subsampled_labels = np.concatenate(subsampled_labels, axis=0)
    
    return subsampled_images, subsampled_labels


# Module-level dataset classes (required for multiprocessing pickling)
class CIFAR10CDataset(Dataset):
    """Dataset for CIFAR-10-C corrupted images with ground truth labels."""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        img = self.images[idx]
        label = int(self.labels[idx])
        # CIFAR-10-C images are in HWC format (uint8)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label  # Use actual ground truth labels


class CIFAR10PDataset(Dataset):
    """Dataset for CIFAR-10-P perturbed images with CIFAR-10 test labels."""
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        img = self.images[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        # Use actual CIFAR-10 test labels if available, otherwise dummy label
        label = int(self.labels[idx]) if self.labels is not None else 0
        return img, label


def download_cifar10c(data_dir: str = './data') -> Path:
    """
    Download and extract CIFAR-10-C dataset from Zenodo.
    
    Args:
        data_dir: Directory to store the dataset
    
    Returns:
        Path to the CIFAR-10-C directory
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    cifar10c_dir = data_path / 'cifar10-c'
    cifar10c_tar = data_path / 'CIFAR-10-C.tar'
    
    # Check if already extracted
    cifar10c_data_dir = cifar10c_dir / 'CIFAR-10-C'
    if cifar10c_data_dir.exists():
        # Check if it has the expected structure: CIFAR-10-C/{corruption_type}.npy
        corruption_types = ['gaussian_noise', 'shot_noise', 'impulse_noise']
        has_data = any((cifar10c_data_dir / f'{ct}.npy').exists() for ct in corruption_types)
        if has_data:
            print(f"CIFAR-10-C already exists at {cifar10c_data_dir}")
            return cifar10c_dir
    
    # Download URL from Zenodo
    # CIFAR-10-C is available at: https://zenodo.org/record/2535967
    # Direct download link for the tar file (Zenodo file ID: 3555552)
    download_url = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
    
    print(f"\n{'='*80}")
    print("Downloading CIFAR-10-C dataset...")
    print(f"Source: https://zenodo.org/record/2535967")
    print(f"Destination: {cifar10c_tar}")
    print(f"{'='*80}\n")
    
    # Handle SSL certificate issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download with progress bar
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100.0 / total_size, 100.0)
            print(f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end='', flush=True)
        
        # Check if tar file exists and is valid
        tar_valid = False
        if cifar10c_tar.exists():
            # Try to validate the tar file
            try:
                with tarfile.open(cifar10c_tar, 'r') as test_tar:
                    test_tar.getmembers()  # Try to read members
                tar_valid = True
                print(f"Using existing tar file: {cifar10c_tar}")
            except (tarfile.TarError, EOFError, OSError) as e:
                print(f"Existing tar file appears corrupted: {e}")
                print("Will re-download...")
                cifar10c_tar.unlink()  # Delete corrupted file
        
        if not tar_valid:
            print("Downloading CIFAR-10-C.tar (this may take a while, ~1.2 GB)...")
            urllib.request.urlretrieve(download_url, str(cifar10c_tar), reporthook=show_progress)
            print("\n✓ Download complete")
            
            # Validate downloaded file
            try:
                with tarfile.open(cifar10c_tar, 'r') as test_tar:
                    test_tar.getmembers()
                print("✓ Downloaded tar file is valid")
            except (tarfile.TarError, EOFError, OSError) as e:
                raise RuntimeError(
                    f"Downloaded tar file is corrupted: {e}. "
                    "Please try downloading manually from https://zenodo.org/record/2535967"
                )
        
        # Extract
        print(f"\nExtracting CIFAR-10-C.tar to {cifar10c_dir}...")
        cifar10c_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(cifar10c_tar, 'r') as tar:
                # Extract with progress
                members = tar.getmembers()
                for member in tqdm(members, desc="Extracting", unit="files"):
                    tar.extract(member, path=cifar10c_dir)
        except (tarfile.TarError, EOFError, OSError) as e:
            # If extraction fails, delete corrupted tar and suggest manual download
            print(f"\n✗ Extraction failed: {e}")
            if cifar10c_tar.exists():
                print(f"Removing corrupted tar file: {cifar10c_tar}")
                cifar10c_tar.unlink()
            raise
        
        print(f"\n✓ CIFAR-10-C extracted to {cifar10c_dir}")
        return cifar10c_dir
        
    except Exception as e:
        print(f"\n✗ Download/extraction failed: {e}")
        print("\n" + "="*80)
        print("ERROR: Could not download CIFAR-10-C automatically.")
        print("="*80)
        print("\nPlease download manually:")
        print("1. Visit: https://zenodo.org/record/2535967")
        print("2. Download 'CIFAR-10-C.tar'")
        print(f"3. Extract to: {cifar10c_dir}")
        print("\nOr use curl/wget:")
        print(f"  curl -L {download_url} -o {cifar10c_tar}")
        print(f"  tar -xf {cifar10c_tar} -C {data_path}")
        print("="*80)
        raise FileNotFoundError(
            f"CIFAR-10-C not found at {cifar10c_dir}. "
            "Please download manually from Zenodo (see instructions above)."
        )


def download_cifar10p(data_dir: str = './data') -> Path:
    """
    Download and extract CIFAR-10-P dataset from Zenodo.
    
    Args:
        data_dir: Directory to store the dataset
    
    Returns:
        Path to the CIFAR-10-P directory
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    cifar10p_dir = data_path / 'cifar10-p'
    cifar10p_tar = data_path / 'CIFAR-10-P.tar'
    
    # Check if already extracted
    cifar10p_data_dir = cifar10p_dir / 'CIFAR-10-P'
    if cifar10p_data_dir.exists():
        # Check if it has the expected structure: CIFAR-10-P/{perturbation_type}.npy
        perturbation_types = ['gaussian_noise', 'shot_noise', 'motion_blur']
        has_data = any((cifar10p_data_dir / f'{pt}.npy').exists() for pt in perturbation_types)
        if has_data:
            print(f"CIFAR-10-P already exists at {cifar10p_data_dir}")
            return cifar10p_dir
    
    # Download URL from Zenodo
    # CIFAR-10-P is available at: https://zenodo.org/record/2535967
    # Direct download link for the tar file (Zenodo file ID: 3555553)
    download_url = "https://zenodo.org/records/2535967/files/CIFAR-10-P.tar?download=1"
    
    print(f"\n{'='*80}")
    print("Downloading CIFAR-10-P dataset...")
    print(f"Source: https://zenodo.org/record/2535967")
    print(f"Destination: {cifar10p_tar}")
    print(f"{'='*80}\n")
    
    # Handle SSL certificate issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download with progress bar
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100.0 / total_size, 100.0)
            print(f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end='', flush=True)
        
        # Check if tar file exists and is valid
        tar_valid = False
        if cifar10p_tar.exists():
            # Try to validate the tar file
            try:
                with tarfile.open(cifar10p_tar, 'r') as test_tar:
                    test_tar.getmembers()  # Try to read members
                tar_valid = True
                print(f"Using existing tar file: {cifar10p_tar}")
            except (tarfile.TarError, EOFError, OSError) as e:
                print(f"Existing tar file appears corrupted: {e}")
                print("Will re-download...")
                cifar10p_tar.unlink()  # Delete corrupted file
        
        if not tar_valid:
            print("Downloading CIFAR-10-P.tar (this may take a while, ~1.2 GB)...")
            urllib.request.urlretrieve(download_url, str(cifar10p_tar), reporthook=show_progress)
            print("\n✓ Download complete")
            
            # Validate downloaded file
            try:
                with tarfile.open(cifar10p_tar, 'r') as test_tar:
                    test_tar.getmembers()
                print("✓ Downloaded tar file is valid")
            except (tarfile.TarError, EOFError, OSError) as e:
                raise RuntimeError(
                    f"Downloaded tar file is corrupted: {e}. "
                    "Please try downloading manually from https://zenodo.org/record/2535967"
                )
        
        # Extract
        print(f"\nExtracting CIFAR-10-P.tar to {cifar10p_dir}...")
        cifar10p_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(cifar10p_tar, 'r') as tar:
                # Extract with progress
                members = tar.getmembers()
                for member in tqdm(members, desc="Extracting", unit="files"):
                    tar.extract(member, path=cifar10p_dir)
        except (tarfile.TarError, EOFError, OSError) as e:
            # If extraction fails, delete corrupted tar and suggest manual download
            print(f"\n✗ Extraction failed: {e}")
            if cifar10p_tar.exists():
                print(f"Removing corrupted tar file: {cifar10p_tar}")
                cifar10p_tar.unlink()
            raise
        
        print(f"\n✓ CIFAR-10-P extracted to {cifar10p_dir}")
        return cifar10p_dir
        
    except Exception as e:
        print(f"\n✗ Download/extraction failed: {e}")
        print("\n" + "="*80)
        print("ERROR: Could not download CIFAR-10-P automatically.")
        print("="*80)
        print("\nPlease download manually:")
        print("1. Visit: https://zenodo.org/record/2535967")
        print("2. Download 'CIFAR-10-P.tar'")
        print(f"3. Extract to: {cifar10p_dir}")
        print("\nOr use curl/wget:")
        print(f"  curl -L {download_url} -o {cifar10p_tar}")
        print(f"  tar -xf {cifar10p_tar} -C {data_path}")
        print("="*80)
        raise FileNotFoundError(
            f"CIFAR-10-P not found at {cifar10p_dir}. "
            "Please download manually from Zenodo (see instructions above)."
        )

def get_cifar10_dataloaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    import ssl
    # Disable SSL verification for dataset downloads (common issue on macOS)
    ssl._create_default_https_context = ssl._create_unverified_context
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # Split train into train/val (90/10)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def get_cifar100_dataloaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load CIFAR-100 dataset.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    import ssl
    # Disable SSL verification for dataset downloads (common issue on macOS)
    ssl._create_default_https_context = ssl._create_unverified_context
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # Split train into train/val (90/10)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def get_imagenet_dataloaders(
    data_dir: str = './data/imagenet',
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load ImageNet-1K dataset.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_dataset = torchvision.datasets.ImageNet(
        root=data_dir, split='train', transform=transform_train
    )
    val_dataset = torchvision.datasets.ImageNet(
        root=data_dir, split='val', transform=transform_val
    )
    
    # Use validation set as test set
    test_dataset = val_dataset
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def load_ood_dataset(
    dataset_name: str,
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 32,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
    # For adversarial datasets
    model: Optional[nn.Module] = None,
    test_loader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    """
    Load an OOD test dataset.
    
    Args:
        dataset_name: Name of OOD dataset
        data_dir: Directory for datasets
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Target image size (for resizing)
        mean: Normalization mean (default: CIFAR-10 values)
        std: Normalization std (default: CIFAR-10 values)
        model: Model for generating adversarial examples (required for adversarial datasets)
        test_loader: ID test loader for generating adversarial examples (required for adversarial datasets)
        device: Device for adversarial generation (required for adversarial datasets)
        cache_dir: Cache directory for adversarial examples (optional)
    
    Returns:
        DataLoader for OOD dataset
    """
    # Default normalization (CIFAR-10)
    if mean is None:
        mean = (0.4914, 0.4822, 0.4465)
    if std is None:
        std = (0.2023, 0.1994, 0.2010)
    
    # Standard transform for OOD datasets (resize to match ID dataset)
    # Convert to RGB to handle grayscale images (e.g., some Places365 images)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(convert_to_rgb),  # Ensure RGB format (module-level function for pickling)
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    dataset_name_lower = dataset_name.lower()
    dataset_name_original = dataset_name  # Keep original for parsing
    
    # Check if this is an adversarial dataset
    if dataset_name_lower.startswith('adversarial:'):
        from src.adversarial import parse_adversarial_dataset_name, generate_adversarial_dataset
        
        if model is None or test_loader is None or device is None:
            raise ValueError(
                "Adversarial datasets require model, test_loader, and device. "
                "Please provide these parameters when loading adversarial datasets."
            )
        
        # Parse attack configuration
        config = parse_adversarial_dataset_name(dataset_name_original)
        if config is None:
            raise ValueError(f"Failed to parse adversarial dataset name: {dataset_name_original}")
        
        # Determine dataset name for cache organization
        # Try to infer from test_loader if possible, otherwise use 'unknown'
        dataset_name_for_cache = 'unknown'
        if hasattr(test_loader.dataset, '__class__'):
            dataset_class_name = test_loader.dataset.__class__.__name__.lower()
            if 'cifar10' in dataset_class_name:
                dataset_name_for_cache = 'cifar10'
            elif 'cifar100' in dataset_class_name:
                dataset_name_for_cache = 'cifar100'
            elif 'imagenet' in dataset_class_name:
                dataset_name_for_cache = 'imagenet'
        
        # Set cache directory
        if cache_dir is None:
            cache_dir = os.path.join(data_dir, 'adversarial_cache')
        
        # Generate adversarial dataset
        adversarial_dataset, stats = generate_adversarial_dataset(
            model=model,
            data_loader=test_loader,
            config=config,
            device=device,
            cache_dir=cache_dir,
            dataset_name=dataset_name_for_cache,
            model_name=None,  # Auto-detect
            max_samples=None,  # Use all samples
            verbose=True,
            mean=mean,
            std=std,
        )
        
        print(f"  Adversarial dataset statistics:")
        print(f"    Attack type: {stats['attack_type']}")
        print(f"    Threat model: {stats['threat_model']}")
        print(f"    Epsilon: {stats['epsilon']}")
        print(f"    Attack success rate: {stats['attack_success_rate']*100:.2f}%")
        
        # Apply normalization to adversarial dataset
        # Adversarial examples are in [0, 1] range, need to normalize them
        from src.adversarial.adversarial_dataset import NormalizedAdversarialDataset
        
        normalized_dataset = NormalizedAdversarialDataset(
            adversarial_dataset.adversarial_images,
            adversarial_dataset.labels,
            mean=mean,
            std=std,
        )
        
        # Create DataLoader
        # Use num_workers=0 for adversarial datasets to avoid pickling issues
        # (though NormalizedAdversarialDataset is now picklable, this is safer)
        loader = DataLoader(
            normalized_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid any pickling issues
        )
        return loader
    
    # CIFAR-10 OOD datasets (32x32)
    if dataset_name_lower == 'svhn':
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        dataset = torchvision.datasets.SVHN(
            root=data_dir, split='test', download=True, transform=transform
        )
    elif dataset_name_lower == 'lsun':
        # Use PyTorch-OOD's LSUNResize
        from pytorch_ood.dataset.img import LSUNResize
        dataset = LSUNResize(root=data_dir, download=True, transform=transform)
    elif dataset_name_lower == 'isun' or dataset_name_lower == 'isun':
        # iSUN - use TinyImageNetResize from PyTorch-OOD
        from pytorch_ood.dataset.img import TinyImageNetResize
        dataset = TinyImageNetResize(root=data_dir, download=True, transform=transform)
    elif dataset_name_lower == 'textures' or dataset_name_lower == 'texture':
        # Describable Textures Dataset (DTD)
        from pytorch_ood.dataset.img import Textures
        dataset = Textures(root=data_dir, download=True, transform=transform)
    elif dataset_name_lower == 'places365' or dataset_name_lower == 'places':
        # Places365
        from pytorch_ood.dataset.img import Places365
        dataset = Places365(root=data_dir, download=True, transform=transform)
    elif dataset_name_lower.startswith('cifar10c') or dataset_name_lower.startswith('cifar10-c'):
        # CIFAR-10-C: Corrupted CIFAR-10 test set
        # Download from: https://zenodo.org/record/2535967
        # Supports individual corruption types: 'cifar10c:gaussian_noise' or 'cifar10c' (all)
        import numpy as np
        from PIL import Image
        
        # Automatically download if not found
        cifar10c_dir = download_cifar10c(data_dir)
        
        # Actual structure: data/cifar10-c/CIFAR-10-C/{corruption_type}.npy
        cifar10c_data_dir = cifar10c_dir / 'CIFAR-10-C'
        
        # Check if a specific corruption type is requested (format: 'cifar10c:gaussian_noise')
        # IMPORTANT: Each corruption type must be specified separately - no concatenation of all types
        if ':' in dataset_name_original:
            # Extract corruption type from dataset name
            corr_type = dataset_name_original.split(':', 1)[1]
        else:
            # If no specific type requested, raise error - user must specify individual corruption type
            raise ValueError(
                "CIFAR-10-C requires specifying a corruption type. "
                "Use format: 'cifar10c:corruption_type' (e.g., 'cifar10c:gaussian_noise'). "
                "Available types: gaussian_noise, shot_noise, impulse_noise, defocus_blur, "
                "glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, "
                "elastic_transform, pixelate, jpeg_compression, speckle_noise, gaussian_blur, "
                "spatter, saturate"
            )
        
        # All corruption types available in CIFAR-10-C
        all_corruption_types = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
            'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
        ]
        
        # Validate corruption type
        if corr_type not in all_corruption_types:
            raise ValueError(
                f"Unknown CIFAR-10-C corruption type: {corr_type}. "
                f"Available types: {all_corruption_types}"
            )
        
        # Load only the specified corruption type (all 5 severities, 50k samples)
        corruption_types = [corr_type]
        
        # Load the specified corruption type (all 5 severities, 50k samples)
        corr_path = cifar10c_data_dir / f'{corr_type}.npy'
        labels_path = cifar10c_data_dir / 'labels.npy'
        
        if not corr_path.exists():
            raise FileNotFoundError(
                f"No CIFAR-10-C data found for corruption type '{corr_type}'. "
                f"Expected: {cifar10c_data_dir}/{corr_type}.npy"
            )
        
        if not labels_path.exists():
            raise FileNotFoundError(
                f"CIFAR-10-C labels file not found. Expected: {labels_path}"
            )
        
        images = np.load(corr_path)
        labels = np.load(labels_path)
        
        # CIFAR-10-C files contain all 5 severities concatenated
        # Shape is (50000, 32, 32, 3) = (5 severities × 10000 samples, H, W, C)
        # Labels are also (50000,) - same order, repeated 5 times (once per severity)
        if images.ndim == 5:
            # If 5D: (n_severities, n_samples, H, W, C) - reshape to concatenate all severities
            # Reshape from (5, 10000, 32, 32, 3) to (50000, 32, 32, 3)
            images = images.reshape(-1, *images.shape[2:])
        # If 4D, use as-is (already has all severities concatenated)
        
        # Verify labels match images
        if len(labels) != len(images):
            raise ValueError(
                f"Label count ({len(labels)}) doesn't match image count ({len(images)}) "
                f"for corruption type '{corr_type}'"
            )
        
        # Subsample to 5K samples (1K per severity level)
        # Use fixed seed with offset per corruption type for reproducibility
        images, labels = subsample_cifar10c_by_severity(
            images, labels,
            corruption_type=corr_type,
            samples_per_severity=1000,
            base_seed=42,
        )
        print(f"  Subsampled CIFAR-10-C ({corr_type}): {len(images)} samples (1K per severity)")
        
        dataset = CIFAR10CDataset(images, labels, transform=transform)
    elif dataset_name_lower.startswith('cifar10p') or dataset_name_lower.startswith('cifar10-p'):
        # CIFAR-10-P: Perturbed CIFAR-10 test set
        # Download from: https://zenodo.org/record/2535967
        # Supports individual perturbation types: 'cifar10p:gaussian_noise' or 'cifar10p' (all)
        import numpy as np
        from PIL import Image
        
        # Automatically download if not found
        cifar10p_dir = download_cifar10p(data_dir)
        
        # Actual structure: data/cifar10-p/CIFAR-10-P/{perturbation_type}.npy
        cifar10p_data_dir = cifar10p_dir / 'CIFAR-10-P'
        
        # Check if a specific perturbation type is requested (format: 'cifar10p:gaussian_noise')
        if ':' in dataset_name_original:
            # Extract perturbation type from dataset name
            pert_type = dataset_name_original.split(':', 1)[1]
        else:
            # Load all perturbation types (legacy behavior)
            pert_type = None
        
        # CIFAR-10-P has perturbation sequences
        # For benchmarking, we'll use the last frame of each sequence (most perturbed)
        all_perturbation_types = [
            'gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur',
            'snow', 'brightness', 'translate', 'rotate', 'tilt', 'scale', 'shear'
        ]
        
        # If specific type requested, only load that one
        if pert_type:
            if pert_type not in all_perturbation_types:
                raise ValueError(
                    f"Unknown CIFAR-10-P perturbation type: {pert_type}. "
                    f"Available types: {all_perturbation_types}"
                )
            perturbation_types = [pert_type]
        else:
            perturbation_types = all_perturbation_types
        
        all_images = []
        all_indices = []  # Track which original test images are used
        
        for pert_type in perturbation_types:
            # Try base name first
            pert_path = cifar10p_data_dir / f'{pert_type}.npy'
            # Also try variants like gaussian_noise_2, gaussian_noise_3
            if not pert_path.exists():
                for variant in [f'{pert_type}_2', f'{pert_type}_3']:
                    variant_path = cifar10p_data_dir / f'{variant}.npy'
                    if variant_path.exists():
                        pert_path = variant_path
                        break
            
            if pert_path.exists():
                # Shape: (n_sequences, n_frames, H, W, C)
                sequences = np.load(pert_path)
                # Use last frame (most perturbed) from each sequence
                last_frames = sequences[:, -1, :, :, :]  # (n_sequences, H, W, C)
                all_images.append(last_frames)
                # Track indices: each perturbation type uses all 10K test images in order
                n_sequences = sequences.shape[0]
                all_indices.append(np.arange(n_sequences))
        
        if not all_images:
            if pert_type:
                raise FileNotFoundError(
                    f"No CIFAR-10-P data found for perturbation type '{pert_type}'. "
                    f"Expected: {cifar10p_data_dir}/{pert_type}.npy"
                )
            else:
                raise FileNotFoundError(
                    f"No CIFAR-10-P data found. Expected structure: {cifar10p_data_dir}/{{perturbation_type}}.npy"
                )
        
        images = np.concatenate(all_images, axis=0)
        
        # Load CIFAR-10 test labels (CIFAR-10-P uses same labels as CIFAR-10 test set)
        # CIFAR-10-P is perturbed versions of CIFAR-10 test images, so labels are the same
        try:
            from torchvision.datasets import CIFAR10
            import torchvision.transforms as tv_transforms
            
            # Load CIFAR-10 test set to get labels (without downloading if already exists)
            cifar10_test = CIFAR10(root=data_dir, train=False, download=False, transform=tv_transforms.ToTensor())
            cifar10_test_labels = np.array([cifar10_test[i][1] for i in range(len(cifar10_test))])
            
            # CIFAR-10-P files contain sequences for all 10K test images in order
            # If we're loading multiple perturbation types, we need to repeat labels
            if len(perturbation_types) > 1:
                # Each perturbation type has 10K images, so repeat labels for each
                labels = np.tile(cifar10_test_labels, len(perturbation_types))
            else:
                # Single perturbation type: use labels directly
                labels = cifar10_test_labels
            
            # Verify alignment
            if len(labels) != len(images):
                print(f"  ⚠️  Warning: Label count ({len(labels)}) doesn't match image count ({len(images)})")
                print(f"     Using labels for first {min(len(labels), len(images))} images only")
                labels = labels[:len(images)]
            
            print(f"  Loaded CIFAR-10 test labels for CIFAR-10-P ({len(labels)} labels)")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not load CIFAR-10 test labels: {e}")
            print(f"     CIFAR-10-P will use dummy labels (0)")
            labels = None
        
        dataset = CIFAR10PDataset(images, labels=labels, transform=transform)
    
    # ImageNet-1K OOD datasets (224x224)
    elif dataset_name_lower == 'inaturalist':
        # iNaturalist - may need custom loader or use available dataset
        # For now, use a placeholder - might need to implement custom loader
        raise NotImplementedError(
            "iNaturalist loader not yet implemented. "
            "May need custom dataset loader."
        )
    elif dataset_name_lower == 'sun':
        # SUN dataset - may need custom loader
        raise NotImplementedError(
            "SUN dataset loader not yet implemented. "
            "May need custom dataset loader."
        )
    else:
        raise ValueError(
            f"Unknown OOD dataset: {dataset_name}. "
            f"Supported: svhn, lsun, isun, textures, places365, cifar10c, cifar10p, inaturalist, sun, "
            f"adversarial:{{attack}}:{{norm}}:{{epsilon}}"
        )
    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return loader


# ============================================================================
# Model Loading
# ============================================================================

def load_classifier(
    model_path: str,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
    use_pretrained: bool = False,
    model_arch: str = 'resnet18',
) -> nn.Module:
    """
    Load a trained classifier model.
    
    Args:
        model_path: Path to model checkpoint (or None to use pre-trained)
        num_classes: Number of classes
        device: Device to load model on
        use_pretrained: If True and model_path doesn't exist, use torchvision pre-trained
        model_arch: Model architecture ('resnet18' or 'resnet50')
    
    Returns:
        Loaded model
    """
    import ssl
    # Disable SSL verification for model downloads (common issue on macOS)
    ssl._create_default_https_context = ssl._create_unverified_context
    
    if device is None:
        device = get_device()
    
    # Create model based on architecture
    if model_arch == 'resnet18':
        model = torchvision.models.resnet18(num_classes=num_classes)
        pretrained_model_fn = torchvision.models.resnet18
    elif model_arch == 'resnet50':
        model = torchvision.models.resnet50(num_classes=num_classes)
        pretrained_model_fn = torchvision.models.resnet50
    else:
        raise ValueError(f"Unsupported model architecture: {model_arch}")
    
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    elif use_pretrained and num_classes == 1000:
        # Use ImageNet pre-trained model
        print(f"Using torchvision pre-trained {model_arch.upper()} (ImageNet)")
        # Use newer weights API if available, fallback to pretrained
        try:
            if model_arch == 'resnet18':
                from torchvision.models import ResNet18_Weights
                model = pretrained_model_fn(weights=ResNet18_Weights.IMAGENET1K_V1)
            elif model_arch == 'resnet50':
                from torchvision.models import ResNet50_Weights
                model = pretrained_model_fn(weights=ResNet50_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            # Fallback to deprecated pretrained parameter
            model = pretrained_model_fn(pretrained=True)
    else:
        print(f"Warning: Model checkpoint not found at {model_path}")
        if use_pretrained:
            print("Using randomly initialized model (not recommended for evaluation)")
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# Helper Functions
# ============================================================================

def _setup_id_dataset(
    id_dataset: str,
    model_path: Optional[str] = None,
    score_model_path: Optional[str] = None,
    batch_size: int = 128,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    skip_stein: bool = False,
    skip_baselines: bool = False,
    train_only_on_correct: bool = False,
    cache_dir: Optional[str] = None,
    use_stein_factory: bool = False,
    stein_classification_scalar_mode: str = 'predicted_class_prob',
    stein_fixed_class_idx: int = 0,
    include_stein_per_dim_l2: bool = False,
    include_stein_per_dim_sum: bool = False,
    include_score_norm: bool = False,
    include_grad_f_norm: bool = False,
    stein_subset_only: bool = False,
    stein_subset_with_baselines: bool = False,
    # DDPM score extraction controls for CIFAR-10 pretrained DDPMScoreWrapper
    cifar10_ddpm_model_id: str = "google/ddpm-cifar10-32",
    cifar10_ddpm_timestep: int = 0,
    cifar10_ddpm_denom: str = "sigma_sq",
    cifar10_ddpm_add_noise: bool = False,
    cifar10_ddpm_noise_seed: int = 0,
    # Stein component ablation preset (type 1) for CIFAR-10 benchmarking
    stein_ablation1: bool = False,
    # Stein component ablation preset mapped to stein_per_dimension_l2 (all classes + L2 over classes)
    stein_ablation1_perdim_l2: bool = False,
) -> Dict[str, Any]:
    """
    Setup ID dataset: load data, model, create & fit detectors, pre-compute ID scores.
    
    This is done once per ID dataset and reused for all OOD datasets.
    
    Returns:
        Dictionary with: detectors, train_loader, val_loader, test_loader,
                        id_scores_cache, image_size, mean, std
    """
    if device is None:
        device = get_device()

    # region agent log
    _agent_log(
        run_id="perf-investigation",
        hypothesis_id="ARGS",
        location="scripts/benchmark_ood_evaluation.py:_setup_id_dataset",
        message="Setup ID dataset called",
        data={
            "id_dataset": str(id_dataset),
            "use_stein_factory": bool(use_stein_factory),
            "stein_classification_scalar_mode": str(stein_classification_scalar_mode),
            "stein_fixed_class_idx": int(stein_fixed_class_idx),
            "include_stein_per_dim_l2": bool(include_stein_per_dim_l2),
            "include_stein_per_dim_sum": bool(include_stein_per_dim_sum),
            "include_score_norm": bool(include_score_norm),
            "include_grad_f_norm": bool(include_grad_f_norm),
            "stein_subset_only": bool(stein_subset_only),
            "stein_subset_with_baselines": bool(stein_subset_with_baselines),
            "stein_ablation1": bool(stein_ablation1),
            "stein_ablation1_perdim_l2": bool(stein_ablation1_perdim_l2),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
        },
    )
    # endregion
    
    # 1. Load ID datasets
    print("\n[1/5] Loading ID datasets...")
    if id_dataset.lower() == 'cifar10':
        train_loader, val_loader, test_loader = get_cifar10_dataloaders(
            batch_size=batch_size, num_workers=num_workers
        )
        num_classes = 10
        image_size = 32
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif id_dataset.lower() == 'cifar100':
        train_loader, val_loader, test_loader = get_cifar100_dataloaders(
            batch_size=batch_size, num_workers=num_workers
        )
        num_classes = 100
        image_size = 32
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif id_dataset.lower() == 'imagenet' or id_dataset.lower() == 'imagenet1k':
        train_loader, val_loader, test_loader = get_imagenet_dataloaders(
            batch_size=batch_size, num_workers=num_workers
        )
        num_classes = 1000
        image_size = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError(f"Unsupported ID dataset: {id_dataset}")
    
    print(f"  ID train set: {len(train_loader.dataset)} samples")
    print(f"  ID val set: {len(val_loader.dataset) if val_loader else 0} samples")
    print(f"  ID test set: {len(test_loader.dataset)} samples")
    
    # 2. Load model
    print("\n[2/5] Loading classifier model...")
    if model_path is None:
        if id_dataset.lower() == 'cifar10' or id_dataset.lower() == 'cifar100':
            model_path = f'checkpoints/{id_dataset}_resnet18.pth'
        elif id_dataset.lower() == 'imagenet' or id_dataset.lower() == 'imagenet1k':
            model_path = f'checkpoints/{id_dataset}_resnet50.pth'
        else:
            model_path = f'checkpoints/{id_dataset}_resnet18.pth'
    
    if not os.path.exists(model_path):
        print(f"\n⚠️  Model checkpoint not found: {model_path}")
        print(f"   Expected location: {os.path.abspath(model_path)}")
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            f"See options in run_benchmark() for how to obtain a model."
        )
    
    # Determine model architecture
    if id_dataset.lower() == 'cifar10' or id_dataset.lower() == 'cifar100':
        model_arch = 'resnet18'
    elif id_dataset.lower() == 'imagenet' or id_dataset.lower() == 'imagenet1k':
        model_arch = 'resnet50'
    else:
        model_arch = 'resnet18'
    
    use_pretrained = (id_dataset.lower() == 'imagenet' or id_dataset.lower() == 'imagenet1k')
    model = load_classifier(
        model_path, 
        num_classes=num_classes, 
        device=device,
        use_pretrained=use_pretrained,
        model_arch=model_arch,
    )
    print(f"  Model loaded: {model_path}")
    
    # 3. Create detectors
    print("\n[3/5] Creating detectors...")
    detectors = {}

    # Stein-subset presets:
    # - stein_subset_only: run ONLY the Stein per-dimension L2 detector (no baselines)
    # - stein_subset_with_baselines: run baselines + ONLY the Stein per-dimension L2 detector
    if stein_subset_only and stein_subset_with_baselines:
        raise ValueError("Choose only one: stein_subset_only or stein_subset_with_baselines")
    if stein_subset_only:
        skip_baselines = True
        use_stein_factory = False
    if stein_subset_with_baselines:
        # Keep baseline detectors enabled; Stein detectors remain independent (no factory).
        use_stein_factory = False
    
    if not skip_baselines:
        # Pass normalization std for ODIN preprocessing (Mahalanobis, ODIN)
        # Convert tuple to list for PyTorch-OOD compatibility
        norm_std_list = list(std) if std else None
        baseline_config = {
            'odin': {'norm_std': norm_std_list},
            'mahalanobis': {'norm_std': norm_std_list},
        }
        baseline_detectors = create_all_baseline_detectors(
            model, 
            device=device,
            config=baseline_config
        )
        # Some baseline detectors are wrappers and may not expose `.model`.
        # Filter those out so downstream logic can reliably access the classifier.
        baseline_detectors = {k: v for k, v in baseline_detectors.items() if hasattr(v, "model")}
        detectors.update(baseline_detectors)
        print(f"  Created {len(baseline_detectors)} baseline detectors")
        if norm_std_list:
            print(f"  Using normalization std: {norm_std_list} for ODIN/Mahalanobis preprocessing")
    
    if not skip_stein:
        from src.models import UNetScore, SmallScoreNet
        from src.ddpm_score import DDPMScoreWrapper
        
        # For CIFAR-10, use pretrained DDPM model
        if id_dataset.lower() == 'cifar10':
            # If a CIFAR-10 score model checkpoint was provided, use it (preferred for ablations/debugging).
            # Otherwise, fall back to pretrained DDPM.
            in_channels = 3  # CIFAR-10 is RGB
            if score_model_path and os.path.exists(score_model_path):
                score_model = UNetScore(in_channels=in_channels)
                score_model.load_state_dict(torch.load(score_model_path, map_location=device))
                score_model = score_model.to(device)
                score_model_loaded = True
                should_train_score_model = False
                is_unet = True
                print(f"  Loaded CIFAR-10 score model from {score_model_path}")
            else:
                # Use pretrained DDPM model from Hugging Face
                score_model = DDPMScoreWrapper(
                    model_id=str(cifar10_ddpm_model_id),
                    timestep=int(cifar10_ddpm_timestep),
                    denom_mode=str(cifar10_ddpm_denom),
                    add_noise=bool(cifar10_ddpm_add_noise),
                    noise_seed=int(cifar10_ddpm_noise_seed),
                    device=device,
                )
                score_model_loaded = True  # Pretrained, no training needed
                should_train_score_model = False
                is_unet = False  # DDPM is not UNetScore, but has similar interface
                print(
                    f"  Using pretrained DDPM model for CIFAR-10 "
                    f"(model_id={cifar10_ddpm_model_id} t={cifar10_ddpm_timestep} "
                    f"denom={cifar10_ddpm_denom} add_noise={cifar10_ddpm_add_noise} seed={cifar10_ddpm_noise_seed})"
                )
        else:
            # For other datasets, use UNetScore or SmallScoreNet
            # Determine number of input channels based on dataset
            if id_dataset.lower() in ['cifar100', 'imagenet', 'imagenet1k']:
                in_channels = 3  # RGB
            else:
                in_channels = 1  # Grayscale (default, e.g., MNIST)
            
            # Try to load score model from checkpoint
            score_model = None
            score_model_loaded = False
            if score_model_path and os.path.exists(score_model_path):
                # Determine score model type from path or use UNetScore by default
                if 'unet' in score_model_path.lower() or 'unet' in id_dataset.lower():
                    score_model = UNetScore(in_channels=in_channels)
                else:
                    score_model = SmallScoreNet()
                score_model.load_state_dict(torch.load(score_model_path, map_location=device))
                score_model = score_model.to(device)
                score_model_loaded = True
                print(f"  Loaded score model from {score_model_path}")
            else:
                # Create a new score model for training
                # Use UNetScore for images (CIFAR-100, ImageNet), SmallScoreNet for smaller datasets
                if id_dataset.lower() in ['cifar100', 'imagenet', 'imagenet1k']:
                    score_model = UNetScore(in_channels=in_channels)
                    print(f"  Created new UNetScore model (RGB, {in_channels} channels, will be trained)")
                else:
                    score_model = SmallScoreNet()
                    print(f"  Created new SmallScoreNet model (will be trained)")
                score_model = score_model.to(device)
            
            # Determine if we should train the score model
            # Only train if no checkpoint was loaded
            should_train_score_model = not score_model_loaded
            is_unet = isinstance(score_model, UNetScore)
        
        # Create SteinDetector with conditional training
        # Score model will be trained during fit() only if should_train_score_model=True
        # Create separate score model instances for each detector to avoid conflicts
        if id_dataset.lower() == 'cifar10':
            # For DDPM, we don't need training config (pretrained)
            score_train_config = {}
            
            # DDPM wrapper can be shared (it's just a wrapper, no training state)
            def create_score_model_copy():
                return score_model  # DDPM can be shared
        else:
            # For UNetScore/SmallScoreNet, set up training config
            score_train_config = {
                'epochs': 50 if is_unet else 5,
                'batch_size': 128,
                'lr': 2e-4 if is_unet else 1e-3,
                'checkpoint_path': score_model_path or f'checkpoints/score_unet_{id_dataset}.pth',
            }
            if is_unet:
                score_train_config.update({
                    'n_levels': 10,
                    'sigma_min': 0.01,
                    'sigma_max': 0.5,
                })
            else:
                score_train_config.update({
                    'noise_sigma': 0.2,
                })
            
            # Create separate score model instances for each detector
            # (they can share weights if loaded from checkpoint, but need separate instances for training)
            def create_score_model_copy():
                if is_unet:
                    new_model = UNetScore(in_channels=in_channels)
                else:
                    new_model = SmallScoreNet()
                if score_model_loaded:
                    # Copy weights from loaded model
                    new_model.load_state_dict(score_model.state_dict())
                return new_model.to(device)
        
        # Stein-subset presets: build ONLY stein_per_dimension_l2 (no SteinFactory).
        if stein_subset_only or stein_subset_with_baselines:
            if use_stein_factory:
                print("  Note: --stein-subset-only set; ignoring --use-stein-factory (detectors are independent).")
            # Force independence (no factory)
            use_stein_factory = False

            # Only requested detector: class-agnostic per-dimension residual with L2 aggregation.
            detectors['stein_per_dimension_l2'] = SteinDetector(
                model=model, score_model=create_score_model_copy(),
                classification_scalar_mode=stein_classification_scalar_mode,
                fixed_class_idx=stein_fixed_class_idx,
                stein_operator_type='per_dimension',
                aggregation='l2',
                skip_laplacian=False,
                num_probes=1,
                device=device,
                train_score_model=should_train_score_model,
                score_train_config=score_train_config,
            )

            print("  Stein subset preset enabled (independent SteinDetector instance):")
            print("   ", sorted(detectors.keys()))

        # Check if we should use factory detector
        elif use_stein_factory:
            # Use SteinFactoryDetector to compute all modes at once
            print("  Using SteinFactoryDetector (computes all modes simultaneously)")
            
            # Create factory detector with all modes
            # Note: skip_laplacian=False by default - special Laplacian formulation handles ResNets with MaxPool
            if stein_ablation1 and stein_ablation1_perdim_l2:
                raise ValueError("Choose only one: stein_ablation1 or stein_ablation1_perdim_l2")

            if stein_ablation1:
                enabled_modes = [
                    # Base component ablations
                    'full', 'full_no_lap', 'lap_only', 'score_only',
                    # Std-normalized / scale-balanced variants (std estimated from training)
                    'lap_only_std', 'full_no_lap_std', 'full_std_balanced',
                ]
            elif stein_ablation1_perdim_l2:
                enabled_modes = [
                    # Full per-dimension L2 (baseline reference)
                    'per_dimension_l2',
                    # Per-dimension L2 component ablations
                    'per_dimension_l2_no_lap',
                    'per_dimension_l2_lap_only',
                    'per_dimension_l2_score_only',
                    # Std-balanced variants (std estimated from training)
                    'per_dimension_l2_lap_only_std',
                    'per_dimension_l2_no_lap_std',
                    'per_dimension_l2_std_balanced',
                ]
            else:
                enabled_modes = ['full', 'full_no_lap', 'first_order', 'first_order_sum']

            factory = SteinFactoryDetector(
                model=model,
                score_model=create_score_model_copy(),
                device=device,
                model_type='classification',
                classification_scalar_mode=stein_classification_scalar_mode,
                fixed_class_idx=stein_fixed_class_idx,
                num_probes=1,
                skip_laplacian=False,  # Special Laplacian formulation handles MaxPool layers
                enabled_modes=enabled_modes,
                compute_baseline=True,
            )
            
            # Store factory for later use
            detectors['stein_factory'] = factory
            
            # Create wrapper detectors that use factory's predict_all()
            # These wrappers allow the evaluation code to work with individual detector names
            class FactoryModeWrapper(SteinDetector):
                """Wrapper that uses factory detector for a specific mode."""
                def __init__(self, factory_detector, mode_name):
                    # Initialize minimal SteinDetector to satisfy interface
                    super().__init__(
                        model=factory_detector.model,
                        score_model=factory_detector.score_model,
                        device=factory_detector.device,
                        classification_scalar_mode=getattr(factory_detector, 'classification_scalar_mode', 'predicted_class_prob'),
                        fixed_class_idx=getattr(factory_detector, 'fixed_class_idx', 0),
                        stein_operator_type=mode_name if mode_name != 'full_no_lap' else 'full',
                        skip_laplacian=(mode_name == 'full_no_lap'),
                        compute_baseline=False,  # Baselines computed by factory
                    )
                    self.factory = factory_detector
                    self.mode_name = mode_name
                    # Don't copy baselines yet - factory hasn't been fitted
                    # They will be lazy-loaded from factory when needed
                    self._baseline = None
                    self._training_std = None
                
                @property
                def baseline(self):
                    """Lazy-load baseline from factory."""
                    if self._baseline is None:
                        baseline = self.factory.get_baseline(self.mode_name)
                        if baseline is not None:
                            self._baseline = baseline
                    return self._baseline
                
                @baseline.setter
                def baseline(self, value):
                    """Allow setting baseline directly."""
                    self._baseline = value
                
                @property
                def training_std(self):
                    """Lazy-load training_std from factory."""
                    if self._training_std is None:
                        training_std = self.factory.get_training_std(self.mode_name)
                        if training_std is not None:
                            self._training_std = training_std
                    return self._training_std
                
                @training_std.setter
                def training_std(self, value):
                    """Allow setting training_std directly."""
                    self._training_std = value
                
                def fit(self, train_loader=None, val_loader=None, **kwargs):
                    # Ensure factory is fitted first
                    if self.factory.baselines is None or len(self.factory.baselines) == 0:
                        # Factory not fitted yet, fit it now
                        if train_loader is None:
                            raise ValueError("train_loader required to fit factory detector")
                        self.factory.fit(train_loader, val_loader=val_loader)
                    
                    # Update baselines from factory (for direct access, not just lazy loading)
                    baseline = self.factory.get_baseline(self.mode_name)
                    training_std = self.factory.get_training_std(self.mode_name)
                    if baseline is not None:
                        self._baseline = baseline
                    if training_std is not None:
                        self._training_std = training_std
                    
                    return self
                
                def predict(self, x):
                    # Use factory's predict_all and return specific mode
                    all_scores = self.factory.predict_all(x)
                    return all_scores[self.mode_name]
            
            # Create individual mode wrappers
            # IMPORTANT: only create wrappers for modes that are actually enabled in the factory,
            # otherwise the wrapper will call predict_all() and then fail when indexing a missing mode.
            if 'full' in enabled_modes:
                detectors['stein_full'] = FactoryModeWrapper(factory, 'full')
            if 'full_no_lap' in enabled_modes:
                detectors['stein_full_no_lap'] = FactoryModeWrapper(factory, 'full_no_lap')
            if stein_ablation1:
                detectors['stein_lap_only'] = FactoryModeWrapper(factory, 'lap_only')
                detectors['stein_score_only'] = FactoryModeWrapper(factory, 'score_only')
                detectors['stein_lap_only_std'] = FactoryModeWrapper(factory, 'lap_only_std')
                detectors['stein_full_no_lap_std'] = FactoryModeWrapper(factory, 'full_no_lap_std')
                detectors['stein_full_std_balanced'] = FactoryModeWrapper(factory, 'full_std_balanced')
            elif stein_ablation1_perdim_l2:
                # Map everything to stein_per_dimension_l2 (all classes + L2 over classes)
                detectors['stein_per_dimension_l2'] = FactoryModeWrapper(factory, 'per_dimension_l2')
                detectors['stein_per_dimension_l2_no_lap'] = FactoryModeWrapper(factory, 'per_dimension_l2_no_lap')
                detectors['stein_per_dimension_l2_lap_only'] = FactoryModeWrapper(factory, 'per_dimension_l2_lap_only')
                detectors['stein_per_dimension_l2_score_only'] = FactoryModeWrapper(factory, 'per_dimension_l2_score_only')
                detectors['stein_per_dimension_l2_lap_only_std'] = FactoryModeWrapper(factory, 'per_dimension_l2_lap_only_std')
                detectors['stein_per_dimension_l2_no_lap_std'] = FactoryModeWrapper(factory, 'per_dimension_l2_no_lap_std')
                detectors['stein_per_dimension_l2_std_balanced'] = FactoryModeWrapper(factory, 'per_dimension_l2_std_balanced')
            else:
                detectors['stein_first_order'] = FactoryModeWrapper(factory, 'first_order')
                detectors['stein_first_order_sum'] = FactoryModeWrapper(factory, 'first_order_sum')
            
            num_stein_detectors = len([k for k in detectors.keys() if k.startswith('stein')])
            if should_train_score_model:
                print(f"  Created SteinFactoryDetector with {num_stein_detectors} modes (score model will be trained)")
            else:
                print(f"  Created SteinFactoryDetector with {num_stein_detectors} modes (using pre-trained score model)")
        else:
            # Original approach: create individual detectors
            detectors['stein_full'] = SteinDetector(
                model=model, score_model=create_score_model_copy(),
                classification_scalar_mode=stein_classification_scalar_mode,
                fixed_class_idx=stein_fixed_class_idx,
                classification_topk=1,
                stein_operator_type='full', skip_laplacian=False,
                num_probes=1, device=device,
                train_score_model=should_train_score_model,  # Only train if not loaded from checkpoint
                score_train_config=score_train_config,
            )
            
            # Top-K ablations: L2 aggregation over top-K per-class Stein residuals (K=1,3,5),
            # matching stein_per_dimension_l2 but restricted to a top-K subset per sample.
            detectors['stein_full_top1'] = SteinDetector(
                model=model, score_model=create_score_model_copy(),
                classification_scalar_mode=stein_classification_scalar_mode,
                fixed_class_idx=stein_fixed_class_idx,
                classification_topk=1,
                stein_operator_type='per_dimension',
                aggregation='topk_l2',
                skip_laplacian=False,
                num_probes=1, device=device,
                train_score_model=should_train_score_model,
                score_train_config=score_train_config,
            )
            detectors['stein_full_top3'] = SteinDetector(
                model=model, score_model=create_score_model_copy(),
                classification_scalar_mode=stein_classification_scalar_mode,
                fixed_class_idx=stein_fixed_class_idx,
                classification_topk=3,
                stein_operator_type='per_dimension',
                aggregation='topk_l2',
                skip_laplacian=False,
                num_probes=1, device=device,
                train_score_model=should_train_score_model,
                score_train_config=score_train_config,
            )
            detectors['stein_full_top5'] = SteinDetector(
                model=model, score_model=create_score_model_copy(),
                classification_scalar_mode=stein_classification_scalar_mode,
                fixed_class_idx=stein_fixed_class_idx,
                classification_topk=5,
                stein_operator_type='per_dimension',
                aggregation='topk_l2',
                skip_laplacian=False,
                num_probes=1, device=device,
                train_score_model=should_train_score_model,
                score_train_config=score_train_config,
            )

            detectors['stein_full_no_lap'] = SteinDetector(
                model=model, score_model=create_score_model_copy(),
                classification_scalar_mode=stein_classification_scalar_mode,
                fixed_class_idx=stein_fixed_class_idx,
                classification_topk=1,
                stein_operator_type='full', skip_laplacian=True,
                num_probes=1, device=device,
                train_score_model=should_train_score_model,  # Only train if not loaded from checkpoint
                score_train_config=score_train_config,
            )
            detectors['stein_first_order'] = SteinDetector(
                model=model, score_model=create_score_model_copy(),
                classification_scalar_mode=stein_classification_scalar_mode,
                fixed_class_idx=stein_fixed_class_idx,
                classification_topk=1,
                stein_operator_type='first_order',
                skip_laplacian=False, device=device,
                train_score_model=should_train_score_model,  # Only train if not loaded from checkpoint
                score_train_config=score_train_config,
            )
            detectors['stein_first_order_sum'] = SteinDetector(
                model=model, score_model=create_score_model_copy(),
                classification_scalar_mode=stein_classification_scalar_mode,
                fixed_class_idx=stein_fixed_class_idx,
                classification_topk=1,
                stein_operator_type='first_order_sum',
                skip_laplacian=False, device=device,
                train_score_model=should_train_score_model,  # Only train if not loaded from checkpoint
                score_train_config=score_train_config,
            )

            # Optional: class-agnostic per-dimension residual with L2 aggregation (avoids single-class sensitivity)
            if include_stein_per_dim_l2:
                detectors['stein_per_dimension_l2'] = SteinDetector(
                    model=model, score_model=create_score_model_copy(),
                    classification_scalar_mode=stein_classification_scalar_mode,
                    fixed_class_idx=stein_fixed_class_idx,
                    stein_operator_type='per_dimension',
                    aggregation='l2',
                    skip_laplacian=False,
                    num_probes=1,
                    device=device,
                    train_score_model=should_train_score_model,
                    score_train_config=score_train_config,
                )

            # Optional: class-agnostic per-dimension residual with SUM aggregation (signed)
            if include_stein_per_dim_sum:
                detectors['stein_per_dimension_sum'] = SteinDetector(
                    model=model, score_model=create_score_model_copy(),
                    classification_scalar_mode=stein_classification_scalar_mode,
                    fixed_class_idx=stein_fixed_class_idx,
                    stein_operator_type='per_dimension',
                    aggregation='sum',
                    skip_laplacian=False,
                    num_probes=1,
                    device=device,
                    train_score_model=should_train_score_model,
                    score_train_config=score_train_config,
                )

            # Optional: score-model probe (||s(x)||). Uses the same score model as Stein variants.
            if include_score_norm:
                detectors['score_norm'] = ScoreNormDetector(
                    score_model=create_score_model_copy(),
                    device=device,
                )

            # Optional: classifier-geometry probe (||∇_x f(x)||). Uses the same f(x) definition as Stein variants.
            if include_grad_f_norm:
                detectors['grad_f_norm'] = GradFNormDetector(
                    model=model,
                    classification_scalar_mode=stein_classification_scalar_mode,
                    fixed_class_idx=stein_fixed_class_idx,
                    device=device,
                )
            
            num_stein_detectors = len([k for k in detectors.keys() if k.startswith('stein')])
            if should_train_score_model:
                print(f"  Created {num_stein_detectors} SteinDetector variant(s) (score model will be trained)")
            else:
                print(f"  Created {num_stein_detectors} SteinDetector variant(s) (using pre-trained score model)")
    
    print(f"  Total detectors: {len(detectors)}")

    # region agent log
    _agent_log(
        run_id="perf-investigation",
        hypothesis_id="DETS",
        location="scripts/benchmark_ood_evaluation.py:_setup_id_dataset",
        message="Detectors constructed",
        data={
            "detector_names": sorted(list(detectors.keys())),
        },
    )
    # endregion
    
    # 4. Fit detectors (once per ID dataset)
    print("\n[4/5] Fitting detectors...")
    detectors = fit_all_detectors(
        detectors, train_loader, val_loader=val_loader, device=device, verbose=True,
        train_only_on_correct=train_only_on_correct,
        classifier_model=model,  # model is the classifier loaded above
        cache_dir=cache_dir,
        id_dataset=id_dataset,
        model_path=model_path,
        score_model_path=score_model_path,
    )
    
    # 5. Pre-compute ID scores (once per ID dataset)
    print("\n[5/5] Pre-computing ID scores...")
    from tqdm import tqdm
    from src.evaluation.detector_cache import (
        get_id_scores_cache_path, load_id_scores_cache, save_id_scores_cache,
        get_detector_config_from_stein_detector, get_detector_config_from_baseline_detector,
        _get_dataset_hash,
    )
    
    id_scores_cache = {}
    dataset_hash = _get_dataset_hash(test_loader)
    
    # Detectors that need gradients (use ODIN preprocessing)
    detectors_needing_grads = {'mahalanobis', 'odin'}

    # If using SteinFactoryDetector, compute ID scores for all factory wrappers in ONE pass.
    # Otherwise, the per-detector loop below will call wrapper.predict() for each mode, which
    # triggers factory.predict_all(x) repeatedly and makes stein_full_no_lap look as slow as stein_full.
    factory_detector = detectors.get('stein_factory', None)
    if isinstance(factory_detector, SteinFactoryDetector):
        # Map wrapper detector name -> mode_name
        factory_wrappers: Dict[str, Any] = {}
        for n, d in detectors.items():
            if hasattr(d, 'factory') and getattr(d, 'factory') is factory_detector:
                mode_name = getattr(d, 'mode_name', None)
                if mode_name is not None:
                    factory_wrappers[n] = d

        if factory_wrappers:
            # 1) Try to load cached wrapper ID scores first.
            missing_wrappers: List[str] = []
            for n, d in factory_wrappers.items():
                if cache_dir is None:
                    missing_wrappers.append(n)
                    continue
                detector_config = get_detector_config_from_stein_detector(d)
                cache_path = get_id_scores_cache_path(
                    Path(cache_dir),
                    id_dataset,
                    model_path,
                    n,
                    detector_config,
                    score_model_path=score_model_path,
                    dataset_hash=dataset_hash,
                )
                cached_scores = load_id_scores_cache(cache_path, device=device)
                if cached_scores is not None:
                    print(f"    {n}: Loaded {len(cached_scores)} ID scores from cache (factory wrapper)")
                    id_scores_cache[n] = cached_scores
                else:
                    missing_wrappers.append(n)

            # 2) If any wrapper missing, compute all missing wrappers in one pass.
            if missing_wrappers:
                print(
                    f"    stein_factory: Computing ID scores once for {len(missing_wrappers)} factory wrapper(s): "
                    f"{sorted(missing_wrappers)}"
                )
                per_wrapper_lists: Dict[str, List[torch.Tensor]] = {n: [] for n in missing_wrappers}
                id_iter = tqdm(test_loader, desc='  Computing ID scores (factory shared)', disable=False)
                with torch.enable_grad():
                    for batch in id_iter:
                        if isinstance(batch, (list, tuple)):
                            x = batch[0]
                        else:
                            x = batch
                        x = x.to(device)
                        all_scores = factory_detector.predict_all(x)  # one shared compute
                        for n in missing_wrappers:
                            mode_name = getattr(factory_wrappers[n], 'mode_name')
                            per_wrapper_lists[n].append(all_scores[mode_name].detach().cpu())

                for n in missing_wrappers:
                    id_scores_cache[n] = torch.cat(per_wrapper_lists[n])
                    print(f"    {n}: {len(id_scores_cache[n])} ID scores computed (factory shared)")

                    # Save to cache
                    if cache_dir is not None:
                        d = factory_wrappers[n]
                        detector_config = get_detector_config_from_stein_detector(d)
                        cache_path = get_id_scores_cache_path(
                            Path(cache_dir),
                            id_dataset,
                            model_path,
                            n,
                            detector_config,
                            score_model_path=score_model_path,
                            dataset_hash=dataset_hash,
                        )
                        save_id_scores_cache(cache_path, id_scores_cache[n])
                        print(f"    {n}: Saved ID scores to cache (factory wrapper)")
    
    for name, detector in detectors.items():
        # If we already computed factory-wrapper ID scores above, skip recomputing here.
        if name in id_scores_cache:
            continue

        # Set model to eval mode if it's a model object (not a function)
        if hasattr(detector, 'model'):
            if hasattr(detector.model, 'eval'):
                detector.model.eval()
        
        id_scores_list = []
        id_iter = tqdm(test_loader, desc=f'  Computing ID scores ({name})', disable=False)
        
        # Check cache for ID scores
        cached_scores = None
        if cache_dir is not None:
            # Get detector config
            if name == 'stein_factory':
                # Skip caching for factory detector itself (individual mode wrappers will be cached)
                print(f"    {name}: Skipping cache (factory detector, modes cached separately)")
                cached_scores = None
            elif name.startswith('stein'):
                detector_config = get_detector_config_from_stein_detector(detector)
                cache_path = get_id_scores_cache_path(
                    Path(cache_dir),
                    id_dataset,
                    model_path,
                    name,
                    detector_config,
                    score_model_path=score_model_path,
                    dataset_hash=dataset_hash,
                )
                cached_scores = load_id_scores_cache(cache_path, device=device)
            else:
                detector_config = get_detector_config_from_baseline_detector(name, detector)
                cache_path = get_id_scores_cache_path(
                    Path(cache_dir),
                    id_dataset,
                    model_path,
                    name,
                    detector_config,
                    score_model_path=score_model_path,
                    dataset_hash=dataset_hash,
                )
                cached_scores = load_id_scores_cache(cache_path, device=device)
            
            if cached_scores is not None:
                print(f"    {name}: Loaded {len(cached_scores)} ID scores from cache")
                id_scores_cache[name] = cached_scores
                continue
        
        # Skip factory detector itself (individual mode wrappers will compute scores)
        if name == 'stein_factory':
            print(f"    {name}: Skipping ID score computation (factory detector, modes computed separately)")
            continue
        
        # Some detectors (Mahalanobis, ODIN) need gradients for preprocessing
        needs_grads = name.lower() in detectors_needing_grads
        
        if needs_grads:
            # Enable gradients for detectors that need them
            for batch in id_iter:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device).requires_grad_(True)
                scores = detector.predict(x)
                id_scores_list.append(scores.detach().cpu())
        else:
            # Use no_grad for efficiency for other detectors
            with torch.no_grad():
                for batch in id_iter:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                    else:
                        x = batch
                    x = x.to(device)
                    scores = detector.predict(x)
                    id_scores_list.append(scores.cpu())
        
        id_scores_cache[name] = torch.cat(id_scores_list)
        print(f"    {name}: {len(id_scores_cache[name])} ID scores computed")
        
        # Save to cache
        if cache_dir is not None:
            # Skip factory detector (individual mode wrappers will be cached)
            if name == 'stein_factory':
                continue
            
            if name.startswith('stein'):
                detector_config = get_detector_config_from_stein_detector(detector)
            else:
                detector_config = get_detector_config_from_baseline_detector(name, detector)
            
            cache_path = get_id_scores_cache_path(
                Path(cache_dir),
                id_dataset,
                model_path,
                name,
                detector_config,
                score_model_path=score_model_path,
                dataset_hash=dataset_hash,
            )
            save_id_scores_cache(cache_path, id_scores_cache[name])
            print(f"    {name}: Saved ID scores to cache")
    
    return {
        'detectors': detectors,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'id_scores_cache': id_scores_cache,
        'image_size': image_size,
        'mean': mean,
        'std': std,
    }


# ============================================================================
# Main Evaluation Function
# ============================================================================

def run_benchmark(
    id_dataset: str,
    ood_dataset: str,
    model_path: Optional[str] = None,
    score_model_path: Optional[str] = None,
    output_dir: str = 'results/benchmark_results',
    batch_size: int = 128,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    skip_stein: bool = False,
    skip_baselines: bool = False,
    detectors: Optional[Dict[str, Any]] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    id_scores_cache: Optional[Dict[str, torch.Tensor]] = None,
    train_only_on_correct: bool = False,
    cache_dir: Optional[str] = None,
    use_stein_factory: bool = False,
    stein_classification_scalar_mode: str = 'predicted_class_prob',
    stein_fixed_class_idx: int = 0,
    include_stein_per_dim_l2: bool = False,
    include_stein_per_dim_sum: bool = False,
    include_score_norm: bool = False,
    include_grad_f_norm: bool = False,
    stein_subset_only: bool = False,
    stein_subset_with_baselines: bool = False,
    # DDPM score extraction controls for CIFAR-10 pretrained DDPMScoreWrapper
    cifar10_ddpm_model_id: str = "google/ddpm-cifar10-32",
    cifar10_ddpm_timestep: int = 0,
    cifar10_ddpm_denom: str = "sigma_sq",
    cifar10_ddpm_add_noise: bool = False,
    cifar10_ddpm_noise_seed: int = 0,
    # Stein component ablation preset (type 1)
    stein_ablation1: bool = False,
    # Stein component ablation preset mapped to stein_per_dimension_l2
    stein_ablation1_perdim_l2: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Run OOD detection benchmark evaluation.
    
    Args:
        id_dataset: In-distribution dataset name ('cifar10', 'cifar100')
        ood_dataset: Out-of-distribution dataset name
        model_path: Path to trained classifier checkpoint
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        device: Device to use
        skip_stein: If True, skip SteinDetector evaluation
        skip_baselines: If True, skip baseline detectors
    
    Returns:
        Dictionary of results: {detector_name: {metric: value}}
    """
    if device is None:
        device = get_device()
    
    print("=" * 60)
    print(f"OOD Detection Benchmark: {id_dataset.upper()} vs {ood_dataset.upper()}")
    print("=" * 60)
    
    # Check if we have pre-computed setup (from run_all_benchmarks)
    if detectors is None or test_loader is None or id_scores_cache is None:
        # Full setup (for single benchmark run)
        setup_result = _setup_id_dataset(
            id_dataset=id_dataset,
            model_path=model_path,
            score_model_path=score_model_path,
            use_stein_factory=use_stein_factory,
            stein_classification_scalar_mode=stein_classification_scalar_mode,
            stein_fixed_class_idx=stein_fixed_class_idx,
            include_stein_per_dim_l2=include_stein_per_dim_l2,
            include_stein_per_dim_sum=include_stein_per_dim_sum,
            include_score_norm=include_score_norm,
            include_grad_f_norm=include_grad_f_norm,
            stein_subset_only=stein_subset_only,
            stein_subset_with_baselines=stein_subset_with_baselines,
            stein_ablation1=stein_ablation1,
            stein_ablation1_perdim_l2=stein_ablation1_perdim_l2,
            cifar10_ddpm_model_id=str(cifar10_ddpm_model_id),
            cifar10_ddpm_timestep=int(cifar10_ddpm_timestep),
            cifar10_ddpm_denom=str(cifar10_ddpm_denom),
            cifar10_ddpm_add_noise=bool(cifar10_ddpm_add_noise),
            cifar10_ddpm_noise_seed=int(cifar10_ddpm_noise_seed),
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            skip_stein=skip_stein,
            skip_baselines=skip_baselines,
            train_only_on_correct=train_only_on_correct,
            cache_dir=cache_dir,
        )
        detectors = setup_result['detectors']
        train_loader = setup_result['train_loader']
        val_loader = setup_result['val_loader']
        test_loader = setup_result['test_loader']
        id_scores_cache = setup_result['id_scores_cache']
        image_size = setup_result['image_size']
        mean = setup_result['mean']
        std = setup_result['std']
    else:
        # Use provided setup (from run_all_benchmarks)
        # Extract image_size, mean, std from test_loader if needed
        # For now, we'll need to get these from the dataset or pass them
        # For simplicity, we'll reload them (they're cheap)
        if id_dataset.lower() == 'cifar10':
            image_size = 32
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif id_dataset.lower() == 'cifar100':
            image_size = 32
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        elif id_dataset.lower() == 'imagenet' or id_dataset.lower() == 'imagenet1k':
            image_size = 224
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            raise ValueError(f"Unsupported ID dataset: {id_dataset}")
    
    # Load OOD dataset
    print("\n[1/1] Loading OOD dataset...")
    # Get model and test_loader for adversarial datasets
    model_for_adv = None
    if ood_dataset.lower().startswith('adversarial:'):
        # Get model from detectors (they all use the same model)
        # Try different detectors until we find one with a model attribute
        model_for_adv = None
        if detectors:
            for detector in detectors.values():
                # GSC and most detectors have a model attribute
                if hasattr(detector, 'model'):
                    model_for_adv = detector.model
                    break
            if model_for_adv is None:
                raise ValueError(
                    "Cannot find model for adversarial generation. "
                    "Ensure at least one detector has a model attribute."
                )
        else:
            raise ValueError("Detectors must be provided for adversarial datasets")
    
    ood_loader = load_ood_dataset(
        ood_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        image_size=image_size,
        mean=mean,
        std=std,
        model=model_for_adv,
        test_loader=test_loader,
        device=device,
        cache_dir=os.path.join('./data', 'adversarial_cache'),  # Default cache location
    )
    print(f"  OOD test set: {len(ood_loader.dataset)} samples")
    
    # Compute classifier accuracy on ID and OOD datasets
    print("\n[2/3] Computing classifier accuracy...")
    from src.evaluation.ood_benchmark import (
        compute_classifier_accuracy,
        compute_classifier_confidence_metrics,
    )
    
    # Get the classifier model from one of the detectors (they all use the same model)
    classifier_model = None
    for detector in detectors.values():
        if hasattr(detector, 'model'):
            classifier_model = detector.model
            break
    
    accuracy_metrics = {}
    if classifier_model is not None:
        # ID accuracy (top-1 and top-5)
        print("  Computing ID test set accuracy...")
        id_acc = compute_classifier_accuracy(classifier_model, test_loader, device=device, verbose=True)
        accuracy_metrics['id_top1_accuracy'] = id_acc.get('top1_accuracy')
        accuracy_metrics['id_top5_accuracy'] = id_acc.get('top5_accuracy')
        if id_acc.get('top1_accuracy') is not None:
            print(f"    ID Top-1 Accuracy: {id_acc['top1_accuracy']:.4f}")
            print(f"    ID Top-5 Accuracy: {id_acc['top5_accuracy']:.4f}")
        else:
            print("    ID Accuracy: Labels not available")
        
        # OOD accuracy (top-1 and top-5) - try to compute if labels are available
        print("  Computing OOD dataset accuracy...")
        ood_acc = compute_classifier_accuracy(
            classifier_model, ood_loader, device=device, verbose=True, dataset_name=ood_dataset
        )
        accuracy_metrics['ood_top1_accuracy'] = ood_acc.get('top1_accuracy')
        accuracy_metrics['ood_top5_accuracy'] = ood_acc.get('top5_accuracy')
        if ood_acc.get('top1_accuracy') is not None:
            print(f"    OOD Top-1 Accuracy: {ood_acc['top1_accuracy']:.4f}")
            print(f"    OOD Top-5 Accuracy: {ood_acc['top5_accuracy']:.4f}")
        else:
            print("    OOD Accuracy: Labels not available (using confidence metrics)")
            # Fallback to confidence metrics if accuracy not available
            ood_confidence = compute_classifier_confidence_metrics(classifier_model, ood_loader, device=device, verbose=True)
            accuracy_metrics['ood_top1_confidence'] = ood_confidence['top1_confidence']
            accuracy_metrics['ood_entropy'] = ood_confidence['entropy']
            print(f"    OOD Top-1 Confidence: {ood_confidence['top1_confidence']:.4f}")
            print(f"    OOD Entropy: {ood_confidence['entropy']:.4f}")
    else:
        print("  Warning: Could not find classifier model for accuracy computation")
        accuracy_metrics['id_top1_accuracy'] = None
        accuracy_metrics['id_top5_accuracy'] = None
        accuracy_metrics['ood_top1_accuracy'] = None
        accuracy_metrics['ood_top5_accuracy'] = None
    
    # Evaluate detectors (using pre-fitted detectors and pre-computed ID scores)
    print("\n[3/3] Evaluating detectors...")
    results = evaluate_all_detectors(
        detectors, test_loader, ood_loader, ood_dataset_name=ood_dataset, device=device, verbose=True,
        id_scores_cache=id_scores_cache,
        classifier_model=classifier_model,
        compute_misclassified_metrics=True,  # Compute both metric sets
    )
    
    # Add accuracy metrics to results
    # Results now have structure: {detector_name: {'dataset_based': {...}, 'misclassified': {...}, 'dataset_and_misclassified': {...}}}
    for detector_name in results:
        if isinstance(results[detector_name], dict) and 'dataset_based' in results[detector_name]:
            # New structure with all three metric sets
            results[detector_name]['dataset_based'].update(accuracy_metrics)
            results[detector_name]['misclassified'].update(accuracy_metrics)
            results[detector_name]['dataset_and_misclassified'].update(accuracy_metrics)
        else:
            # Old structure (backward compatibility)
            results[detector_name].update(accuracy_metrics)
    
    # Save and display results
    print("\nSaving results...")
    save_results(
        results=results,
        output_dir=output_dir,
        dataset_name=id_dataset,
        ood_dataset_name=ood_dataset,
        additional_info={
            'model_path': model_path,
            'batch_size': batch_size,
        },
    )
    
    print_results_summary(results, id_dataset, ood_dataset)
    
    return results


def run_all_benchmarks(
    model_paths: Dict[str, str],
    score_model_paths: Optional[Dict[str, str]] = None,
    output_dir: str = 'results/benchmark_results',
    batch_size: int = 128,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    skip_stein: bool = False,
    skip_baselines: bool = False,
    train_only_on_correct: bool = False,
    cache_dir: Optional[str] = None,
    use_stein_factory: bool = False,
    stein_classification_scalar_mode: str = 'predicted_class_prob',
    stein_fixed_class_idx: int = 0,
    include_stein_per_dim_l2: bool = False,
    include_stein_per_dim_sum: bool = False,
    include_score_norm: bool = False,
    include_grad_f_norm: bool = False,
    stein_subset_only: bool = False,
    stein_subset_with_baselines: bool = False,
    # DDPM score extraction controls for CIFAR-10 pretrained DDPMScoreWrapper
    cifar10_ddpm_model_id: str = "google/ddpm-cifar10-32",
    cifar10_ddpm_timestep: int = 0,
    cifar10_ddpm_denom: str = "sigma_sq",
    cifar10_ddpm_add_noise: bool = False,
    cifar10_ddpm_noise_seed: int = 0,
    # Stein component ablation preset (type 1)
    stein_ablation1: bool = False,
    # Stein component ablation preset mapped to stein_per_dimension_l2
    stein_ablation1_perdim_l2: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run all benchmarks for CIFAR-10 evaluation.
    
    Evaluation Set: ID=CIFAR-10, OOD=[SVHN, LSUN, iSUN, Textures, Places365]
    
    Note: ImageNet-1K evaluation has been removed (focusing on CIFAR-10 with pretrained DDPM).
    
    Args:
        model_paths: Dictionary mapping dataset names to model paths
                    e.g., {'cifar10': 'checkpoints/cifar10_resnet.pth', 'imagenet': '...'}
        score_model_paths: Optional dictionary mapping dataset names to score model paths
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        device: Device to use
        skip_stein: If True, skip SteinDetector evaluation
        skip_baselines: If True, skip baseline detectors
    
    Returns:
        Nested dictionary: {id_dataset: {ood_dataset: {detector: metrics}}}
    """
    if device is None:
        device = get_device()
    
    all_results = {}
    
    # Evaluation Set 1: CIFAR-10
    print("\n" + "=" * 80)
    print("EVALUATION SET 1: CIFAR-10 as ID")
    print("=" * 80)
    
    id_dataset_1 = 'cifar10'
    # Expand CIFAR-10-C and CIFAR-10-P into individual corruption/perturbation types
    # Each type will be treated as a separate OOD test (avoids duplication and enables granular analysis)
    cifar10c_corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]
    cifar10c_datasets = [f'cifar10c:{ct}' for ct in cifar10c_corruption_types]
    
    cifar10p_perturbation_types = [
        'gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur',
        'snow', 'brightness', 'translate', 'rotate', 'tilt', 'scale', 'shear'
    ]
    cifar10p_datasets = [f'cifar10p:{pt}' for pt in cifar10p_perturbation_types]
    
    ood_datasets_1 = cifar10c_datasets + cifar10p_datasets + ['svhn', 'lsun', 'isun', 'textures', 'places365']
    
    model_path_1 = model_paths.get(id_dataset_1)
    score_model_path_1 = score_model_paths.get(id_dataset_1) if score_model_paths else None
    
    # Setup once for this ID dataset (load data, model, create & fit detectors, pre-compute ID scores)
    print(f"\n{'='*80}")
    print(f"Setting up for ID dataset: {id_dataset_1.upper()}")
    print(f"{'='*80}")
    print("(This will be reused for all OOD datasets)")
    
    setup_result = _setup_id_dataset(
        id_dataset=id_dataset_1,
        model_path=model_path_1,
        score_model_path=score_model_path_1,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        skip_stein=skip_stein,
        skip_baselines=skip_baselines,
        train_only_on_correct=train_only_on_correct,
        cache_dir=cache_dir or './cache',
        use_stein_factory=use_stein_factory,
        stein_classification_scalar_mode=stein_classification_scalar_mode,
        stein_fixed_class_idx=stein_fixed_class_idx,
        include_stein_per_dim_l2=include_stein_per_dim_l2,
        include_stein_per_dim_sum=include_stein_per_dim_sum,
        include_score_norm=include_score_norm,
        include_grad_f_norm=include_grad_f_norm,
        stein_subset_only=stein_subset_only,
        stein_subset_with_baselines=stein_subset_with_baselines,
        stein_ablation1=stein_ablation1,
        stein_ablation1_perdim_l2=stein_ablation1_perdim_l2,
        cifar10_ddpm_model_id=str(cifar10_ddpm_model_id),
        cifar10_ddpm_timestep=int(cifar10_ddpm_timestep),
        cifar10_ddpm_denom=str(cifar10_ddpm_denom),
        cifar10_ddpm_add_noise=bool(cifar10_ddpm_add_noise),
        cifar10_ddpm_noise_seed=int(cifar10_ddpm_noise_seed),
    )
    
    detectors_1 = setup_result['detectors']
    train_loader_1 = setup_result['train_loader']
    val_loader_1 = setup_result['val_loader']
    test_loader_1 = setup_result['test_loader']
    id_scores_cache_1 = setup_result['id_scores_cache']
    image_size_1 = setup_result['image_size']
    mean_1 = setup_result['mean']
    std_1 = setup_result['std']
    
    all_results[id_dataset_1] = {}
    
    # Now evaluate against each OOD dataset (reusing fitted detectors and ID scores)
    for ood_dataset in ood_datasets_1:
        print(f"\n{'='*80}")
        print(f"Running: {id_dataset_1.upper()} vs {ood_dataset.upper()}")
        print(f"{'='*80}")
        
        try:
            results = run_benchmark(
                id_dataset=id_dataset_1,
                ood_dataset=ood_dataset,
                model_path=model_path_1,
                score_model_path=score_model_path_1,
                output_dir=output_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                skip_stein=skip_stein,
                skip_baselines=skip_baselines,
                detectors=detectors_1,  # Reuse fitted detectors
                train_loader=train_loader_1,  # Reuse (not needed, but for consistency)
                val_loader=val_loader_1,  # Reuse (not needed, but for consistency)
                test_loader=test_loader_1,  # Reuse test loader
                id_scores_cache=id_scores_cache_1,  # Reuse pre-computed ID scores
                use_stein_factory=use_stein_factory,
                stein_classification_scalar_mode=stein_classification_scalar_mode,
                stein_fixed_class_idx=stein_fixed_class_idx,
                include_stein_per_dim_l2=include_stein_per_dim_l2,
                include_stein_per_dim_sum=include_stein_per_dim_sum,
                include_score_norm=include_score_norm,
                include_grad_f_norm=include_grad_f_norm,
                stein_subset_only=stein_subset_only,
                stein_subset_with_baselines=stein_subset_with_baselines,
                stein_ablation1=stein_ablation1,
                stein_ablation1_perdim_l2=stein_ablation1_perdim_l2,
                cifar10_ddpm_model_id=str(cifar10_ddpm_model_id),
                cifar10_ddpm_timestep=int(cifar10_ddpm_timestep),
                cifar10_ddpm_denom=str(cifar10_ddpm_denom),
                cifar10_ddpm_add_noise=bool(cifar10_ddpm_add_noise),
                cifar10_ddpm_noise_seed=int(cifar10_ddpm_noise_seed),
            )
            all_results[id_dataset_1][ood_dataset] = results
        except Exception as e:
            print(f"Error evaluating {id_dataset_1} vs {ood_dataset}: {e}")
            import traceback
            traceback.print_exc()
            all_results[id_dataset_1][ood_dataset] = {}
    
    # Note: ImageNet-1K evaluation removed (as requested)
    # Focus on CIFAR-10 evaluation only
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save aggregated results as JSON (convert numpy arrays to lists)
    aggregated_json_path = os.path.join(output_dir, 'all_benchmarks_results.json')
    with open(aggregated_json_path, 'w') as f:
        json.dump(_convert_to_json_serializable(all_results), f, indent=2)
    print(f"\nAll results saved to: {aggregated_json_path}")
    
    # Create comprehensive CSV summary with all metrics including accuracies
    csv_path = os.path.join(output_dir, 'all_benchmarks_summary.csv')
    _save_comprehensive_csv(all_results, csv_path)
    print(f"Comprehensive summary CSV saved to: {csv_path}")
    
    return all_results


def run_all_adversarial_benchmarks(
    model_paths: Dict[str, str],
    score_model_paths: Optional[Dict[str, str]] = None,
    output_dir: str = 'results/benchmark_results',
    batch_size: int = 128,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    skip_stein: bool = False,
    skip_baselines: bool = False,
    train_only_on_correct: bool = False,
    cache_dir: Optional[str] = None,
    use_stein_factory: bool = False,
    stein_classification_scalar_mode: str = 'predicted_class_prob',
    stein_fixed_class_idx: int = 0,
    include_stein_per_dim_l2: bool = False,
    include_stein_per_dim_sum: bool = False,
    include_score_norm: bool = False,
    include_grad_f_norm: bool = False,
    stein_subset_only: bool = False,
    stein_subset_with_baselines: bool = False,
    # DDPM score extraction controls for CIFAR-10 pretrained DDPMScoreWrapper
    cifar10_ddpm_model_id: str = "google/ddpm-cifar10-32",
    cifar10_ddpm_timestep: int = 0,
    cifar10_ddpm_denom: str = "sigma_sq",
    cifar10_ddpm_add_noise: bool = False,
    cifar10_ddpm_noise_seed: int = 0,
    # Stein component ablation preset (type 1)
    stein_ablation1: bool = False,
    # Stein component ablation preset mapped to stein_per_dimension_l2
    stein_ablation1_perdim_l2: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run all adversarial attack benchmarks.
    
    Tests various adversarial attacks (AutoAttack, PGD, FGSM) with different
    threat models (L∞, L2) and epsilon values.
    
    Args:
        model_paths: Dictionary mapping dataset names to model paths
        score_model_paths: Optional dictionary mapping dataset names to score model paths
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        device: Device to use
        skip_stein: If True, skip SteinDetector evaluation
        skip_baselines: If True, skip baseline detectors
        train_only_on_correct: If True, train score model only on correctly classified samples
        cache_dir: Cache directory for adversarial examples and detector baselines
    
    Returns:
        Nested dictionary: {id_dataset: {ood_dataset: {detector: metrics}}}
    """
    if device is None:
        device = get_device()
    
    all_results = {}
    
    # Define comprehensive set of adversarial attacks
    # Format: adversarial:{attack_type}:{threat_model}:{epsilon}
    adversarial_datasets = [
        # AutoAttack (L∞) - most comprehensive
        'adversarial:autoattack:linf:8/255',
        'adversarial:autoattack:linf:4/255',
        'adversarial:autoattack:linf:2/255',
        
        # PGD (L∞) - strong iterative attack
        'adversarial:pgd:linf:8/255:steps=50',
        'adversarial:pgd:linf:4/255:steps=50',
        'adversarial:pgd:linf:2/255:steps=50',
        
        # FGSM (L∞) - fast single-step attack
        'adversarial:fgsm:linf:8/255',
        'adversarial:fgsm:linf:4/255',
        
        # PGD (L2) - different norm
        'adversarial:pgd:l2:0.5:steps=50',
        'adversarial:pgd:l2:1.0:steps=50',
    ]
    
    # For now, focus on CIFAR-10 (can be extended)
    id_dataset = 'cifar10'
    
    print("\n" + "=" * 80)
    print(f"ADVERSARIAL BENCHMARKS: {id_dataset.upper()} as ID")
    print("=" * 80)
    print(f"Testing {len(adversarial_datasets)} adversarial attack configurations")
    
    model_path = model_paths.get(id_dataset)
    score_model_path = score_model_paths.get(id_dataset) if score_model_paths else None
    
    if model_path is None:
        raise ValueError(f"Model path required for {id_dataset} when running adversarial benchmarks")
    
    # Setup once for this ID dataset (load data, model, create & fit detectors, pre-compute ID scores)
    print(f"\n{'='*80}")
    print(f"Setting up for ID dataset: {id_dataset.upper()}")
    print(f"{'='*80}")
    print("(This will be reused for all adversarial OOD datasets)")
    
    setup_result = _setup_id_dataset(
        id_dataset=id_dataset,
        model_path=model_path,
        score_model_path=score_model_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        skip_stein=skip_stein,
        skip_baselines=skip_baselines,
        train_only_on_correct=train_only_on_correct,
        cache_dir=cache_dir or './cache',
        use_stein_factory=use_stein_factory,
        stein_classification_scalar_mode=stein_classification_scalar_mode,
        stein_fixed_class_idx=stein_fixed_class_idx,
        include_stein_per_dim_l2=include_stein_per_dim_l2,
        include_stein_per_dim_sum=include_stein_per_dim_sum,
        include_score_norm=include_score_norm,
        include_grad_f_norm=include_grad_f_norm,
        stein_subset_only=stein_subset_only,
        stein_subset_with_baselines=stein_subset_with_baselines,
        stein_ablation1=stein_ablation1,
        stein_ablation1_perdim_l2=stein_ablation1_perdim_l2,
        cifar10_ddpm_model_id=str(cifar10_ddpm_model_id),
        cifar10_ddpm_timestep=int(cifar10_ddpm_timestep),
        cifar10_ddpm_denom=str(cifar10_ddpm_denom),
        cifar10_ddpm_add_noise=bool(cifar10_ddpm_add_noise),
        cifar10_ddpm_noise_seed=int(cifar10_ddpm_noise_seed),
    )
    
    detectors = setup_result['detectors']
    train_loader = setup_result['train_loader']
    val_loader = setup_result['val_loader']
    test_loader = setup_result['test_loader']
    id_scores_cache = setup_result['id_scores_cache']
    
    all_results[id_dataset] = {}
    
    # Now evaluate against each adversarial dataset (reusing fitted detectors and ID scores)
    for ood_dataset in adversarial_datasets:
        print(f"\n{'='*80}")
        print(f"Running: {id_dataset.upper()} vs {ood_dataset.upper()}")
        print(f"{'='*80}")
        
        try:
            results = run_benchmark(
                id_dataset=id_dataset,
                ood_dataset=ood_dataset,
                model_path=model_path,
                score_model_path=score_model_path,
                output_dir=output_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                skip_stein=skip_stein,
                skip_baselines=skip_baselines,
                detectors=detectors,  # Reuse fitted detectors
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,  # Reuse test loader (needed for adversarial generation)
                id_scores_cache=id_scores_cache,  # Reuse pre-computed ID scores
                train_only_on_correct=train_only_on_correct,
                use_stein_factory=use_stein_factory,
                cache_dir=cache_dir or './cache',
                stein_classification_scalar_mode=stein_classification_scalar_mode,
                stein_fixed_class_idx=stein_fixed_class_idx,
                include_stein_per_dim_l2=include_stein_per_dim_l2,
                include_stein_per_dim_sum=include_stein_per_dim_sum,
                include_score_norm=include_score_norm,
                include_grad_f_norm=include_grad_f_norm,
                stein_subset_only=stein_subset_only,
                stein_subset_with_baselines=stein_subset_with_baselines,
                stein_ablation1=stein_ablation1,
                stein_ablation1_perdim_l2=stein_ablation1_perdim_l2,
                cifar10_ddpm_model_id=str(cifar10_ddpm_model_id),
                cifar10_ddpm_timestep=int(cifar10_ddpm_timestep),
                cifar10_ddpm_denom=str(cifar10_ddpm_denom),
                cifar10_ddpm_add_noise=bool(cifar10_ddpm_add_noise),
                cifar10_ddpm_noise_seed=int(cifar10_ddpm_noise_seed),
            )
            all_results[id_dataset][ood_dataset] = results
        except Exception as e:
            print(f"Error evaluating {id_dataset} vs {ood_dataset}: {e}")
            import traceback
            traceback.print_exc()
            all_results[id_dataset][ood_dataset] = {}
    
    # Save aggregated results as JSON (convert numpy arrays to lists)
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    aggregated_json_path = os.path.join(output_dir, 'all_adversarial_benchmarks_results.json')
    with open(aggregated_json_path, 'w') as f:
        json.dump(_convert_to_json_serializable(all_results), f, indent=2)
    print(f"\nAll adversarial results saved to: {aggregated_json_path}")
    
    # Create comprehensive CSV summary
    csv_path = os.path.join(output_dir, 'all_adversarial_benchmarks_summary.csv')
    _save_comprehensive_csv(all_results, csv_path)
    print(f"Comprehensive summary CSV saved to: {csv_path}")
    
    return all_results


def run_all_cifar_corruptions_benchmarks(
    model_paths: Dict[str, str],
    score_model_paths: Optional[Dict[str, str]] = None,
    output_dir: str = 'results/benchmark_results',
    batch_size: int = 128,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    skip_stein: bool = False,
    skip_baselines: bool = False,
    train_only_on_correct: bool = False,
    cache_dir: Optional[str] = None,
    use_stein_factory: bool = False,
    stein_classification_scalar_mode: str = 'predicted_class_prob',
    stein_fixed_class_idx: int = 0,
    include_stein_per_dim_l2: bool = False,
    include_stein_per_dim_sum: bool = False,
    include_score_norm: bool = False,
    include_grad_f_norm: bool = False,
    stein_subset_only: bool = False,
    stein_subset_with_baselines: bool = False,
    # DDPM score extraction controls for CIFAR-10 pretrained DDPMScoreWrapper
    cifar10_ddpm_model_id: str = "google/ddpm-cifar10-32",
    cifar10_ddpm_timestep: int = 0,
    cifar10_ddpm_denom: str = "sigma_sq",
    cifar10_ddpm_add_noise: bool = False,
    cifar10_ddpm_noise_seed: int = 0,
    # Stein component ablation preset (type 1)
    stein_ablation1: bool = False,
    # Stein component ablation preset mapped to stein_per_dimension_l2
    stein_ablation1_perdim_l2: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run all CIFAR-10-C and CIFAR-10-P corruption/perturbation benchmarks.
    
    Tests all corruption types from CIFAR-10-C and perturbation types from CIFAR-10-P.
    Does NOT include other OOD datasets (SVHN, LSUN, iSUN, Textures, Places365).
    
    Args:
        model_paths: Dictionary mapping dataset names to model paths
        score_model_paths: Optional dictionary mapping dataset names to score model paths
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        device: Device to use
        skip_stein: If True, skip SteinDetector evaluation
        skip_baselines: If True, skip baseline detectors
        train_only_on_correct: If True, train score model only on correctly classified samples
        cache_dir: Cache directory for detector baselines and ID scores
    
    Returns:
        Nested dictionary: {id_dataset: {ood_dataset: {detector: metrics}}}
    """
    if device is None:
        device = get_device()
    
    all_results = {}
    
    # Define all CIFAR-10-C and CIFAR-10-P corruption/perturbation types
    cifar10c_corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]
    cifar10c_datasets = [f'cifar10c:{ct}' for ct in cifar10c_corruption_types]
    
    cifar10p_perturbation_types = [
        'gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur',
        'snow', 'brightness', 'translate', 'rotate', 'tilt', 'scale', 'shear'
    ]
    cifar10p_datasets = [f'cifar10p:{pt}' for pt in cifar10p_perturbation_types]
    
    # Only CIFAR-10-C and CIFAR-10-P (no other OOD datasets)
    ood_datasets = cifar10c_datasets + cifar10p_datasets
    
    # For now, focus on CIFAR-10
    id_dataset = 'cifar10'
    
    print("\n" + "=" * 80)
    print(f"CIFAR-10 CORRUPTIONS BENCHMARKS: {id_dataset.upper()} as ID")
    print("=" * 80)
    print(f"Testing {len(ood_datasets)} corruption/perturbation types")
    print(f"  CIFAR-10-C: {len(cifar10c_datasets)} corruption types")
    print(f"  CIFAR-10-P: {len(cifar10p_datasets)} perturbation types")
    
    model_path = model_paths.get(id_dataset)
    score_model_path = score_model_paths.get(id_dataset) if score_model_paths else None
    
    if model_path is None:
        raise ValueError(f"Model path required for {id_dataset} when running CIFAR corruption benchmarks")
    
    # Setup once for this ID dataset (load data, model, create & fit detectors, pre-compute ID scores)
    print(f"\n{'='*80}")
    print(f"Setting up for ID dataset: {id_dataset.upper()}")
    print(f"{'='*80}")
    print("(This will be reused for all corruption/perturbation OOD datasets)")
    
    setup_result = _setup_id_dataset(
        id_dataset=id_dataset,
        model_path=model_path,
        score_model_path=score_model_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        skip_stein=skip_stein,
        skip_baselines=skip_baselines,
        train_only_on_correct=train_only_on_correct,
        cache_dir=cache_dir or './cache',
        use_stein_factory=use_stein_factory,
        stein_classification_scalar_mode=stein_classification_scalar_mode,
        stein_fixed_class_idx=stein_fixed_class_idx,
        include_stein_per_dim_l2=include_stein_per_dim_l2,
        include_stein_per_dim_sum=include_stein_per_dim_sum,
        include_score_norm=include_score_norm,
        include_grad_f_norm=include_grad_f_norm,
        stein_subset_only=stein_subset_only,
        stein_subset_with_baselines=stein_subset_with_baselines,
        stein_ablation1=stein_ablation1,
        stein_ablation1_perdim_l2=stein_ablation1_perdim_l2,
        cifar10_ddpm_model_id=str(cifar10_ddpm_model_id),
        cifar10_ddpm_timestep=int(cifar10_ddpm_timestep),
        cifar10_ddpm_denom=str(cifar10_ddpm_denom),
        cifar10_ddpm_add_noise=bool(cifar10_ddpm_add_noise),
        cifar10_ddpm_noise_seed=int(cifar10_ddpm_noise_seed),
    )
    
    detectors = setup_result['detectors']
    train_loader = setup_result['train_loader']
    val_loader = setup_result['val_loader']
    test_loader = setup_result['test_loader']
    id_scores_cache = setup_result['id_scores_cache']
    
    all_results[id_dataset] = {}
    
    # Now evaluate against each corruption/perturbation dataset (reusing fitted detectors and ID scores)
    for ood_dataset in ood_datasets:
        print(f"\n{'='*80}")
        print(f"Running: {id_dataset.upper()} vs {ood_dataset.upper()}")
        print(f"{'='*80}")
        
        try:
            results = run_benchmark(
                id_dataset=id_dataset,
                ood_dataset=ood_dataset,
                model_path=model_path,
                score_model_path=score_model_path,
                output_dir=output_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                skip_stein=skip_stein,
                skip_baselines=skip_baselines,
                detectors=detectors,  # Reuse fitted detectors
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,  # Reuse test loader
                id_scores_cache=id_scores_cache,  # Reuse pre-computed ID scores
                train_only_on_correct=train_only_on_correct,
                cache_dir=cache_dir or './cache',
                use_stein_factory=use_stein_factory,
                stein_classification_scalar_mode=stein_classification_scalar_mode,
                stein_fixed_class_idx=stein_fixed_class_idx,
                include_stein_per_dim_l2=include_stein_per_dim_l2,
                include_stein_per_dim_sum=include_stein_per_dim_sum,
                include_score_norm=include_score_norm,
                include_grad_f_norm=include_grad_f_norm,
                stein_subset_only=stein_subset_only,
                stein_subset_with_baselines=stein_subset_with_baselines,
                stein_ablation1=stein_ablation1,
                stein_ablation1_perdim_l2=stein_ablation1_perdim_l2,
                cifar10_ddpm_model_id=str(cifar10_ddpm_model_id),
                cifar10_ddpm_timestep=int(cifar10_ddpm_timestep),
                cifar10_ddpm_denom=str(cifar10_ddpm_denom),
                cifar10_ddpm_add_noise=bool(cifar10_ddpm_add_noise),
                cifar10_ddpm_noise_seed=int(cifar10_ddpm_noise_seed),
            )
            all_results[id_dataset][ood_dataset] = results
        except Exception as e:
            print(f"Error evaluating {id_dataset} vs {ood_dataset}: {e}")
            import traceback
            traceback.print_exc()
            all_results[id_dataset][ood_dataset] = {}
    
    # Save aggregated results as JSON (convert numpy arrays to lists)
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    aggregated_json_path = os.path.join(output_dir, 'all_cifar_corruptions_benchmarks_results.json')
    with open(aggregated_json_path, 'w') as f:
        json.dump(_convert_to_json_serializable(all_results), f, indent=2)
    print(f"\nAll CIFAR corruption results saved to: {aggregated_json_path}")
    
    # Create comprehensive CSV summary
    csv_path = os.path.join(output_dir, 'all_cifar_corruptions_benchmarks_summary.csv')
    _save_comprehensive_csv(all_results, csv_path)
    print(f"Comprehensive summary CSV saved to: {csv_path}")
    
    return all_results


def _convert_to_json_serializable(obj):
    """
    Recursively convert numpy arrays and torch tensors to JSON-serializable types.
    
    Args:
        obj: Object to convert (dict, list, numpy array, torch tensor, etc.)
    
    Returns:
        JSON-serializable version of obj
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: _convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else obj.detach().cpu().item()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Try to convert to string as fallback
        return str(obj)


def _save_comprehensive_csv(
    all_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    csv_path: str,
) -> None:
    """
    Save comprehensive CSV with all metrics including accuracies.
    
    Format: ID_Dataset, OOD_Dataset, Detector, AUROC, FPR95, ID_Top1_Acc, ID_Top5_Acc, OOD_Top1_Acc, OOD_Top5_Acc, ...
    """
    # Collect all metric names from all results
    all_metric_names = set()
    for id_dataset_results in all_results.values():
        for ood_dataset_results in id_dataset_results.values():
            for detector_results in ood_dataset_results.values():
                all_metric_names.update(detector_results.keys())
    
    # Sort metrics: put accuracies first, then OOD metrics
    priority_metrics = [
        'id_top1_accuracy', 'id_top5_accuracy', 'ood_top1_accuracy', 'ood_top5_accuracy',
        'AUROC', 'FPR95', 'AUPR_IN', 'AUPR_OUT', 'AUTC',
        'ood_top1_confidence', 'ood_entropy',
    ]
    other_metrics = sorted([m for m in all_metric_names if m not in priority_metrics])
    metric_order = [m for m in priority_metrics if m in all_metric_names] + other_metrics
    
    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['ID_Dataset', 'OOD_Dataset', 'Detector'] + metric_order
        writer.writerow(header)
        
        # Data rows
        for id_dataset, id_dataset_results in all_results.items():
            for ood_dataset, ood_dataset_results in id_dataset_results.items():
                for detector_name, detector_results in ood_dataset_results.items():
                    row = [id_dataset, ood_dataset, detector_name]
                    for metric in metric_order:
                        value = detector_results.get(metric)
                        if value is None:
                            row.append('N/A')
                        elif isinstance(value, float):
                            row.append(f'{value:.6f}')
                        else:
                            row.append(str(value))
                    writer.writerow(row)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='OOD Detection Benchmark Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        '--id-dataset',
        type=str,
        default='cifar10',
        choices=['cifar10', 'cifar100'],
        help='In-distribution dataset',
    )
    parser.add_argument(
        '--ood-dataset',
        type=str,
        default=None,
        help='Out-of-distribution dataset (e.g., svhn, lsun, isun). Required if not using --run-all',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained classifier checkpoint (used for single benchmark or as default for --run-all)',
    )
    parser.add_argument(
        '--cifar10-model-path',
        type=str,
        default=None,
        help='Path to CIFAR-10 classifier checkpoint (overrides --model-path for CIFAR-10)',
    )
    parser.add_argument(
        '--imagenet-model-path',
        type=str,
        default=None,
        help='Path to ImageNet-1K classifier checkpoint (overrides --model-path for ImageNet)',
    )
    parser.add_argument(
        '--score-model-path',
        type=str,
        default=None,
        help='Path to trained score model checkpoint (used for single benchmark or as default for --run-all)',
    )
    parser.add_argument(
        '--cifar10-score-model-path',
        type=str,
        default=None,
        help='Path to CIFAR-10 score model checkpoint (overrides --score-model-path for CIFAR-10)',
    )
    parser.add_argument(
        '--imagenet-score-model-path',
        type=str,
        default=None,
        help='Path to ImageNet-1K score model checkpoint (overrides --score-model-path for ImageNet)',
    )
    parser.add_argument(
        '--run-all',
        action='store_true',
        help='Run all benchmarks for both evaluation sets',
    )
    parser.add_argument(
        '--run-all-adversarial',
        action='store_true',
        help='Run all adversarial attack benchmarks (AutoAttack, PGD, FGSM with various epsilons)',
    )
    parser.add_argument(
        '--run-all-cifar-corruptions',
        action='store_true',
        help='Run all CIFAR-10-C and CIFAR-10-P corruption/perturbation benchmarks',
    )
    parser.add_argument(
        '--run-all-ood-suite',
        action='store_true',
        help='Run the full OOD suite: adversarial + cifar10c + cifar10p + classic ood (svhn/lsun/isun/textures/places365).',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/benchmark_results',
        help='Directory to save results',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for evaluation',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers',
    )
    parser.add_argument(
        '--skip-stein',
        action='store_true',
        help='Skip SteinDetector evaluation',
    )
    parser.add_argument(
        '--skip-baselines',
        action='store_true',
        help='Skip baseline detectors',
    )
    parser.add_argument(
        '--use-stein-factory',
        action='store_true',
        help='Use SteinFactoryDetector to compute all Stein modes simultaneously (faster, ~4x speedup)',
    )
    parser.add_argument(
        '--stein-ablation1',
        action='store_true',
        help=(
            "Run Stein component ablation (type 1) via SteinFactoryDetector shared compute. "
            "Adds detector rows: stein_full, stein_full_no_lap, stein_lap_only, stein_score_only, "
            "stein_lap_only_std, stein_full_no_lap_std, stein_full_std_balanced."
        ),
    )
    parser.add_argument(
        '--stein-ablation1-perdim-l2',
        action='store_true',
        help=(
            "Run the same ablations as --stein-ablation1, but mapped to stein_per_dimension_l2: "
            "compute per-class terms for all classes and aggregate with L2 over classes. "
            "Adds detector rows: stein_per_dimension_l2, stein_per_dimension_l2_no_lap, stein_per_dimension_l2_lap_only, "
            "stein_per_dimension_l2_score_only, "
            "and std-balanced variants stein_per_dimension_l2_lap_only_std, stein_per_dimension_l2_no_lap_std, "
            "stein_per_dimension_l2_std_balanced."
        ),
    )
    parser.add_argument(
        '--stein-classification-scalar-mode',
        type=str,
        default='predicted_class_prob',
        choices=['predicted_class_prob', 'fixed_class_prob'],
        help=(
            "Scalar f(x) for Stein detectors in classification. "
            "'predicted_class_prob' uses predicted-class softmax prob (argmax per sample). "
            "'fixed_class_prob' uses a fixed class index (more stable Laplacian for ResNet softmax Laplacian)."
        ),
    )
    parser.add_argument(
        '--stein-fixed-class-idx',
        type=int,
        default=0,
        help="If --stein-classification-scalar-mode=fixed_class_prob, use this class index for f(x).",
    )
    parser.add_argument(
        '--include-stein-per-dim-l2',
        action='store_true',
        help='Also evaluate class-agnostic per-dimension Stein residual with L2 aggregation (classification only).',
    )
    parser.add_argument(
        '--include-stein-per-dim-sum',
        action='store_true',
        help='Also evaluate class-agnostic per-dimension Stein residual with SUM aggregation (classification only).',
    )
    parser.add_argument(
        '--include-score-norm',
        action='store_true',
        help='Also evaluate ||s(x)|| as a diagnostic for score-model quality (uses the Stein score model).',
    )
    parser.add_argument(
        '--include-grad-f-norm',
        action='store_true',
        help='Also evaluate ||∇_x f(x)|| as a diagnostic for classifier test-function geometry (uses the same f(x) as Stein).',
    )
    parser.add_argument(
        '--stein-subset-only',
        action='store_true',
        help=(
            "Run ONLY stein_per_dimension_l2 (no baselines). "
            "This detector computes per-class Stein residuals and aggregates with L2 over classes."
        ),
    )
    parser.add_argument(
        '--stein-subset-with-baselines',
        action='store_true',
        help=(
            "Run baseline detectors (msp/energy/odin/mahalanobis/knn/...) PLUS ONLY stein_per_dimension_l2. "
            "Stein detector is fitted independently (no SteinFactory)."
        ),
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, auto-detect if None)',
    )
    parser.add_argument(
        '--train-only-on-correct',
        action='store_true',
        help='Train detectors only on correctly classified ID samples',
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./cache',
        help='Directory for caching detector baselines and ID scores (default: ./cache)',
    )

    # CIFAR-10 DDPM score extraction controls (pretrained HuggingFace DDPM wrapper).
    # These MUST be part of cache keys (handled via detector_cache config) to avoid stale reuse.
    parser.add_argument("--cifar10-ddpm-model-id", type=str, default="google/ddpm-cifar10-32")
    parser.add_argument("--cifar10-ddpm-timestep", type=int, default=0)
    parser.add_argument("--cifar10-ddpm-denom", type=str, default="sigma_sq", choices=["sigma", "sigma_sq"])
    parser.add_argument(
        "--cifar10-ddpm-add-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled (default), query DDPM UNet on x_t via scheduler.add_noise(x0, eps, t).",
    )
    parser.add_argument("--cifar10-ddpm-noise-seed", type=int, default=0)
    
    args = parser.parse_args()

    # region agent log
    _agent_maybe_clear_debug_log()
    _agent_log(
        run_id="perf-investigation",
        hypothesis_id="RUN",
        location="scripts/benchmark_ood_evaluation.py:main",
        message="Benchmark script start",
        data={
            "id_dataset": str(args.id_dataset),
            "ood_dataset": str(args.ood_dataset),
            "use_stein_factory": bool(args.use_stein_factory),
            "stein_classification_scalar_mode": str(args.stein_classification_scalar_mode),
            "stein_fixed_class_idx": int(args.stein_fixed_class_idx),
            "include_stein_per_dim_l2": bool(args.include_stein_per_dim_l2),
            "include_stein_per_dim_sum": bool(args.include_stein_per_dim_sum),
            "include_score_norm": bool(args.include_score_norm),
            "include_grad_f_norm": bool(args.include_grad_f_norm),
            "stein_ablation1": bool(args.stein_ablation1),
            "stein_ablation1_perdim_l2": bool(args.stein_ablation1_perdim_l2),
            "agent_clear_debug_log": (os.environ.get("AGENT_CLEAR_DEBUG_LOG", "0") == "1"),
        },
    )
    # endregion
    
    device = get_device() if args.device is None else torch.device(args.device)
    
    # Handle full OOD suite (adversarial + cifar corruptions + classics)
    if args.run_all_ood_suite:
        if args.cifar10_model_path:
            model_paths = {'cifar10': args.cifar10_model_path}
        elif args.model_path:
            model_paths = {'cifar10': args.model_path}
        else:
            parser.error("--model-path or --cifar10-model-path required for --run-all-ood-suite")

        score_model_paths = None
        if args.cifar10_score_model_path:
            score_model_paths = {'cifar10': args.cifar10_score_model_path}
        elif args.score_model_path:
            score_model_paths = {'cifar10': args.score_model_path}

        combined_results = {}

        print("\n" + "=" * 80)
        print("RUNNING FULL OOD SUITE (ADVERSARIAL + CORRUPTIONS + CLASSICS)")
        print("=" * 80)

        adversarial_results = run_all_adversarial_benchmarks(
            model_paths=model_paths,
            score_model_paths=score_model_paths,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            skip_stein=args.skip_stein,
            skip_baselines=args.skip_baselines,
            train_only_on_correct=args.train_only_on_correct,
            cache_dir=args.cache_dir,
            use_stein_factory=args.use_stein_factory,
            stein_classification_scalar_mode=args.stein_classification_scalar_mode,
            stein_fixed_class_idx=args.stein_fixed_class_idx,
            include_stein_per_dim_l2=args.include_stein_per_dim_l2,
            include_stein_per_dim_sum=args.include_stein_per_dim_sum,
            include_score_norm=args.include_score_norm,
            include_grad_f_norm=args.include_grad_f_norm,
            stein_subset_only=args.stein_subset_only,
            stein_subset_with_baselines=args.stein_subset_with_baselines,
            stein_ablation1=bool(args.stein_ablation1),
            stein_ablation1_perdim_l2=bool(args.stein_ablation1_perdim_l2),
            cifar10_ddpm_model_id=str(args.cifar10_ddpm_model_id),
            cifar10_ddpm_timestep=int(args.cifar10_ddpm_timestep),
            cifar10_ddpm_denom=str(args.cifar10_ddpm_denom),
            cifar10_ddpm_add_noise=bool(args.cifar10_ddpm_add_noise),
            cifar10_ddpm_noise_seed=int(args.cifar10_ddpm_noise_seed),
        )
        for id_dataset, ood_results in adversarial_results.items():
            combined_results.setdefault(id_dataset, {}).update(ood_results)

        nonadv_results = run_all_benchmarks(
            model_paths=model_paths,
            score_model_paths=score_model_paths,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            skip_stein=args.skip_stein,
            skip_baselines=args.skip_baselines,
            train_only_on_correct=args.train_only_on_correct,
            cache_dir=args.cache_dir,
            use_stein_factory=args.use_stein_factory,
            stein_classification_scalar_mode=args.stein_classification_scalar_mode,
            stein_fixed_class_idx=args.stein_fixed_class_idx,
            include_stein_per_dim_l2=args.include_stein_per_dim_l2,
            include_stein_per_dim_sum=args.include_stein_per_dim_sum,
            include_score_norm=args.include_score_norm,
            include_grad_f_norm=args.include_grad_f_norm,
            stein_subset_only=args.stein_subset_only,
            stein_subset_with_baselines=args.stein_subset_with_baselines,
            stein_ablation1=bool(args.stein_ablation1),
            stein_ablation1_perdim_l2=bool(args.stein_ablation1_perdim_l2),
            cifar10_ddpm_model_id=str(args.cifar10_ddpm_model_id),
            cifar10_ddpm_timestep=int(args.cifar10_ddpm_timestep),
            cifar10_ddpm_denom=str(args.cifar10_ddpm_denom),
            cifar10_ddpm_add_noise=bool(args.cifar10_ddpm_add_noise),
            cifar10_ddpm_noise_seed=int(args.cifar10_ddpm_noise_seed),
        )
        for id_dataset, ood_results in nonadv_results.items():
            combined_results.setdefault(id_dataset, {}).update(ood_results)

        combined_json_path = os.path.join(args.output_dir, 'all_ood_suite_results.json')
        with open(combined_json_path, 'w') as f:
            json.dump(_convert_to_json_serializable(combined_results), f, indent=2)
        print(f"\nCombined results saved to: {combined_json_path}")

        combined_csv_path = os.path.join(args.output_dir, 'all_ood_suite_summary.csv')
        _save_comprehensive_csv(combined_results, combined_csv_path)
        print(f"Combined summary CSV saved to: {combined_csv_path}")

        print("\n" + "=" * 80)
        print("Full OOD suite complete!")
        print("=" * 80)

    # Handle combined flags: --run-all-adversarial and --run-all-cifar-corruptions
    # These can be used together or separately
    elif args.run_all_adversarial or args.run_all_cifar_corruptions:
        # Determine which model paths to use
        if args.cifar10_model_path:
            model_paths = {'cifar10': args.cifar10_model_path}
        elif args.model_path:
            model_paths = {'cifar10': args.model_path}
        else:
            parser.error("--model-path or --cifar10-model-path required for --run-all-adversarial or --run-all-cifar-corruptions")
        
        score_model_paths = None
        if args.cifar10_score_model_path:
            score_model_paths = {'cifar10': args.cifar10_score_model_path}
        elif args.score_model_path:
            score_model_paths = {'cifar10': args.score_model_path}
        
        combined_results = {}
        
        # Run adversarial benchmarks if requested
        if args.run_all_adversarial:
            print("\n" + "=" * 80)
            print("RUNNING ADVERSARIAL BENCHMARKS")
            print("=" * 80)
            adversarial_results = run_all_adversarial_benchmarks(
                model_paths=model_paths,
                score_model_paths=score_model_paths,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                skip_stein=args.skip_stein,
                skip_baselines=args.skip_baselines,
                train_only_on_correct=args.train_only_on_correct,
                cache_dir=args.cache_dir,
                use_stein_factory=args.use_stein_factory,
                stein_classification_scalar_mode=args.stein_classification_scalar_mode,
                stein_fixed_class_idx=args.stein_fixed_class_idx,
                include_stein_per_dim_l2=args.include_stein_per_dim_l2,
                include_stein_per_dim_sum=args.include_stein_per_dim_sum,
                include_score_norm=args.include_score_norm,
                include_grad_f_norm=args.include_grad_f_norm,
                stein_subset_only=args.stein_subset_only,
                stein_subset_with_baselines=args.stein_subset_with_baselines,
                stein_ablation1=bool(args.stein_ablation1),
                stein_ablation1_perdim_l2=bool(args.stein_ablation1_perdim_l2),
                cifar10_ddpm_model_id=str(args.cifar10_ddpm_model_id),
                cifar10_ddpm_timestep=int(args.cifar10_ddpm_timestep),
                cifar10_ddpm_denom=str(args.cifar10_ddpm_denom),
                cifar10_ddpm_add_noise=bool(args.cifar10_ddpm_add_noise),
                cifar10_ddpm_noise_seed=int(args.cifar10_ddpm_noise_seed),
            )
            # Merge results
            for id_dataset, ood_results in adversarial_results.items():
                if id_dataset not in combined_results:
                    combined_results[id_dataset] = {}
                combined_results[id_dataset].update(ood_results)
            print("\n" + "=" * 80)
            print("Adversarial benchmark evaluations complete!")
            print("=" * 80)
        
        # Run CIFAR corruption benchmarks if requested
        if args.run_all_cifar_corruptions:
            print("\n" + "=" * 80)
            print("RUNNING CIFAR-10 CORRUPTION BENCHMARKS")
            print("=" * 80)
            corruption_results = run_all_cifar_corruptions_benchmarks(
                model_paths=model_paths,
                score_model_paths=score_model_paths,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                skip_stein=args.skip_stein,
                skip_baselines=args.skip_baselines,
                train_only_on_correct=args.train_only_on_correct,
                cache_dir=args.cache_dir,
                use_stein_factory=args.use_stein_factory,
                stein_classification_scalar_mode=args.stein_classification_scalar_mode,
                stein_fixed_class_idx=args.stein_fixed_class_idx,
                include_stein_per_dim_l2=args.include_stein_per_dim_l2,
                include_stein_per_dim_sum=args.include_stein_per_dim_sum,
                include_score_norm=args.include_score_norm,
                include_grad_f_norm=args.include_grad_f_norm,
                stein_subset_only=args.stein_subset_only,
                stein_subset_with_baselines=args.stein_subset_with_baselines,
                stein_ablation1=bool(args.stein_ablation1),
                stein_ablation1_perdim_l2=bool(args.stein_ablation1_perdim_l2),
                cifar10_ddpm_model_id=str(args.cifar10_ddpm_model_id),
                cifar10_ddpm_timestep=int(args.cifar10_ddpm_timestep),
                cifar10_ddpm_denom=str(args.cifar10_ddpm_denom),
                cifar10_ddpm_add_noise=bool(args.cifar10_ddpm_add_noise),
                cifar10_ddpm_noise_seed=int(args.cifar10_ddpm_noise_seed),
            )
            # Merge results
            for id_dataset, ood_results in corruption_results.items():
                if id_dataset not in combined_results:
                    combined_results[id_dataset] = {}
                combined_results[id_dataset].update(ood_results)
            print("\n" + "=" * 80)
            print("CIFAR-10 corruption benchmark evaluations complete!")
            print("=" * 80)
        
        # Save combined results if both were run
        if args.run_all_adversarial and args.run_all_cifar_corruptions:
            combined_json_path = os.path.join(args.output_dir, 'all_adversarial_and_corruptions_benchmarks_results.json')
            with open(combined_json_path, 'w') as f:
                json.dump(_convert_to_json_serializable(combined_results), f, indent=2)
            print(f"\nCombined results saved to: {combined_json_path}")
            
            combined_csv_path = os.path.join(args.output_dir, 'all_adversarial_and_corruptions_benchmarks_summary.csv')
            _save_comprehensive_csv(combined_results, combined_csv_path)
            print(f"Combined summary CSV saved to: {combined_csv_path}")
        
        print("\n" + "=" * 80)
        print("All requested benchmark evaluations complete!")
        print("=" * 80)
    elif args.run_all:
        # Run all benchmarks (CIFAR-10 only, using pretrained DDPM)
        # Priority: dataset-specific path > general path > default
        cifar10_model = (
            args.cifar10_model_path or 
            args.model_path or 
            'checkpoints/cifar10_resnet18.pth'
        )
        
        model_paths = {
            'cifar10': cifar10_model,
        }
        
        # Score model path (optional). If provided, we will use it for Stein/score-based detectors.
        score_model_paths = None
        cifar10_score_model = args.cifar10_score_model_path or args.score_model_path
        if cifar10_score_model:
            score_model_paths = {'cifar10': cifar10_score_model}
        
        all_results = run_all_benchmarks(
            model_paths=model_paths,
            score_model_paths=score_model_paths,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            skip_stein=args.skip_stein,
            skip_baselines=args.skip_baselines,
            train_only_on_correct=args.train_only_on_correct,
            cache_dir=args.cache_dir,
            use_stein_factory=args.use_stein_factory,
            stein_classification_scalar_mode=args.stein_classification_scalar_mode,
            stein_fixed_class_idx=args.stein_fixed_class_idx,
            include_stein_per_dim_l2=args.include_stein_per_dim_l2,
            include_stein_per_dim_sum=args.include_stein_per_dim_sum,
            include_score_norm=args.include_score_norm,
            include_grad_f_norm=args.include_grad_f_norm,
            stein_subset_only=args.stein_subset_only,
            stein_subset_with_baselines=args.stein_subset_with_baselines,
            stein_ablation1=bool(args.stein_ablation1),
            stein_ablation1_perdim_l2=bool(args.stein_ablation1_perdim_l2),
            cifar10_ddpm_model_id=str(args.cifar10_ddpm_model_id),
            cifar10_ddpm_timestep=int(args.cifar10_ddpm_timestep),
            cifar10_ddpm_denom=str(args.cifar10_ddpm_denom),
            cifar10_ddpm_add_noise=bool(args.cifar10_ddpm_add_noise),
            cifar10_ddpm_noise_seed=int(args.cifar10_ddpm_noise_seed),
        )
        
        print("\n" + "=" * 80)
        print("All benchmark evaluations complete!")
        print("=" * 80)
    else:
        # Run single benchmark
        if args.ood_dataset is None:
            parser.error("--ood-dataset is required when not using --run-all")
        
        results = run_benchmark(
            id_dataset=args.id_dataset,
            ood_dataset=args.ood_dataset,
            model_path=args.model_path,
            score_model_path=args.score_model_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            skip_stein=args.skip_stein,
            skip_baselines=args.skip_baselines,
            train_only_on_correct=args.train_only_on_correct,
            cache_dir=args.cache_dir,
            use_stein_factory=args.use_stein_factory,
            stein_classification_scalar_mode=args.stein_classification_scalar_mode,
            stein_fixed_class_idx=args.stein_fixed_class_idx,
            include_stein_per_dim_l2=args.include_stein_per_dim_l2,
            include_stein_per_dim_sum=args.include_stein_per_dim_sum,
            include_score_norm=args.include_score_norm,
            include_grad_f_norm=args.include_grad_f_norm,
            stein_subset_only=args.stein_subset_only,
            stein_subset_with_baselines=args.stein_subset_with_baselines,
            stein_ablation1=bool(args.stein_ablation1),
            stein_ablation1_perdim_l2=bool(args.stein_ablation1_perdim_l2),
            cifar10_ddpm_model_id=str(args.cifar10_ddpm_model_id),
            cifar10_ddpm_timestep=int(args.cifar10_ddpm_timestep),
            cifar10_ddpm_denom=str(args.cifar10_ddpm_denom),
            cifar10_ddpm_add_noise=bool(args.cifar10_ddpm_add_noise),
            cifar10_ddpm_noise_seed=int(args.cifar10_ddpm_noise_seed),
        )
        
        print("\n" + "=" * 60)
        print("Benchmark evaluation complete!")
        print("=" * 60)


if __name__ == '__main__':
    main()

