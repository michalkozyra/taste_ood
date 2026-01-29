"""
Data loading and transformation utilities for Stein shift detection.
"""

import random
from pathlib import Path
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


def get_mnist_dataloaders(batch_size=128, holdout_fraction=0.1, data_dir='./data', device=None, padding_size=18):
    """
    Load MNIST dataset and create train/val/test dataloaders.
    
    Args:
        batch_size: Batch size for dataloaders
        holdout_fraction: Fraction of training data to use for validation
        data_dir: Directory for MNIST data
        device: Device to determine pin_memory setting (only for CUDA)
        padding_size: Padding size in pixels (default: 18, which gives 64x64 images from 28x28)
    
    Returns:
        train_loader, val_loader, test_loader, train_ds, val_ds, test_set
    """
    transform = transforms.Compose([
        transforms.Pad(padding=padding_size, fill=0),  # Pad 28x28 MNIST images (padding_size pixels on each side)
        transforms.ToTensor()
    ])
    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # create a held-out validation portion from train (or keep for score training)
    n = len(train_full)
    idxs = list(range(n))
    random.shuffle(idxs)
    hold = int(holdout_fraction * n)
    train_idx = idxs[hold:]
    val_idx = idxs[:hold]
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(train_full, val_idx)

    # pin_memory only works with CUDA, not MPS
    # For small datasets like MNIST, num_workers=0 is often faster (avoids multiprocessing overhead)
    pin_mem = (device is not None and device.type == 'cuda')
    num_workers = 0 if (device is not None and device.type == 'mps') else 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_set


def create_translated_dataset(dataset: Dataset, max_shift=None):
    """
    Return a new Dataset wrapper that applies a random translation to each image.
    Translation vector is sampled from a product of two independent uniform(-18, +18) distributions.
    
    Args:
        dataset: Base dataset to wrap
        max_shift: Deprecated parameter (kept for compatibility, but not used)
    
    Returns:
        TranslatedDataset wrapper
    """
    class TranslatedDataset(Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            x, y = self.ds[idx]
            # Sample translation from uniform(-18, +18) for each component independently
            tx = random.uniform(-18, 18)
            ty = random.uniform(-18, 18)
            # torchvision.functional.affine: angle=0, translate=(tx,ty), scale=1.0, shear=0
            x_t = TF.affine(x, angle=0.0, translate=(tx,ty), scale=1.0, shear=0.0, fill=0)
            return x_t, y
    return TranslatedDataset(dataset)


def create_translated_dataset_fixed_size(dataset: Dataset, translation_size: float):
    """
    Return a new Dataset wrapper that applies translation with fixed L∞ norm (max metric).
    For translation_size t, samples directions where max(|tx|, |ty|) = t.
    
    Args:
        dataset: Base dataset to wrap
        translation_size: L∞ norm of translation vector (max(|tx|, |ty|) = translation_size)
    
    Returns:
        TranslatedDataset wrapper
    """
    class TranslatedDatasetFixed(Dataset):
        def __init__(self, ds, t_size):
            self.ds = ds
            self.t_size = t_size
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            x, y = self.ds[idx]
            # Sample direction uniformly, then scale to have max(|tx|, |ty|) = t_size
            # Sample angle uniformly in [0, 2π)
            angle = random.uniform(0, 2 * 3.141592653589793)
            # For L∞ norm: if we want max(|tx|, |ty|) = t_size, we can:
            # - Sample uniformly on the L∞ ball boundary
            # One approach: sample uniformly on [-t_size, t_size] for one component,
            # then set the other to ±t_size
            if random.random() < 0.5:
                # tx in [-t_size, t_size], ty = ±t_size
                tx = random.uniform(-self.t_size, self.t_size)
                ty = random.choice([-self.t_size, self.t_size])
            else:
                # ty in [-t_size, t_size], tx = ±t_size
                ty = random.uniform(-self.t_size, self.t_size)
                tx = random.choice([-self.t_size, self.t_size])
            # torchvision.functional.affine: angle=0, translate=(tx,ty), scale=1.0, shear=0
            x_t = TF.affine(x, angle=0.0, translate=(tx,ty), scale=1.0, shear=0.0, fill=0)
            return x_t, y
    return TranslatedDatasetFixed(dataset, translation_size)


def create_rotated_dataset(dataset: Dataset, angle=None):
    """
    Return a new Dataset wrapper that applies random rotation to each image.
    Rotation angle is sampled from uniform(-90, +90) degrees for each image.
    
    Args:
        dataset: Base dataset to wrap
        angle: Deprecated parameter (kept for compatibility, but not used)
    
    Returns:
        RotatedDataset wrapper
    """
    class RotatedDataset(Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            x, y = self.ds[idx]
            # Sample rotation angle from uniform(-90, +90) degrees
            angle = random.uniform(-90, 90)
            x_r = TF.rotate(x, angle=angle, fill=0)
            return x_r, y
    return RotatedDataset(dataset)


def create_rotated_dataset_fixed_angle(dataset: Dataset, angle: float):
    """
    Return a new Dataset wrapper that applies a fixed rotation angle to each image.
    
    Args:
        dataset: Base dataset to wrap
        angle: Rotation angle in degrees (fixed for all images)
    
    Returns:
        RotatedDataset wrapper
    """
    class RotatedDatasetFixed(Dataset):
        def __init__(self, ds, fixed_angle):
            self.ds = ds
            self.fixed_angle = fixed_angle
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            x, y = self.ds[idx]
            x_r = TF.rotate(x, angle=self.fixed_angle, fill=0)
            return x_r, y
    return RotatedDatasetFixed(dataset, angle)


def load_adain_dataset(adain_dir, alpha, padding_size=18, batch_size=128, shuffle=False, num_workers=0):
    """
    Load AdaIN-stylized MNIST dataset from preprocessed images.
    
    Args:
        adain_dir: Base directory containing AdaIN images (e.g., './data/adain')
        alpha: Alpha level to load (e.g., 0.1, 0.2, ..., 1.0)
        padding_size: Padding size to match MNIST format (default: 18, gives 64x64 from 28x28)
        batch_size: Batch size for dataloader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for dataloader
    
    Returns:
        DataLoader for AdaIN dataset
    """
    from torchvision.datasets import ImageFolder
    
    alpha_dir = Path(adain_dir) / f'alpha_{alpha:.1f}'
    
    if not alpha_dir.exists():
        raise FileNotFoundError(
            f"AdaIN dataset not found at {alpha_dir}. "
            f"Please run preprocess_adain.py first to generate images."
        )
    
    # Use same transforms as MNIST for compatibility
    # AdaIN images are saved as 224x224 grayscale, but MNIST with padding=18 is 64x64
    # Resize to match MNIST padded size for compatibility
    target_size = 28 + 2 * padding_size  # e.g., 28 + 2*18 = 64
    
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),  # Resize 224x224 -> 64x64 to match MNIST
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale (should already be, but safe)
        transforms.ToTensor()  # Same as MNIST - no normalization needed
    ])
    
    dataset = ImageFolder(root=str(alpha_dir), transform=transform)
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False  # Don't use pin_memory for compatibility
    )
    
    return loader, dataset


def load_histogram_matching_dataset(hist_dir, alpha, batch_size=128, shuffle=False, num_workers=0, target_size=None):
    """
    Load histogram-matched MNIST dataset from preprocessed images.
    
    Args:
        hist_dir: Base directory containing histogram-matched images (e.g., './data/histogram_matching')
        alpha: Alpha level to load (e.g., 0.1, 0.2, ..., 1.0)
        batch_size: Batch size for dataloader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for dataloader
        target_size: Optional target size to resize to (if None, uses original size from disk)
    
    Returns:
        DataLoader and Dataset for histogram-matched images
    """
    from torchvision.datasets import ImageFolder
    
    alpha_dir = Path(hist_dir) / f'alpha_{alpha:.1f}'
    
    if not alpha_dir.exists():
        raise FileNotFoundError(
            f"Histogram matching dataset not found at {alpha_dir}. "
            f"Please run preprocess_histogram_matching.py first to generate images."
        )
    
    # Images are saved at 256x256 (upscaled from 28x28 MNIST)
    # Optionally resize to target_size if specified (e.g., for compatibility with models expecting 28x28)
    if target_size is not None:
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.ToTensor()  # Convert to tensor [0, 1]
        ])
    else:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.ToTensor()  # Convert to tensor [0, 1]
        ])
    
    dataset = ImageFolder(root=str(alpha_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
    
    return loader, dataset


def get_mnist_dataloaders_32x32(batch_size=128, data_dir='./data', device=None):
    """
    Load MNIST dataset and create train/test dataloaders with 32x32 resize.
    
    Args:
        batch_size: Batch size for dataloaders
        data_dir: Directory for MNIST data
        device: Device to determine pin_memory setting (only for CUDA)
    
    Returns:
        train_loader, test_loader, train_ds, test_set
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor()
    ])
    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # pin_memory only works with CUDA, not MPS
    pin_mem = (device is not None and device.type == 'cuda')
    num_workers = 0 if (device is not None and device.type == 'mps') else 2
    train_loader = DataLoader(train_full, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    return train_loader, test_loader, train_full, test_set


def get_fashion_mnist_dataloaders_32x32(batch_size=128, data_dir='./data', device=None):
    """
    Load Fashion-MNIST dataset and create train/test dataloaders with 32x32 resize.
    
    Args:
        batch_size: Batch size for dataloaders
        data_dir: Directory for Fashion-MNIST data
        device: Device to determine pin_memory setting (only for CUDA)
    
    Returns:
        train_loader, test_loader, train_ds, test_set
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor()
    ])
    train_full = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    # pin_memory only works with CUDA, not MPS
    pin_mem = (device is not None and device.type == 'cuda')
    num_workers = 0 if (device is not None and device.type == 'mps') else 2
    train_loader = DataLoader(train_full, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    return train_loader, test_loader, train_full, test_set


def create_mixed_test_set(mnist_test, fashion_test, alpha):
    """
    Create a mixed test set with alpha fraction of MNIST and (1-alpha) fraction of Fashion-MNIST.
    
    Args:
        mnist_test: MNIST test dataset
        fashion_test: Fashion-MNIST test dataset
        alpha: Fraction of MNIST in the mix (0.0 = all Fashion-MNIST, 1.0 = all MNIST)
    
    Returns:
        MixedDataset wrapper
    """
    class MixedDataset(Dataset):
        def __init__(self, mnist_ds, fashion_ds, mix_alpha):
            self.mnist_ds = mnist_ds
            self.fashion_ds = fashion_ds
            self.mix_alpha = mix_alpha
            self.mnist_size = len(mnist_ds)
            self.fashion_size = len(fashion_ds)
            # Determine how many samples from each dataset
            # We'll use the smaller test set size as reference
            min_size = min(self.mnist_size, self.fashion_size)
            self.n_mnist = int(min_size * mix_alpha)
            self.n_fashion = int(min_size * (1 - mix_alpha))
            self.total_size = self.n_mnist + self.n_fashion
            
            # Randomly sample indices
            self.mnist_indices = list(range(self.mnist_size))
            self.fashion_indices = list(range(self.fashion_size))
            random.shuffle(self.mnist_indices)
            random.shuffle(self.fashion_indices)
            self.mnist_indices = self.mnist_indices[:self.n_mnist]
            self.fashion_indices = self.fashion_indices[:self.n_fashion]
            
        def __len__(self):
            return self.total_size
        
        def __getitem__(self, idx):
            if idx < self.n_mnist:
                x, y = self.mnist_ds[self.mnist_indices[idx]]
                return x, y, 0  # 0 indicates MNIST
            else:
                x, y = self.fashion_ds[self.fashion_indices[idx - self.n_mnist]]
                return x, y, 1  # 1 indicates Fashion-MNIST
    
    return MixedDataset(mnist_test, fashion_test, alpha)

