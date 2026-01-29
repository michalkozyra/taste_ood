"""
Adversarial dataset generation and caching.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .attack_config import AttackConfig
from .robustbench_attacks import create_attack, generate_adversarial_batch


class AdversarialDataset(Dataset):
    """
    PyTorch Dataset for adversarial examples.
    
    Stores adversarial examples and their original labels.
    Images are stored in [0, 1] range (not normalized).
    """
    
    def __init__(
        self,
        adversarial_images: Tensor,
        labels: Tensor,
        transform: Optional[torch.nn.Module] = None,
    ):
        """
        Initialize adversarial dataset.
        
        Args:
            adversarial_images: Adversarial images (N, C, H, W) in [0, 1] range
            labels: Original labels (N,)
            transform: Optional transform to apply
        """
        self.adversarial_images = adversarial_images
        self.labels = labels
        self.transform = transform
        
        if len(adversarial_images) != len(labels):
            raise ValueError(
                f"Mismatch: {len(adversarial_images)} images but {len(labels)} labels"
            )
    
    def __len__(self) -> int:
        return len(self.adversarial_images)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = self.adversarial_images[idx]
        label = self.labels[idx].item()
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class NormalizedAdversarialDataset(AdversarialDataset):
    """
    AdversarialDataset with normalization applied.
    
    This is a picklable version for use with DataLoader multiprocessing.
    Images are expected to be in [0, 1] range and will be normalized.
    """
    
    def __init__(
        self,
        adversarial_images: Tensor,
        labels: Tensor,
        mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465),  # CIFAR-10 default
        std: Tuple[float, ...] = (0.2023, 0.1994, 0.2010),   # CIFAR-10 default
    ):
        """
        Initialize normalized adversarial dataset.
        
        Args:
            adversarial_images: Adversarial images (N, C, H, W) in [0, 1] range
            labels: Original labels (N,)
            mean: Normalization mean (per channel)
            std: Normalization std (per channel)
        """
        # Fix shape before passing to parent (ensure [N, C, H, W])
        # Handle cases where images might have extra dimensions from old cache format
        if adversarial_images.dim() == 5 and adversarial_images.size(1) == 1:
            adversarial_images = adversarial_images.squeeze(1)
        if adversarial_images.dim() != 4:
            raise ValueError(f"Expected adversarial_images with shape [N, C, H, W], got {adversarial_images.shape}")
        
        super().__init__(adversarial_images, labels, transform=None)
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        # Get image and label from parent (without transform, since we set transform=None)
        img = self.adversarial_images[idx]
        label = self.labels[idx].item()
        
        # Squeeze any extra singleton dimensions (safety check for edge cases)
        if img.dim() > 3:
            img = torch.squeeze(img)
        
        # Apply normalization (images are in [0, 1] range)
        # mean/std are stored as [1, C, 1, 1] for batch processing
        # For single image [C, H, W], we need [C, 1, 1] for proper broadcasting
        mean = self.mean.to(img.device) if hasattr(img, 'device') else self.mean
        std = self.std.to(img.device) if hasattr(img, 'device') else self.std
        
        # Reshape mean/std from [1, C, 1, 1] to [C, 1, 1] for single image broadcasting
        mean = mean.squeeze(0)  # [1, C, 1, 1] -> [C, 1, 1]
        std = std.squeeze(0)    # [1, C, 1, 1] -> [C, 1, 1]
        
        # Normalize: [C, H, W] - [C, 1, 1] = [C, H, W] (correct broadcasting)
        img = (img - mean) / std
        
        return img, label


def _get_cache_path(
    cache_dir: str,
    dataset_name: str,
    model_name: str,
    config: AttackConfig,
    data_hash: str,
) -> Path:
    """
    Generate cache path for adversarial examples.
    
    Args:
        cache_dir: Base cache directory
        dataset_name: Name of source dataset (e.g., 'cifar10')
        model_name: Name/architecture of model
        config: Attack configuration
        data_hash: Hash of data subset used
    
    Returns:
        Path to cache file
    """
    # Create cache directory structure
    cache_base = Path(cache_dir) / dataset_name / model_name / config.attack_type / config.threat_model
    
    # Create cache key from config
    config_str = f"eps={config.epsilon}"
    if config.steps:
        config_str += f"_steps={config.steps}"
    if config.step_size:
        config_str += f"_step={config.step_size}"
    if config.restarts:
        config_str += f"_restarts={config.restarts}"
    if config.seed:
        config_str += f"_seed={config.seed}"
    
    cache_key = f"{config_str}_{data_hash}.pt"
    cache_path = cache_base / cache_key
    
    return cache_path


def _compute_data_hash(data_loader: DataLoader, max_samples: int = 1000) -> str:
    """
    Compute hash of data loader for cache key.
    
    Args:
        data_loader: DataLoader to hash
        max_samples: Maximum number of samples to use for hashing
    
    Returns:
        Hash string
    """
    # Collect sample indices and labels
    indices = []
    labels = []
    count = 0
    
    for batch_idx, (x, y) in enumerate(data_loader):
        batch_size = x.size(0)
        for i in range(batch_size):
            if count >= max_samples:
                break
            indices.append(batch_idx * data_loader.batch_size + i)
            labels.append(y[i].item())
            count += 1
        if count >= max_samples:
            break
    
    # Create hash
    hash_input = f"{indices}_{labels}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:16]


def _generate_autoattack_batch(attack, x: Tensor, y: Tensor, device: torch.device) -> Tensor:
    """
    Helper function to generate AutoAttack adversarial examples for a batch.
    
    AutoAttack's run_standard_evaluation processes the full dataset, so we need
    a workaround for batch processing.
    """
    # For batch processing with AutoAttack, we'll use the individual attacks
    # that AutoAttack uses internally
    try:
        from autoattack.autopgd_base import APGDAttack
    except ImportError:
        raise ImportError(
            "AutoAttack is not installed. Install it with: "
            "pip install git+https://github.com/fra31/auto-attack.git"
        )
    
    # Use APGDAttack as a proxy (AutoAttack uses APGDAttack internally)
    # This is a simplification - full AutoAttack would use multiple attacks
    if hasattr(attack, 'apgd'):
        # If attack has individual attack components, use APGDAttack
        return attack.apgd.perturb(x, y)
    else:
        # Fallback: create a simple APGDAttack attack
        # APGDAttack expects norm as a string: 'Linf' or 'L2'
        norm = attack.norm if hasattr(attack, 'norm') else 'Linf'
        apgd = APGDAttack(
            attack.model,
            norm=norm,
            eps=attack.eps,
            n_iter=100,
            n_restarts=1,
            device=device,
            verbose=False,
        )
        return apgd.perturb(x, y)


def generate_adversarial_dataset(
    model: nn.Module,
    data_loader: DataLoader,
    config: AttackConfig,
    device: torch.device,
    cache_dir: Optional[str] = None,
    dataset_name: str = 'unknown',
    model_name: Optional[str] = None,
    max_samples: Optional[int] = None,
    verbose: bool = True,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> Tuple[AdversarialDataset, Dict[str, Any]]:
    """
    Generate adversarial examples from a data loader.
    
    Args:
        model: Target model to attack
        data_loader: DataLoader with clean images (may be normalized)
        config: Attack configuration
        device: Device to run attack on
        cache_dir: Directory to cache adversarial examples (None = no caching)
        dataset_name: Name of source dataset (for cache organization)
        model_name: Name of model (for cache organization, auto-detected if None)
        max_samples: Maximum number of samples to generate (None = all)
        verbose: Whether to show progress
        mean: Normalization mean (for denormalization, default: CIFAR-10)
        std: Normalization std (for denormalization, default: CIFAR-10)
    
    Returns:
        adversarial_dataset: AdversarialDataset with generated examples (in [0, 1] range)
        stats: Dictionary with generation statistics
    """
    model.eval()
    
    # Auto-detect model name if not provided
    if model_name is None:
        model_name = model.__class__.__name__
    
    # Check cache
    if cache_dir is not None:
        data_hash = _compute_data_hash(data_loader)
        cache_path = _get_cache_path(cache_dir, dataset_name, model_name, config, data_hash)
        
        if cache_path.exists():
            if verbose:
                print(f"  Loading adversarial examples from cache: {cache_path}")
            # Set weights_only=False for our own cache files (they contain AttackConfig objects)
            cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)
            adversarial_images = cache_data['adversarial_images']
            labels = cache_data['labels']
            stats = cache_data.get('stats', {})
            
            # Fix shape if images have extra dimension (from old cache format)
            if adversarial_images.dim() == 5 and adversarial_images.size(1) == 1:
                adversarial_images = adversarial_images.squeeze(1)
            if adversarial_images.dim() != 4:
                raise ValueError(f"Expected adversarial_images with shape [N, C, H, W], got {adversarial_images.shape}")
            
            dataset = AdversarialDataset(adversarial_images, labels)
            return dataset, stats
    
    # Generate adversarial examples
    if verbose:
        print(f"  Generating adversarial examples using {config.attack_type} ({config.threat_model}, ε={config.epsilon})...")
    
    # Create attack (with model wrapper for normalization)
    attack = create_attack(config, model, device, mean=mean, std=std)
    
    # Handle normalization: attacks work on [0, 1] images, but data_loader may provide normalized images
    # Default CIFAR-10 normalization if not provided
    if mean is None:
        mean = (0.4914, 0.4822, 0.4465)
    if std is None:
        std = (0.2023, 0.1994, 0.2010)
    
    mean_tensor = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, 3, 1, 1)
    
    # Collect all adversarial examples
    all_adv_images = []
    all_labels = []
    total_samples = 0
    successful_attacks = 0
    
    # Process in batches
    pbar = tqdm(data_loader, desc='  Generating', disable=not verbose)
    for batch_idx, (x, y) in enumerate(pbar):
        if max_samples is not None and total_samples >= max_samples:
            break
        
        x = x.to(device)
        y = y.to(device)
        
        # Denormalize images to [0, 1] range for adversarial attacks
        # x_normalized = (x_original - mean) / std
        # x_original = x_normalized * std + mean
        x_denorm = x * std_tensor + mean_tensor
        x_denorm = torch.clamp(x_denorm, 0, 1)  # Ensure in [0, 1]
        
        # Generate adversarial examples
        try:
            if config.attack_type == 'autoattack':
                # AutoAttack expects [0, 1] images and handles them internally
                # For batch processing, we'll use the perturb method if available
                if hasattr(attack, 'perturb'):
                    x_adv = attack.perturb(x_denorm, y)
                else:
                    # Fallback: process batch by batch
                    x_adv = _generate_autoattack_batch(attack, x_denorm, y, device)
            else:
                x_adv, y_orig = generate_adversarial_batch(attack, x_denorm, y, config)
            
            # Ensure adversarial examples are in [0, 1] range
            x_adv = torch.clamp(x_adv, 0, 1)
            
            # Fix shape: ensure x_adv is [B, C, H, W] (remove any extra dimensions)
            if x_adv.dim() == 5 and x_adv.size(1) == 1:
                x_adv = x_adv.squeeze(1)
            if x_adv.dim() != 4:
                raise ValueError(f"Expected x_adv with shape [B, C, H, W], got {x_adv.shape}")
            
            # Check attack success (if model predictions changed)
            # Note: Model expects normalized inputs, so we need to normalize x_adv for prediction
            with torch.no_grad():
                # Normalize for model prediction
                x_norm = (x - mean_tensor) / std_tensor
                x_adv_norm = (x_adv - mean_tensor) / std_tensor
                
                clean_preds = model(x_norm).argmax(dim=1)
                adv_preds = model(x_adv_norm).argmax(dim=1)
                successful = (clean_preds != adv_preds).sum().item()
                successful_attacks += successful
            
            all_adv_images.append(x_adv.cpu())
            all_labels.append(y.cpu())
            total_samples += x.size(0)
            
            if verbose:
                pbar.set_postfix({
                    'success_rate': f'{successful_attacks/total_samples*100:.1f}%',
                    'samples': total_samples
                })
        
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to generate adversarial examples for batch {batch_idx}: {e}")
            continue
    
    # Concatenate all batches
    if len(all_adv_images) == 0:
        raise RuntimeError("Failed to generate any adversarial examples")
    
    adversarial_images = torch.cat(all_adv_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Ensure images have correct shape [N, C, H, W] (remove any extra dimensions)
    if adversarial_images.dim() == 5 and adversarial_images.size(1) == 1:
        adversarial_images = adversarial_images.squeeze(1)
    if adversarial_images.dim() != 4:
        raise ValueError(f"Expected adversarial_images with shape [N, C, H, W], got {adversarial_images.shape}")
    
    # Limit to max_samples if specified
    if max_samples is not None:
        adversarial_images = adversarial_images[:max_samples]
        labels = labels[:max_samples]
    
    # Compute statistics
    attack_success_rate = successful_attacks / total_samples if total_samples > 0 else 0.0
    
    stats = {
        'total_samples': len(adversarial_images),
        'attack_success_rate': attack_success_rate,
        'attack_type': config.attack_type,
        'threat_model': config.threat_model,
        'epsilon': config.epsilon,
    }
    
    # Create dataset
    dataset = AdversarialDataset(adversarial_images, labels)
    
    # Save to cache
    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            'adversarial_images': adversarial_images,
            'labels': labels,
            'stats': stats,
            'config': config,
        }
        torch.save(cache_data, cache_path)
        if verbose:
            print(f"  Cached adversarial examples to: {cache_path}")
    
    if verbose:
        print(f"  Generated {len(adversarial_images)} adversarial examples")
        print(f"  Attack success rate: {attack_success_rate*100:.2f}%")
    
    return dataset, stats
