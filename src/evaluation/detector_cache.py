"""
Caching utilities for detector baselines and ID scores.

This module provides functions to cache and load:
1. Detector baselines (computed during fit())
2. ID scores (computed on test set)
3. Detector fitting state

Cache keys are based on:
- Model path (or hash of model state)
- ID dataset name
- Detector configuration
- Score model path (if applicable)
- Dataset characteristics (size, etc.)
"""

import os
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils import get_device


def _get_model_hash(model_path: str) -> str:
    """Get a hash of the model file for cache key."""
    if not os.path.exists(model_path):
        return "unknown"
    
    # Use file size and modification time as a simple hash
    # For more robustness, could hash the actual model weights
    stat = os.stat(model_path)
    content = f"{model_path}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _get_detector_config_hash(detector_config: Dict[str, Any]) -> str:
    """Get a hash of detector configuration."""
    # Sort keys for consistent hashing
    config_str = str(sorted(detector_config.items()))
    return hashlib.md5(config_str.encode()).hexdigest()[:16]


def _get_dataset_hash(loader: DataLoader) -> str:
    """Get a simple hash of dataset characteristics."""
    dataset = loader.dataset
    # Use dataset length and batch size as a simple identifier
    # For more robustness, could hash actual data samples
    content = f"{len(dataset)}_{loader.batch_size}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def get_detector_cache_path(
    cache_dir: Path,
    id_dataset: str,
    model_path: str,
    detector_name: str,
    detector_config: Dict[str, Any],
    score_model_path: Optional[str] = None,
    dataset_hash: Optional[str] = None,
) -> Path:
    """
    Generate cache path for a detector's baseline and state.
    
    Args:
        cache_dir: Base cache directory
        id_dataset: ID dataset name (e.g., 'cifar10')
        model_path: Path to classifier model
        detector_name: Detector name (e.g., 'stein_full')
        detector_config: Detector configuration dict
        score_model_path: Path to score model (optional)
        dataset_hash: Hash of training dataset (optional)
    
    Returns:
        Path to cache file
    """
    model_hash = _get_model_hash(model_path)
    config_hash = _get_detector_config_hash(detector_config)
    
    # Build cache key components
    parts = [
        id_dataset,
        detector_name,
        model_hash,
        config_hash,
    ]
    
    if score_model_path:
        score_hash = _get_model_hash(score_model_path)
        parts.append(score_hash)
    
    if dataset_hash:
        parts.append(dataset_hash)
    
    # Create safe filename
    cache_key = "_".join(parts)
    safe_key = cache_key.replace('/', '_').replace('\\', '_')
    
    # Path: cache_dir / id_dataset / detector_name / {cache_key}.pt
    cache_path = cache_dir / id_dataset / detector_name / f"{safe_key}.pt"
    return cache_path


def get_id_scores_cache_path(
    cache_dir: Path,
    id_dataset: str,
    model_path: str,
    detector_name: str,
    detector_config: Dict[str, Any],
    score_model_path: Optional[str] = None,
    dataset_hash: Optional[str] = None,
) -> Path:
    """
    Generate cache path for ID scores.
    
    Uses same structure as detector cache but with '_id_scores' suffix.
    """
    base_path = get_detector_cache_path(
        cache_dir, id_dataset, model_path, detector_name,
        detector_config, score_model_path, dataset_hash
    )
    # Add suffix before extension
    return base_path.parent / f"{base_path.stem}_id_scores{base_path.suffix}"


def save_detector_cache(
    cache_path: Path,
    baseline: Optional[Tensor],
    training_std: Optional[Tensor],
    detector_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save detector baseline and state to cache.
    
    Args:
        cache_path: Path to cache file
        baseline: Baseline value (scalar tensor)
        training_std: Training standard deviation (scalar tensor)
        detector_state: Additional detector state (optional)
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    cache_data = {
        'baseline': baseline.cpu() if baseline is not None else None,
        'training_std': training_std.cpu() if training_std is not None else None,
        'detector_state': detector_state or {},
    }
    
    torch.save(cache_data, cache_path)


def load_detector_cache(
    cache_path: Path,
    device: Optional[torch.device] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load detector baseline and state from cache.
    
    Args:
        cache_path: Path to cache file
        device: Device to move tensors to
    
    Returns:
        Dictionary with 'baseline', 'training_std', 'detector_state', or None if not found
    """
    if not cache_path.exists():
        return None
    
    if device is None:
        device = get_device()
    
    # Set weights_only=False for our own cache files
    cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)
    
    # Move tensors to device
    if cache_data.get('baseline') is not None:
        cache_data['baseline'] = cache_data['baseline'].to(device)
    if cache_data.get('training_std') is not None:
        cache_data['training_std'] = cache_data['training_std'].to(device)
    
    return cache_data


def save_id_scores_cache(
    cache_path: Path,
    id_scores: Tensor,
) -> None:
    """
    Save ID scores to cache.
    
    Args:
        cache_path: Path to cache file
        id_scores: ID scores tensor (N,)
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as CPU tensor to save space
    torch.save({'id_scores': id_scores.cpu()}, cache_path)


def load_id_scores_cache(
    cache_path: Path,
    device: Optional[torch.device] = None,
) -> Optional[Tensor]:
    """
    Load ID scores from cache.
    
    Args:
        cache_path: Path to cache file
        device: Device to move tensor to
    
    Returns:
        ID scores tensor (N,) or None if not found
    """
    if not cache_path.exists():
        return None
    
    if device is None:
        device = get_device()
    
    # Set weights_only=False for our own cache files
    cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)
    id_scores = cache_data['id_scores']
    
    # Move to device (but keep on CPU for now, move when needed)
    return id_scores  # Keep on CPU, move to device when used


def get_detector_config_from_stein_detector(detector) -> Dict[str, Any]:
    """
    Extract configuration dict from a SteinDetector or SteinFactoryDetector instance.
    
    Args:
        detector: SteinDetector or SteinFactoryDetector instance
    
    Returns:
        Configuration dictionary
    """
    from src.detector import SteinFactoryDetector
    
    # Check if it's a factory detector
    if isinstance(detector, SteinFactoryDetector):
        return {
            'detector_type': 'stein_factory',
            'enabled_modes': detector.enabled_modes,
            'skip_laplacian': detector.skip_laplacian,
            'num_probes': detector.num_probes,
            'compute_baseline': detector.compute_baseline,
            'baseline_subset_size': detector.baseline_subset_size,
            # New: ensure ID-score/baseline caches change when f(x) definition changes
            'classification_scalar_mode': getattr(detector, 'classification_scalar_mode', 'predicted_class_prob'),
            'fixed_class_idx': int(getattr(detector, 'fixed_class_idx', 0)),
            'classification_topk': int(getattr(detector, 'classification_topk', 1)),
        }
    else:
        # Regular SteinDetector
        cfg = {
            'stein_operator_type': detector.stein_operator_type,
            # Include aggregation for per_dimension (and any future aggregation-aware modes) so caches
            # invalidate correctly between L2 vs SUM, etc.
            'aggregation': getattr(detector, 'aggregation', None),
            'skip_laplacian': detector.skip_laplacian,
            'num_probes': detector.num_probes,
            'compute_baseline': detector.compute_baseline,
            'baseline_subset_size': detector.baseline_subset_size,
            # New: ensure ID-score/baseline caches change when f(x) definition changes
            'classification_scalar_mode': getattr(detector, 'classification_scalar_mode', 'predicted_class_prob'),
            'fixed_class_idx': int(getattr(detector, 'fixed_class_idx', 0)),
            'classification_topk': int(getattr(detector, 'classification_topk', 1)),
        }

        # If the Stein detector uses a DDPM-based score wrapper (pretrained, no score_model_path),
        # we MUST include its config in the cache key, otherwise changing timestep/denom/add_noise/seed
        # would incorrectly reuse old cached baselines/ID scores.
        try:
            from src.ddpm_score import DDPMScoreWrapper  # type: ignore

            sm = getattr(detector, "score_model", None)
            if isinstance(sm, DDPMScoreWrapper):
                cfg["score_model_kind"] = "ddpm"
                cfg["ddpm_model_id"] = str(getattr(sm, "model_id", ""))
                cfg["ddpm_timestep"] = int(getattr(sm, "timestep", 0))
                cfg["ddpm_denom_mode"] = str(getattr(sm, "denom_mode", ""))
                cfg["ddpm_add_noise"] = bool(getattr(sm, "add_noise", False))
                cfg["ddpm_noise_seed"] = int(getattr(sm, "noise_seed", 0))
        except Exception:
            pass

        return cfg


def get_detector_config_from_baseline_detector(detector_name: str, detector) -> Dict[str, Any]:
    """
    Extract configuration dict from a baseline detector instance.
    
    Args:
        detector_name: Name of detector (e.g., 'msp', 'odin')
        detector: Detector instance
    
    Returns:
        Configuration dictionary
    """
    config = {'detector_type': detector_name}
    
    # Extract relevant attributes based on detector type
    if detector_name == 'odin':
        if hasattr(detector, 'temperature'):
            config['temperature'] = detector.temperature
        if hasattr(detector, 'eps'):
            config['eps'] = detector.eps
    elif detector_name == 'mahalanobis':
        if hasattr(detector, 'eps'):
            config['eps'] = detector.eps
    elif detector_name == 'energy':
        if hasattr(detector, 'temperature'):
            config['temperature'] = detector.temperature
    elif detector_name == 'knn':
        if hasattr(detector, 'n_neighbors'):
            config['n_neighbors'] = detector.n_neighbors
    elif detector_name == 'react':
        if hasattr(detector, 'threshold'):
            config['threshold'] = detector.threshold
        # Note: detector function is not easily serializable, so we skip it
        # The default (EnergyBased.score) will be used if not specified
    elif detector_name == 'gsc':
        if hasattr(detector, 'num_perturbations'):
            config['num_perturbations'] = detector.num_perturbations
        if hasattr(detector, 'eps'):
            config['eps'] = detector.eps
        if hasattr(detector, 'mask_percentile'):
            config['mask_percentile'] = detector.mask_percentile
    
    return config
