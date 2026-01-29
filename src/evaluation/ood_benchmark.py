"""
OOD Detection Benchmark Evaluation Pipeline.

This module provides utilities for evaluating OOD detectors (including SteinDetector)
against baseline methods on standard benchmarks, computing AUROC and FPR95 metrics.
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Literal
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
import json
import time

from pytorch_ood.utils import OODMetrics
from pytorch_ood.api import Detector

from ..detector import SteinDetector
from ..detector.baselines import create_all_baseline_detectors
from ..utils import get_device


# OOD Definition Modes
OODDefinitionMode = Literal['dataset', 'misclassified', 'dataset_and_misclassified', 'softmax_based']

# region agent log
_AGENT_DEBUG_LOG_PATH = "/Users/michalkozyra/Developer/PhD/stein_shift_detection/.cursor/debug.log"
_AGENT_CODE_VERSION = "alt_tail_diag_v1"
def _agent_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        with open(_AGENT_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps({
                "timestamp": 0,
                "sessionId": "debug-session",
                "runId": run_id,
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
            }) + "\n")
    except Exception:
        pass
# endregion


# Hardcoded mapping: detectors that return higher scores for OOD
# True = higher scores mean more OOD (need to negate for OODMetrics)
# False = higher scores mean more ID (don't negate)
# None = use two-sided test (for Stein detectors)
DETECTOR_SCORE_DIRECTION = {
    'msp': True,  # MaxSoftmax: higher = more OOD
    'odin': True,  # ODIN: higher = more OOD
    'energy': True,  # Energy: higher = more OOD
    'mahalanobis': True,  # Mahalanobis: higher distance = more OOD
    'knn': True,  # KNN: returns distance from sklearn.kneighbors (higher distance = more OOD)
    'score_norm': True,  # ||s(x)|| diagnostic: higher magnitude often = more OOD (probe only)
    # Stein detectors: use two-sided test (handled separately)
    'stein_full': None,  # None = use two-sided test
    'stein_full_no_lap': None,
    'stein_first_order': None,
    'stein_first_order_sum': None,
    'stein_ensemble': None,  # Ensemble also uses two-sided test
}


def _compute_stein_absolute_difference_scores(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute OOD scores for Stein detectors using absolute difference from baseline.
    
    Stein scores are already computed as |residual - baseline| in SteinDetector.predict().
    Higher absolute difference = more OOD.
    
    For OODMetrics compatibility (which expects higher = more ID), we negate the scores.
    
    Args:
        id_scores: ID scores (N_id,) - already |residual - baseline|
        ood_scores: OOD scores (N_ood,) - already |residual - baseline|
    
    Returns:
        id_scores_transformed: Transformed ID scores for OODMetrics (negated)
        ood_scores_transformed: Transformed OOD scores for OODMetrics (negated)
    """
    # Scores are already |residual - baseline|, where higher = more OOD
    # OODMetrics expects: higher = more ID
    # So we negate: -|residual - baseline|, where higher (less negative) = more ID
    id_scores_transformed = -id_scores
    ood_scores_transformed = -ood_scores
    
    return id_scores_transformed, ood_scores_transformed


def _compute_two_sided_test_scores(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    alpha: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute OOD scores using two-sided statistical test against ID distribution.
    
    Uses percentile-based approach (works for non-symmetric distributions):
    - For scores within ID distribution: p-value = 2 * min(percentile, 1 - percentile)
    - For scores outside ID distribution: preserves ordering using distance from boundaries
      * Below min: p-value = 1 / (1 + distance_below * n_id)
      * Above max: p-value = 1 / (1 + distance_above * n_id)
    - This prevents all extreme values from being clamped to the same score
    
    Lower p-value = more extreme (more OOD).
    Converts to OODMetrics-compatible scores where higher = more ID.
    
    Args:
        id_scores: ID scores (N_id,)
        ood_scores: OOD scores (N_ood,)
        alpha: Significance level (default 0.05 for 95% confidence, not used but kept for API)
    
    Returns:
        id_scores_transformed: Transformed ID scores for OODMetrics
        ood_scores_transformed: Transformed OOD scores for OODMetrics
    """
    # Sort ID scores for percentile computation
    id_scores_sorted = torch.sort(id_scores)[0]
    n_id = int(id_scores_sorted.numel())

    # Get min and max ID scores for handling out-of-boundary values
    id_min = id_scores_sorted[0]
    id_max = id_scores_sorted[-1]
    id_range = id_max - id_min
    eps = 1e-10

    def _two_sided_logp(scores: torch.Tensor) -> torch.Tensor:
        # Inclusive percentile: P(ID <= score)
        # Use searchsorted for O(N log N) instead of O(N^2) counting.
        counts_le = torch.searchsorted(id_scores_sorted, scores, right=True).float()
        percentile = counts_le / float(n_id)

        below = scores < id_min
        above_or_equal = scores >= id_max  # includes max, avoids p=0 from the inside formula
        inside = ~(below | above_or_equal)

        # inside: p = 2 * min(p, 1-p)
        p_inside = 2.0 * torch.minimum(percentile, 1.0 - percentile)

        # below min: preserve ordering by distance below min
        distance_below = (id_min - scores) / (id_range + eps)
        p_below = 1.0 / (1.0 + distance_below * float(n_id))

        # at/above max: preserve ordering by distance above max (distance=0 at the max → p=1)
        distance_above = (scores - id_max) / (id_range + eps)
        p_above = 1.0 / (1.0 + distance_above * float(n_id))

        p = torch.where(inside, p_inside, torch.where(below, p_below, p_above))
        return torch.log(p + eps)

    id_scores_transformed = _two_sided_logp(id_scores)
    ood_scores_transformed = _two_sided_logp(ood_scores)
    return id_scores_transformed, ood_scores_transformed


def _compute_upper_tail_test_scores(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    alpha: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ID-likeness scores using a one-sided (upper-tail) statistical test against the ID distribution.

    This is appropriate for detectors where the raw score is inherently non-negative and "larger = more OOD"
    (e.g. norms / L2-aggregated residuals). In such cases, the lower tail should NOT be treated as "extreme".

    We compute the empirical CDF F(x) from ID scores, then:
      p_value(x) = 1 - F(x)   (upper tail)
    For scores above the max ID score, we preserve ordering using distance from the max boundary (same spirit
    as `_compute_two_sided_test_scores`), avoiding clamping all extreme values to the same p-value.

    Returns log(p_value) so that higher = more ID-like, as expected by OODMetrics.
    """
    id_scores_sorted = torch.sort(id_scores)[0]
    n_id = int(id_scores_sorted.numel())

    id_min = id_scores_sorted[0]
    id_max = id_scores_sorted[-1]
    id_range = id_max - id_min
    eps = 1e-10

    def _upper_tail_logp(scores: torch.Tensor) -> torch.Tensor:
        # Strict percentile: P(ID < score)
        counts_lt = torch.searchsorted(id_scores_sorted, scores, right=False).float()
        percentile_left = counts_lt / float(n_id)

        above = scores > id_max  # strictly above max → percentile_left == 1
        inside_or_below = ~above

        # p = 1 - F_left(x)
        p_inside = 1.0 - percentile_left

        # above max: preserve ordering by distance above max
        distance_above = (scores - id_max) / (id_range + eps)
        p_above = 1.0 / (1.0 + distance_above * float(n_id))

        p = torch.where(inside_or_below, p_inside, p_above)
        return torch.log(p + eps)

    id_scores_transformed = _upper_tail_logp(id_scores)
    ood_scores_transformed = _upper_tail_logp(ood_scores)
    return id_scores_transformed, ood_scores_transformed


def compute_classifier_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    top_k: int = 5,
    dataset_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute top-1 and top-k classification accuracy for a model on a dataset.
    
    Args:
        model: Classifier model
        dataloader: DataLoader with (x, y) batches
        device: Device to use (auto-detect if None)
        verbose: If True, show progress bar
        top_k: Compute top-k accuracy (default: 5 for top-5)
        dataset_name: Optional dataset name for warning about label mismatches
    
    Returns:
        Dictionary with 'top1_accuracy' and 'top5_accuracy' (or None if labels unavailable)
    """
    if device is None:
        device = get_device()
    
    # Datasets that don't have meaningful CIFAR-10 labels
    # These datasets have their own label spaces that don't correspond to CIFAR-10 classes
    ood_datasets_without_cifar10_labels = {
        'lsun', 'isun', 'tinyimagenet', 'textures', 'dtd', 'places365', 'places',
    }
    
    # Check if this is an OOD dataset without meaningful CIFAR-10 labels
    if dataset_name:
        dataset_name_lower = dataset_name.lower()
        # Check for CIFAR-10-C (has correct labels)
        is_cifar10c = dataset_name_lower.startswith('cifar10c') or dataset_name_lower.startswith('cifar10-c')
        # Check for CIFAR-10-P (has correct labels - uses CIFAR-10 test labels)
        is_cifar10p = dataset_name_lower.startswith('cifar10p') or dataset_name_lower.startswith('cifar10-p')
        # Check for SVHN (has digit labels, not CIFAR-10 class labels)
        is_svhn = dataset_name_lower == 'svhn'
        
        # Check if labels are meaningful
        if not is_cifar10c and not is_cifar10p and (dataset_name_lower in ood_datasets_without_cifar10_labels or 
                                any(d in dataset_name_lower for d in ood_datasets_without_cifar10_labels)):
            if verbose:
                print(f"  ⚠️  Warning: {dataset_name} does not have CIFAR-10 class labels.")
                print(f"     Accuracy computation will be meaningless (labels are from different label space).")
                print(f"     Returning None for accuracy metrics.")
            return {'top1_accuracy': None, 'top5_accuracy': None}
        elif is_svhn:
            if verbose:
                print(f"  ⚠️  Warning: SVHN has digit labels (0-9), not CIFAR-10 class labels.")
                print(f"     Accuracy computation may not be meaningful without label mapping.")
    
    model.eval()
    correct_top1 = 0
    correct_topk = 0
    total = 0
    
    iterator = tqdm(dataloader, desc='Computing accuracy', disable=not verbose)
    with torch.no_grad():
        for batch in iterator:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch, None
            
            x = x.to(device)
            if y is not None:
                y = y.to(device)
                logits = model(x)
                
                # Check if labels are in valid range for CIFAR-10 (0-9)
                # This helps detect label mismatches
                if dataset_name and not is_cifar10c and not is_cifar10p and y.max().item() >= 10:
                    if verbose:
                        print(f"  ⚠️  Warning: Labels out of CIFAR-10 range (0-9). Max label: {y.max().item()}")
                        print(f"     This suggests labels are from a different label space.")
                        print(f"     Returning None for accuracy metrics.")
                    return {'top1_accuracy': None, 'top5_accuracy': None}
                
                # Top-1 accuracy
                pred_top1 = logits.argmax(dim=1)
                correct_top1 += (pred_top1 == y).sum().item()
                
                # Top-k accuracy
                _, pred_topk = logits.topk(min(top_k, logits.size(1)), dim=1)  # (B, k)
                correct_topk += pred_topk.eq(y.view(-1, 1).expand_as(pred_topk)).sum().item()
                
                total += y.size(0)
            else:
                # No labels available (e.g., OOD dataset without ID labels)
                return {'top1_accuracy': None, 'top5_accuracy': None}
    
    if total == 0:
        return {'top1_accuracy': 0.0, 'top5_accuracy': 0.0}
    
    return {
        'top1_accuracy': correct_top1 / total,
        'top5_accuracy': correct_topk / total,
    }


def _compute_classifier_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute classifier predictions, ground truth labels, and softmax probabilities.
    
    Args:
        model: Classifier model
        dataloader: DataLoader with (x, y) batches
        device: Device to use (auto-detect if None)
        verbose: If True, show progress bar
    
    Returns:
        predictions: (N,) predicted class indices
        ground_truth: (N,) true class indices (or -1 if not available)
        softmax: (N, num_classes) softmax probabilities
    """
    if device is None:
        device = get_device()
    
    model.eval()
    all_predictions = []
    all_ground_truth = []
    all_softmax = []
    
    iterator = tqdm(dataloader, desc='Computing predictions', disable=not verbose)
    with torch.no_grad():
        for batch in iterator:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                x, y = batch, None
            
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_predictions.append(preds.cpu())
            all_softmax.append(probs.cpu())
            
            if y is not None:
                all_ground_truth.append(y.cpu())
            else:
                # No ground truth available - use -1 as placeholder
                all_ground_truth.append(torch.full((x.size(0),), -1, dtype=torch.long))
    
    predictions = torch.cat(all_predictions)
    ground_truth = torch.cat(all_ground_truth)
    softmax = torch.cat(all_softmax)
    
    return predictions, ground_truth, softmax


def _filter_correctly_classified_samples(
    dataloader: DataLoader,
    classifier_model: nn.Module,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> DataLoader:
    """
    Filter DataLoader to only include correctly classified samples.
    
    Args:
        dataloader: Original DataLoader with (x, y) batches
        classifier_model: Classifier model to check predictions
        device: Device to use (auto-detect if None)
        verbose: If True, show progress
    
    Returns:
        Filtered DataLoader containing only correctly classified samples
    """
    if device is None:
        device = get_device()
    
    classifier_model.eval()
    correct_indices = []
    dataset = dataloader.dataset
    
    iterator = tqdm(dataloader, desc='Filtering correctly classified samples', disable=not verbose)
    
    current_idx = 0
    with torch.no_grad():
        for batch in iterator:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                # Skip batches without labels
                current_idx += len(batch[0]) if isinstance(batch, (list, tuple)) else len(batch)
                continue
            
            x = x.to(device)
            y = y.to(device)
            
            # Get predictions
            logits = classifier_model(x)
            predictions = logits.argmax(dim=1)
            
            # Find correct predictions
            correct_mask = (predictions == y).cpu()
            
            # Add indices for correct samples
            batch_indices = list(range(current_idx, current_idx + len(x)))
            for i, is_correct in enumerate(correct_mask):
                if is_correct:
                    correct_indices.append(batch_indices[i])
            
            current_idx += len(x)
    
    if len(correct_indices) == 0:
        raise ValueError("No correctly classified samples found in the dataset!")
    
    # Create filtered dataset
    filtered_dataset = Subset(dataset, correct_indices)
    
    # Create new DataLoader with same settings as original
    # Detect shuffle from sampler type (RandomSampler = shuffled, SequentialSampler = not shuffled)
    is_shuffled = isinstance(dataloader.sampler, RandomSampler) if hasattr(dataloader, 'sampler') else False
    
    filtered_loader = DataLoader(
        filtered_dataset,
        batch_size=dataloader.batch_size,
        shuffle=is_shuffled,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory if hasattr(dataloader, 'pin_memory') else False,
        drop_last=dataloader.drop_last if hasattr(dataloader, 'drop_last') else False,
    )
    
    if verbose:
        print(f"  Filtered dataset: {len(correct_indices)} / {len(dataset)} samples are correctly classified")
    
    return filtered_loader


def _compute_classifier_losses(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-sample cross-entropy loss for all samples in dataloader.
    
    Args:
        model: Classifier model
        dataloader: DataLoader with (x, y) batches
        device: Device to use (auto-detect if None)
        verbose: If True, show progress bar
    
    Returns:
        losses: (N,) per-sample cross-entropy losses
        ground_truth: (N,) true class indices (or -1 if not available)
    """
    if device is None:
        device = get_device()
    
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss
    all_losses = []
    all_ground_truth = []
    
    iterator = tqdm(dataloader, desc='Computing losses', disable=not verbose)
    with torch.no_grad():
        for batch in iterator:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                x, y = batch, None
            
            x = x.to(device)
            logits = model(x)
            
            if y is not None:
                y = y.to(device)
                # Compute per-sample loss
                losses = criterion(logits, y)
                all_losses.append(losses.cpu())
                all_ground_truth.append(y.cpu())
            else:
                # No ground truth available - cannot compute loss
                # Return NaN losses and -1 placeholder for ground truth
                batch_size = x.size(0)
                all_losses.append(torch.full((batch_size,), float('nan'), dtype=torch.float32))
                all_ground_truth.append(torch.full((batch_size,), -1, dtype=torch.long))
    
    losses = torch.cat(all_losses)
    ground_truth = torch.cat(all_ground_truth)
    
    return losses, ground_truth


def _compute_spearman_correlation(
    scores: torch.Tensor,
    losses: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Compute Spearman correlation coefficient and p-value between scores and losses.
    
    Args:
        scores: (N,) detector scores
        losses: (N,) classifier losses (must align with scores)
        device: Device for tensors (not used, but kept for consistency)
    
    Returns:
        Dictionary with:
        - 'spearman_correlation': float (correlation coefficient, -1 to 1)
        - 'spearman_pvalue': float (p-value for significance test)
    """
    # Convert to numpy arrays for scipy
    scores_np = scores.cpu().numpy().flatten()
    losses_np = losses.cpu().numpy().flatten()
    
    # Filter out NaN/Inf values
    valid_mask = np.isfinite(scores_np) & np.isfinite(losses_np)
    
    if not valid_mask.any():
        # No valid values
        return {
            'spearman_correlation': float('nan'),
            'spearman_pvalue': float('nan'),
        }
    
    scores_clean = scores_np[valid_mask]
    losses_clean = losses_np[valid_mask]
    
    # Check for constant values (no variance)
    if np.std(scores_clean) == 0 or np.std(losses_clean) == 0:
        # Constant values - correlation undefined
        return {
            'spearman_correlation': float('nan'),
            'spearman_pvalue': float('nan'),
        }
    
    # Compute Spearman correlation
    try:
        correlation, pvalue = spearmanr(scores_clean, losses_clean)
        
        # Handle case where spearmanr returns NaN
        if np.isnan(correlation) or np.isnan(pvalue):
            return {
                'spearman_correlation': float('nan'),
                'spearman_pvalue': float('nan'),
            }
        
        return {
            'spearman_correlation': float(correlation),
            'spearman_pvalue': float(pvalue),
        }
    except Exception as e:
        # Handle any unexpected errors
        return {
            'spearman_correlation': float('nan'),
            'spearman_pvalue': float('nan'),
        }


def _assign_ood_labels(
    ood_definition_mode: OODDefinitionMode,
    n_id_samples: int,
    n_ood_samples: int,
    id_predictions: Optional[torch.Tensor] = None,
    ood_predictions: Optional[torch.Tensor] = None,
    id_ground_truth: Optional[torch.Tensor] = None,
    ood_ground_truth: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Assign OOD labels based on definition mode.
    
    Args:
        ood_definition_mode: Mode for defining OOD samples
        n_id_samples: Number of ID dataset samples
        n_ood_samples: Number of OOD dataset samples
        id_predictions: (N_id,) predicted class indices for ID samples
        ood_predictions: (N_ood,) predicted class indices for OOD samples
        id_ground_truth: (N_id,) true class indices for ID samples
        ood_ground_truth: (N_ood,) true class indices for OOD samples (if available)
        device: Device for tensors
    
    Returns:
        all_labels: Tensor of shape (N_id + N_ood,) where:
                    -1 = ID (in-distribution)
                    0+ = OOD (out-of-distribution)
    """
    if device is None:
        device = get_device()
    
    if ood_definition_mode == 'dataset':
        # Current behavior: OOD = samples from OOD dataset
        all_labels = torch.cat([
            torch.full((n_id_samples,), -1, dtype=torch.long, device=device),  # All ID = -1
            torch.zeros(n_ood_samples, dtype=torch.long, device=device),  # All OOD = 0
        ])
    
    elif ood_definition_mode == 'misclassified':
        # OOD = misclassified samples (regardless of dataset)
        if id_predictions is None or id_ground_truth is None:
            raise ValueError("For 'misclassified' mode, id_predictions and id_ground_truth are required")
        
        # ID samples: -1 if correctly classified, 0 if misclassified
        id_correct = (id_predictions.to(device) == id_ground_truth.to(device))
        id_labels = torch.where(id_correct, -1, 0)
        
        # OOD samples: -1 if correctly classified, 0 if misclassified
        # Note: For OOD samples, "correctly classified" means prediction matches ground truth (if available)
        if ood_ground_truth is not None and (ood_ground_truth >= 0).any():
            # Some OOD samples have ground truth
            ood_has_gt = (ood_ground_truth >= 0)
            ood_correct = (ood_predictions.to(device) == ood_ground_truth.to(device))
            ood_labels = torch.where(ood_has_gt.to(device), 
                                    torch.where(ood_correct, -1, 0),
                                    torch.zeros(n_ood_samples, dtype=torch.long, device=device))  # No GT = assume OOD
        else:
            # No ground truth for OOD samples - treat all as OOD (0)
            ood_labels = torch.zeros(n_ood_samples, dtype=torch.long, device=device)
        
        all_labels = torch.cat([id_labels, ood_labels])
    
    elif ood_definition_mode == 'dataset_and_misclassified':
        # OOD = (from OOD dataset) AND (misclassified)
        if ood_predictions is None or ood_ground_truth is None:
            raise ValueError("For 'dataset_and_misclassified' mode, ood_predictions and ood_ground_truth are required")
        
        # ID samples: Always -1 (ID dataset, regardless of classification)
        id_labels = torch.full((n_id_samples,), -1, dtype=torch.long, device=device)
        
        # OOD samples: 0 if misclassified, -1 if correctly classified
        ood_has_gt = (ood_ground_truth >= 0)
        ood_correct = (ood_predictions.to(device) == ood_ground_truth.to(device))
        ood_labels = torch.where(ood_has_gt.to(device),
                                torch.where(ood_correct, -1, 0),
                                torch.zeros(n_ood_samples, dtype=torch.long, device=device))  # No GT = assume OOD
        
        all_labels = torch.cat([id_labels, ood_labels])
    
    elif ood_definition_mode == 'softmax_based':
        # Future: OOD based on softmax properties
        raise NotImplementedError("'softmax_based' mode not yet implemented")
    
    else:
        raise ValueError(f"Unknown OOD definition mode: {ood_definition_mode}")
    
    return all_labels


def compute_classifier_confidence_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute classifier confidence metrics (top-1 confidence, entropy) for a dataset.
    Useful for OOD datasets where we can't compute accuracy.
    
    Args:
        model: Classifier model
        dataloader: DataLoader with (x, y) batches
        device: Device to use (auto-detect if None)
        verbose: If True, show progress bar
    
    Returns:
        Dictionary with 'top1_confidence' (mean) and 'entropy' (mean)
    """
    if device is None:
        device = get_device()
    
    import torch.nn.functional as F
    
    model.eval()
    all_confidences = []
    all_entropies = []
    
    iterator = tqdm(dataloader, desc='Computing confidence metrics', disable=not verbose)
    with torch.no_grad():
        for batch in iterator:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            
            # Top-1 confidence (max probability)
            top1_conf = probs.max(dim=1)[0]  # (B,)
            all_confidences.append(top1_conf.cpu())
            
            # Entropy: -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)  # (B,)
            all_entropies.append(entropy.cpu())
    
    all_confidences = torch.cat(all_confidences)
    all_entropies = torch.cat(all_entropies)
    
    return {
        'top1_confidence': all_confidences.mean().item(),
        'entropy': all_entropies.mean().item(),
    }


def compute_ood_metrics(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    detector_name: Optional[str] = None,
    ood_dataset_name: Optional[str] = None,
    device: Optional[torch.device] = None,
    use_two_sided_test: bool = False,
    alpha: float = 0.05,
    ood_definition_mode: OODDefinitionMode = 'dataset',
    id_predictions: Optional[torch.Tensor] = None,
    ood_predictions: Optional[torch.Tensor] = None,
    id_ground_truth: Optional[torch.Tensor] = None,
    ood_ground_truth: Optional[torch.Tensor] = None,
    id_losses: Optional[torch.Tensor] = None,
    ood_losses: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute OOD detection metrics (AUROC, FPR95, etc.) using PyTorch-OOD's OODMetrics.
    
    Args:
        id_scores: OOD scores for in-distribution samples (N_id,)
        ood_scores: OOD scores for out-of-distribution samples (N_ood,)
        detector_name: Name of the detector (e.g., 'msp', 'stein_full')
                     Used to determine score direction from hardcoded mapping
        device: Device to use (auto-detect if None)
        use_two_sided_test: If True, use two-sided statistical test (for Stein detectors)
        alpha: Significance level for two-sided test (default 0.05 = 95% confidence)
        ood_definition_mode: How to define OOD samples:
                           - 'dataset': OOD = samples from OOD dataset (default)
                           - 'misclassified': OOD = misclassified samples (regardless of dataset)
                           - 'dataset_and_misclassified': OOD = (from OOD dataset) AND (misclassified)
        id_predictions: (N_id,) predicted class indices for ID samples (required for non-dataset modes)
        ood_predictions: (N_ood,) predicted class indices for OOD samples (required for non-dataset modes)
        id_ground_truth: (N_id,) true class indices for ID samples (required for non-dataset modes)
        ood_ground_truth: (N_ood,) true class indices for OOD samples (optional, -1 if not available)
    
    Returns:
        Dictionary with metrics: {'AUROC': float, 'FPR95': float, ...}
    
    Note:
        OODMetrics expects: Higher scores = More ID (in-distribution)
        Most OOD detectors return: Higher scores = More OOD (need to negate)
        Stein detectors use two-sided test against ID distribution
    """
    if device is None:
        device = get_device()
    
    # OODMetrics computes: P(score_ID > score_OOD)
    # It expects: Higher scores = More ID (in-distribution)
    # Most OOD detectors return: Higher scores = More OOD
    # So we need to negate the scores
    
    # Ensure scores are 1D (squeeze any extra dimensions)
    id_scores = id_scores.squeeze()
    ood_scores = ood_scores.squeeze()
    
    # Ensure float32 for MPS compatibility (MPS doesn't support float64)
    id_scores = id_scores.float()
    ood_scores = ood_scores.float()

    # region agent log
    # Debug ReAct performance: detect score collapse / near-constant distributions.
    if detector_name is not None and detector_name.lower() == "react":
        try:
            _agent_log(
                run_id="react-investigation",
                hypothesis_id="H_REACT_SCORE_COLLAPSE",
                location="src/evaluation/ood_benchmark.py:compute_ood_metrics",
                message="react_raw_score_stats",
                data={
                    "ood_dataset_name": ood_dataset_name,
                    "ood_definition_mode": ood_definition_mode,
                    "n_id": int(id_scores.numel()),
                    "n_ood": int(ood_scores.numel()),
                    "id_mean": float(id_scores.mean().item()) if id_scores.numel() else None,
                    "id_std": float(id_scores.std().item()) if id_scores.numel() else None,
                    "id_min": float(id_scores.min().item()) if id_scores.numel() else None,
                    "id_max": float(id_scores.max().item()) if id_scores.numel() else None,
                    "ood_mean": float(ood_scores.mean().item()) if ood_scores.numel() else None,
                    "ood_std": float(ood_scores.std().item()) if ood_scores.numel() else None,
                    "ood_min": float(ood_scores.min().item()) if ood_scores.numel() else None,
                    "ood_max": float(ood_scores.max().item()) if ood_scores.numel() else None,
                },
            )
        except Exception:
            pass
    # endregion
    
    # Determine score direction: check if detector uses two-sided test or has hardcoded direction
    use_two_sided = use_two_sided_test
    higher_is_ood = True  # Default value
    stein_tail: Literal["two_sided", "upper"] = "two_sided"
    is_stein = False
    
    if detector_name is not None:
        detector_key = detector_name.lower()
        # Check if this is a Stein detector (should use two-sided test)
        if detector_key.startswith('stein') or DETECTOR_SCORE_DIRECTION.get(detector_key) is None:
            use_two_sided = True
            is_stein = True
        else:
            # Use hardcoded direction
            use_two_sided = False
            higher_is_ood = DETECTOR_SCORE_DIRECTION.get(detector_key, True)  # Default to True if unknown

    # region agent log
    _agent_log(
        run_id="perf-investigation",
        hypothesis_id="TAIL",
        location="src/evaluation/ood_benchmark.py:compute_ood_metrics",
        message="Tail modes requested",
        data={
            "code_version": _AGENT_CODE_VERSION,
            "detector_name": detector_name,
            "ood_dataset_name": ood_dataset_name,
            "ood_definition_mode": ood_definition_mode,
            "is_stein": bool(is_stein),
            "use_two_sided": bool(use_two_sided),
        },
    )
    # endregion

    # Assign labels once (reused across tail modes)
    all_labels = _assign_ood_labels(
        ood_definition_mode=ood_definition_mode,
        n_id_samples=len(id_scores),
        n_ood_samples=len(ood_scores),
        id_predictions=id_predictions,
        ood_predictions=ood_predictions,
        id_ground_truth=id_ground_truth,
        ood_ground_truth=ood_ground_truth,
        device=device,
    )

    def _run_metrics(id_scores_for_metrics: torch.Tensor, ood_scores_for_metrics: torch.Tensor) -> Dict[str, float]:
        all_scores = torch.cat([id_scores_for_metrics, ood_scores_for_metrics]).to(device)
        with torch.no_grad():
            order = torch.argsort(all_scores, stable=True)
            ranks = torch.empty_like(all_scores, dtype=torch.float32)
            ranks[order] = torch.arange(all_scores.numel(), device=all_scores.device, dtype=torch.float32)
            denom = max(int(all_scores.numel() - 1), 1)
            all_scores01 = ranks / float(denom)
        metrics = OODMetrics(device=str(device))
        metrics.update(all_scores01, all_labels)
        results = metrics.compute()
        return {
            'AUROC': results.get('AUROC', None),
            'FPR95': results.get('FPR95TPR', None),
            'AUPR_IN': results.get('AUPR-IN', None),
            'AUPR_OUT': results.get('AUPR-OUT', None),
            'AUTC': results.get('AUTC', None),
        }

    # Apply appropriate transformation and compute metrics
    if use_two_sided and is_stein:
        # Always compute BOTH tail modes for Stein detectors.
        id_two, ood_two = _compute_two_sided_test_scores(id_scores, ood_scores, alpha=alpha)
        id_up, ood_up = _compute_upper_tail_test_scores(id_scores, ood_scores, alpha=alpha)

        met_two = _run_metrics(id_two, ood_two)
        met_up = _run_metrics(id_up, ood_up)

        # Choose which tail is used for the *primary* AUROC/FPR95 outputs.
        #
        # IMPORTANT: Tail policy is DATASET-dependent (not detector-dependent):
        # - adversarial:* => two-sided
        # - everything else (svhn/lsun/isun/textures/places365/cifar10c/cifar10p/...) => upper
        #
        # We still compute BOTH tails and export both, but the primary AUROC/FPR95 follows this dataset policy.
        ood_key = str(ood_dataset_name).lower() if ood_dataset_name is not None else ""
        tail_used: Literal["two_sided", "upper"] = "two_sided" if ood_key.startswith("adversarial:") else "upper"

        if tail_used == "upper":
            id_scores_for_metrics, ood_scores_for_metrics = id_up, ood_up
            stein_tail = "upper"
            mapped_results = dict(met_up)
        else:
            id_scores_for_metrics, ood_scores_for_metrics = id_two, ood_two
            stein_tail = "two_sided"
            mapped_results = dict(met_two)

        mapped_results.update({
            'stein_tail': stein_tail,
            # Explicitly exposed tail modes (all metrics)
            'AUROC_two_sided': met_two.get('AUROC'),
            'FPR95_two_sided': met_two.get('FPR95'),
            'AUPR_IN_two_sided': met_two.get('AUPR_IN'),
            'AUPR_OUT_two_sided': met_two.get('AUPR_OUT'),
            'AUTC_two_sided': met_two.get('AUTC'),
            'AUROC_upper': met_up.get('AUROC'),
            'FPR95_upper': met_up.get('FPR95'),
            'AUPR_IN_upper': met_up.get('AUPR_IN'),
            'AUPR_OUT_upper': met_up.get('AUPR_OUT'),
            'AUTC_upper': met_up.get('AUTC'),
        })
        # Also expose the transformed score vectors used for metrics so callers (CSV exporter, analysis)
        # can reuse them without recomputing.
        id_scores_for_metrics_upper, ood_scores_for_metrics_upper = id_up, ood_up
    elif use_two_sided:
        # Forced by caller: two-sided only.
        id_scores_for_metrics, ood_scores_for_metrics = _compute_two_sided_test_scores(id_scores, ood_scores, alpha=alpha)
        mapped_results = _run_metrics(id_scores_for_metrics, ood_scores_for_metrics)
        mapped_results['stein_tail'] = "two_sided"
        id_scores_for_metrics_upper = None
        ood_scores_for_metrics_upper = None
    else:
        # Use simple negation based on hardcoded direction
        if higher_is_ood:
            id_scores_for_metrics = -id_scores
            ood_scores_for_metrics = -ood_scores
        else:
            id_scores_for_metrics = id_scores
            ood_scores_for_metrics = ood_scores
        mapped_results = _run_metrics(id_scores_for_metrics, ood_scores_for_metrics)
        mapped_results['stein_tail'] = None
        id_scores_for_metrics_upper = None
        ood_scores_for_metrics_upper = None

    # Keep legacy diagnostic keys present (now unused)
    mapped_results.setdefault('AUROC_two_sided_alt', None)
    mapped_results.setdefault('FPR95_two_sided_alt', None)

    # region agent log
    _agent_log(
        run_id="perf-investigation",
        hypothesis_id="MET",
        location="src/evaluation/ood_benchmark.py:compute_ood_metrics",
        message="OOD metrics computed",
        data={
            "code_version": _AGENT_CODE_VERSION,
            "detector_name": detector_name,
            "ood_definition_mode": str(ood_definition_mode),
            "use_two_sided": bool(use_two_sided),
            "stein_tail": stein_tail if use_two_sided else None,
            "auroc": float(mapped_results['AUROC']) if mapped_results.get('AUROC') is not None else None,
            "fpr95": float(mapped_results['FPR95']) if mapped_results.get('FPR95') is not None else None,
            "auroc_two_sided": float(mapped_results.get('AUROC_two_sided')) if mapped_results.get('AUROC_two_sided') is not None else None,
            "auroc_upper": float(mapped_results.get('AUROC_upper')) if mapped_results.get('AUROC_upper') is not None else None,
            "fpr95_two_sided": float(mapped_results.get('FPR95_two_sided')) if mapped_results.get('FPR95_two_sided') is not None else None,
            "fpr95_upper": float(mapped_results.get('FPR95_upper')) if mapped_results.get('FPR95_upper') is not None else None,
            "n_id": int(id_scores.numel()),
            "n_ood": int(ood_scores.numel()),
        },
    )
    # endregion
    
    # Compute correlation metrics if losses are provided
    if id_losses is not None and ood_losses is not None:
        # Verify alignment
        if len(id_scores_for_metrics) != len(id_losses):
            raise ValueError(f"ID scores and losses length mismatch: {len(id_scores_for_metrics)} vs {len(id_losses)}")
        if len(ood_scores_for_metrics) != len(ood_losses):
            raise ValueError(f"OOD scores and losses length mismatch: {len(ood_scores_for_metrics)} vs {len(ood_losses)}")
        
        # Combined correlation (all samples)
        all_scores_combined = torch.cat([id_scores_for_metrics, ood_scores_for_metrics])
        all_losses_combined = torch.cat([id_losses, ood_losses])
        combined_corr = _compute_spearman_correlation(all_scores_combined, all_losses_combined, device)
        mapped_results['spearman_correlation_all'] = combined_corr['spearman_correlation']
        mapped_results['spearman_pvalue_all'] = combined_corr['spearman_pvalue']
        
        # ID-only correlation
        id_corr = _compute_spearman_correlation(id_scores_for_metrics, id_losses, device)
        mapped_results['spearman_correlation_id'] = id_corr['spearman_correlation']
        mapped_results['spearman_pvalue_id'] = id_corr['spearman_pvalue']
        
        # OOD-only correlation
        ood_corr = _compute_spearman_correlation(ood_scores_for_metrics, ood_losses, device)
        mapped_results['spearman_correlation_ood'] = ood_corr['spearman_correlation']
        mapped_results['spearman_pvalue_ood'] = ood_corr['spearman_pvalue']

        # If Stein, also report correlations for upper-tail transformed scores (so both modes are comparable).
        if use_two_sided and is_stein:
            try:
                id_up_corr, ood_up_corr = _compute_upper_tail_test_scores(id_scores, ood_scores, alpha=alpha)
                mapped_results['spearman_correlation_all_upper'] = _compute_spearman_correlation(
                    torch.cat([id_up_corr, ood_up_corr]), all_losses_combined, device
                )['spearman_correlation']
                mapped_results['spearman_correlation_id_upper'] = _compute_spearman_correlation(id_up_corr, id_losses, device)['spearman_correlation']
                mapped_results['spearman_correlation_ood_upper'] = _compute_spearman_correlation(ood_up_corr, ood_losses, device)['spearman_correlation']
            except Exception:
                mapped_results['spearman_correlation_all_upper'] = float('nan')
                mapped_results['spearman_correlation_id_upper'] = float('nan')
                mapped_results['spearman_correlation_ood_upper'] = float('nan')
    
    return mapped_results


def evaluate_detector(
    detector: Detector,
    id_loader: DataLoader,
    ood_loader: DataLoader,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a single detector on ID and OOD datasets.
    
    Args:
        detector: OOD detector (must implement predict() method)
        id_loader: DataLoader for in-distribution test set
        ood_loader: DataLoader for out-of-distribution test set
        device: Device to use (auto-detect if None)
        verbose: If True, show progress bar
    
    Returns:
        Dictionary with metrics: {'AUROC': float, 'FPR95': float, ...}
    """
    if device is None:
        device = get_device()
    
    if hasattr(detector, 'model') and hasattr(detector.model, 'eval'):
        detector.model.eval()
    
    # Detectors that need gradients (use ODIN preprocessing or gradient-based methods)
    detector_name = getattr(detector, '__class__', type(detector)).__name__.lower()
    needs_grads = 'mahalanobis' in detector_name or 'odin' in detector_name or 'gsc' in detector_name
    
    # Compute scores for ID samples
    id_scores_list = []
    id_iter = tqdm(id_loader, desc='Computing ID scores', disable=not verbose)
    if needs_grads:
        for batch in id_iter:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device).requires_grad_(True)
            scores = detector.predict(x)
            id_scores_list.append(scores.detach().cpu())
    else:
        with torch.no_grad():
            for batch in id_iter:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                scores = detector.predict(x)
                id_scores_list.append(scores.cpu())
    
    id_scores = torch.cat(id_scores_list).squeeze()
    
    # Compute scores for OOD samples
    ood_scores_list = []
    ood_iter = tqdm(ood_loader, desc='Computing OOD scores', disable=not verbose)
    if needs_grads:
        for batch in ood_iter:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device).requires_grad_(True)
            scores = detector.predict(x)
            ood_scores_list.append(scores.detach().cpu())
    else:
        with torch.no_grad():
            for batch in ood_iter:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                scores = detector.predict(x)
                ood_scores_list.append(scores.cpu())
    
    ood_scores = torch.cat(ood_scores_list).squeeze()
    
    # Compute metrics
    # Note: detector_name not available in this function, will use default behavior
    metrics = compute_ood_metrics(id_scores, ood_scores, detector_name=None, device=device)
    
    return metrics


def evaluate_all_detectors(
    detectors: Dict[str, Detector],
    id_loader: DataLoader,
    ood_loader: DataLoader,
    ood_dataset_name: Optional[str] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    id_scores_cache: Optional[Dict[str, torch.Tensor]] = None,
    classifier_model: Optional[nn.Module] = None,
    compute_misclassified_metrics: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate multiple detectors on ID and OOD datasets.
    
    Args:
        detectors: Dictionary mapping detector names to detector instances
        id_loader: DataLoader for in-distribution test set
        ood_loader: DataLoader for out-of-distribution test set
        device: Device to use (auto-detect if None)
        verbose: If True, show progress bars
        id_scores_cache: Optional cache of pre-computed ID scores {detector_name: scores}
                       If provided, will reuse instead of recomputing
    
    Returns:
        Dictionary mapping detector names to metric dictionaries
        Format: {'detector_name': {'AUROC': float, 'FPR95': float, ...}}
    """
    results = {}
    
    # If cache provided, use it; otherwise compute ID scores for each detector
    if id_scores_cache is None:
        id_scores_cache = {}
    
    # Compute predictions and losses once (same for all detectors)
    id_predictions = None
    ood_predictions = None
    id_ground_truth = None
    ood_ground_truth = None
    id_losses = None
    ood_losses = None
    
    if compute_misclassified_metrics:
        if classifier_model is None:
            # Try to extract classifier from first detector
            first_detector = next(iter(detectors.values()))
            if hasattr(first_detector, 'model'):
                classifier_model = first_detector.model
            else:
                raise ValueError(
                    "classifier_model is required for compute_misclassified_metrics=True. "
                    "Either pass classifier_model or ensure detectors have a 'model' attribute."
                )
        
        # Compute predictions for both datasets (once for all detectors)
        if verbose:
            print("\nComputing classifier predictions and losses (shared across all detectors)...")
        id_predictions, id_ground_truth, _ = _compute_classifier_predictions(
            classifier_model, id_loader, device=device, verbose=verbose
        )
        ood_predictions, ood_ground_truth, _ = _compute_classifier_predictions(
            classifier_model, ood_loader, device=device, verbose=verbose
        )
        
        # Compute losses for both datasets (once for all detectors)
        id_losses, id_ground_truth_loss = _compute_classifier_losses(
            classifier_model, id_loader, device=device, verbose=verbose
        )
        ood_losses, ood_ground_truth_loss = _compute_classifier_losses(
            classifier_model, ood_loader, device=device, verbose=verbose
        )
        
        # Verify alignment
        assert torch.equal(id_ground_truth, id_ground_truth_loss), "Ground truth mismatch for ID"
        assert torch.equal(ood_ground_truth, ood_ground_truth_loss), "Ground truth mismatch for OOD"
        if verbose:
            print(f"  ✓ ID samples: {len(id_predictions)} predictions, {len(id_losses)} losses")
            print(f"  ✓ OOD samples: {len(ood_predictions)} predictions, {len(ood_losses)} losses")
    
    # Check if we're using factory mode and pre-compute OOD scores for all modes
    from src.detector import SteinFactoryDetector
    factory_detector = None
    factory_ood_scores_cache = {}
    factory_id_scores_cache = {}
    
    # Find factory detector and its wrappers
    for name, detector in detectors.items():
        if isinstance(detector, SteinFactoryDetector):
            factory_detector = detector
            factory_name = name
            break
        # Check if it's a wrapper (has factory attribute)
        if hasattr(detector, 'factory') and isinstance(detector.factory, SteinFactoryDetector):
            factory_detector = detector.factory
            factory_name = 'stein_factory'  # Assume standard name
            break
    
    # If using factory, pre-compute ID and OOD scores once for all modes
    if factory_detector is not None:
        if verbose:
            print(f"\n[Factory Mode] Pre-computing scores for all Stein modes...")
        if hasattr(factory_detector, 'model') and hasattr(factory_detector.model, 'eval'):
            factory_detector.model.eval()
        
        # Pre-compute ID scores for all modes (if not already cached)
        id_scores_all_modes = {}
        if factory_name not in id_scores_cache or not all(
            f'stein_{mode}' in id_scores_cache for mode in factory_detector.enabled_modes
        ):
            if verbose:
                print(f"  Pre-computing ID scores for all modes...")
            id_iter = tqdm(id_loader, desc=f'  Computing ID scores (factory)', disable=not verbose)
            with torch.no_grad():
                for batch in id_iter:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                    else:
                        x = batch
                    x = x.to(device)
                    # Compute all modes at once
                    all_scores = factory_detector.predict_all(x)
                    # Accumulate scores for each mode
                    for mode_name, scores in all_scores.items():
                        if mode_name not in id_scores_all_modes:
                            id_scores_all_modes[mode_name] = []
                        id_scores_all_modes[mode_name].append(scores.cpu())
            
            # Concatenate all batches for each mode and cache
            for mode_name in id_scores_all_modes:
                mode_scores = torch.cat(id_scores_all_modes[mode_name]).squeeze()
                wrapper_name = f'stein_{mode_name}' if mode_name != 'full_no_lap' else 'stein_full_no_lap'
                id_scores_cache[wrapper_name] = mode_scores
                factory_id_scores_cache[mode_name] = mode_scores
            
            if verbose:
                print(f"  ✓ Pre-computed ID scores for {len(factory_id_scores_cache)} modes")
        else:
            if verbose:
                print(f"  Using cached ID scores for factory modes")
        
        # Pre-compute OOD scores for all modes
        ood_scores_all_modes = {}
        ood_iter = tqdm(ood_loader, desc=f'  Computing OOD scores (factory)', disable=not verbose)
        with torch.no_grad():
            for batch in ood_iter:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                # Compute all modes at once
                all_scores = factory_detector.predict_all(x)
                # Accumulate scores for each mode
                for mode_name, scores in all_scores.items():
                    if mode_name not in ood_scores_all_modes:
                        ood_scores_all_modes[mode_name] = []
                    ood_scores_all_modes[mode_name].append(scores.cpu())
        
        # Concatenate all batches for each mode
        for mode_name in ood_scores_all_modes:
            factory_ood_scores_cache[mode_name] = torch.cat(ood_scores_all_modes[mode_name]).squeeze()
        
        if verbose:
            print(f"  ✓ Pre-computed OOD scores for {len(factory_ood_scores_cache)} modes")
    
    for name, detector in detectors.items():
        # Skip factory detector itself (only evaluate wrappers)
        if isinstance(detector, SteinFactoryDetector):
            if verbose:
                print(f"\nSkipping {name} (factory detector - evaluating individual mode wrappers instead)")
            continue
        
        if verbose:
            print(f"\nEvaluating {name}...")
        
        # Check if detector needs gradients
        detector_name = getattr(detector, '__class__', type(detector)).__name__.lower()
        needs_grads = 'mahalanobis' in detector_name or 'odin' in detector_name or 'gsc' in detector_name
        
        # Check if this is a factory wrapper
        is_factory_wrapper = hasattr(detector, 'factory') and isinstance(detector.factory, SteinFactoryDetector)
        wrapper_mode = None
        if is_factory_wrapper:
            wrapper_mode = getattr(detector, 'mode_name', None)
        
        # Get or compute ID scores
        if name in id_scores_cache:
            if verbose:
                print(f"  Using cached ID scores")
            id_scores = id_scores_cache[name].squeeze()  # Ensure 1D
        else:
            # Compute ID scores
            if verbose:
                print(f"  Computing ID scores...")
            if hasattr(detector, 'model') and hasattr(detector.model, 'eval'):
                detector.model.eval()
            id_scores_list = []
            id_iter = tqdm(id_loader, desc=f'  Computing ID scores ({name})', disable=not verbose)
            if needs_grads:
                for batch in id_iter:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                    else:
                        x = batch
                    x = x.to(device).requires_grad_(True)
                    scores = detector.predict(x)
                    id_scores_list.append(scores.detach().cpu())
            else:
                with torch.no_grad():
                    for batch in id_iter:
                        if isinstance(batch, (list, tuple)):
                            x = batch[0]
                        else:
                            x = batch
                        x = x.to(device)
                        scores = detector.predict(x)
                        id_scores_list.append(scores.cpu())
            id_scores = torch.cat(id_scores_list).squeeze()
            id_scores_cache[name] = id_scores  # Cache for reuse

        
        # Compute OOD scores
        # For factory wrappers, use pre-computed scores
        if is_factory_wrapper and wrapper_mode and wrapper_mode in factory_ood_scores_cache:
            if verbose:
                print(f"  Using pre-computed OOD scores from factory (mode: {wrapper_mode})")
            ood_scores = factory_ood_scores_cache[wrapper_mode]
        else:
            # Compute OOD scores normally
            # Note: This processes all OOD samples in batches and stores them in memory.
            # This is necessary to compute metrics (AUROC, FPR95, etc.) which require
            # comparing all ID vs OOD scores. For large OOD datasets, this may use significant memory.
            if verbose:
                print(f"  Computing OOD scores...")
            if hasattr(detector, 'model') and hasattr(detector.model, 'eval'):
                detector.model.eval()
            ood_scores_list = []
            ood_iter = tqdm(ood_loader, desc=f'  Computing OOD scores ({name})', disable=not verbose)
            if needs_grads:
                for batch in ood_iter:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                    else:
                        x = batch
                    x = x.to(device).requires_grad_(True)
                    scores = detector.predict(x)
                    ood_scores_list.append(scores.detach().cpu())
            else:
                with torch.no_grad():
                    for batch in ood_iter:
                        if isinstance(batch, (list, tuple)):
                            x = batch[0]
                        else:
                            x = batch
                        x = x.to(device)
                        scores = detector.predict(x)
                        ood_scores_list.append(scores.cpu())
            ood_scores = torch.cat(ood_scores_list).squeeze()
        # Ensure scores are 1D
        ood_scores = ood_scores.squeeze()
        
        # Verify alignment with pre-computed losses
        if compute_misclassified_metrics:
            assert len(id_scores) == len(id_losses), f"Score/loss length mismatch for ID: {len(id_scores)} vs {len(id_losses)}"
            assert len(ood_scores) == len(ood_losses), f"Score/loss length mismatch for OOD: {len(ood_scores)} vs {len(ood_losses)}"
        
        # Compute metrics with dataset-based OOD definition (default)
        metrics_dataset = compute_ood_metrics(
            id_scores, ood_scores, 
            detector_name=name, 
            ood_dataset_name=ood_dataset_name,
            device=device,
            ood_definition_mode='dataset',
            id_losses=id_losses,
            ood_losses=ood_losses,
        )

        # Debug diagnostic: for signed Stein variants, compare against baseline-centered absolute residual.
        # Hypothesis: current evaluation uses raw signed residuals (two-sided), but the intended detector
        # score might be |residual - baseline| (non-negative, upper-tail). We only compute this for
        # dataset-based mode to keep overhead low.
        baseline_abs_diag = None
        try:
            if name in {"stein_full", "stein_full_no_lap", "stein_per_dimension_sum", "stein_first_order_sum"}:
                b = getattr(detector, "baseline", None)
                if b is not None:
                    b_val = float(b.detach().cpu().item()) if hasattr(b, "detach") else float(b)
                    abs_id = (id_scores.float() - b_val).abs()
                    abs_ood = (ood_scores.float() - b_val).abs()
                    baseline_abs_diag = compute_ood_metrics(
                        abs_id, abs_ood,
                        detector_name=None,  # simple higher-is-OOD negation + rank-normalized metrics
                        ood_dataset_name=ood_dataset_name,
                        device=device,
                        ood_definition_mode='dataset',
                    )
                    # region agent log
                    _agent_log(
                        run_id="perf-investigation",
                        hypothesis_id="BASE",
                        location="src/evaluation/ood_benchmark.py:evaluate_all_detectors",
                        message="Stein baseline abs-centered diagnostic (dataset mode)",
                        data={
                            "code_version": _AGENT_CODE_VERSION,
                            "detector_name": name,
                            "baseline": b_val,
                            "auroc_main": float(metrics_dataset.get("AUROC")) if isinstance(metrics_dataset, dict) and metrics_dataset.get("AUROC") is not None else None,
                            "fpr95_main": float(metrics_dataset.get("FPR95")) if isinstance(metrics_dataset, dict) and metrics_dataset.get("FPR95") is not None else None,
                            "auroc_abs_centered": float(baseline_abs_diag.get("AUROC")) if isinstance(baseline_abs_diag, dict) and baseline_abs_diag.get("AUROC") is not None else None,
                            "fpr95_abs_centered": float(baseline_abs_diag.get("FPR95")) if isinstance(baseline_abs_diag, dict) and baseline_abs_diag.get("FPR95") is not None else None,
                        },
                    )
                    # endregion
        except Exception:
            baseline_abs_diag = None
        
        if compute_misclassified_metrics:
            # Compute metrics with misclassified-based OOD definition
            metrics_misclassified = compute_ood_metrics(
                id_scores, ood_scores,
                detector_name=name,
                ood_dataset_name=ood_dataset_name,
                device=device,
                ood_definition_mode='misclassified',
                id_predictions=id_predictions,
                ood_predictions=ood_predictions,
                id_ground_truth=id_ground_truth,
                ood_ground_truth=ood_ground_truth,
                id_losses=id_losses,
                ood_losses=ood_losses,
            )
            
            # Compute metrics with dataset AND misclassified-based OOD definition
            metrics_both = compute_ood_metrics(
                id_scores, ood_scores,
                detector_name=name,
                ood_dataset_name=ood_dataset_name,
                device=device,
                ood_definition_mode='dataset_and_misclassified',
                id_predictions=id_predictions,
                ood_predictions=ood_predictions,
                id_ground_truth=id_ground_truth,
                ood_ground_truth=ood_ground_truth,
                id_losses=id_losses,
                ood_losses=ood_losses,
            )
            
            # Return all three sets of metrics
            results[name] = {
                'dataset_based': metrics_dataset,
                'misclassified': metrics_misclassified,
                'dataset_and_misclassified': metrics_both,
            }
            
            if verbose:
                print(f"  Dataset-based - AUROC: {metrics_dataset.get('AUROC', 'N/A'):.4f}, FPR95: {metrics_dataset.get('FPR95', 'N/A'):.4f}")
                print(f"  Misclassified - AUROC: {metrics_misclassified.get('AUROC', 'N/A'):.4f}, FPR95: {metrics_misclassified.get('FPR95', 'N/A'):.4f}")
                print(f"  Dataset+Misclassified - AUROC: {metrics_both.get('AUROC', 'N/A'):.4f}, FPR95: {metrics_both.get('FPR95', 'N/A'):.4f}")
        else:
            # Return only dataset-based metrics (backward compatible)
            # Store as dict to allow adding scores
            results[name] = metrics_dataset.copy() if isinstance(metrics_dataset, dict) else metrics_dataset

        # Attach baseline-abs diagnostic metrics (if computed)
        if baseline_abs_diag is not None and isinstance(results.get(name), dict):
            results[name]["_diagnostics_baseline_abs_centered_dataset"] = baseline_abs_diag
            
            if verbose:
                print(f"  AUROC: {metrics_dataset.get('AUROC', 'N/A'):.4f}")
                print(f"  FPR95: {metrics_dataset.get('FPR95', 'N/A'):.4f}")
        
        # Store raw scores for this detector
        # Convert to numpy for easier serialization
        # Ensure results[name] is a dict
        if not isinstance(results[name], dict):
            results[name] = {'metrics': results[name]}
        
        # Get classification correctness if available
        id_correct = None
        ood_correct = None
        if compute_misclassified_metrics and id_predictions is not None and id_ground_truth is not None:
            id_correct = (id_predictions == id_ground_truth).cpu().numpy()
        if compute_misclassified_metrics and ood_predictions is not None and ood_ground_truth is not None:
            ood_correct = (ood_predictions == ood_ground_truth).cpu().numpy()
        
        results[name]['_scores'] = {
            'id_scores': id_scores.cpu().numpy(),
            'ood_scores': ood_scores.cpu().numpy(),
            'id_correct': id_correct,  # None if not available
            'ood_correct': ood_correct,  # None if not available
        }
    
    return results


def fit_all_detectors(
    detectors: Dict[str, Detector],
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    train_only_on_correct: bool = False,
    classifier_model: Optional[nn.Module] = None,
    cache_dir: Optional[str] = None,
    id_dataset: Optional[str] = None,
    model_path: Optional[str] = None,
    score_model_path: Optional[str] = None,
) -> Dict[str, Detector]:
    """
    Fit all detectors on training data.
    
    Args:
        detectors: Dictionary mapping detector names to detector instances
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data (only used by SteinDetector)
        device: Device to use (auto-detect if None)
        verbose: If True, show progress
        train_only_on_correct: If True, filter training data to only include correctly classified samples
        classifier_model: Classifier model required if train_only_on_correct=True
    
    Returns:
        Dictionary of fitted detectors (same as input, but detectors are fitted)
    """
    if device is None:
        device = get_device()
    
    # Filter training data if requested
    if train_only_on_correct:
        if classifier_model is None:
            # Try to extract classifier from first detector
            first_detector = next(iter(detectors.values()))
            if hasattr(first_detector, 'model'):
                classifier_model = first_detector.model
            else:
                raise ValueError(
                    "classifier_model is required for train_only_on_correct=True. "
                    "Either pass classifier_model or ensure detectors have a 'model' attribute."
                )
        
        if verbose:
            print(f"\nFiltering training data to only include correctly classified samples...")
        train_loader = _filter_correctly_classified_samples(
            train_loader, classifier_model, device=device, verbose=verbose
        )
    
    for name, detector in detectors.items():
        if verbose:
            print(f"\nFitting {name}...")
        
        # Check if detector needs fitting
        if hasattr(detector, 'fit'):
            try:
                # Different detectors have different fit() signatures
                if name == 'knn':
                    # KNN.fit(loader, device='cpu')
                    detector.fit(train_loader, device=str(device))
                elif name == 'mahalanobis':
                    # Mahalanobis.fit(data_loader, device=None)
                    detector.fit(train_loader, device=device)
                elif name.startswith('stein'):
                    # SteinDetector.fit(train_loader, val_loader=val_loader, ...)
                    # Pass cache info if available
                    # Note: Ensemble detectors don't have individual baselines, so skip caching for them
                    from src.detector.stein_ensemble import SteinEnsembleDetector
                    from src.detector import SteinFactoryDetector
                    is_ensemble = isinstance(detector, SteinEnsembleDetector)
                    is_factory = isinstance(detector, SteinFactoryDetector)
                    
                    # Skip caching for factory detector itself (individual mode wrappers will be cached)
                    if cache_dir is not None and id_dataset is not None and model_path is not None and not is_ensemble and not is_factory:
                        from .detector_cache import (
                            get_detector_cache_path, load_detector_cache, save_detector_cache,
                            get_detector_config_from_stein_detector, _get_dataset_hash,
                        )
                        from pathlib import Path
                        
                        # Get detector config
                        detector_config = get_detector_config_from_stein_detector(detector)
                        dataset_hash = _get_dataset_hash(train_loader)
                        
                        # Check cache
                        cache_path = get_detector_cache_path(
                            Path(cache_dir),
                            id_dataset,
                            model_path,
                            name,
                            detector_config,
                            score_model_path=score_model_path,
                            dataset_hash=dataset_hash,
                        )
                        cached_data = load_detector_cache(cache_path, device=device)
                        
                        if cached_data is not None:
                            # Load from cache
                            detector.baseline = cached_data.get('baseline')
                            detector.training_std = cached_data.get('training_std')
                            if verbose:
                                print(f"  Loaded baseline from cache: {detector.baseline.item():.6e}" if detector.baseline is not None else "  No baseline in cache")
                                print(f"  Loaded training_std from cache: {detector.training_std.item():.6e}" if detector.training_std is not None else "  No training_std in cache")
                            
                            # Still need to fit for score model training (if needed)
                            # But skip baseline computation if already cached
                            if detector.baseline is not None and detector.training_std is not None:
                                # Temporarily disable baseline computation
                                original_compute_baseline = detector.compute_baseline
                                detector.compute_baseline = False
                                detector.fit(train_loader, val_loader=val_loader)
                                detector.compute_baseline = original_compute_baseline
                            else:
                                # Fit normally (will compute baseline)
                                detector.fit(train_loader, val_loader=val_loader)
                                # Save to cache
                                if detector.baseline is not None:
                                    save_detector_cache(
                                        cache_path,
                                        detector.baseline,
                                        detector.training_std,
                                    )
                                    if verbose:
                                        print(f"  Saved baseline to cache")
                        else:
                            # No cache, fit normally
                            detector.fit(train_loader, val_loader=val_loader)
                            # Save to cache
                            if detector.baseline is not None:
                                save_detector_cache(
                                    cache_path,
                                    detector.baseline,
                                    detector.training_std,
                                )
                                if verbose:
                                    print(f"  Saved baseline to cache")
                    else:
                        # No cache info, fit normally (or ensemble detector)
                        detector.fit(train_loader, val_loader=val_loader)
                else:
                    # Other detectors (MSP, ODIN, Energy) don't need fit()
                    # But try generic fit() call if it exists
                    try:
                        detector.fit(train_loader)
                    except TypeError:
                        # Some detectors don't accept any arguments
                        detector.fit()
            except Exception as e:
                if verbose:
                    print(f"  Warning: {name} fit() failed: {e}")
                    import traceback
                    traceback.print_exc()
                # Some detectors might not need fit() or might fail
                # Continue anyway
                pass
        elif verbose:
            print(f"  {name} does not require fitting")
    
    return detectors


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    dataset_name: str,
    ood_dataset_name: str,
    additional_info: Optional[Dict[str, Any]] = None,
    save_scores: bool = True,
) -> None:
    """
    Save evaluation results to files.
    
    Args:
        results: Dictionary mapping detector names to metric dictionaries
                Can be either: {detector: {metric: value}} or 
                {detector: {'dataset_based': {...}, 'misclassified': {...}, 'dataset_and_misclassified': {...}}}
        output_dir: Directory to save results
        dataset_name: Name of in-distribution dataset
        ood_dataset_name: Name of out-of-distribution dataset
        additional_info: Optional dictionary with additional information to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize dataset names for file paths (replace colons and other problematic characters)
    def sanitize_filename(name: str) -> str:
        return name.replace(':', '_').replace('/', '_').replace('\\', '_')
    
    dataset_name_safe = sanitize_filename(dataset_name)
    ood_dataset_name_safe = sanitize_filename(ood_dataset_name)
    
    # Prepare results for JSON/CSV (exclude scores, they're saved separately as NPZ)
    results_for_json = {}
    for detector_name, detector_results in results.items():
        if isinstance(detector_results, dict) and '_scores' in detector_results:
            # Create a copy without scores
            results_for_json[detector_name] = {k: v for k, v in detector_results.items() if k != '_scores'}
        else:
            results_for_json[detector_name] = detector_results
    
    # Save as JSON
    output_data = {
        'dataset': dataset_name,
        'ood_dataset': ood_dataset_name,
        'results': results_for_json,
    }
    if additional_info:
        output_data['additional_info'] = additional_info
    
    json_path = os.path.join(output_dir, f'{dataset_name_safe}_vs_{ood_dataset_name_safe}_results.json')
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)  # default=str for tensor serialization
    
    # Save scores separately as numpy arrays
    if save_scores:
        scores_data = {}
        for detector_name, detector_results in results.items():
            if isinstance(detector_results, dict) and '_scores' in detector_results:
                scores_data[detector_name] = detector_results['_scores']
        
        if scores_data:
            scores_path = os.path.join(output_dir, f'{dataset_name_safe}_vs_{ood_dataset_name_safe}_scores.npz')
            # Save in flat format: detector_name_id and detector_name_ood
            scores_flat = {}
            for detector_name, scores_dict in scores_data.items():
                scores_flat[f'{detector_name}_id'] = scores_dict['id_scores']
                scores_flat[f'{detector_name}_ood'] = scores_dict['ood_scores']
                if scores_dict.get('id_correct') is not None:
                    scores_flat[f'{detector_name}_id_correct'] = scores_dict['id_correct']
                if scores_dict.get('ood_correct') is not None:
                    scores_flat[f'{detector_name}_ood_correct'] = scores_dict['ood_correct']
            np.savez_compressed(scores_path, **scores_flat)

            # Also save a single consolidated CSV with RAW scores for all detectors.
            # This is convenient for notebook-level post-processing without relying on per-detector files.
            try:
                # Build base rows once (ID then OOD), aligned by index.
                # Note: correctness flags may be None depending on dataset label availability.
                first_det = next(iter(scores_data.values()))
                n_id = int(len(first_det['id_scores']))
                n_ood = int(len(first_det['ood_scores']))

                datapoint_id = [f"id_{i}" for i in range(n_id)] + [f"ood_{i}" for i in range(n_ood)]
                is_ood = [0] * n_id + [1] * n_ood

                # Prefer classifier correctness flags if available for any detector; otherwise leave empty.
                id_correct_any = None
                ood_correct_any = None
                for _det_name, _scores in scores_data.items():
                    if _scores.get('id_correct') is not None and id_correct_any is None:
                        id_correct_any = _scores.get('id_correct')
                    if _scores.get('ood_correct') is not None and ood_correct_any is None:
                        ood_correct_any = _scores.get('ood_correct')
                is_correct = None
                if id_correct_any is not None and ood_correct_any is not None:
                    is_correct = [int(x) for x in id_correct_any] + [int(x) for x in ood_correct_any]

                # Add raw score columns for each detector.
                rows = {
                    "datapoint_id": datapoint_id,
                    "is_ood": is_ood,
                }
                if is_correct is not None:
                    rows["is_classified_correctly"] = is_correct

                for detector_name, scores_dict in scores_data.items():
                    rows[f"{detector_name}_score"] = (
                        [float(x) for x in scores_dict['id_scores']] + [float(x) for x in scores_dict['ood_scores']]
                    )

                combined_scores_csv = os.path.join(
                    output_dir,
                    f'{dataset_name_safe}_vs_{ood_dataset_name_safe}_scores_long.csv'
                )
                import pandas as _pd
                _pd.DataFrame(rows).to_csv(combined_scores_csv, index=False)
            except Exception:
                # Best-effort only; per-detector detailed CSVs / NPZ remain the source of truth.
                pass
            
            # Also save detailed CSV files per detector with all information
            for detector_name, scores_dict in scores_data.items():
                csv_detailed_path = os.path.join(output_dir, f'{dataset_name_safe}_vs_{ood_dataset_name_safe}_{detector_name}_detailed.csv')
                with open(csv_detailed_path, 'w') as f:
                    # Header
                    f.write('datapoint_id,score,is_ood,is_classified_correctly,score_space,stein_logp_two_sided,stein_logp_upper,stein_oodness_two_sided,stein_oodness_upper\n')

                    det_key = detector_name.lower()
                    is_stein = det_key.startswith('stein')

                    # Precompute Stein transforms once (so CSV is fully reproducible for both tail modes).
                    id_logp_two = None
                    ood_logp_two = None
                    id_logp_up = None
                    ood_logp_up = None
                    if is_stein:
                        try:
                            id_scores_t = torch.as_tensor(scores_dict['id_scores'], dtype=torch.float32)
                            ood_scores_t = torch.as_tensor(scores_dict['ood_scores'], dtype=torch.float32)
                            id_logp_two_t, ood_logp_two_t = _compute_two_sided_test_scores(id_scores_t, ood_scores_t)
                            id_logp_up_t, ood_logp_up_t = _compute_upper_tail_test_scores(id_scores_t, ood_scores_t)
                            id_logp_two = id_logp_two_t.detach().cpu().numpy()
                            ood_logp_two = ood_logp_two_t.detach().cpu().numpy()
                            id_logp_up = id_logp_up_t.detach().cpu().numpy()
                            ood_logp_up = ood_logp_up_t.detach().cpu().numpy()
                        except Exception:
                            id_logp_two = ood_logp_two = id_logp_up = ood_logp_up = None
                    
                    # ID samples
                    id_scores = scores_dict['id_scores']
                    id_correct = scores_dict.get('id_correct')
                    for i, score in enumerate(id_scores):
                        datapoint_id = f'id_{i}'
                        is_ood = 0
                        is_correct = int(id_correct[i]) if id_correct is not None else 'N/A'
                        score_space = 'raw_stein' if is_stein else 'raw'
                        logp_two = f"{float(id_logp_two[i]):.8f}" if id_logp_two is not None else ''
                        logp_up = f"{float(id_logp_up[i]):.8f}" if id_logp_up is not None else ''
                        oodness_two = f"{float(-id_logp_two[i]):.8f}" if id_logp_two is not None else ''
                        oodness_up = f"{float(-id_logp_up[i]):.8f}" if id_logp_up is not None else ''
                        f.write(f'{datapoint_id},{float(score):.8f},{is_ood},{is_correct},{score_space},{logp_two},{logp_up},{oodness_two},{oodness_up}\n')
                    
                    # OOD samples
                    ood_scores = scores_dict['ood_scores']
                    ood_correct = scores_dict.get('ood_correct')
                    for i, score in enumerate(ood_scores):
                        datapoint_id = f'ood_{i}'
                        is_ood = 1
                        is_correct = int(ood_correct[i]) if ood_correct is not None else 'N/A'
                        score_space = 'raw_stein' if is_stein else 'raw'
                        logp_two = f"{float(ood_logp_two[i]):.8f}" if ood_logp_two is not None else ''
                        logp_up = f"{float(ood_logp_up[i]):.8f}" if ood_logp_up is not None else ''
                        oodness_two = f"{float(-ood_logp_two[i]):.8f}" if ood_logp_two is not None else ''
                        oodness_up = f"{float(-ood_logp_up[i]):.8f}" if ood_logp_up is not None else ''
                        f.write(f'{datapoint_id},{float(score):.8f},{is_ood},{is_correct},{score_space},{logp_two},{logp_up},{oodness_two},{oodness_up}\n')
    
    # Save as CSV table(s)
    # Check if results have nested structure (both metric sets)
    has_nested = results_for_json and isinstance(next(iter(results_for_json.values())), dict) and 'dataset_based' in next(iter(results_for_json.values()))
    
    if has_nested:
        # Save three CSV files: one for each metric set
        for metric_set_name in ['dataset_based', 'misclassified', 'dataset_and_misclassified']:
            csv_path = os.path.join(output_dir, f'{dataset_name_safe}_vs_{ood_dataset_name_safe}_results_{metric_set_name}.csv')
            with open(csv_path, 'w') as f:
                # Header
                first_detector = next(iter(results_for_json.values()))
                if metric_set_name in first_detector:
                    metric_names = list(first_detector[metric_set_name].keys())
                    f.write('Detector,' + ','.join(metric_names) + '\n')
                    
                    # Data rows
                    for detector_name, detector_results in results_for_json.items():
                        if metric_set_name in detector_results:
                            values = [str(detector_results[metric_set_name].get(m, 'N/A')) for m in metric_names]
                            f.write(f'{detector_name},' + ','.join(values) + '\n')
    else:
        # Save single CSV file (backward compatible)
        csv_path = os.path.join(output_dir, f'{dataset_name_safe}_vs_{ood_dataset_name_safe}_results.csv')
        with open(csv_path, 'w') as f:
            # Header
            if results_for_json:
                first_detector = next(iter(results_for_json.values()))
                metric_names = list(first_detector.keys())
                f.write('Detector,' + ','.join(metric_names) + '\n')
            
            # Data rows
            for detector_name, metrics in results_for_json.items():
                values = [str(metrics.get(metric, 'N/A')) for metric in metric_names]
                f.write(f'{detector_name},' + ','.join(values) + '\n')
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    if has_nested:
        print(f"  CSV (Dataset-Based): {os.path.join(output_dir, f'{dataset_name_safe}_vs_{ood_dataset_name_safe}_results_dataset_based.csv')}")
        print(f"  CSV (Misclassified): {os.path.join(output_dir, f'{dataset_name_safe}_vs_{ood_dataset_name_safe}_results_misclassified.csv')}")
        print(f"  CSV (Dataset+Misclassified): {os.path.join(output_dir, f'{dataset_name_safe}_vs_{ood_dataset_name_safe}_results_dataset_and_misclassified.csv')}")
    else:
        print(f"  CSV: {csv_path}")
    if save_scores and scores_data:
        print(f"  Scores (NPZ): {scores_path}")
        print(f"    Saved scores for {len(scores_data)} detector(s): {', '.join(scores_data.keys())}")
        for detector_name, scores_dict in scores_data.items():
            id_count = scores_dict['id_scores'].shape[0]
            ood_count = scores_dict['ood_scores'].shape[0]
            has_correctness = scores_dict.get('id_correct') is not None
            print(f"      {detector_name}: ID={id_count} samples, OOD={ood_count} samples" + 
                  (f" (with correctness info)" if has_correctness else ""))
            # Print detailed CSV path
            csv_detailed_path = os.path.join(output_dir, f'{dataset_name_safe}_vs_{ood_dataset_name_safe}_{detector_name}_detailed.csv')
            print(f"        Detailed CSV: {csv_detailed_path}")


def create_results_table(
    results: Dict[str, Dict[str, float]],
    metric: str = 'AUROC',
) -> str:
    """
    Create a formatted table string from results.
    
    Args:
        results: Dictionary mapping detector names to metric dictionaries
        metric: Metric name to display
    
    Returns:
        Formatted table string
    """
    lines = []
    lines.append(f"{'Detector':<20} {metric:>10}")
    lines.append("-" * 32)
    
    for detector_name, metrics in results.items():
        value = metrics.get(metric, None)
        if value is not None:
            lines.append(f"{detector_name:<20} {value:>10.4f}")
        else:
            lines.append(f"{detector_name:<20} {'N/A':>10}")
    
    return "\n".join(lines)


def print_results_summary(
    results: Dict[str, Dict[str, float]],
    dataset_name: str,
    ood_dataset_name: str,
) -> None:
    """
    Print summary of evaluation results including accuracies.
    
    Args:
        results: Dictionary mapping detector names to metric dictionaries
        dataset_name: Name of in-distribution dataset
        ood_dataset_name: Name of out-of-distribution dataset
    """
    print("\n" + "=" * 80)
    print(f"Results Summary: {dataset_name.upper()} vs {ood_dataset_name.upper()}")
    print("=" * 80)
    
    # Check if results have nested structure
    has_nested = results and isinstance(next(iter(results.values())), dict) and 'dataset_based' in next(iter(results.values()))
    
    # Print accuracy metrics if available
    if results:
        first_detector_results = next(iter(results.values()))
        # Handle nested structure
        if has_nested and 'dataset_based' in first_detector_results:
            first_detector_results = first_detector_results['dataset_based']
        
        if 'id_top1_accuracy' in first_detector_results or 'id_top5_accuracy' in first_detector_results:
            print("\nClassifier Accuracy:")
            print("-" * 80)
            id_top1 = first_detector_results.get('id_top1_accuracy')
            id_top5 = first_detector_results.get('id_top5_accuracy')
            ood_top1 = first_detector_results.get('ood_top1_accuracy')
            ood_top5 = first_detector_results.get('ood_top5_accuracy')
            
            if id_top1 is not None:
                print(f"  ID Top-1 Accuracy:  {id_top1:.4f}")
            if id_top5 is not None:
                print(f"  ID Top-5 Accuracy:  {id_top5:.4f}")
            if ood_top1 is not None:
                print(f"  OOD Top-1 Accuracy: {ood_top1:.4f}")
            if ood_top5 is not None:
                print(f"  OOD Top-5 Accuracy: {ood_top5:.4f}")
            if ood_top1 is None and 'ood_top1_confidence' in first_detector_results:
                print(f"  OOD Top-1 Confidence: {first_detector_results.get('ood_top1_confidence', 'N/A'):.4f}")
                print(f"  OOD Entropy:         {first_detector_results.get('ood_entropy', 'N/A'):.4f}")
    
    # Print OOD detection metrics
    if has_nested:
        # Print both metric sets
        print("\nOOD Detection Metrics - Dataset-Based:")
        print("-" * 80)
        dataset_results = {name: res['dataset_based'] for name, res in results.items() if 'dataset_based' in res}
        print("\nAUROC (Area Under ROC Curve):")
        print(create_results_table(dataset_results, metric='AUROC'))
        print("\nFPR95 (False Positive Rate at 95% TPR):")
        print(create_results_table(dataset_results, metric='FPR95'))
        
        # Display correlation metrics if available
        if dataset_results and any('spearman_correlation_all' in res for res in dataset_results.values()):
            print("\nSpearman Correlation with Classifier Loss:")
            print("  Combined (all samples):")
            print(create_results_table(dataset_results, metric='spearman_correlation_all'))
            print("  ID samples only:")
            print(create_results_table(dataset_results, metric='spearman_correlation_id'))
            print("  OOD samples only:")
            print(create_results_table(dataset_results, metric='spearman_correlation_ood'))
        
        print("\nOOD Detection Metrics - Misclassified-Based:")
        print("-" * 80)
        misclassified_results = {name: res['misclassified'] for name, res in results.items() if 'misclassified' in res}
        print("\nAUROC (Area Under ROC Curve):")
        print(create_results_table(misclassified_results, metric='AUROC'))
        print("\nFPR95 (False Positive Rate at 95% TPR):")
        print(create_results_table(misclassified_results, metric='FPR95'))
        
        # Display correlation metrics if available
        if misclassified_results and any('spearman_correlation_all' in res for res in misclassified_results.values()):
            print("\nSpearman Correlation with Classifier Loss:")
            print("  Combined (all samples):")
            print(create_results_table(misclassified_results, metric='spearman_correlation_all'))
            print("  ID samples only:")
            print(create_results_table(misclassified_results, metric='spearman_correlation_id'))
            print("  OOD samples only:")
            print(create_results_table(misclassified_results, metric='spearman_correlation_ood'))
        
        print("\nOOD Detection Metrics - Dataset AND Misclassified:")
        print("-" * 80)
        both_results = {name: res['dataset_and_misclassified'] for name, res in results.items() if 'dataset_and_misclassified' in res}
        print("\nAUROC (Area Under ROC Curve):")
        print(create_results_table(both_results, metric='AUROC'))
        print("\nFPR95 (False Positive Rate at 95% TPR):")
        print(create_results_table(both_results, metric='FPR95'))
        
        # Display correlation metrics if available
        if both_results and any('spearman_correlation_all' in res for res in both_results.values()):
            print("\nSpearman Correlation with Classifier Loss:")
            print("  Combined (all samples):")
            print(create_results_table(both_results, metric='spearman_correlation_all'))
            print("  ID samples only:")
            print(create_results_table(both_results, metric='spearman_correlation_id'))
            print("  OOD samples only:")
            print(create_results_table(both_results, metric='spearman_correlation_ood'))
        
        # Best performers for each metric set
        if dataset_results:
            auroc_scores = {name: metrics.get('AUROC', 0.0) for name, metrics in dataset_results.items() if metrics.get('AUROC') is not None}
            if auroc_scores:
                best_auroc = max(auroc_scores.items(), key=lambda x: x[1])
                print(f"\nBest AUROC (Dataset-Based): {best_auroc[0]} ({best_auroc[1]:.4f})")
        
        if misclassified_results:
            auroc_scores = {name: metrics.get('AUROC', 0.0) for name, metrics in misclassified_results.items() if metrics.get('AUROC') is not None}
            if auroc_scores:
                best_auroc = max(auroc_scores.items(), key=lambda x: x[1])
                print(f"Best AUROC (Misclassified): {best_auroc[0]} ({best_auroc[1]:.4f})")
        
        if both_results:
            auroc_scores = {name: metrics.get('AUROC', 0.0) for name, metrics in both_results.items() if metrics.get('AUROC') is not None}
            if auroc_scores:
                best_auroc = max(auroc_scores.items(), key=lambda x: x[1])
                print(f"Best AUROC (Dataset+Misclassified): {best_auroc[0]} ({best_auroc[1]:.4f})")
    else:
        # Print single metric set (backward compatible)
        print("\nOOD Detection Metrics:")
        print("-" * 80)
        print("\nAUROC (Area Under ROC Curve):")
        print(create_results_table(results, metric='AUROC'))
        print("\nFPR95 (False Positive Rate at 95% TPR):")
        print(create_results_table(results, metric='FPR95'))
        
        # Best performers
        if results:
            auroc_scores = {
                name: metrics.get('AUROC', 0.0)
                for name, metrics in results.items()
                if metrics.get('AUROC') is not None
            }
            if auroc_scores:
                best_auroc = max(auroc_scores.items(), key=lambda x: x[1])
                print(f"\nBest AUROC: {best_auroc[0]} ({best_auroc[1]:.4f})")
            
            fpr95_scores = {
                name: metrics.get('FPR95', float('inf'))
                for name, metrics in results.items()
                if metrics.get('FPR95') is not None
            }
            if fpr95_scores:
                best_fpr95 = min(fpr95_scores.items(), key=lambda x: x[1])
                print(f"Best FPR95: {best_fpr95[0]} ({best_fpr95[1]:.4f})")

