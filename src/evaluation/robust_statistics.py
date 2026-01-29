"""
Robust statistics utilities for OOD detection.
Provides robust alternatives to mean/std for handling outliers.
"""

import torch
import numpy as np
from typing import Tuple


def median_absolute_deviation(x: torch.Tensor, dim: int = None, keepdim: bool = False) -> torch.Tensor:
    """
    Compute Median Absolute Deviation (MAD).
    
    MAD = median(|x - median(x)|)
    
    This is a robust measure of scale, similar to standard deviation but
    less sensitive to outliers.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute (None = all elements)
        keepdim: Whether to keep the dimension
    
    Returns:
        MAD value(s)
    """
    if dim is None:
        # Compute over all elements
        median = x.median()
        mad = (x - median).abs().median()
        return mad
    else:
        median = x.median(dim=dim, keepdim=True)[0]
        mad = (x - median).abs().median(dim=dim, keepdim=keepdim)[0]
        return mad


def robust_baseline_and_threshold(
    residuals: torch.Tensor,
    method: str = 'median_mad',
    multiplier: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute robust baseline and threshold for OOD detection.
    
    Provides alternatives to mean + 2*std that are less sensitive to outliers:
    - 'median_mad': median + multiplier * MAD (recommended)
    - 'median_iqr': median + multiplier * IQR
    - 'trimmed_mean': trimmed mean + multiplier * trimmed std
    
    Args:
        residuals: Stein residuals (N,)
        method: Method to use ('median_mad', 'median_iqr', 'trimmed_mean')
        multiplier: Multiplier for the robust scale measure (default 2.0)
    
    Returns:
        baseline: Robust baseline (scalar tensor)
        threshold: Robust threshold (scalar tensor)
    """
    residuals_flat = residuals.flatten()
    
    if method == 'median_mad':
        # Median + k * MAD
        # This is the most robust option, similar to z-score but using median/MAD
        baseline = residuals_flat.median()
        mad = median_absolute_deviation(residuals_flat)
        threshold = baseline + multiplier * mad
        
    elif method == 'median_iqr':
        # Median + k * IQR (Interquartile Range)
        # IQR = Q3 - Q1, where Q3 is 75th percentile and Q1 is 25th percentile
        baseline = residuals_flat.median()
        q1 = residuals_flat.quantile(0.25)
        q3 = residuals_flat.quantile(0.75)
        iqr = q3 - q1
        threshold = baseline + multiplier * iqr
        
    elif method == 'trimmed_mean':
        # Trimmed mean + k * trimmed std
        # Remove top and bottom 5% of values, then compute mean/std
        trim_percent = 0.05
        sorted_residuals = torch.sort(residuals_flat)[0]
        n = len(sorted_residuals)
        trim_n = int(n * trim_percent)
        trimmed = sorted_residuals[trim_n:-trim_n] if trim_n > 0 else sorted_residuals
        baseline = trimmed.mean()
        threshold = baseline + multiplier * trimmed.std()
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'median_mad', 'median_iqr', 'trimmed_mean'")
    
    return baseline, threshold


def robust_percentile_based_scoring(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    use_robust_percentiles: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute OOD scores using robust percentile-based approach.
    
    This is similar to the two-sided test but uses robust percentiles
    (based on median/MAD) to handle outliers better.
    
    Args:
        id_scores: ID scores (N_id,)
        ood_scores: OOD scores (N_ood,)
        use_robust_percentiles: If True, use robust percentile computation
    
    Returns:
        id_scores_transformed: Transformed ID scores (higher = more ID)
        ood_scores_transformed: Transformed OOD scores (higher = more ID)
    """
    if use_robust_percentiles:
        # Use robust normalization: (score - median) / MAD
        id_median = id_scores.median()
        id_mad = median_absolute_deviation(id_scores)
        
        # Normalize both ID and OOD scores
        id_normalized = (id_scores - id_median) / (id_mad + 1e-8)
        ood_normalized = (ood_scores - id_median) / (id_mad + 1e-8)
        
        # Convert to percentiles using standard normal CDF approximation
        # For robust z-scores, we can use erf approximation
        # But simpler: use the fact that ~68% of data is within 1 MAD of median
        # So we can map MAD units to approximate percentiles
        
        # For two-sided test: p-value = 2 * min(percentile, 1 - percentile)
        # We'll use the absolute value of normalized scores as a proxy
        id_abs_norm = id_normalized.abs()
        ood_abs_norm = ood_normalized.abs()
        
        # Convert to scores where higher = more ID (lower absolute deviation = more ID)
        # Use negative of absolute normalized score
        id_scores_transformed = -id_abs_norm
        ood_scores_transformed = -ood_abs_norm
        
    else:
        # Use standard percentile-based approach (existing method)
        from .ood_benchmark import _compute_two_sided_test_scores
        id_scores_transformed, ood_scores_transformed = _compute_two_sided_test_scores(
            id_scores, ood_scores
        )
    
    return id_scores_transformed, ood_scores_transformed
