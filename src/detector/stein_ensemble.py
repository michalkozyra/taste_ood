"""
Ensemble Stein Detector.

Combines multiple Stein detectors by normalizing their scores by training std
and summing them.
"""

from typing import List, Optional, TypeVar
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pytorch_ood.api import Detector

from .stein import SteinDetector

Self = TypeVar("Self")


class SteinEnsembleDetector(Detector):
    """
    Ensemble of Stein detectors.
    
    Combines multiple SteinDetector instances by:
    1. Getting raw scores from each detector (already baseline-corrected)
    2. Normalizing each by its training std
    3. Summing the normalized scores
    
    This allows combining detectors with different scales.
    """
    
    def __init__(
        self,
        detectors: List[SteinDetector],
        name: Optional[str] = None,
    ):
        """
        Initialize ensemble detector.
        
        Args:
            detectors: List of SteinDetector instances to ensemble
            name: Optional name for the ensemble (for logging)
        """
        super().__init__()
        
        if len(detectors) < 2:
            raise ValueError("Ensemble requires at least 2 detectors")
        
        self.detectors = detectors
        self.name = name or f"stein_ensemble_{len(detectors)}"
        self.device = detectors[0].device
        
        # Verify all detectors are on the same device
        for i, det in enumerate(detectors):
            if det.device != self.device:
                raise ValueError(f"All detectors must be on the same device. Detector {i} is on {det.device}, expected {self.device}")
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        score_train_loader: Optional[DataLoader] = None,
    ) -> Self:
        """
        Fit all detectors in the ensemble.
        
        If a detector is already fitted (has baseline and training_std), it will be skipped.
        This allows the ensemble to reuse detectors that were fitted separately.
        
        Args:
            train_loader: Training data for all detectors
            val_loader: Validation data (optional)
            score_train_loader: Training data for score models (optional)
        
        Returns:
            self (for method chaining)
        """
        print(f"\nFitting ensemble detector: {self.name}")
        print(f"  Ensemble contains {len(self.detectors)} detectors")
        
        for i, detector in enumerate(self.detectors):
            # Get detector type name for logging (handle wrappers that might not have stein_operator_type)
            detector_type = getattr(detector, 'stein_operator_type', getattr(detector, 'mode_name', 'unknown'))
            
            # Check if detector is already fitted
            # Note: For factory wrappers, baselines are lazy-loaded, so we need to access them
            # to trigger the lazy loading, but this might return None if factory not fitted yet
            baseline = detector.baseline if hasattr(detector, 'baseline') else None
            training_std = detector.training_std if hasattr(detector, 'training_std') else None
            
            if baseline is not None and training_std is not None:
                print(f"\n  Detector {i+1}/{len(self.detectors)} ({detector_type}) already fitted, skipping")
                print(f"    Baseline: {baseline.item():.6e}")
                print(f"    Training std: {training_std.item():.6e}")
            else:
                print(f"\n  Fitting detector {i+1}/{len(self.detectors)}: {detector_type}")
                detector.fit(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    score_train_loader=score_train_loader,
                )
                
                # Re-check baselines after fitting (for lazy-loaded properties)
                baseline = detector.baseline if hasattr(detector, 'baseline') else None
                training_std = detector.training_std if hasattr(detector, 'training_std') else None
                
                # Verify that training std was computed
                if training_std is None:
                    raise RuntimeError(
                        f"Detector {i+1} ({detector_type}) did not compute training_std. "
                        "Ensure compute_baseline=True and fit() was called."
                    )
                
                print(f"    Baseline: {baseline.item():.6e}")
                print(f"    Training std: {training_std.item():.6e}")
        
        return self
    
    def predict(self, x: Tensor) -> Tensor:
        """
        Compute ensemble scores.
        
        For each detector:
        1. Get raw residual scores from detector.predict() (raw residuals, not baseline-corrected)
        2. Apply mean removal: (residual - baseline)
        3. Normalize by training std: (residual - baseline) / training_std
        4. Sum normalized scores
        
        Args:
            x: Input tensor of shape (N, ...)
        
        Returns:
            scores: Ensemble scores of shape (N,)
                  Higher scores = more likely OOD
        """
        if any(det.baseline is None or det.training_std is None for det in self.detectors):
            raise RuntimeError(
                "All detectors must be fitted before prediction. Call fit() first."
            )
        
        # Get scores from each detector
        all_scores = []
        
        for detector in self.detectors:
            # Get raw residual scores (not baseline-corrected)
            raw_residuals = detector.predict(x)  # (N,) - raw residuals
            
            # Apply mean removal (baseline correction)
            residuals_centered = raw_residuals - detector.baseline  # (N,)
            
            # Normalize by training std
            normalized_scores = residuals_centered / (detector.training_std + 1e-8)  # Add small epsilon for numerical stability
            
            all_scores.append(normalized_scores)
        
        # Sum normalized scores
        ensemble_scores = sum(all_scores)  # (N,)
        
        return ensemble_scores
    
    def fit_features(self, x: Tensor, y: Tensor) -> Self:
        """
        Not applicable for Stein ensemble (requires full forward pass).
        """
        raise NotImplementedError(
            "Stein ensemble detector requires full model forward pass, "
            "use fit() with DataLoader instead"
        )
    
    def predict_features(self, x: Tensor) -> Tensor:
        """
        Not applicable for Stein ensemble (requires full forward pass).
        """
        raise NotImplementedError(
            "Stein ensemble detector requires full model forward pass, "
            "use predict() with input tensors instead"
        )
