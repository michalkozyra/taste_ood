"""
Baseline OOD detectors from PyTorch-OOD for comparison with SteinDetector.

This module provides utilities to instantiate and configure standard OOD detection
methods: MSP, ODIN, Energy, Mahalanobis, and Deep Nearest Neighbours (KNN).
"""

from typing import Optional, Dict, Any, List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_ood.detector import (
    MaxSoftmax,
    ODIN,
    EnergyBased,
    Mahalanobis,
    KNN,
    ReAct,
)
from pytorch_ood.api import Detector

from ..utils import get_device


def extract_features_before_fc(model: nn.Module, x: Tensor) -> Tensor:
    """
    Extract features from a model before the final fully connected layer.
    
    This function handles common architectures:
    - ResNet: extracts features after avgpool, before fc
    - Custom models: attempts to extract from 'features' attribute or before 'classifier'
    
    Args:
        model: The neural network model
        x: Input tensor
    
    Returns:
        Feature tensor (flattened)
    """
    model.eval()
    
    # Check if it's a ResNet (torchvision models)
    if hasattr(model, 'layer4') and hasattr(model, 'avgpool') and hasattr(model, 'fc'):
        # ResNet architecture: extract features before fc layer
        with torch.set_grad_enabled(x.requires_grad):
            # Preserve gradient requirement for ODIN preprocessing (Mahalanobis)
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)  # Flatten to (B, features)
        return x
    elif hasattr(model, 'features') and hasattr(model, 'classifier'):
        # Custom architecture with separate features and classifier (e.g., ClassifierNet)
        x = model.features(x)
        x = torch.flatten(x, 1)
        return x
    else:
        # Fallback: try to extract from intermediate layer if possible
        # For now, return logits as fallback (but this is suboptimal)
        # In practice, models should be structured to allow feature extraction
        return model(x)


def create_msp_detector(
    model: nn.Module,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> MaxSoftmax:
    """
    Create Maximum Softmax Probability (MSP) detector.
    
    Args:
        model: Trained classifier model
        temperature: Temperature scaling parameter (default: 1.0)
        device: Device to use (auto-detect if None)
    
    Returns:
        Configured MaxSoftmax detector
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    return MaxSoftmax(model=model, t=temperature)


def create_odin_detector(
    model: nn.Module,
    temperature: float = 1000.0,
    eps: float = 0.05,
    norm_std: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
) -> ODIN:
    """
    Create ODIN (Out-of-Distribution Detector using Input Preprocessing) detector.
    
    Args:
        model: Trained classifier model
        temperature: Temperature scaling parameter (default: 1000.0)
        eps: Perturbation magnitude for input preprocessing (default: 0.05)
        norm_std: Standard deviations for normalization (optional)
        device: Device to use (auto-detect if None)
    
    Returns:
        Configured ODIN detector
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    return ODIN(
        model=model,
        temperature=temperature,
        eps=eps,
        norm_std=norm_std,
    )


def create_energy_detector(
    model: nn.Module,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> EnergyBased:
    """
    Create Energy-based detector.
    
    Args:
        model: Trained classifier model
        temperature: Temperature scaling parameter (default: 1.0)
        device: Device to use (auto-detect if None)
    
    Returns:
        Configured EnergyBased detector
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    return EnergyBased(model=model, t=temperature)


def create_mahalanobis_detector(
    model: nn.Module,
    eps: float = 0.002,
    norm_std: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
) -> Mahalanobis:
    """
    Create Mahalanobis distance detector.
    
    Args:
        model: Trained classifier model (or feature extractor)
        eps: Perturbation magnitude for input preprocessing (default: 0.002)
        norm_std: Standard deviations for normalization (optional)
        device: Device to use (auto-detect if None)
    
    Returns:
        Configured Mahalanobis detector
    
    Note:
        Mahalanobis requires a feature extractor. If model is a full classifier,
        you may need to extract features from an intermediate layer.
        
        Features are normalized to unit length (L2 normalization) before computing
        Mahalanobis distances.
        
        IMPORTANT: Mahalanobis uses ODIN preprocessing which requires gradients.
        The feature extractor must NOT use torch.no_grad() to allow gradients
        to flow through the model for input preprocessing.
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()
    
    # Ensure model can compute gradients (even though parameters don't need gradients)
    # This is necessary for ODIN preprocessing which computes gradients w.r.t. input
    for param in model.parameters():
        param.requires_grad = False  # Don't update parameters
    # But we still need the computation graph for input gradients
    
    # Mahalanobis expects a callable that returns features
    # Use intermediate features (before FC layer) for better distance-based separation
    # NOTE: We cannot use torch.no_grad() here because Mahalanobis uses ODIN preprocessing
    # which requires gradients w.r.t. input. The model parameters don't need gradients,
    # but we need to allow gradients to flow through the model for input preprocessing.
    def feature_extractor(x: Tensor) -> Tensor:
        model.eval()
        # Don't use torch.no_grad() - Mahalanobis needs gradients for ODIN preprocessing
        # Model parameters don't need gradients (they're frozen), but we need gradients w.r.t. input
        # Extract features before final FC layer for better distance-based separation
        features = extract_features_before_fc(model, x)
        # Normalize to unit length (L2 normalization)
        features_norm = torch.norm(features, p=2, dim=1, keepdim=True)
        # Avoid division by zero
        features_norm = torch.clamp(features_norm, min=1e-8)
        features = features / features_norm
        return features
    
    return Mahalanobis(
        model=feature_extractor,
        eps=eps,
        norm_std=norm_std,
    )


def create_knn_detector(
    model: nn.Module,
    n_neighbors: int = 1,
    device: Optional[torch.device] = None,
    **knn_kwargs: Any,
) -> KNN:
    """
    Create Deep Nearest Neighbours (KNN) detector.
    
    Args:
        model: Trained classifier model (or feature extractor)
        n_neighbors: Number of nearest neighbors (default: 1)
                     Note: KNN detector hardcodes n_neighbors=1, so this parameter
                     is currently ignored. To use different values, you would need
                     to modify the KNN detector implementation.
        device: Device to use (auto-detect if None)
        **knn_kwargs: Additional arguments for KNN detector.
                     Note: n_neighbors and n_jobs are hardcoded in KNN detector
                     and cannot be overridden via kwargs.
    
    Returns:
        Configured KNN detector
    
    Note:
        KNN requires a feature extractor. If model is a full classifier,
        you may need to extract features from an intermediate layer.
        
        Features are normalized to unit length (L2 normalization) before computing
        nearest neighbor distances.
        
        Known limitation: The PyTorch-OOD KNN detector hardcodes n_neighbors=1
        and n_jobs=-1, so these cannot be customized via this function.
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    
    # KNN expects a callable that returns features
    # Use intermediate features (before FC layer) for better distance-based separation
    def feature_extractor(x: Tensor) -> Tensor:
        model.eval()
        with torch.no_grad():
            # Extract features before final FC layer for better distance-based separation
            features = extract_features_before_fc(model, x)
            # Normalize to unit length (L2 normalization)
            features_norm = torch.norm(features, p=2, dim=1, keepdim=True)
            # Avoid division by zero
            features_norm = torch.clamp(features_norm, min=1e-8)
            features = features / features_norm
            return features
    
    # KNN detector hardcodes n_neighbors=1 and n_jobs=-1 in its __init__,
    # then passes **knn_kwargs to sklearn NearestNeighbors.
    # We cannot override these hardcoded values, so we filter them out.
    # Filter out parameters that are hardcoded in KNN detector
    filtered_kwargs = {
        k: v for k, v in knn_kwargs.items()
        if k not in ['n_neighbors', 'n_jobs']
    }
    
    # Warn if user tried to override hardcoded parameters
    if n_neighbors != 1:
        import warnings
        warnings.warn(
            f"KNN detector hardcodes n_neighbors=1. Requested value {n_neighbors} will be ignored.",
            UserWarning
        )
    
    if 'n_neighbors' in knn_kwargs or 'n_jobs' in knn_kwargs:
        import warnings
        warnings.warn(
            "KNN detector hardcodes n_neighbors=1 and n_jobs=-1. "
            "These parameters in knn_kwargs will be ignored.",
            UserWarning
        )
    
    return KNN(
        model=feature_extractor,
        **filtered_kwargs,
    )


def create_react_detector(
    model: nn.Module,
    threshold: float = 1.0,
    detector: Optional[Callable[[Tensor], Tensor]] = None,
    device: Optional[torch.device] = None,
) -> ReAct:
    """
    Create ReAct (Rectified Activations) detector using pytorch_ood's implementation.
    
    Args:
        model: Trained classifier model
        threshold: Activation clipping threshold (default: 1.0)
        detector: Optional detector function mapping logits to OOD scores.
                 If None, uses EnergyBased.score as default.
        device: Device to use (auto-detect if None)
    
    Returns:
        Configured ReAct detector from pytorch_ood
    
    Note:
        ReAct requires splitting the model into backbone (up to penultimate layer)
        and head (final FC layer). The backbone extracts features, ReAct clips
        activations above the threshold, then the head produces logits.
        
        ReAct doesn't require fitting (fit() is a no-op).
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()
    
    # Ensure model can compute gradients (for potential use in detector)
    for param in model.parameters():
        param.requires_grad = False
    
    # Create backbone callable: extracts features before final FC layer
    def backbone(x: Tensor) -> Tensor:
        model.eval()
        # Extract features before final FC layer
        features = extract_features_before_fc(model, x)
        # #region agent log
        # Very small, rate-limited logging to debug ReAct behavior (clipping threshold vs feature scale).
        try:
            import os, json, time
            if os.environ.get("AGENT_DEBUG_REACT", "0") == "1":
                cnt = getattr(backbone, "_agent_cnt", 0)
                if cnt < int(os.environ.get("AGENT_DEBUG_REACT_MAX", "2")):
                    setattr(backbone, "_agent_cnt", cnt + 1)
                    # fraction of activations that would be clipped (ReAct clips > threshold)
                    frac_gt = float((features > float(threshold)).float().mean().item()) if features.numel() else float("nan")
                    entry = {
                        "timestamp": 0,
                        "sessionId": "debug-session",
                        "runId": "react-debug",
                        "hypothesisId": "H_REACT_THRESHOLD",
                        "location": "src/detector/baselines.py:create_react_detector.backbone",
                        "message": "react_backbone_feature_stats",
                        "data": {
                            "threshold": float(threshold),
                            "x_requires_grad": bool(x.requires_grad),
                            "features_shape": list(features.shape),
                            "features_mean": float(features.mean().item()) if features.numel() else float("nan"),
                            "features_std": float(features.std().item()) if features.numel() else float("nan"),
                            "features_min": float(features.min().item()) if features.numel() else float("nan"),
                            "features_max": float(features.max().item()) if features.numel() else float("nan"),
                            "frac_features_gt_threshold": frac_gt,
                        },
                    }
                    with open("/Users/michalkozyra/Developer/PhD/stein_shift_detection/.cursor/debug.log", "a") as f:
                        f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        # #endregion
        return features
    
    # Create head callable: takes features and produces logits
    def head(features: Tensor) -> Tensor:
        model.eval()
        # Check if it's a ResNet
        if hasattr(model, 'fc'):
            # ResNet: use the fc layer directly
            logits = model.fc(features)
            return logits
        elif hasattr(model, 'classifier'):
            # Custom architecture with classifier
            logits = model.classifier(features)
            return logits
        else:
            raise ValueError(
                "ReAct requires a model with a clear separation between "
                "backbone (features) and head (classifier). "
                "ResNet models (with 'fc' attribute) or models with 'classifier' "
                "attribute are supported."
            )
    
    # Use default detector if not provided
    if detector is None:
        detector = EnergyBased.score
    
    # Use pytorch_ood's ReAct implementation
    return ReAct(
        backbone=backbone,
        head=head,
        threshold=threshold,
        detector=detector,
    )


class GSCDetector(Detector):
    """
    Gradient Short-Circuit (GSC) detector for OOD detection.
    
    Based on the ICCV 2025 paper "Gradient Short-Circuit: Efficient Out-of-Distribution 
    Detection via Feature Intervention". This method measures gradient consistency across
    perturbations and short-circuits (masks) features with high variance.
    
    Higher scores indicate higher likelihood of being OOD.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_perturbations: int = 5,
        eps: float = 1e-3,
        mask_percentile: float = 90.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize GSC detector.
        
        Args:
            model: Trained classifier model
            num_perturbations: Number of perturbations to sample for gradient consistency (default: 5)
            eps: Magnitude of input perturbation (default: 1e-3)
            mask_percentile: Percentile threshold for feature masking (default: 90.0)
            device: Device to use (auto-detect if None)
        """
        super().__init__()
        if device is None:
            device = get_device()
        self.model = model.to(device)
        self.device = device
        self.num_perturbations = num_perturbations
        self.eps = eps
        self.mask_percentile = mask_percentile
        
        # Ensure model can compute gradients
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def fit(self, *args, **kwargs):
        """GSC doesn't require fitting."""
        return self
    
    def predict(self, x: Tensor) -> Tensor:
        """
        Compute OOD scores for input batch.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            OOD scores (B,) - higher scores indicate more likely OOD
        """
        # #region agent log
        import json
        debug_log_path = "/Users/michalkozyra/Developer/PhD/stein_shift_detection/.cursor/debug.log"
        def _log(msg, data=None):
            try:
                with open(debug_log_path, "a") as f:
                    log_entry = {
                        "timestamp": 0,
                        "location": "GSCDetector.predict",
                        "message": msg,
                        "data": data or {},
                        "sessionId": "gsc-debug",
                        "runId": "run1",
                        "hypothesisId": "A"
                    }
                    f.write(json.dumps(log_entry) + "\n")
            except: pass
        # #endregion
        
        self.model.eval()
        batch_size = x.shape[0]
        x = x.to(self.device).requires_grad_(True)
        _log("Starting GSC predict", {"batch_size": batch_size})
        
        # 1. Extract features and get logits (with gradients enabled)
        with torch.enable_grad():
            features = extract_features_before_fc(self.model, x)
            if not features.requires_grad:
                features = features.requires_grad_(True)
            
            # Get logits from features
            logits = self._get_logits_from_features(features)
            predicted_classes = logits.argmax(dim=1)
        
        # 2. Sample perturbations and compute gradients
        grads_list = []
        for _ in range(self.num_perturbations):
            # Create perturbed input
            x_perturb = x + self.eps * torch.randn_like(x)
            x_perturb = x_perturb.detach().requires_grad_(True)
            
            # Extract features from perturbed input (with gradients enabled)
            # We need to ensure the entire forward pass preserves gradients
            # The model parameters don't need gradients, but we need gradients w.r.t. inputs/features
            with torch.enable_grad():
                # Extract features - this should preserve gradients if x_perturb.requires_grad=True
                features_perturb = extract_features_before_fc(self.model, x_perturb)
                
                # Get logits from perturbed features
                logits_perturb = self._get_logits_from_features(features_perturb)
                
                # According to the paper, we should compute gradient of predicted class logit
                # Use the original predicted class for consistency across perturbations
                logit_pred_perturb = logits_perturb.gather(1, predicted_classes.unsqueeze(1)).squeeze(1)
                
                # Compute gradients w.r.t. features (inside gradient context)
                # Gradient of predicted class logit w.r.t. features
                grad_h = torch.autograd.grad(
                    outputs=logit_pred_perturb.sum(),
                    inputs=features_perturb,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]
            
            grads_list.append(grad_h.detach())
        
        # Stack gradients: (num_perturbations, batch_size, feature_dim)
        grads = torch.stack(grads_list, dim=0)
        
        # 3. Measure gradient variance (inconsistency metric)
        grad_var = torch.var(grads, dim=0)  # (batch_size, feature_dim)
        _log("Computed gradient variance", {
            "grad_var_mean": float(grad_var.mean().item()),
            "grad_var_std": float(grad_var.std().item()),
            "grad_var_shape": list(grad_var.shape)
        })
        
        # 4. Identify feature dimensions to mask (short-circuit)
        # Find threshold for each sample based on percentile
        thresh = torch.quantile(
            grad_var, 
            self.mask_percentile / 100.0, 
            dim=1, 
            keepdim=True
        )  # (batch_size, 1)
        mask = (grad_var > thresh).float()  # (batch_size, feature_dim)
        _log("Computed mask", {
            "mask_mean": float(mask.mean().item()),
            "mask_sum_per_sample": [float(mask[i].sum().item()) for i in range(min(3, batch_size))],
            "features_to_mask_ratio": float(mask.mean().item())
        })
        
        # 5. Compute first-order Taylor approximation of changed logits
        # According to the paper, we need the full Jacobian J_F = ∇_F y for Taylor approximation
        # We need to recompute logits in a gradient-enabled context
        with torch.enable_grad():
            # Ensure features require grad
            if not features.requires_grad:
                features = features.requires_grad_(True)
            
            # Recompute logits from features to ensure gradient connection
            logits_for_grad = self._get_logits_from_features(features)
            
            # Compute full Jacobian: gradients of all logits w.r.t. features
            # This is needed for proper Taylor approximation: y' ≈ y + J_F · ΔF
            num_classes = logits_for_grad.shape[1]
            jacobian_list = []
            for c in range(num_classes):
                logit_c = logits_for_grad[:, c]
                grad_c = torch.autograd.grad(
                    outputs=logit_c.sum(),
                    inputs=features,
                    retain_graph=(c < num_classes - 1),  # Keep graph until last class
                    create_graph=False,
                    allow_unused=False,
                )[0]
                jacobian_list.append(grad_c)
            
            # Stack to get Jacobian: (batch_size, num_classes, feature_dim)
            jacobian = torch.stack(jacobian_list, dim=1)  # (batch_size, num_classes, feature_dim)
        
        # Short-circuit: mask features (set to zero)
        # Mask should zero out features with high variance (inconsistent gradients)
        delta_h = -features * mask  # (batch_size, feature_dim)
        
        _log("Computed delta_h", {
            "delta_h_mean": float(delta_h.mean().item()),
            "delta_h_std": float(delta_h.std().item()),
            "features_masked_count": [int((mask[i] > 0).sum().item()) for i in range(min(3, batch_size))],
            "jacobian_shape": list(jacobian.shape)
        })
        
        # Approximate new logits using full Jacobian: y' ≈ y + J_F · ΔF
        # For each sample: (num_classes, feature_dim) @ (feature_dim,) -> (num_classes,)
        # We need to compute: jacobian[i] @ delta_h[i] for each sample i
        logit_changes = torch.bmm(
            jacobian,  # (batch_size, num_classes, feature_dim)
            delta_h.unsqueeze(2)  # (batch_size, feature_dim, 1)
        ).squeeze(2)  # (batch_size, num_classes)
        
        _log("Computed logit changes", {
            "logit_changes_mean": float(logit_changes.mean().item()),
            "logit_changes_std": float(logit_changes.std().item()),
            "logit_changes_shape": list(logit_changes.shape)
        })
        
        # Approximate new logits: original + changes
        logits_detached = logits.detach() if logits.requires_grad else logits
        logits_approx = logits_detached + logit_changes.detach()
        
        # 6. Compute OOD score as confidence drop
        # Convert original logits to probabilities for confidence
        probs_orig = F.softmax(logits_detached, dim=1)
        conf_orig = probs_orig.gather(1, predicted_classes.unsqueeze(1)).squeeze(1)
        
        # Approximate new confidence using modified logits
        probs_approx = F.softmax(logits_approx, dim=1)
        conf_approx = probs_approx.gather(1, predicted_classes.unsqueeze(1)).squeeze(1)
        
        _log("Computed confidences", {
            "conf_orig_mean": float(conf_orig.mean().item()),
            "conf_approx_mean": float(conf_approx.mean().item()),
            "conf_drop_mean": float((conf_orig - conf_approx).mean().item())
        })
        
        # Score = original_confidence - approximated_confidence
        # Higher drop = more OOD
        score = conf_orig - conf_approx
        
        _log("Computed final scores", {
            "conf_orig_mean": float(conf_orig.mean().item()),
            "conf_approx_mean": float(conf_approx.mean().item()),
            "score_mean": float(score.mean().item()),
            "score_std": float(score.std().item()),
            "score_range": [float(score.min().item()), float(score.max().item())],
            "logit_changes_mean": float(logit_changes.mean().item()) if 'logit_changes' in locals() else None
        })
        
        return score
    
    def fit_features(self, x: Tensor, y: Tensor):  # noqa: D401
        """Not applicable (requires full input); use fit() or treat as no-op."""
        return self
    
    def predict_features(self, x: Tensor) -> Tensor:
        """Not applicable for feature-based API; use predict(x) on inputs."""
        raise NotImplementedError("GSCDetector operates on input tensors; use predict().")
    
    def _get_logits_from_features(self, features: Tensor) -> Tensor:
        """Get logits from features by applying the final FC layer."""
        if hasattr(self.model, 'fc'):
            return self.model.fc(features)
        elif hasattr(self.model, 'classifier'):
            if isinstance(self.model.classifier, nn.Module):
                return self.model.classifier(features)
            else:
                return self.model.classifier(features)
        else:
            # Fallback: this shouldn't happen if extract_features_before_fc works
            raise ValueError("Cannot extract logits from features for this model architecture")


def create_gsc_detector(
    model: nn.Module,
    num_perturbations: int = 5,
    eps: float = 1e-3,
    mask_percentile: float = 90.0,
    device: Optional[torch.device] = None,
) -> GSCDetector:
    """
    Create Gradient Short-Circuit (GSC) detector.
    
    Args:
        model: Trained classifier model
        num_perturbations: Number of perturbations for gradient consistency (default: 5)
        eps: Perturbation magnitude (default: 1e-3)
        mask_percentile: Percentile threshold for feature masking (default: 90.0)
        device: Device to use (auto-detect if None)
    
    Returns:
        Configured GSC detector
    
    Note:
        GSC requires gradients, so the model must allow gradient computation
        (similar to Mahalanobis). The model parameters don't need gradients,
        but gradients w.r.t. input and features are required.
    """
    if device is None:
        device = get_device()
    
    return GSCDetector(
        model=model,
        num_perturbations=num_perturbations,
        eps=eps,
        mask_percentile=mask_percentile,
        device=device,
    )


def create_all_baseline_detectors(
    model: nn.Module,
    device: Optional[torch.device] = None,
    config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Detector]:
    """
    Create all baseline detectors with default or custom configurations.
    
    Args:
        model: Trained classifier model
        device: Device to use (auto-detect if None)
        config: Optional dictionary with detector-specific configurations.
                Format: {'detector_name': {'param': value, ...}}
                Example: {'odin': {'temperature': 1000.0, 'eps': 0.05}}
    
    Returns:
        Dictionary mapping detector names to detector instances
    """
    if device is None:
        device = get_device()
    
    if config is None:
        config = {}
    
    detectors = {}
    
    # MSP
    msp_config = config.get('msp', {})
    detectors['msp'] = create_msp_detector(
        model=model,
        temperature=msp_config.get('temperature', 1.0),
        device=device,
    )
    
    # ODIN
    odin_config = config.get('odin', {})
    detectors['odin'] = create_odin_detector(
        model=model,
        temperature=odin_config.get('temperature', 1000.0),
        eps=odin_config.get('eps', 0.05),
        norm_std=odin_config.get('norm_std', None),
        device=device,
    )
    
    # Energy
    energy_config = config.get('energy', {})
    detectors['energy'] = create_energy_detector(
        model=model,
        temperature=energy_config.get('temperature', 1.0),
        device=device,
    )
    
    # Mahalanobis
    mahalanobis_config = config.get('mahalanobis', {})
    detectors['mahalanobis'] = create_mahalanobis_detector(
        model=model,
        eps=mahalanobis_config.get('eps', 0.002),
        norm_std=mahalanobis_config.get('norm_std', None),
        device=device,
    )
    
    # KNN (Deep Nearest Neighbours)
    knn_config = config.get('knn', {})
    detectors['knn'] = create_knn_detector(
        model=model,
        n_neighbors=knn_config.get('n_neighbors', 1),  # Note: currently ignored due to KNN limitation
        device=device,
        **{k: v for k, v in knn_config.items() if k not in ['n_neighbors', 'n_jobs']},
    )
    
    # ReAct
    react_config = config.get('react', {})
    detectors['react'] = create_react_detector(
        model=model,
        threshold=react_config.get('threshold', 1.0),
        detector=react_config.get('detector', None),
        device=device,
    )
    
    # GSC (Gradient Short-Circuit)
    gsc_config = config.get('gsc', {})
    detectors['gsc'] = create_gsc_detector(
        model=model,
        num_perturbations=gsc_config.get('num_perturbations', 5),
        eps=gsc_config.get('eps', 1e-3),
        mask_percentile=gsc_config.get('mask_percentile', 90.0),
        device=device,
    )
    
    return detectors


def get_detector_config_template() -> Dict[str, Dict[str, Any]]:
    """
    Get a template configuration dictionary for all baseline detectors.
    
    Returns:
        Dictionary with default configurations for each detector
    """
    return {
        'msp': {
            'temperature': 1.0,
        },
        'odin': {
            'temperature': 1000.0,
            'eps': 0.05,
            'norm_std': None,  # Will use dataset-specific values if available
        },
        'energy': {
            'temperature': 1.0,
        },
        'mahalanobis': {
            'eps': 0.002,
            'norm_std': None,  # Will use dataset-specific values if available
        },
        'knn': {
            'n_neighbors': 1,
        },
        'react': {
            'threshold': 1.0,
            'detector': None,  # Uses EnergyBased.score by default
        },
        'gsc': {
            'num_perturbations': 5,
            'eps': 1e-3,
            'mask_percentile': 90.0,
        },
    }

