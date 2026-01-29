"""
Stein Factory Detector - computes shared components once and generates all Stein flavors.

This detector computes f, grad f, lap f, and s once per batch, then generates
all supported Stein operator modes from these shared components, significantly
reducing compute time when evaluating multiple Stein variants.
"""

from typing import Optional, Literal, Callable, Dict, List, TypeVar
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset
import random
from tqdm import tqdm
import json
import time

from pytorch_ood.api import Detector, ModelNotSetException

from ..gradients import compute_grad_f, hutchinson_laplacian, softmax_laplacian_approx
from ..eval_functions import score_at_x
from ..utils import get_device, is_resnet_model
from ..evaluation.robust_statistics import robust_baseline_and_threshold

Self = TypeVar("Self")


# region agent log
_AGENT_DEBUG_LOG_PATH = "/Users/michalkozyra/Developer/PhD/stein_shift_detection/.cursor/debug.log"


def _agent_log(payload: dict) -> None:
    try:
        payload = dict(payload)
        payload["timestamp"] = 0
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def _agent_run_id() -> str:
    try:
        return str(os.environ.get("AGENT_RUN_ID", "run1"))
    except Exception:
        return "run1"


# endregion agent log


class SteinFactoryDetector(Detector):
    """
    Factory detector that computes shared Stein components once
    and generates all supported Stein operator flavors.
    
    This saves compute time by avoiding redundant forward/backward passes
    when evaluating multiple Stein variants.
    
    Supported modes:
    - 'full': Laplacian(f(x)) + s(x)^T grad f(x)
    - 'full_no_lap': s(x)^T grad f(x) (Laplacian skipped)
    - 'first_order': ||grad f(x) + s(x) * f(x)||_2
    - 'first_order_sum': sum(grad f(x) + s(x) * f(x))
    - 'per_dimension': Per-class residuals with aggregation (classification only)
    """
    
    # All supported modes.
    # NOTE:
    # - per_dimension_l2 matches SteinDetector(stein_operator_type='per_dimension', aggregation='l2')
    # - *_fixed0 / *_top1 / *_all are multi-scalar ablations used by the benchmark preset.
    SUPPORTED_MODES = [
        'full', 'full_no_lap', 'first_order', 'first_order_sum',
        'per_dimension', 'per_dimension_l2',
        # Component ablations (type 1)
        'lap_only', 'grad_only', 'score_only',
        # Std-normalized / scale-balanced variants (estimated from training)
        'lap_only_std', 'full_no_lap_std', 'full_std_balanced',
        # Per-dimension L2 ablations (map everything to stein_per_dimension_l2)
        'per_dimension_l2_no_lap',
        'per_dimension_l2_lap_only',
        'per_dimension_l2_grad_only',
        'per_dimension_l2_score_only',
        'per_dimension_l2_lap_only_std',
        'per_dimension_l2_no_lap_std',
        'per_dimension_l2_std_balanced',
        'full_fixed0', 'full_top1',
        'first_order_fixed0', 'first_order_top1', 'first_order_all',
    ]

    _STD_DERIVED_MODES = {
        'lap_only_std', 'full_no_lap_std', 'full_std_balanced',
        'per_dimension_l2_lap_only_std', 'per_dimension_l2_no_lap_std', 'per_dimension_l2_std_balanced',
    }
    
    def __init__(
        self,
        model: nn.Module,
        score_model: Optional[nn.Module] = None,
        score_function: Optional[Callable[[Tensor], Tensor]] = None,
        device: Optional[torch.device] = None,
        model_type: Literal['classification', 'regression'] = 'classification',
        # Classification scalar f(x) choice (only used when model_type='classification')
        classification_scalar_mode: Literal['predicted_class_prob', 'fixed_class_prob', 'topk_sum_prob'] = 'predicted_class_prob',
        fixed_class_idx: int = 0,
        classification_topk: int = 1,
        num_probes: int = 1,
        skip_laplacian: bool = False,  # For models with MaxPool (ResNet)
        enabled_modes: Optional[List[str]] = None,  # None = all modes
        compute_baseline: bool = True,
        baseline_subset_size: Optional[int] = None,
    ):
        """
        Initialize SteinFactoryDetector.
        
        Args:
            model: Model (f(x)) - required
            score_model: Trained score model (s(x) = grad log p(x)) - required if score_function is None
            score_function: Analytical score function s(x) -> Tensor - required if score_model is None
            device: Device to use (auto-detect if None)
            model_type: 'classification' or 'regression'
            num_probes: Number of Hutchinson probes for Laplacian (only for 'full' mode)
            skip_laplacian: If True, assume Laplacian is zero (e.g., for ReLU networks)
            enabled_modes: List of modes to compute. If None, enables all supported modes.
            compute_baseline: If True, compute training baseline for correction
            baseline_subset_size: Use subset of training data for baseline (faster, optional)
        """
        super().__init__()
        
        # Validate score function source
        if score_model is None and score_function is None:
            raise ValueError(
                "Either score_model or score_function must be provided. "
                "Stein operator requires a score function s(x) = grad log p(x)."
            )
        
        # Validate enabled modes
        if enabled_modes is None:
            enabled_modes = self.SUPPORTED_MODES.copy()
        else:
            # Filter out per_dimension for regression
            if model_type != 'classification' and 'per_dimension' in enabled_modes:
                enabled_modes = [m for m in enabled_modes if m != 'per_dimension']
                print("Warning: per_dimension mode only supported for classification, removed from enabled_modes")
            
            # Validate all modes are supported
            invalid_modes = [m for m in enabled_modes if m not in self.SUPPORTED_MODES]
            if invalid_modes:
                raise ValueError(f"Invalid modes: {invalid_modes}. Supported: {self.SUPPORTED_MODES}")
        
        self.model = model
        self.score_model = score_model
        self.score_function = score_function
        self.device = device if device is not None else get_device()
        self.model_type = model_type
        self.classification_scalar_mode = classification_scalar_mode
        self.fixed_class_idx = int(fixed_class_idx)
        self.classification_topk = int(classification_topk)
        if self.classification_topk < 1:
            raise ValueError(f"classification_topk must be >= 1, got {self.classification_topk}")
        self.num_probes = num_probes
        self.skip_laplacian = skip_laplacian
        self.enabled_modes = enabled_modes
        self.compute_baseline = compute_baseline
        self.baseline_subset_size = baseline_subset_size
        
        # Will be set during fit()
        self.baselines: Dict[str, Tensor] = {}  # Mode -> baseline tensor
        self.training_stds: Dict[str, Tensor] = {}  # Mode -> training std tensor
        self.sigmas: Optional[Tensor] = None
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        score_train_loader: Optional[DataLoader] = None,
    ) -> Self:
        """
        Fit the detector by computing baselines for all enabled modes.
        
        Args:
            train_loader: Training data
            val_loader: Validation data (optional, not used currently)
            score_train_loader: Training data for score model (optional, not used currently)
        
        Returns:
            self (for method chaining)
        """
        if self.compute_baseline:
            self._compute_baselines(train_loader)
        
        return self
    
    def _compute_baselines(self, train_loader: DataLoader) -> None:
        """
        Compute Stein residual baselines on training data for all enabled modes.
        """
        if not self.compute_baseline:
            return
        
        if self.model is None:
            raise ModelNotSetException("Model must be set to compute baseline")
        
        # Optionally use subset for speed
        if self.baseline_subset_size is not None:
            dataset = train_loader.dataset
            subset_size = min(self.baseline_subset_size, len(dataset))
            indices = random.sample(range(len(dataset)), subset_size)
            subset = Subset(dataset, indices)
            loader = DataLoader(
                subset,
                batch_size=train_loader.batch_size,
                shuffle=False,
                num_workers=getattr(train_loader, 'num_workers', 0),
                pin_memory=getattr(train_loader, 'pin_memory', False)
            )
        else:
            loader = train_loader
        
        # Collect scores for all modes.
        #
        # NOTE: Some modes are derived using training stds (e.g. lap_only_std), so we cannot
        # compute them during the first pass (circular dependency). We compute base-mode
        # distributions first, then derive the std-normalized variants afterward.
        enabled_base_modes = [m for m in self.enabled_modes if m not in self._STD_DERIVED_MODES]
        all_scores = {mode: [] for mode in enabled_base_modes}
        
        self.model.eval()
        if self.score_model is not None:
            self.score_model.eval()
        
        with torch.enable_grad():
            for batch in tqdm(loader, desc='Computing Stein baselines (factory)'):
                # Handle both (x, y) and (x,) tuples
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device)
                x_req = x.clone().detach().requires_grad_(True)
                
                # Compute shared components once
                components = self._compute_shared_components(x_req)
                
                # Generate all mode scores
                # IMPORTANT: std-derived modes depend on training stds, which we are estimating here.
                # Temporarily disable std-derived modes while collecting the base distributions.
                _old_enabled_modes = self.enabled_modes
                try:
                    self.enabled_modes = enabled_base_modes
                    scores = self._generate_all_stein_scores(components)
                finally:
                    self.enabled_modes = _old_enabled_modes
                
                # Collect scores for each mode
                for mode in enabled_base_modes:
                    all_scores[mode].append(scores[mode].detach().cpu())
        
        # Compute baselines per mode
        for mode in enabled_base_modes:
            mode_scores = torch.cat(all_scores[mode], dim=0)
            
            # Use robust statistics
            baseline_robust, threshold_robust = robust_baseline_and_threshold(
                mode_scores,
                method='median_mad',
                multiplier=2.0
            )
            
            # Store baselines and stds
            self.baselines[mode] = baseline_robust.to(self.device)
            self.training_stds[mode] = mode_scores.std().to(self.device)
            
            # Print summary
            baseline_mean = mode_scores.mean()
            baseline_std = mode_scores.std()
            print(f'  {mode}: baseline={baseline_robust.item():.6e} (MAD), '
                  f'mean={baseline_mean.item():.6e}, std={baseline_std.item():.6e}')

        # Derive std-normalized / scale-balanced variants after base stds are available.
        #
        # These are lightweight transforms of the already-collected base-mode score vectors,
        # so this adds no extra gradient compute.
        eps = 1e-8
        if any(m in self.enabled_modes for m in self._STD_DERIVED_MODES):
            derived: Dict[str, torch.Tensor] = {}

            # Scalar (top-1 / fixed / topk-sum) scale-balancing
            if any(m in self.enabled_modes for m in {'lap_only_std', 'full_no_lap_std', 'full_std_balanced'}):
                if 'lap_only' not in all_scores or 'full_no_lap' not in all_scores:
                    raise ValueError(
                        "Std-derived modes (scalar) require 'lap_only' and 'full_no_lap' to be enabled "
                        "(so their training stds can be estimated)."
                    )
                lap_scores = torch.cat(all_scores['lap_only'], dim=0)
                dot_scores = torch.cat(all_scores['full_no_lap'], dim=0)
                std_lap = lap_scores.std().clamp_min(eps)
                std_dot = dot_scores.std().clamp_min(eps)
                self.training_stds['lap_only'] = std_lap.to(self.device)
                self.training_stds['full_no_lap'] = std_dot.to(self.device)
                if 'lap_only_std' in self.enabled_modes:
                    derived['lap_only_std'] = lap_scores / std_lap
                if 'full_no_lap_std' in self.enabled_modes:
                    derived['full_no_lap_std'] = dot_scores / std_dot
                if 'full_std_balanced' in self.enabled_modes:
                    derived['full_std_balanced'] = (lap_scores / std_lap) + (dot_scores / std_dot)

            # Per-dimension L2 scale-balancing (maps to stein_per_dimension_l2 family)
            if any(m in self.enabled_modes for m in {'per_dimension_l2_lap_only_std', 'per_dimension_l2_no_lap_std', 'per_dimension_l2_std_balanced'}):
                if 'per_dimension_l2_lap_only' not in all_scores or 'per_dimension_l2_no_lap' not in all_scores:
                    raise ValueError(
                        "Std-derived modes (per_dimension_l2) require 'per_dimension_l2_lap_only' and "
                        "'per_dimension_l2_no_lap' to be enabled (so their training stds can be estimated)."
                    )
                lap_scores = torch.cat(all_scores['per_dimension_l2_lap_only'], dim=0)
                dot_scores = torch.cat(all_scores['per_dimension_l2_no_lap'], dim=0)
                std_lap = lap_scores.std().clamp_min(eps)
                std_dot = dot_scores.std().clamp_min(eps)
                self.training_stds['per_dimension_l2_lap_only'] = std_lap.to(self.device)
                self.training_stds['per_dimension_l2_no_lap'] = std_dot.to(self.device)
                if 'per_dimension_l2_lap_only_std' in self.enabled_modes:
                    derived['per_dimension_l2_lap_only_std'] = lap_scores / std_lap
                if 'per_dimension_l2_no_lap_std' in self.enabled_modes:
                    derived['per_dimension_l2_no_lap_std'] = dot_scores / std_dot
                if 'per_dimension_l2_std_balanced' in self.enabled_modes:
                    derived['per_dimension_l2_std_balanced'] = (lap_scores / std_lap) + (dot_scores / std_dot)

            for mode, mode_scores in derived.items():
                baseline_robust, _threshold_robust = robust_baseline_and_threshold(
                    mode_scores,
                    method='median_mad',
                    multiplier=2.0,
                )
                self.baselines[mode] = baseline_robust.to(self.device)
                self.training_stds[mode] = mode_scores.std().to(self.device)

                baseline_mean = mode_scores.mean()
                baseline_std = mode_scores.std()
                print(
                    f'  {mode}: baseline={baseline_robust.item():.6e} (MAD), '
                    f'mean={baseline_mean.item():.6e}, std={baseline_std.item():.6e}'
                )
    
    def predict_all(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Compute Stein residual scores for all enabled modes.
        
        Args:
            x: Input tensor of shape (N, ...)
        
        Returns:
            scores: Dict mapping mode name -> scores tensor of shape (N,)
                   Higher scores = more likely OOD
        """
        if self.model is None:
            raise ModelNotSetException("Model must be set or trained")
        
        # Set model to eval mode
        self.model.eval()
        
        # Ensure on correct device and enable gradients
        x = x.to(self.device)
        x_req = x.clone().detach().requires_grad_(True)
        
        # Compute shared components and generate all mode scores with gradients enabled.
        # IMPORTANT: callers may wrap prediction in torch.no_grad() for performance. Factory modes
        # rely on torch.autograd.grad internally, so we must force-enable grads here.
        with torch.enable_grad():
            components = self._compute_shared_components(x_req)
            scores = self._generate_all_stein_scores(components)
        
        # Apply baseline correction if available
        if self.compute_baseline and self.baselines:
            for mode in scores.keys():
                if mode in self.baselines:
                    # Return raw residuals (baseline correction handled in evaluation)
                    # But we could apply it here if needed
                    pass
        
        return scores
    
    def predict(self, x: Tensor, mode: Optional[str] = None) -> Tensor:
        """
        Compute Stein residual scores for a single mode (backward compatibility).
        
        Args:
            x: Input tensor of shape (N, ...)
            mode: Mode to compute. If None, uses 'full' or first enabled mode.
        
        Returns:
            scores: Stein residual scores of shape (N,)
                   Higher scores = more likely OOD
        """
        if mode is None:
            # Default to 'full' if available, otherwise first enabled mode
            mode = 'full' if 'full' in self.enabled_modes else self.enabled_modes[0]
        
        if mode not in self.enabled_modes:
            raise ValueError(f"Mode '{mode}' not enabled. Enabled modes: {self.enabled_modes}")
        
        all_scores = self.predict_all(x)
        return all_scores[mode]
    
    def _compute_shared_components(self, x_req: Tensor) -> Dict[str, Tensor]:
        """
        Compute all shared components once.
        
        Returns:
            components: Dict with keys:
                - 'f': Model output (B, num_classes) or (B,)
                - 'f_scalar': Scalar function value (B,)
                - 'grad_f': Gradient (B, C, H, W) or (B, features)
                - 'grad_f_flat': Flattened gradient (B, D)
                - 'lap_f': Laplacian (B,) - only if needed
                - 's': Score function (B, D) where D is flattened input dim
                - 'x_req': Input with requires_grad (for consistency)
        """
        self.model.eval()
        if self.score_model is not None:
            self.score_model.eval()
        
        # 1. Forward pass and (optionally) grad f (handles both classification and regression)
        if self.model_type == 'classification':
            if not x_req.requires_grad:
                x_req = x_req.clone().detach().to(self.device).requires_grad_(True)
            else:
                x_req = x_req.to(self.device)
            self.model.eval()
            with torch.enable_grad():
                logits = self.model(x_req)  # (B, K)
                probs = F.softmax(logits, dim=1)

            # Default scalar definition (legacy) - needed by old modes ('full', 'first_order', ...)
            if self.classification_scalar_mode == 'predicted_class_prob':
                k_def = torch.argmax(probs, dim=1)
                f_scalar = probs[torch.arange(probs.size(0), device=probs.device), k_def]
            elif self.classification_scalar_mode == 'fixed_class_prob':
                k_def = torch.full((probs.size(0),), fill_value=self.fixed_class_idx, device=probs.device, dtype=torch.long)
                k_def = torch.clamp(k_def, 0, probs.size(1) - 1)
                f_scalar = probs[torch.arange(probs.size(0), device=probs.device), k_def]
            elif self.classification_scalar_mode == 'topk_sum_prob':
                K = probs.size(1)
                kk = min(self.classification_topk, K)
                topk_idx = torch.topk(probs, k=kk, dim=1).indices
                f_scalar = probs.gather(1, topk_idx).sum(dim=1)
            else:
                raise ValueError(f"Unknown classification_scalar_mode={self.classification_scalar_mode!r}")

            grads = torch.autograd.grad(f_scalar.sum(), x_req, create_graph=True)[0]
            x_req_grad = x_req
        else:  # regression
            if not x_req.requires_grad:
                x_req = x_req.requires_grad_(True)
            f_scalar = self.model(x_req)  # (B,)
            grads = torch.autograd.grad(
                f_scalar.sum(), x_req, create_graph=True, retain_graph=True
            )[0]  # (B, ...)
            x_req_grad = x_req
            logits = None
            probs = None
        
        # Flatten gradient
        grad_f_flat = grads.view(x_req_grad.size(0), -1)  # (B, D)
        
        # 2. Compute Laplacian (only if needed by any enabled mode that uses it)
        lap_f = None
        lap_method = None
        k = None
        needs_lap = any(
            m in self.enabled_modes
            for m in ('full', 'lap_only', 'lap_only_std', 'full_std_balanced')
        )
        if needs_lap and (not self.skip_laplacian):
            if self.model_type == 'classification' and is_resnet_model(self.model):
                # Use softmax Laplacian approximation for ResNet models
                if logits is None:
                    with torch.enable_grad():
                        logits = self.model(x_req_grad)
                        probs = F.softmax(logits, dim=1)
                if self.classification_scalar_mode == 'fixed_class_prob':
                    k = torch.full(
                        (probs.size(0),),
                        fill_value=self.fixed_class_idx,
                        device=probs.device,
                        dtype=torch.long,
                    )
                    k = torch.clamp(k, 0, probs.size(1) - 1)
                elif self.classification_scalar_mode == 'topk_sum_prob':
                    K = probs.size(1)
                    kk = min(self.classification_topk, K)
                    k = torch.topk(probs, k=kk, dim=1).indices  # (B, kk)
                else:
                    k = torch.argmax(probs, dim=1)  # (B,)
                num_classes = logits.size(1)
                lap_f = softmax_laplacian_approx(
                    x_req_grad, self.model, k=k, topk=num_classes, device=self.device
                )
                lap_method = "softmax_laplacian_approx"
            else:
                # Use Hutchinson estimator
                lap_f = hutchinson_laplacian(
                    x_req_grad, grads,
                    num_probes=self.num_probes,
                    device=self.device
                )
                lap_method = "hutchinson_laplacian"
        
        # 3. Compute score function
        s = self._get_score(x_req_grad)  # (B, D)
        
        return {
            'f': logits if self.model_type == 'classification' else f_scalar,
            'f_scalar': f_scalar,
            'probs': probs if self.model_type == 'classification' else None,
            'grad_f': grads,
            'grad_f_flat': grad_f_flat,
            'lap_f': lap_f,
            's': s,
            'x_req': x_req_grad,
        }
    
    def _generate_all_stein_scores(self, components: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Generate all enabled Stein flavors from shared components.
        
        Args:
            components: Dict from _compute_shared_components()
        
        Returns:
            scores: Dict mapping mode name -> scores tensor (B,)
        """
        scores = {}
        
        grad_f_flat = components['grad_f_flat']  # (B, D)
        s = components['s']  # (B, D)
        f_scalar = components['f_scalar']  # (B,)
        lap_f = components['lap_f']  # (B,) or None

        # region agent log
        # Debug: explain why dot-term tracks grad-only (first few batches only).
        try:
            n = int(getattr(self, "_agent_log_count_dot_vs_grad", 0))
            if n < 4:
                g_norm = torch.norm(grad_f_flat.detach().float(), p=2, dim=1)  # (B,)
                s_norm = torch.norm(s.detach().float(), p=2, dim=1)  # (B,)
                dot = (s.detach().float() * grad_f_flat.detach().float()).sum(dim=1)  # (B,)
                denom = (s_norm * g_norm).clamp_min(1e-12)
                cos = (dot / denom).clamp(-10.0, 10.0)

                def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
                    a = a - a.mean()
                    b = b - b.mean()
                    return float((a * b).mean().item() / ((a.std().item() * b.std().item()) + 1e-12))

                payload = {
                    "sessionId": "debug-session",
                    "runId": _agent_run_id(),
                    "hypothesisId": "H_DOT_COLLAPSE",
                    "location": "src/detector/stein_factory.py:SteinFactoryDetector._generate_all_stein_scores:dot_stats",
                    "message": "dot vs grad/score stats (why no_lap ~= grad_only)",
                    "data": {
                        "enabled_modes": list(self.enabled_modes),
                        "B": int(g_norm.numel()),
                        "score_model": type(self.score_model).__name__ if self.score_model is not None else None,
                        "score_model_cfg": {
                            "timestep": int(getattr(self.score_model, "timestep", -1)) if self.score_model is not None else None,
                            "denom_mode": str(getattr(self.score_model, "denom_mode", "")) if self.score_model is not None else None,
                            "add_noise": bool(getattr(self.score_model, "add_noise", False)) if self.score_model is not None else None,
                        },
                        "g_norm_mean": float(g_norm.mean().item()),
                        "g_norm_std": float(g_norm.std().item()),
                        "s_norm_mean": float(s_norm.mean().item()),
                        "s_norm_std": float(s_norm.std().item()),
                        "dot_mean": float(dot.mean().item()),
                        "dot_std": float(dot.std().item()),
                        "abs_cos_mean": float(cos.abs().mean().item()),
                        "abs_cos_std": float(cos.abs().std().item()),
                        "corr_dot_gnorm": _corr(dot, g_norm),
                        "corr_dot_snorm": _corr(dot.abs(), s_norm),
                        "corr_gnorm_snorm": _corr(g_norm, s_norm),
                    },
                }
                _agent_log(payload)
                setattr(self, "_agent_log_count_dot_vs_grad", n + 1)
        except Exception:
            pass
        # endregion agent log
        
        # Full operator (with Laplacian)
        if 'full' in self.enabled_modes:
            if lap_f is None:
                # If Laplacian wasn't computed (skip_laplacian=True), use zeros
                lap_f = torch.zeros(f_scalar.size(0), device=self.device)
            s_dot_grad = (s * grad_f_flat).sum(dim=1)  # (B,)
            scores['full'] = lap_f + s_dot_grad
        
        # Full operator (without Laplacian)
        if 'full_no_lap' in self.enabled_modes:
            s_dot_grad = (s * grad_f_flat).sum(dim=1)  # (B,)
            scores['full_no_lap'] = s_dot_grad

        # Component ablations (type 1)
        if 'lap_only' in self.enabled_modes:
            if lap_f is None:
                # If Laplacian wasn't computed (skip_laplacian=True), use zeros
                scores['lap_only'] = torch.zeros(f_scalar.size(0), device=self.device)
            else:
                scores['lap_only'] = lap_f

        if 'grad_only' in self.enabled_modes:
            scores['grad_only'] = torch.norm(grad_f_flat, p=2, dim=1)

        if 'score_only' in self.enabled_modes:
            scores['score_only'] = torch.norm(s, p=2, dim=1)
        
        # First-order L2 norm
        if 'first_order' in self.enabled_modes:
            f_expanded = f_scalar.unsqueeze(1)  # (B, 1)
            stein_term = f_expanded * s + grad_f_flat  # (B, D)
            scores['first_order'] = torch.norm(stein_term, dim=1)  # (B,)
        
        # First-order sum
        if 'first_order_sum' in self.enabled_modes:
            f_expanded = f_scalar.unsqueeze(1)  # (B, 1)
            stein_term = f_expanded * s + grad_f_flat  # (B, D)
            scores['first_order_sum'] = stein_term.sum(dim=1)  # (B,)

        # Std-normalized / scale-balanced variants (scalar family).
        # These use training stds estimated from the ID training distribution for scalar lap/dot terms.
        eps = 1e-8
        scalar_std_modes = {'lap_only_std', 'full_no_lap_std', 'full_std_balanced'}
        if any(m in self.enabled_modes for m in scalar_std_modes):
            if (not self.training_stds) or ('lap_only' not in self.training_stds) or ('full_no_lap' not in self.training_stds):
                raise RuntimeError(
                    "Std-normalized SteinFactory modes (scalar) require fitted training stds. "
                    "Call fit() before predict_all(), and ensure 'lap_only' and 'full_no_lap' are enabled."
                )
            std_lap = self.training_stds['lap_only'].clamp_min(eps)
            std_dot = self.training_stds['full_no_lap'].clamp_min(eps)

            # Use already-computed base terms if present, otherwise compute on the fly.
            lap_term = scores.get('lap_only', None)
            if lap_term is None:
                if lap_f is None:
                    lap_term = torch.zeros(f_scalar.size(0), device=self.device)
                else:
                    lap_term = lap_f

            dot_term = scores.get('full_no_lap', None)
            if dot_term is None:
                dot_term = (s * grad_f_flat).sum(dim=1)

            if 'lap_only_std' in self.enabled_modes:
                scores['lap_only_std'] = lap_term / std_lap
            if 'full_no_lap_std' in self.enabled_modes:
                scores['full_no_lap_std'] = dot_term / std_dot
            if 'full_std_balanced' in self.enabled_modes:
                scores['full_std_balanced'] = (lap_term / std_lap) + (dot_term / std_dot)
        
        # === Multi-scalar ablation modes (fixed class, top1, all classes) ===
        # These are used by the benchmark "subset" preset.
        if self.model_type == 'classification':
            probs = components['probs']  # (B, K)
            x_req = components['x_req']
            K = probs.size(1)

            need_fixed0 = any(m in self.enabled_modes for m in ['full_fixed0', 'first_order_fixed0'])
            need_top1 = any(m in self.enabled_modes for m in ['full_top1', 'first_order_top1'])

            # fixed class idx
            if need_fixed0:
                k0 = torch.full((probs.size(0),), fill_value=self.fixed_class_idx, device=probs.device, dtype=torch.long)
                k0 = torch.clamp(k0, 0, K - 1)
                f0 = probs[torch.arange(probs.size(0), device=probs.device), k0]  # (B,)
                g0 = torch.autograd.grad(f0.sum(), x_req, create_graph=True, retain_graph=True)[0]
                g0_flat = g0.view(x_req.size(0), -1)
                if any(m == 'full_fixed0' for m in self.enabled_modes):
                    if self.skip_laplacian:
                        lap0 = torch.zeros_like(f0)
                    else:
                        if is_resnet_model(self.model):
                            lap0 = softmax_laplacian_approx(x_req, self.model, k=k0, topk=K, device=self.device)
                        else:
                            lap0 = hutchinson_laplacian(x_req, g0, num_probes=self.num_probes, device=self.device)
                    scores['full_fixed0'] = lap0 + (s * g0_flat).sum(dim=1)
                if any(m == 'first_order_fixed0' for m in self.enabled_modes):
                    scores['first_order_fixed0'] = torch.norm(g0_flat + f0.unsqueeze(1) * s, dim=1)

            # top1 predicted class (argmax)
            if need_top1:
                k1 = torch.argmax(probs, dim=1)  # (B,)
                f1 = probs[torch.arange(probs.size(0), device=probs.device), k1]  # (B,)
                g1 = torch.autograd.grad(f1.sum(), x_req, create_graph=True, retain_graph=True)[0]
                g1_flat = g1.view(x_req.size(0), -1)
                if any(m == 'full_top1' for m in self.enabled_modes):
                    if self.skip_laplacian:
                        lap1 = torch.zeros_like(f1)
                    else:
                        if is_resnet_model(self.model):
                            lap1 = softmax_laplacian_approx(x_req, self.model, k=k1, topk=K, device=self.device)
                        else:
                            lap1 = hutchinson_laplacian(x_req, g1, num_probes=self.num_probes, device=self.device)
                    scores['full_top1'] = lap1 + (s * g1_flat).sum(dim=1)
                if any(m == 'first_order_top1' for m in self.enabled_modes):
                    scores['first_order_top1'] = torch.norm(g1_flat + f1.unsqueeze(1) * s, dim=1)

            # all classes: f(x)=sum_k p_k(x)=1, so grad f = 0 and first_order reduces to ||s||_2
            if 'first_order_all' in self.enabled_modes:
                scores['first_order_all'] = torch.norm(s, dim=1)

        # Per-dimension (classification only): returns per-class residuals aggregated by sum
        if 'per_dimension' in self.enabled_modes:
            if self.model_type != 'classification':
                raise ValueError("per_dimension only for classification")
            scores['per_dimension'] = self._compute_per_dimension_from_components(components, aggregation='sum')

        # Per-dimension L2 (classification only): matches stein_per_dimension_l2
        if 'per_dimension_l2' in self.enabled_modes:
            if self.model_type != 'classification':
                raise ValueError("per_dimension_l2 only for classification")
            scores['per_dimension_l2'] = self._compute_per_dimension_from_components(components, aggregation='l2')

        # Per-dimension L2 ablations: compute per-class terms once, then aggregate with L2 over classes.
        need_perdim_l2_ablation = any(
            m in self.enabled_modes
            for m in (
                'per_dimension_l2_no_lap',
                'per_dimension_l2_lap_only',
                'per_dimension_l2_grad_only',
                'per_dimension_l2_score_only',
                'per_dimension_l2_lap_only_std',
                'per_dimension_l2_no_lap_std',
                'per_dimension_l2_std_balanced',
            )
        )
        if need_perdim_l2_ablation:
            if self.model_type != 'classification':
                raise ValueError("per_dimension_l2_* modes only supported for classification")

            x_req = components['x_req']
            probs = components['probs']  # (B, K)
            if probs is None:
                raise RuntimeError("per_dimension_l2_* modes require classification probs in components")
            K = probs.size(1)

            # Score once
            s = components['s']  # (B, D)

            # Compute per-class laplacians and per-class s·grad(p_c) once
            lap_per_dim = torch.zeros(probs.size(0), K, device=self.device)
            s_dot_grad_per_dim = torch.zeros(probs.size(0), K, device=self.device)
            grad_norm_per_dim = torch.zeros(probs.size(0), K, device=self.device)

            for c in range(K):
                f_c = probs[:, c]  # (B,)
                grad_c = torch.autograd.grad(
                    f_c.sum(), x_req, create_graph=True, retain_graph=True
                )[0]  # (B, ...)
                grad_c_flat = grad_c.view(x_req.size(0), -1)  # (B, D)
                s_dot_grad_per_dim[:, c] = (s * grad_c_flat).sum(dim=1)
                grad_norm_per_dim[:, c] = torch.norm(grad_c_flat, p=2, dim=1)

                if self.skip_laplacian:
                    lap_c = torch.zeros(probs.size(0), device=self.device)
                else:
                    if is_resnet_model(self.model):
                        k_c = torch.full((probs.size(0),), fill_value=c, device=probs.device, dtype=torch.long)
                        k_c = torch.clamp(k_c, 0, K - 1)
                        lap_c = softmax_laplacian_approx(x_req, self.model, k=k_c, topk=K, device=self.device)
                    else:
                        retain_for_next = (c < K - 1) or (self.score_model is not None or self.score_function is not None)
                        lap_c = hutchinson_laplacian(
                            x_req, grad_c,
                            num_probes=self.num_probes,
                            device=self.device,
                            retain_graph_after=retain_for_next,
                        )
                lap_per_dim[:, c] = lap_c

            # Base aggregated terms
            lap_l2 = torch.norm(lap_per_dim, p=2, dim=1)               # (B,)
            dot_l2 = torch.norm(s_dot_grad_per_dim, p=2, dim=1)         # (B,)
            full_l2 = torch.norm(lap_per_dim + s_dot_grad_per_dim, p=2, dim=1)  # (B,)
            grad_l2_over_classes = torch.norm(grad_norm_per_dim, p=2, dim=1)    # (B,)
            score_l2 = torch.norm(s, p=2, dim=1) * (float(K) ** 0.5)            # (B,)

            # region agent log
            # Debug: per-dimension L2 ablations often look similar if dot_l2 tracks grad_l2_over_classes.
            try:
                n2 = int(getattr(self, "_agent_log_count_perdim", 0))
                if n2 < 4:
                    a = dot_l2.detach().float()
                    b = grad_l2_over_classes.detach().float()
                    c = score_l2.detach().float()

                    def _corr(a_: torch.Tensor, b_: torch.Tensor) -> float:
                        a_ = a_ - a_.mean()
                        b_ = b_ - b_.mean()
                        return float((a_ * b_).mean().item() / ((a_.std().item() * b_.std().item()) + 1e-12))

                    _agent_log(
                        {
                            "sessionId": "debug-session",
                            "runId": _agent_run_id(),
                            "hypothesisId": "H_PERDIM_COLLAPSE",
                            "location": "src/detector/stein_factory.py:SteinFactoryDetector._generate_all_stein_scores:perdim_l2_stats",
                            "message": "perdim_l2 dot vs grad vs score correlations",
                            "data": {
                                "B": int(a.numel()),
                                "K_classes": int(K),
                                "dot_l2_mean": float(a.mean().item()),
                                "dot_l2_std": float(a.std().item()),
                                "grad_l2_mean": float(b.mean().item()),
                                "grad_l2_std": float(b.std().item()),
                                "score_l2_mean": float(c.mean().item()),
                                "score_l2_std": float(c.std().item()),
                                "corr_dotl2_gradl2": _corr(a, b),
                                "corr_dotl2_scorel2": _corr(a, c),
                                "corr_gradl2_scorel2": _corr(b, c),
                            },
                        }
                    )
                    setattr(self, "_agent_log_count_perdim", n2 + 1)
            except Exception:
                pass
            # endregion agent log

            if 'per_dimension_l2_lap_only' in self.enabled_modes:
                scores['per_dimension_l2_lap_only'] = lap_l2
            if 'per_dimension_l2_no_lap' in self.enabled_modes:
                scores['per_dimension_l2_no_lap'] = dot_l2
            if 'per_dimension_l2_grad_only' in self.enabled_modes:
                scores['per_dimension_l2_grad_only'] = grad_l2_over_classes
            if 'per_dimension_l2_score_only' in self.enabled_modes:
                scores['per_dimension_l2_score_only'] = score_l2

            # Std-balanced variants (use training stds estimated on ID training set)
            eps = 1e-8
            need_std = any(
                m in self.enabled_modes
                for m in ('per_dimension_l2_lap_only_std', 'per_dimension_l2_no_lap_std', 'per_dimension_l2_std_balanced')
            )
            if need_std:
                if (not self.training_stds) or ('per_dimension_l2_lap_only' not in self.training_stds) or ('per_dimension_l2_no_lap' not in self.training_stds):
                    raise RuntimeError(
                        "Std-normalized per_dimension_l2 modes require fitted training stds. "
                        "Call fit() before predict_all(), and ensure per_dimension_l2_lap_only and per_dimension_l2_no_lap are enabled."
                    )
                std_lap = self.training_stds['per_dimension_l2_lap_only'].clamp_min(eps)
                std_dot = self.training_stds['per_dimension_l2_no_lap'].clamp_min(eps)
                if 'per_dimension_l2_lap_only_std' in self.enabled_modes:
                    scores['per_dimension_l2_lap_only_std'] = lap_l2 / std_lap
                if 'per_dimension_l2_no_lap_std' in self.enabled_modes:
                    scores['per_dimension_l2_no_lap_std'] = dot_l2 / std_dot
                if 'per_dimension_l2_std_balanced' in self.enabled_modes:
                    scores['per_dimension_l2_std_balanced'] = (lap_l2 / std_lap) + (dot_l2 / std_dot)
        
        return scores
    
    def _compute_per_dimension_from_components(self, components: Dict[str, Tensor], aggregation: Literal['sum', 'l2'] = 'sum') -> Tensor:
        """
        Compute per-dimension Stein residuals from shared components.
        
        This is more complex as it requires per-class gradients and Laplacians.
        For now, we'll compute it separately, but could optimize further.
        """
        x_req = components['x_req']
        probs = components['probs']  # (B, num_classes)
        num_classes = probs.size(1)
        
        # Compute Laplacian for each class
        lap_per_dim = torch.zeros(probs.size(0), num_classes, device=self.device)
        grads_per_dim_list = []
        
        for c in range(num_classes):
            # Compute gradient for this class
            f_c = probs[:, c]  # (B,)
            grad_c = torch.autograd.grad(
                f_c.sum(), x_req, create_graph=True, retain_graph=True
            )[0]  # (B, ...)
            grads_per_dim_list.append(grad_c)
            
            # Compute Laplacian for this class
            if self.skip_laplacian:
                lap_c = torch.zeros(probs.size(0), device=self.device)
            else:
                # Prefer the softmax Laplacian approximation for ResNet classifiers (avoids second-order through MaxPool).
                if is_resnet_model(self.model):
                    k = torch.full((probs.size(0),), fill_value=c, device=probs.device, dtype=torch.long)
                    k = torch.clamp(k, 0, probs.size(1) - 1)
                    lap_c = softmax_laplacian_approx(x_req, self.model, k=k, topk=num_classes, device=self.device)
                else:
                    retain_for_next = (c < num_classes - 1) or (self.score_model is not None or self.score_function is not None)
                    lap_c = hutchinson_laplacian(
                        x_req, grad_c,
                        num_probes=self.num_probes,
                        device=self.device,
                        retain_graph_after=retain_for_next
                    )
            lap_per_dim[:, c] = lap_c
        
        # Stack gradients
        grads_per_dim = torch.stack(grads_per_dim_list, dim=1)  # (B, num_classes, ...)
        
        # Get score
        s = components['s']  # (B, D)
        
        # Compute s^T grad f_c for each class
        s_dot_grad_per_dim = torch.zeros(probs.size(0), num_classes, device=self.device)
        for c in range(num_classes):
            grad_c = grads_per_dim[:, c]  # (B, ...)
            grad_c_flat = grad_c.view(x_req.size(0), -1)  # (B, D)
            s_dot_grad_per_dim[:, c] = (s * grad_c_flat).sum(dim=1)  # (B,)
        
        # Per-dimension Stein residual
        r_per_dim = lap_per_dim + s_dot_grad_per_dim  # (B, num_classes)
        
        # Aggregate
        if aggregation == 'sum':
            residuals = r_per_dim.sum(dim=1)  # (B,)
        elif aggregation == 'l2':
            residuals = torch.norm(r_per_dim, p=2, dim=1)  # (B,)
        else:
            raise ValueError(f"Unknown per-dimension aggregation: {aggregation}")
        
        return residuals
    
    def _get_score(self, x_req: Tensor) -> Tensor:
        """
        Get score function s(x).
        Supports both trained score model and analytical score function.
        
        Returns:
            s: Score tensor of shape (N, D) where D is flattened input dimension
        """
        if self.score_function is not None:
            # Analytical score function
            s = self.score_function(x_req)  # (N, ...) or (N, D)
            # Flatten to (N, D)
            if s.dim() == 1:
                s = s.unsqueeze(1) if s.size(0) == x_req.size(0) else s
            return s.view(x_req.size(0), -1)
        elif self.score_model is not None:
            # Trained score model
            sigmas = self._get_sigmas()
            if sigmas is not None:
                return score_at_x(self.score_model, x_req, sigmas, self.device, use_sigma_min=True)
            else:
                s = self.score_model(x_req)  # (N, 1, H, W) or (N, ...)
                return s.view(x_req.size(0), -1)
        else:
            raise RuntimeError(
                "Neither score_model nor score_function is available. "
                "This should have been caught during initialization."
            )
    
    def _is_unet_score(self) -> bool:
        """Check if score model is UNetScore (annealed)"""
        if self.score_model is None:
            return False
        from src.models import UNetScore
        return isinstance(self.score_model, UNetScore)
    
    def _get_sigmas(self) -> Optional[Tensor]:
        """Get noise levels for UNetScore"""
        if self.sigmas is not None:
            return self.sigmas
        elif self._is_unet_score():
            if hasattr(self.score_model, 'sigmas'):
                return self.score_model.sigmas
            else:
                # Create default schedule
                import math
                n_levels = getattr(self.score_model, 'n_levels', 10)
                sigma_min = getattr(self.score_model, 'sigma_min', 0.01)
                sigma_max = getattr(self.score_model, 'sigma_max', 0.5)
                return torch.exp(
                    torch.linspace(math.log(sigma_max), math.log(sigma_min), steps=n_levels)
                ).to(self.device)
        return None
    
    def get_baseline(self, mode: str) -> Optional[Tensor]:
        """Get baseline for a specific mode."""
        return self.baselines.get(mode)
    
    def get_training_std(self, mode: str) -> Optional[Tensor]:
        """Get training std for a specific mode."""
        return self.training_stds.get(mode)
    
    def fit_features(self, x: Tensor, y: Tensor) -> Self:
        """Not applicable for Stein (requires full forward pass)."""
        raise NotImplementedError(
            "Stein factory detector requires full model forward pass, "
            "use fit() with DataLoader instead"
        )
    
    def predict_features(self, x: Tensor) -> Tensor:
        """Not applicable for Stein (requires full forward pass)."""
        raise NotImplementedError(
            "Stein factory detector requires full model forward pass, "
            "use predict() with input tensors instead"
        )
