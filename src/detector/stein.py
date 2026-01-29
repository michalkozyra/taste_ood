"""
Stein-based Out-of-Distribution Detector for PyTorch-OOD.

Implements Stein residual computation for OOD detection:
- Full operator: Laplacian(f(x)) + s(x)^T grad f(x)
- First-order operator (L2): ||grad f(x) + s(x) * f(x)||_2
- First-order operator (sum): sum(grad f(x) + s(x) * f(x))
- Per-dimension operator: Per-class residuals with aggregation (classification only)

Supports:
- Classification and regression models
- Trained score models and analytical score functions
- Tabular and image data
"""

from typing import Optional, Literal, Callable, Tuple, TypeVar
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

from ..gradients import compute_grad_f, compute_grad_f_per_dim, hutchinson_laplacian, softmax_laplacian_approx
from ..eval_functions import score_at_x
from ..training import train_classifier, train_score_model, train_score_model_annealed
from ..utils import get_device, is_resnet_model

Self = TypeVar("Self")

# region agent log
_AGENT_DEBUG_LOG_PATH = "/Users/michalkozyra/Developer/PhD/stein_shift_detection/.cursor/debug.log"
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


class SteinDetector(Detector):
    """
    Stein-based OOD detector using Stein's identity.
    
    Computes Stein residuals for OOD detection. Supports multiple operator types,
    model types (classification/regression), and score function sources (trained/analytical).
    
    Higher residuals indicate higher likelihood of being OOD.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        score_model: Optional[nn.Module] = None,
        score_function: Optional[Callable[[Tensor], Tensor]] = None,
        device: Optional[torch.device] = None,
        # Model type
        model_type: Literal['classification', 'regression'] = 'classification',
        # Classification scalar f(x) choice (only used when model_type='classification')
        # NOTE: Using predicted_class_prob makes the scalar depend on argmax class selection,
        # which can be unstable / violate assumptions behind some Stein-based intuition.
        classification_scalar_mode: Literal['predicted_class_prob', 'fixed_class_prob', 'topk_sum_prob'] = 'predicted_class_prob',
        fixed_class_idx: int = 0,
        classification_topk: int = 1,
        # Stein computation parameters
        stein_operator_type: Literal['full', 'first_order', 'first_order_sum', 'per_dimension', 'first_order_per_dimension'] = 'full',
        num_probes: int = 1,
        aggregation: Literal['sum', 'l2', 'topk_l2'] = 'sum',  # per_dimension; also used by first_order_per_dimension ('sum'|'l2')
        skip_laplacian: bool = False,  # If True, assume Laplacian is zero (e.g., for ReLU networks)
        # Training parameters (optional, for fit())
        train_classifier: bool = False,
        train_score_model: bool = False,
        classifier_train_config: Optional[dict] = None,
        score_train_config: Optional[dict] = None,
        # Baseline computation
        compute_baseline: bool = True,
        baseline_subset_size: Optional[int] = None,  # None = use all training data
    ):
        """
        Initialize SteinDetector.
        
        Args:
            model: Model (f(x)) - can be None if training
            score_model: Trained score model (s(x) = grad log p(x)) - required if score_function is None
            score_function: Analytical score function s(x) -> Tensor - required if score_model is None
            device: Device to use (auto-detect if None)
            model_type: 'classification' or 'regression'
            stein_operator_type: 'full', 'first_order', 'first_order_sum', 'per_dimension', or 'first_order_per_dimension'
            num_probes: Number of Hutchinson probes for Laplacian (only for 'full' and 'per_dimension')
            aggregation: 'sum', 'l2', or 'topk_l2' - used for 'per_dimension' and 'first_order_per_dimension'
            skip_laplacian: If True, assume Laplacian is zero (e.g., for ReLU networks)
            train_classifier: If True, fit() will train classifier
            train_score_model: If True, fit() will train score model
            classifier_train_config: Dict with training params (epochs, lr, batch_size, etc.)
            score_train_config: Dict with training params (epochs, lr, batch_size, noise_sigma, etc.)
            compute_baseline: If True, compute training baseline for correction
            baseline_subset_size: Use subset of training data for baseline (faster, optional)
        """
        super().__init__()
        
        # Validate score function source - require at least one
        if score_model is None and score_function is None:
            raise ValueError(
                "Either score_model or score_function must be provided. "
                "Stein operator requires a score function s(x) = grad log p(x)."
            )
        
        # Validate class-wise operators
        if stein_operator_type in ('per_dimension', 'first_order_per_dimension') and model_type != 'classification':
            raise ValueError(f"{stein_operator_type} operator only supported for classification")
        
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
        self.stein_operator_type = stein_operator_type
        self.num_probes = num_probes
        self.aggregation = aggregation
        self.skip_laplacian = skip_laplacian
        self.train_classifier = train_classifier
        self.train_score_model = train_score_model
        self.classifier_train_config = classifier_train_config or {}
        self.score_train_config = score_train_config or {}
        self.compute_baseline = compute_baseline
        self.baseline_subset_size = baseline_subset_size
        
        # Will be set during fit()
        self.baseline: Optional[Tensor] = None
        self.training_std: Optional[Tensor] = None  # Training std for normalization
        self.sigmas: Optional[Tensor] = None
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        score_train_loader: Optional[DataLoader] = None,
    ) -> Self:
        """
        Fit the detector:
        1. Train classifier (if needed)
        2. Train score model (if needed)
        3. Compute training baseline
        
        Args:
            train_loader: Training data for classifier
            val_loader: Validation data for classifier (optional)
            score_train_loader: Training data for score model (optional, uses train_loader if None)
        
        Returns:
            self (for method chaining)
        """
        # 1. Train classifier if needed
        if self.train_classifier:
            if self.model is None:
                raise ValueError("model must be set if train_classifier=True")
            
            config = self.classifier_train_config
            self.model = train_classifier(
                self.model,
                train_loader,
                val_loader or train_loader,
                device=self.device,
                epochs=config.get('epochs', 5),
                lr=config.get('lr', 1e-3),
                checkpoint_path=config.get('checkpoint_path', None),
            )
        
        # 2. Train score model if needed
        if self.train_score_model:
            if self.score_model is None:
                raise ValueError("score_model must be set if train_score_model=True")
            
            config = self.score_train_config
            score_loader = score_train_loader or train_loader
            
            # Check if UNetScore (annealed) or SmallScoreNet (simple)
            if self._is_unet_score():
                # Use validation dataset if available for evaluation
                val_dataset = val_loader.dataset if val_loader is not None else None
                self.score_model, self.sigmas = train_score_model_annealed(
                    self.score_model,
                    score_loader.dataset,
                    device=self.device,
                    epochs=config.get('epochs', 50),
                    batch_size=config.get('batch_size', 128),
                    lr=config.get('lr', 2e-4),
                    n_levels=config.get('n_levels', 10),
                    sigma_min=config.get('sigma_min', 0.01),
                    sigma_max=config.get('sigma_max', 0.5),
                    ckpt=config.get('checkpoint_path', None),
                    val_dataset=val_dataset,
                    eval_every=config.get('eval_every', 5),
                )
            else:  # SmallScoreNet
                self.score_model = train_score_model(
                    self.score_model,
                    score_loader.dataset,
                    device=self.device,
                    epochs=config.get('epochs', 5),
                    batch_size=config.get('batch_size', 128),
                    lr=config.get('lr', 1e-3),
                    noise_sigma=config.get('noise_sigma', 0.2),
                    ckpt=config.get('checkpoint_path', None),
                )
        
        # 3. Compute training baseline
        if self.compute_baseline:
            self.baseline = self._compute_baseline(train_loader)
        
        return self
    
    def _compute_baseline(self, train_loader: DataLoader) -> Optional[Tensor]:
        """
        Compute Stein residual baseline on training data.
        
        Returns:
            baseline: Mean Stein residual on training data (scalar tensor)
                     None if compute_baseline=False
        """
        if not self.compute_baseline:
            return None
        
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
        
        # Compute Stein residuals on training data
        all_residuals = []
        
        self.model.eval()
        if self.score_model is not None:
            self.score_model.eval()
        
        with torch.enable_grad():
            for batch in tqdm(loader, desc='Computing Stein baseline'):
                # Handle both (x, y) and (x,) tuples
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device)
                x_req = x.clone().detach().requires_grad_(True)
                residuals = self._compute_stein_residuals_batch(x_req)
                all_residuals.append(residuals.detach().cpu())
        
        all_residuals_cat = torch.cat(all_residuals, dim=0)
        
        # Use robust statistics: median + 2*MAD instead of mean + 2*std
        # This is less sensitive to outliers
        from ..evaluation.robust_statistics import median_absolute_deviation, robust_baseline_and_threshold
        
        # Compute robust baseline and threshold
        baseline_robust, threshold_robust = robust_baseline_and_threshold(
            all_residuals_cat,
            method='median_mad',
            multiplier=2.0
        )
        
        # Keep mean/std for backward compatibility and reporting
        baseline_mean = all_residuals_cat.mean()
        baseline_std = all_residuals_cat.std()
        
        # Store training std for normalization (used in ensemble)
        self.training_std = baseline_std.to(self.device)
        
        print(f'Stein baseline computed (robust): {baseline_robust.item():.6e} (MAD: {median_absolute_deviation(all_residuals_cat).item():.6e})')
        print(f'  Traditional: mean={baseline_mean.item():.6e}, std={baseline_std.item():.6e}')
        print(f'  Raw residuals range: [{all_residuals_cat.min().item():.6e}, {all_residuals_cat.max().item():.6e}]')
        print(f'  Robust threshold (median + 2*MAD): {threshold_robust.item():.6e}')
        
        # Store robust baseline (use median as baseline, threshold for reference)
        # We'll use median as the baseline for centering
        return baseline_robust.to(self.device)
    
    def predict(self, x: Tensor) -> Tensor:
        """
        Compute Stein residual scores for input batch.
        
        Args:
            x: Input tensor of shape (N, ...) - can be images (N, C, H, W) or tabular (N, features)
        
        Returns:
            scores: Stein residual scores of shape (N,)
                   Higher scores = more likely OOD
        """
        if self.model is None:
            raise ModelNotSetException("Model must be set or trained")
        
        # Set model to eval mode (but keep gradients enabled for input)
        self.model.eval()
        
        # Ensure on correct device and enable gradients
        x = x.to(self.device)
        x_req = x.clone().detach().requires_grad_(True)
        
        # Compute Stein residuals with gradients enabled
        # Note: model.eval() is fine, but we need gradients w.r.t. input
        with torch.enable_grad():
            residuals = self._compute_stein_residuals_batch(x_req)
        
        # Return raw residuals (baseline correction and transformation handled in evaluation)
        # The evaluation code will use percentile-based two-sided test for non-symmetric distributions
        return residuals
    
    def _compute_stein_residuals_batch(self, x_req: Tensor) -> Tensor:
        """
        Compute Stein residuals for a batch.
        
        Args:
            x_req: Input tensor with requires_grad=True (N, ...)
        
        Returns:
            residuals: Stein residuals of shape (N,)
        """
        self.model.eval()
        if self.score_model is not None:
            self.score_model.eval()
        
        if self.stein_operator_type == 'per_dimension':
            if self.model_type != 'classification':
                raise ValueError("per_dimension operator only supported for classification")
            residuals = self._compute_per_dimension_residuals(x_req)
        elif self.stein_operator_type == 'first_order_per_dimension':
            if self.model_type != 'classification':
                raise ValueError("first_order_per_dimension only supported for classification")
            residuals = self._compute_first_order_per_dimension_residuals(x_req)
        elif self.stein_operator_type == 'first_order':
            residuals = self._compute_first_order_residuals(x_req)
        elif self.stein_operator_type == 'first_order_sum':
            residuals = self._compute_first_order_sum_residuals(x_req)
        else:  # 'full'
            residuals = self._compute_full_residuals(x_req)
        
        return residuals
    
    def _compute_grad_f(self, x_req: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute gradient of f w.r.t. input.
        Handles both classification and regression.
        
        Returns:
            grads: Gradient tensor (same shape as x_req)
            f_scalar: Scalar function value (B,)
            x_req: Input with requires_grad
        """
        if self.model_type == 'classification':
            # Compute gradient for chosen scalar function f(x).
            #
            # predicted_class_prob: f(x)=softmax(logits)_argmax  (data-dependent class selection)
            # fixed_class_prob:     f(x)=softmax(logits)_fixed_class_idx (fixed test function)
            # topk_sum_prob:        f(x)=sum_{i in topK} softmax(logits)_i (data-dependent set selection)
            if self.classification_scalar_mode == 'predicted_class_prob':
                return compute_grad_f(x_req, self.model, self.device)
            elif self.classification_scalar_mode == 'fixed_class_prob':
                # Inline version of compute_grad_f, but with a fixed class index.
                if not x_req.requires_grad:
                    x_req = x_req.clone().detach().to(self.device).requires_grad_(True)
                else:
                    x_req = x_req.to(self.device)
                self.model.eval()
                with torch.enable_grad():
                    logits = self.model(x_req)  # (B, K)
                    probs = F.softmax(logits, dim=1)
                    k = torch.full(
                        (probs.size(0),),
                        fill_value=self.fixed_class_idx,
                        device=probs.device,
                        dtype=torch.long,
                    )
                    k = torch.clamp(k, 0, probs.size(1) - 1)
                    f_vals = probs[torch.arange(probs.size(0), device=probs.device), k]  # (B,)
                    grads = torch.autograd.grad(f_vals.sum(), x_req, create_graph=True)[0]
                return grads, f_vals, x_req
            elif self.classification_scalar_mode == 'topk_sum_prob':
                if not x_req.requires_grad:
                    x_req = x_req.clone().detach().to(self.device).requires_grad_(True)
                else:
                    x_req = x_req.to(self.device)
                self.model.eval()
                with torch.enable_grad():
                    logits = self.model(x_req)  # (B, K)
                    probs = F.softmax(logits, dim=1)
                    K = probs.size(1)
                    k = min(self.classification_topk, K)
                    topk_idx = torch.topk(probs, k=k, dim=1).indices  # (B, k)
                    f_vals = probs.gather(1, topk_idx).sum(dim=1)  # (B,)
                    grads = torch.autograd.grad(f_vals.sum(), x_req, create_graph=True)[0]
                return grads, f_vals, x_req
            else:
                raise ValueError(f"Unknown classification_scalar_mode={self.classification_scalar_mode!r}")
        else:  # regression
            # For regression: f(x) is model output directly
            # Ensure x_req has requires_grad (should already, but double-check)
            if not x_req.requires_grad:
                x_req = x_req.requires_grad_(True)
            f_scalar = self.model(x_req)  # (B,)
            # Compute gradient with create_graph=True for Laplacian computation
            grads = torch.autograd.grad(
                f_scalar.sum(), x_req, create_graph=True, retain_graph=True
            )[0]  # (B, ...)
            return grads, f_scalar, x_req
    
    def _get_score(self, x_req: Tensor) -> Tensor:
        """
        Get score function s(x).
        Supports both trained score model and analytical score function.
        
        Returns:
            s: Score tensor of shape (N, D) where D is flattened input dimension
        """
        if self.score_function is not None:
            # Analytical score function (e.g., s(x) = -x for standard Gaussian)
            s = self.score_function(x_req)  # (N, ...) or (N, D)
            # Flatten to (N, D)
            if s.dim() == 1:
                # Already flat or needs expansion
                s = s.unsqueeze(1) if s.size(0) == x_req.size(0) else s
            return s.view(x_req.size(0), -1)
        elif self.score_model is not None:
            # Trained score model
            sigmas = self._get_sigmas()
            if sigmas is not None:
                s = score_at_x(self.score_model, x_req, sigmas, self.device, use_sigma_min=True)
                return s
            else:
                s = self.score_model(x_req)  # (N, 1, H, W) or (N, ...)
                return s.view(x_req.size(0), -1)
        else:
            # This should never happen due to validation in __init__, but keep for safety
            raise RuntimeError(
                "Neither score_model nor score_function is available. "
                "This should have been caught during initialization."
            )
    
    def _compute_full_residuals(self, x_req: Tensor) -> Tensor:
        """Compute full Stein operator: Laplacian + score^T grad"""
        # Compute grad f (handles both classification and regression)
        # Note: compute_grad_f returns a new x_req, so we use that
        grads, f_scalar, x_req_grad = self._compute_grad_f(x_req)
        g_flat = grads.view(x_req_grad.size(0), -1)  # (N, D)
        
        # Compute Laplacian (use x_req_grad returned from _compute_grad_f)
        lap_method = "unknown"
        if self.skip_laplacian:
            # Assume Laplacian is zero (e.g., for ReLU networks)
            lap = torch.zeros(x_req_grad.size(0), device=self.device)
            lap_method = "skip_laplacian"
        elif self.model_type == 'classification' and is_resnet_model(self.model):
            # Use softmax Laplacian approximation for ResNet models
            # This bypasses second-order backprop which fails with MaxPool layers
            # Get class indices k used by scalar f(x) for Laplacian of softmax output.
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
                    # (B, kk) class indices for the top-kk probs
                    k = torch.topk(probs, k=kk, dim=1).indices
                else:
                    k = torch.argmax(probs, dim=1)  # (B,)
                num_classes = logits.size(1)  # Get number of classes

            lap = softmax_laplacian_approx(
                x_req_grad, self.model, k=k, topk=num_classes, device=self.device
            )
            lap_method = "softmax_laplacian_approx"
        else:
            # Use Hutchinson estimator for non-ResNet models or regression
            lap = hutchinson_laplacian(
                x_req_grad, grads,
                num_probes=self.num_probes,
                device=self.device
            )
            lap_method = "hutchinson_laplacian"
        
        # Get score and compute score dot grad (use x_req_grad for consistency)
        s = self._get_score(x_req_grad)  # (N, D)
        s_dot_grad = (s * g_flat).sum(dim=1)  # (N,)
        
        # Full Stein residual
        residuals = lap + s_dot_grad

        # region agent log
        _agent_log(
            run_id="perf-investigation",
            hypothesis_id="TERM",
            location="src/detector/stein.py:_compute_full_residuals",
            message="Stein full term summary",
            data={
                "classification_scalar_mode": getattr(self, "classification_scalar_mode", None),
                "fixed_class_idx": int(getattr(self, "fixed_class_idx", -1)),
                "lap_method": lap_method,
                "k_unique_count": int(torch.unique(k).numel()) if ("k" in locals()) else None,
                "lap_std": float(lap.std().detach().cpu()),
                "s_dot_grad_std": float(s_dot_grad.std().detach().cpu()),
                "residual_std": float(residuals.std().detach().cpu()),
            },
        )
        # endregion
        
        return residuals
    
    def _compute_first_order_residuals(self, x_req: Tensor) -> Tensor:
        """Compute first-order Stein operator: ||grad f + score * f||_2"""
        # Compute grad f and f_scalar
        # Note: compute_grad_f returns a new x_req, so we use that
        grads, f_scalar, x_req_grad = self._compute_grad_f(x_req)  # f_scalar: (N,)
        grad_f_flat = grads.view(x_req_grad.size(0), -1)  # (N, D)
        
        # Get score (use x_req_grad for consistency)
        s = self._get_score(x_req_grad)  # (N, D)
        
        # First-order operator: f * s + grad_f
        f_expanded = f_scalar.unsqueeze(1)  # (N, 1)
        stein_term = f_expanded * s + grad_f_flat  # (N, D)
        residuals = torch.norm(stein_term, dim=1)  # (N,) - L2 norm
        
        return residuals
    
    def _compute_first_order_sum_residuals(self, x_req: Tensor) -> Tensor:
        """Compute first-order Stein operator with sum aggregation: sum(grad f + score * f)"""
        # Compute grad f and f_scalar
        # Note: compute_grad_f returns a new x_req, so we use that
        grads, f_scalar, x_req_grad = self._compute_grad_f(x_req)  # f_scalar: (N,)
        grad_f_flat = grads.view(x_req_grad.size(0), -1)  # (N, D)
        
        # Get score (use x_req_grad for consistency)
        s = self._get_score(x_req_grad)  # (N, D)
        
        # First-order operator: f * s + grad_f
        f_expanded = f_scalar.unsqueeze(1)  # (N, 1)
        stein_term = f_expanded * s + grad_f_flat  # (N, D)
        residuals = stein_term.sum(dim=1)  # (N,) - Sum aggregation instead of L2 norm
        
        return residuals

    def _compute_first_order_per_dimension_residuals(self, x_req: Tensor) -> Tensor:
        """
        Expensive all-classes first-order analogue of stein_per_dimension_l2.

        For each class c:
          v_c(x) = ∇_x p_c(x) + s(x) * p_c(x)   (vector in input space)
          a_c(x) = ||v_c(x)||_2                (scalar)
        Aggregate across classes:
          score(x) = ||a(x)||_2  (or sum if aggregation='sum')
        """
        if self.model_type != 'classification':
            raise ValueError("first_order_per_dimension only supported for classification")

        logits = self.model(x_req)  # (N, K)
        probs = F.softmax(logits, dim=1)  # (N, K)
        num_classes = probs.size(1)

        # Score function once
        s = self._get_score(x_req)  # (N, D)

        a_per_class = torch.zeros(probs.size(0), num_classes, device=self.device)
        for c in range(num_classes):
            f_c = probs[:, c]  # (N,)
            grad_c = torch.autograd.grad(
                f_c.sum(), x_req, create_graph=True, retain_graph=True
            )[0]  # (N, ...)
            grad_c_flat = grad_c.view(x_req.size(0), -1)  # (N, D)
            v_c = grad_c_flat + f_c.unsqueeze(1) * s  # (N, D)
            a_per_class[:, c] = torch.norm(v_c, p=2, dim=1)  # (N,)

        if self.aggregation == 'l2':
            residuals = torch.norm(a_per_class, p=2, dim=1)  # (N,)
        elif self.aggregation == 'sum':
            residuals = a_per_class.sum(dim=1)  # (N,)
        else:
            raise ValueError(f"aggregation={self.aggregation!r} not supported for first_order_per_dimension (use 'l2' or 'sum').")

        return residuals
    
    def _compute_per_dimension_residuals(self, x_req: Tensor) -> Tensor:
        """
        Compute per-dimension Stein residuals (classification only).
        For each class c: r_c(x) = Laplacian(f_c(x)) + s(x)^T grad f_c(x)
        Then aggregate using specified method.
        
        Note: This follows the same approach as compute_stein_residuals_per_dim in eval_functions.py
        """
        if self.model_type != 'classification':
            raise ValueError("per_dimension operator only supported for classification")
        
        # Get logits
        logits = self.model(x_req)  # (N, num_classes)
        num_classes = logits.size(1)
        
        # Compute Laplacian for each class
        lap_per_dim = torch.zeros(logits.size(0), num_classes, device=self.device)
        grads_per_dim_list = []
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)  # (N, num_classes)
        
        for c in range(num_classes):
            # Compute gradient for this class only (using probability, not logit)
            f_c = probs[:, c]  # (N,)
            # Always retain graph since we need it for Laplacian and potentially score computation
            grad_c = torch.autograd.grad(
                f_c.sum(), x_req, create_graph=True, retain_graph=True
            )[0]  # (N, ...)
            grads_per_dim_list.append(grad_c)
            
            # Compute Laplacian for this class
            if self.skip_laplacian:
                # Assume Laplacian is zero (e.g., for ReLU networks)
                lap_c = torch.zeros(logits.size(0), device=self.device)
            else:
                # For ResNet-style models, avoid second-order autograd through MaxPool by using
                # the analytical softmax Laplacian approximation (same motivation as in full mode).
                if self.model_type == 'classification' and is_resnet_model(self.model):
                    k = torch.full(
                        (logits.size(0),),
                        fill_value=int(c),
                        device=logits.device,
                        dtype=torch.long,
                    )
                    k = torch.clamp(k, 0, num_classes - 1)
                    lap_c = softmax_laplacian_approx(
                        x_req, self.model, k=k, topk=num_classes, device=self.device
                    )
                else:
                    # Retain graph if we have more classes to process (for score computation later)
                    retain_for_next = (c < num_classes - 1) or (self.score_model is not None or self.score_function is not None)
                    lap_c = hutchinson_laplacian(
                        x_req, grad_c,
                        num_probes=self.num_probes,
                        device=self.device,
                        retain_graph_after=retain_for_next
                    )
            lap_per_dim[:, c] = lap_c
        
        # Stack gradients for score computation
        grads_per_dim = torch.stack(grads_per_dim_list, dim=1)  # (N, num_classes, ...)
        
        # Get score
        s = self._get_score(x_req)  # (N, D)
        
        # Compute s^T grad f_c for each class
        s_dot_grad_per_dim = torch.zeros(logits.size(0), num_classes, device=self.device)
        for c in range(num_classes):
            grad_c = grads_per_dim[:, c]  # (N, ...)
            grad_c_flat = grad_c.view(x_req.size(0), -1)  # (N, D)
            s_dot_grad_per_dim[:, c] = (s * grad_c_flat).sum(dim=1)  # (N,)
        
        # Per-dimension Stein residual
        r_per_dim = lap_per_dim + s_dot_grad_per_dim  # (N, num_classes)
        
        # Aggregate
        if self.aggregation == 'sum':
            residuals = r_per_dim.sum(dim=1)  # (N,)
        elif self.aggregation == 'l2':
            residuals = torch.norm(r_per_dim, p=2, dim=1)  # (N,)
        elif self.aggregation == 'topk_l2':
            # L2 aggregation over the top-K predicted classes (by softmax probability) per sample.
            # Intended to match stein_per_dimension_l2 behavior, restricted to a top-K subset.
            kk = min(int(self.classification_topk), int(num_classes))
            topk_idx = torch.topk(probs, k=kk, dim=1).indices  # (N, kk)
            r_topk = r_per_dim.gather(dim=1, index=topk_idx)  # (N, kk)
            residuals = torch.norm(r_topk, p=2, dim=1)  # (N,)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # region agent log
        _agent_log(
            run_id="perf-investigation",
            hypothesis_id="TERM",
            location="src/detector/stein.py:_compute_per_dimension_residuals",
            message="Stein per-dim term summary",
            data={
                "aggregation": getattr(self, "aggregation", None),
                "lap_method": "softmax_laplacian_approx" if (not self.skip_laplacian and is_resnet_model(self.model)) else ("skip_laplacian" if self.skip_laplacian else "hutchinson_laplacian"),
                "r_per_dim_abs_mean": float(r_per_dim.abs().mean().detach().cpu()),
                "residual_std": float(residuals.std().detach().cpu()),
            },
        )
        # endregion
        
        return residuals
    
    def _is_unet_score(self) -> bool:
        """Check if score model is UNetScore (annealed)"""
        if self.score_model is None:
            return False
        # Check by class name (most reliable)
        from src.models import UNetScore
        return isinstance(self.score_model, UNetScore)
    
    def _get_sigmas(self) -> Optional[Tensor]:
        """Get noise levels for UNetScore"""
        if self.sigmas is not None:
            return self.sigmas
        elif self._is_unet_score():
            # Get sigmas from model or create default schedule
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
    
    def fit_features(self, x: Tensor, y: Tensor) -> Self:
        """
        Not applicable for Stein (requires full forward pass).
        """
        raise NotImplementedError(
            "Stein detector requires full model forward pass, "
            "use fit() with DataLoader instead"
        )
    
    def predict_features(self, x: Tensor) -> Tensor:
        """
        Not applicable for Stein (requires full forward pass).
        """
        raise NotImplementedError(
            "Stein detector requires full model forward pass, "
            "use predict() with input tensors instead"
        )

