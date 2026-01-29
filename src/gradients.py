"""
Gradient computation and Hutchinson estimators for Stein residual computation.
"""

from typing import Optional
import os

# region agent log
_AGENT_DEBUG_LOG_PATH = "/Users/michalkozyra/Developer/PhD/stein_shift_detection/.cursor/debug.log"
_AGENT_DEBUG_GRADIENTS = (os.environ.get("AGENT_DEBUG_GRADIENTS", "0") == "1")
_AGENT_DEBUG_GRADIENTS_MAX = int(os.environ.get("AGENT_DEBUG_GRADIENTS_MAX", "10"))
_agent_gradients_log_count = 0


def _agent_log_gradients(payload: dict) -> None:
    """
    Low-overhead debug logger for hot paths.
    Disabled by default; enable with AGENT_DEBUG_GRADIENTS=1 and cap with AGENT_DEBUG_GRADIENTS_MAX.
    """
    global _agent_gradients_log_count
    if not _AGENT_DEBUG_GRADIENTS:
        return
    if _agent_gradients_log_count >= _AGENT_DEBUG_GRADIENTS_MAX:
        return
    _agent_gradients_log_count += 1
    try:
        import json, time
        payload["timestamp"] = 0
        payload.setdefault("sessionId", "debug-session")
        with open(_AGENT_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# endregion
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_grad_f(x: torch.Tensor, f_model: nn.Module, device: torch.device):
    """
    Compute gradient of scalar output w.r.t. input images.
    For f_model that outputs logits, pick a scalar summary per image:
      - by default we use the predicted class probability (softmax).
    Returns:
      grads: same shape as x (B,C,H,W)
      scalar_outputs: (B,) values used as the scalar f(x)
      x_req: input tensor with requires_grad=True
    """
    # Ensure input has requires_grad and is on correct device
    if not x.requires_grad:
        x = x.clone().detach().to(device).requires_grad_(True)
    else:
        x = x.to(device)
    
    # Set model to eval mode (but keep gradients enabled for input)
    f_model.eval()
    
    # Compute forward pass with gradients enabled
    # Model parameters don't need gradients, but input does
    with torch.enable_grad():
        logits = f_model(x)                           # (B, num_classes)
        probs = F.softmax(logits, dim=1)              # (B, num_classes)
        # scalar: probability of predicted class (softmax)
        pred_classes = torch.argmax(probs, dim=1)
        f_vals = probs[torch.arange(probs.size(0)), pred_classes]   # (B,)
        grads = torch.autograd.grad(f_vals.sum(), x, create_graph=True)[0]  # (B,C,H,W)
    
    return grads, f_vals, x


def compute_grad_f_per_dim(x: torch.Tensor, f_model: nn.Module, device: torch.device):
    """
    Compute gradient of each output dimension (class probability) w.r.t. input images.
    For f_model that outputs logits, compute gradients for all class probabilities.
    Memory-efficient: computes and returns all gradients, but caller should use them immediately.
    
    Returns:
      grads_per_dim: (B, num_classes, C, H, W) - gradient for each class probability
      probs: (B, num_classes) - softmax probabilities for each class
      x_req: input tensor with requires_grad=True
    """
    x = x.clone().detach().to(device).requires_grad_(True)
    logits = f_model(x)                           # (B, num_classes)
    probs = F.softmax(logits, dim=1)              # (B, num_classes)
    num_classes = probs.size(1)
    B = probs.size(0)
    
    # Compute gradient for each class probability
    # Only retain graph for the last one to minimize memory
    grads_list = []
    for c in range(num_classes):
        f_c = probs[:, c]  # (B,) - probability of class c for each sample
        retain = (c < num_classes - 1)  # Only retain for intermediate classes
        grad_c = torch.autograd.grad(f_c.sum(), x, create_graph=True, retain_graph=retain)[0]  # (B,C,H,W)
        grads_list.append(grad_c)
    
    # Stack: (B, num_classes, C, H, W)
    grads_per_dim = torch.stack(grads_list, dim=1)
    return grads_per_dim, probs, x


def hutchinson_laplacian(x_req_grad: torch.Tensor, grads: torch.Tensor, num_probes: int, device: torch.device, probe='rademacher', retain_graph_after=False, create_graph=False):
    """
    Hutchinson estimator for Laplacian (trace of Hessian) per sample.
    Inputs:
        x_req_grad: input tensor with requires_grad=True (B,C,H,W)
        grads: grad f wrt x, with create_graph=True (B,C,H,W)
        num_probes: number of random probes for estimation
        device: device to use
        probe: 'rademacher' or 'gaussian'
        retain_graph_after: if True, retain graph even after last probe (for multi-class computation)
        create_graph: if True, create graph for backpropagation (needed during training)
    Returns:
        lap_est: (B,) tensor approximating Laplacian per sample
    """
    B = x_req_grad.shape[0]
    flat_dim = grads.view(B, -1).shape[1]
    lap = torch.zeros(B, device=device)
    for probe_idx in range(num_probes):
        if probe == 'rademacher':
            v = (torch.randint(0,2,(B, flat_dim), device=device).float()*2.0 - 1.0).requires_grad_(False)
        else:
            v = torch.randn(B, flat_dim, device=device).requires_grad_(False)
        v_reshaped = v.view_as(grads)  # shape (B,C,H,W)
        inner = (grads * v_reshaped).view(B, -1).sum(dim=1)  # (B,)
        # Hessian-vector product: grad(inner, x)
        # Retain graph if we have more probes OR if retain_graph_after is True OR if create_graph is True
        # (when create_graph=True, we need to retain graph to allow backpropagation through all probes)
        retain = (probe_idx < num_probes - 1) or retain_graph_after or create_graph
        Hv = torch.autograd.grad(inner.sum(), x_req_grad, retain_graph=retain, create_graph=create_graph)[0]  # (B,C,H,W)
        vtHv = (v_reshaped * Hv).view(B, -1).sum(dim=1)  # (B,)
        lap += vtHv
    lap = lap / float(num_probes)
    return lap


def hutchinson_laplacian_per_dim(x_req_grad: torch.Tensor, grads_per_dim: torch.Tensor, num_probes: int, device: torch.device, probe='rademacher'):
    """
    Hutchinson estimator for Laplacian (trace of Hessian) per sample and per output dimension.
    Memory-efficient: computes Laplacian for each class using the same approach as the single-class version.
    Inputs:
        x_req_grad: input tensor with requires_grad=True (B,C,H,W)
        grads_per_dim: grad f_c wrt x for each class c, with create_graph=True (B, num_classes, C, H, W)
        num_probes: number of random probes for estimation
        device: device to use
        probe: 'rademacher' or 'gaussian'
    Returns:
        lap_est: (B, num_classes) tensor approximating Laplacian per sample per class
    """
    B, num_classes = grads_per_dim.shape[0], grads_per_dim.shape[1]
    C, H, W = grads_per_dim.shape[2], grads_per_dim.shape[3], grads_per_dim.shape[4]
    flat_dim = C * H * W
    lap = torch.zeros(B, num_classes, device=device)
    
    # Compute Laplacian for each class separately (similar to single-class version)
    # This is more memory efficient than trying to compute all at once
    for c in range(num_classes):
        grad_c = grads_per_dim[:, c, :, :, :]  # (B, C, H, W)
        lap_c = torch.zeros(B, device=device)
        
        for probe_idx in range(num_probes):
            if probe == 'rademacher':
                v = (torch.randint(0,2,(B, flat_dim), device=device).float()*2.0 - 1.0).requires_grad_(False)
            else:
                v = torch.randn(B, flat_dim, device=device).requires_grad_(False)
            
            v_reshaped = v.view_as(grad_c)  # shape (B,C,H,W)
            inner = (grad_c * v_reshaped).view(B, -1).sum(dim=1)  # (B,)
            # Hessian-vector product: grad(inner, x)
            # Only retain graph if we have more probes or more classes
            retain = (probe_idx < num_probes - 1) or (c < num_classes - 1)
            Hv = torch.autograd.grad(inner.sum(), x_req_grad, retain_graph=retain, create_graph=False)[0]  # (B,C,H,W)
            vtHv = (v_reshaped * Hv).view(B, -1).sum(dim=1)  # (B,)
            lap_c += vtHv
        
        lap[:, c] = lap_c / float(num_probes)
    
    return lap


def hutchinson_divergence(x_req_grad: torch.Tensor, s_flat: torch.Tensor, num_probes: int, device: torch.device, probe='rademacher'):
    """
    Hutchinson estimator for divergence of score function: div(s) = trace(J)
    where J_ij = d/dx_j s_i is the Jacobian of s w.r.t. x.
    
    Inputs:
        x_req_grad: input tensor with requires_grad=True (B,C,H,W)
        s_flat: score function flattened to (B, D) where D = C*H*W
        num_probes: number of random probes for estimation
        device: device to use
        probe: 'rademacher' or 'gaussian'
    Returns:
        div_est: (B,) tensor approximating divergence per sample
    """
    B = x_req_grad.shape[0]
    D = s_flat.shape[1]
    div_est = torch.zeros(B, device=device)
    
    for probe_idx in range(num_probes):
        if probe == 'rademacher':
            v = (torch.randint(0, 2, (B, D), device=device).float() * 2.0 - 1.0).requires_grad_(False)
        else:
            v = torch.randn(B, D, device=device).requires_grad_(False)
        
        # Compute v^T s (dot product per sample)
        v_dot_s = (v * s_flat).sum(dim=1)  # (B,)
        
        # Jacobian-vector product: Jv = grad(v^T s, x)
        # Only retain graph if we have more probes to compute
        retain = (probe_idx < num_probes - 1)
        Jv = torch.autograd.grad(
            v_dot_s.sum(), 
            x_req_grad, 
            retain_graph=retain, 
            create_graph=False
        )[0]  # (B, C, H, W)
        
        # Flatten Jv to match s_flat dimensions
        Jv_flat = Jv.view(B, -1)  # (B, D)
        
        # v^T Jv estimates trace(J) = div(s)
        v_Jv = (v * Jv_flat).sum(dim=1)  # (B,)
        div_est += v_Jv
    
    div_est = div_est / float(num_probes)
    return div_est


def softmax_laplacian_approx(
    x_req: torch.Tensor,
    model: nn.Module,
    k: Optional[torch.Tensor] = None,
    topk: int = 5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute Laplacian of softmax output using only first-order autograd.
    
    This bypasses second-order backprop through the network by:
    1. Computing first-order gradients of selected logits
    2. Using analytical softmax Hessian
    3. Combining via Gram matrix
    
    Args:
        x_req: Input tensor with requires_grad=True (B, C, H, W)
        model: Classification model that outputs logits
        k: Selected class indices per sample (B,) OR a set of class indices per sample (B, M).
           If (B, M), this function returns the Laplacian of sum_m softmax(z)_{k_m} for each sample.
           If None, uses predicted class (argmax) per sample.
        topk: Number of top logits to use for approximation (default: 5)
        device: Device to use (auto-detect if None)
    
    Returns:
        lap: Laplacian per sample (B,)
    """
    if device is None:
        device = x_req.device
    
    model.eval()
    B = x_req.size(0)
    
    # Compute logits and probabilities
    with torch.enable_grad():
        logits = model(x_req)  # (B, K) where K = num_classes
        probs = F.softmax(logits, dim=1)  # (B, K)
        num_classes = logits.size(1)
        
        # Determine selected class indices per sample.
        # - If k is None: use predicted class (B,)
        # - If k is (B,): use that class
        # - If k is (B, M): treat it as a set and compute Laplacian of the sum of those probs
        if k is None:
            k = torch.argmax(probs, dim=1)  # (B,)
        else:
            k = k.to(device)
        
        # Clamp topk to valid range
        topk = min(topk, num_classes)
        
        # Get top-K logit indices per sample
        _, topk_indices = torch.topk(logits, k=topk, dim=1)  # (B, topk)
        
        # Compute first-order gradients for selected logits (optimized)
        # Strategy: Compute gradients for all top-K logits at once, then select per sample
        # This is more efficient than computing one sample at a time
        grads_list = []
        for i in range(topk):
            # Get logit indices for this position across all samples
            logit_indices = topk_indices[:, i]  # (B,)
            
            # Use advanced indexing to select logits per sample in batch
            batch_indices = torch.arange(B, device=device)  # (B,)
            selected_logits = logits[batch_indices, logit_indices]  # (B,) - selected logit per sample
            
            # Compute gradient - sum across batch is fine because we want gradient w.r.t. each input
            # Each element of selected_logits depends on a different x[b], so grad will be correct
            retain = (i < topk - 1)
            grad_i = torch.autograd.grad(
                selected_logits.sum(), x_req, retain_graph=retain, create_graph=False
            )[0]  # (B, C, H, W) - gradient for each sample
            
            grad_i_flat = grad_i.view(B, -1)  # (B, D)
            grads_list.append(grad_i_flat)
        
        # Build Gram matrix G: (B, topk, topk) - vectorized
        # G[b, i, j] = ⟨g_i, g_j⟩ for sample b
        # Stack all gradients: (topk, B, D)
        grads_stack = torch.stack(grads_list, dim=0)  # (topk, B, D)
        # Compute all pairwise dot products: (B, topk, topk)
        # Use einsum: i,j are topk indices (dim 0), b is batch index (dim 1), d is feature dimension (dim 2)
        # We want G[b, i, j] = sum_d grads_stack[i, b, d] * grads_stack[j, b, d]
        # einsum 'ibd,jbd->bij' gives (B, topk, topk) directly
        G = torch.einsum('ibd,jbd->bij', grads_stack, grads_stack)  # (B, topk, topk)
        
        # Compute softmax Hessian H: (B, topk, topk) - vectorized
        # H_ij = p_k * [(δ_ki - p_i) * (δ_kj - p_j) - δ_ij * p_j]
        
        # Get probabilities for top-K indices: (B, topk)
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, topk)  # (B, topk)
        p_topk = probs[batch_indices, topk_indices]  # (B, topk) - probabilities of top-K classes
        # region agent log
        _agent_log_gradients({
            "runId": "post-fix",
            "hypothesisId": "H1",
            "location": "src/gradients.py:softmax_laplacian_approx",
            "message": "softmax_laplacian_approx inputs",
            "data": {
                "B": int(B),
                "num_classes": int(num_classes),
                "topk": int(topk),
                "k_dim": int(k.dim()) if isinstance(k, torch.Tensor) else None,
                "k_shape": list(k.shape) if isinstance(k, torch.Tensor) else None,
            },
        })
        # endregion

        # Create delta_ij: (topk, topk) identity matrix, expanded to (B, topk, topk)
        delta_ij = torch.eye(topk, device=device).unsqueeze(0).expand(B, -1, -1)  # (B, topk, topk)

        # Expand p_i and p_j for broadcasting: (B, 1, topk) and (B, topk, 1)
        # NOTE: These must be defined before constructing H (bugfix for UnboundLocalError).
        p_i = p_topk.unsqueeze(1)  # (B, 1, topk)
        p_j = p_topk.unsqueeze(2)  # (B, topk, 1)
        
        # Build delta tensors and p_k depending on whether k is scalar or a set.
        # We will construct a single H (B, topk, topk) corresponding to either:
        #  - y_k (single class), or
        #  - sum_m y_{k_m} (set of classes), using linearity.
        if k.dim() == 1:
            # Single class per sample
            p_k = probs[torch.arange(B, device=device), k]  # (B,)
            p_k_expanded = p_k.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)

            # delta_ki: (B, topk)
            delta_ki = (topk_indices == k.unsqueeze(1)).float()
            delta_kj = delta_ki.unsqueeze(2)        # (B, topk, 1)
            delta_ki_expanded = delta_ki.unsqueeze(1)  # (B, 1, topk)

            # Hessian for single y_k
            H = p_k_expanded * (
                (delta_ki_expanded - p_i) * (delta_kj - p_j)
                - delta_ij * p_j
                + (p_i * p_j)
            )
        elif k.dim() == 2:
            # Set of classes per sample: k is (B, M)
            M = k.size(1)
            batch = torch.arange(B, device=device).unsqueeze(1).expand(B, M)
            p_km = probs[batch, k]  # (B, M)

            # delta_ki_m: (B, M, topk)
            delta_ki_m = (topk_indices.unsqueeze(1) == k.unsqueeze(2)).float()
            # Expand to (B, M, topk, topk)
            delta_ki_exp = delta_ki_m.unsqueeze(2)  # (B, M, 1, topk)
            delta_kj_exp = delta_ki_m.unsqueeze(3)  # (B, M, topk, 1)

            # Broadcast p_i, p_j to (B, 1, topk, topk) then to (B, M, topk, topk)
            p_i4 = p_i.unsqueeze(1)  # (B, 1, 1, topk)
            p_j4 = p_j.unsqueeze(1)  # (B, 1, topk, 1)
            delta_ij4 = delta_ij.unsqueeze(1)  # (B, 1, topk, topk)

            p_km4 = p_km.unsqueeze(2).unsqueeze(3)  # (B, M, 1, 1)

            Hm = p_km4 * (
                (delta_ki_exp - p_i4) * (delta_kj_exp - p_j4)
                - delta_ij4 * p_j4
                + (p_i4 * p_j4)
            )  # (B, M, topk, topk)

            H = Hm.sum(dim=1)  # (B, topk, topk)
        else:
            raise ValueError(f"Expected k to have dim 1 or 2, got k.dim()={k.dim()}")
        
        # Compute Laplacian: Δf_k = trace(H * G) = Σ_ij H_ij G_ij
        lap = (H * G).sum(dim=(1, 2))  # (B,)
        # region agent log
        _agent_log_gradients({
            "runId": "post-fix",
            "hypothesisId": "H1",
            "location": "src/gradients.py:softmax_laplacian_approx",
            "message": "softmax_laplacian_approx output",
            "data": {
                "lap_abs_mean": float(lap.abs().mean().detach().cpu()),
                "lap_std": float(lap.std().detach().cpu()),
            },
        })
        # endregion
    
    return lap

