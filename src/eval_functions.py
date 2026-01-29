"""
Evaluation functions for computing Stein residuals and score model outputs.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from .gradients import compute_grad_f, compute_grad_f_per_dim, hutchinson_laplacian, hutchinson_laplacian_per_dim, hutchinson_divergence


def score_at_x(score_model: nn.Module, x: torch.Tensor, sigmas: torch.Tensor, 
                device: torch.device, use_sigma_min: bool = True):
    """
    Helper function to get score s(x) from UNetScore model.
    For evaluation, we typically use the minimum sigma (low noise) to get the clean score.
    
    Args:
        score_model: UNetScore model
        x: input tensor (B, C, H, W)
        sigmas: tensor of noise levels from training schedule
        device: device to use
        use_sigma_min: if True, use minimum sigma (low noise); if False, use model's default
    
    Returns:
        s: score output (B, 1, H, W) flattened to (B, D)
    """
    score_model.eval()
    if use_sigma_min:
        # Use minimum sigma for low-noise evaluation (closer to true score)
        sigma_eval = sigmas.min().item()
        sigma_tensor = torch.tensor(sigma_eval, device=device).expand(x.size(0))
    else:
        # Use a small fixed sigma
        sigma_tensor = torch.tensor(0.01, device=device).expand(x.size(0))
    
    s = score_model(x, sigma_tensor)  # (B, 1, H, W)
    return s.view(x.size(0), -1)  # (B, D)


@torch.no_grad()
def compute_score_outputs(dataloader: DataLoader, score_model: Optional[nn.Module],
                          device: torch.device, max_batches=None, sigmas=None):
    """
    Compute score model outputs s(x) = grad log p(x) for each sample.
    
    Args:
        sigmas: Optional tensor of noise levels (for UNetScore models). If provided, uses score_at_x helper.
    
    Returns:
       score_norms: list of ||s(x)||^2 values (squared norm of score)
       score_outputs: list of score vectors (flattened)
       labels: list of true labels
    """
    if score_model is None:
        return None, None, None
    
    score_model.eval()
    score_norms = []
    score_outputs_list = []
    labs = []
    
    pbar = tqdm(dataloader, desc='Computing score outputs')
    for b_idx, (x, y) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)
        # Use score_at_x helper if sigmas provided (UNetScore), otherwise direct call (SmallScoreNet)
        if sigmas is not None:
            s_flat = score_at_x(score_model, x, sigmas, device, use_sigma_min=True)  # (B, D)
        else:
            s = score_model(x)  # (B, 1, H, W)
            s_flat = s.view(x.size(0), -1)  # (B, D)
        # Compute squared norm of score: ||s(x)||^2
        s_norm_sq = (s_flat ** 2).sum(dim=1)  # (B,)
        score_norms.extend(s_norm_sq.detach().cpu().numpy().tolist())
        score_outputs_list.append(s_flat.detach().cpu().numpy())
        labs.extend(y.detach().cpu().numpy().tolist())
        if max_batches is not None and b_idx+1 >= max_batches:
            break
    
    score_outputs = np.vstack(score_outputs_list) if score_outputs_list else np.array([])
    return np.array(score_norms), score_outputs, np.array(labs)


@torch.no_grad()
def compute_log_likelihood_score_model(dataloader: DataLoader, score_model: Optional[nn.Module],
                                      device: torch.device, max_batches=None, sigmas=None):
    """
    Compute log-likelihood of data under the score model.
    
    For a score model where s(x) = grad_x log p(x), we approximate log p(x) using:
    log p(x) ≈ -0.5 * ||s(x)||^2 - constant
    
    This is exact for isotropic Gaussian distributions. For general distributions,
    this is an approximation based on the energy-based model perspective.
    
    Args:
        sigmas: Optional tensor of noise levels (for UNetScore models). If provided, uses score_at_x helper.
    
    Returns:
       log_likelihoods: array of log-likelihood values per sample
       labels: array of true labels
    """
    if score_model is None:
        return None, None
    
    score_model.eval()
    log_likelihoods = []
    labs = []
    
    pbar = tqdm(dataloader, desc='Computing log-likelihoods')
    for b_idx, (x, y) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)
        # Use score_at_x helper if sigmas provided (UNetScore), otherwise direct call (SmallScoreNet)
        if sigmas is not None:
            s_flat = score_at_x(score_model, x, sigmas, device, use_sigma_min=True)  # (B, D)
        else:
            s = score_model(x)  # (B, 1, H, W)
            s_flat = s.view(x.size(0), -1)  # (B, D)
        
        # Approximate log p(x) ≈ -0.5 * ||s(x)||^2
        # This is exact for isotropic Gaussian, approximate for general distributions
        s_norm_sq = (s_flat ** 2).sum(dim=1)  # (B,)
        log_p = -0.5 * s_norm_sq  # (B,)
        log_likelihoods.extend(log_p.detach().cpu().numpy().tolist())
        labs.extend(y.detach().cpu().numpy().tolist())
        if max_batches is not None and b_idx+1 >= max_batches:
            break
    
    return np.array(log_likelihoods), np.array(labs)


@torch.no_grad()
def compute_stein_residuals(dataloader: DataLoader, f_model: nn.Module, score_model: Optional[nn.Module],
                            device: torch.device, num_probes=1, max_batches=None, sigmas=None):
    """
    Compute per-sample Stein residuals r(x) on dataset in dataloader.
    Uses: r(x) = Laplacian(f(x)) + s(x)^T grad f(x)
    If score_model is None -> fallback to simple baseline (use zero score) and warn.
    
    Args:
        sigmas: Optional tensor of noise levels (for UNetScore models). If provided, uses score_at_x helper.
    
    Returns:
       residues: list of r values
       f_vals: list of scalar f(x) used
       labels: list of true labels
    """
    f_model.eval()
    if score_model is not None:
        score_model.eval()
    residues = []
    fvals = []
    labs = []
    with torch.enable_grad():
        pbar = tqdm(dataloader, desc='Computing residuals')
        for b_idx, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            # compute grad f (requires create_graph to compute HVP later)
            grads, f_scalar, x_req = compute_grad_f(x, f_model, device)
            # compute Hutchinson Laplacian via HVPs
            lap = hutchinson_laplacian(x_req, grads, num_probes=num_probes, device=device)
            if score_model is not None:
                # Use score_at_x helper if sigmas provided (UNetScore), otherwise direct call (SmallScoreNet)
                if sigmas is not None:
                    s = score_at_x(score_model, x_req, sigmas, device, use_sigma_min=True)  # (B, D)
                else:
                    s = score_model(x_req).view(x.size(0), -1)  # (B, D)
                g_flat = grads.view(x.size(0), -1)                     # (B, D)
                s_dot_grad = (s * g_flat).sum(dim=1)                   # (B,)
            else:
                # fallback: score unknown -> we use zero vector (this just yields Laplacian)
                s_dot_grad = torch.zeros_like(lap)
            r = lap + s_dot_grad
            residues.extend(r.detach().cpu().numpy().tolist())
            fvals.extend(f_scalar.detach().cpu().numpy().tolist())
            labs.extend(y.detach().cpu().numpy().tolist())
            if max_batches is not None and b_idx+1 >= max_batches:
                break
    return np.array(residues), np.array(fvals), np.array(labs)


@torch.no_grad()
def compute_stein_residuals_classic(dataloader: DataLoader, f_model: nn.Module, score_model: Optional[nn.Module],
                                    device: torch.device, num_probes=1, max_batches=None):
    """
    Compute per-sample Stein residuals using the classic Stein operator.
    Uses: r(x) = s(x)^T grad f(x) + f(x) * div(s(x))
    This should be approximately zero-centered on in-distribution data when the score function
    is well-trained and matches the data distribution.
    If score_model is None -> fallback to simple baseline (use zero score) and warn.
    
    Args:
        num_probes: Number of Hutchinson probes for divergence estimation (default: 1)
    
    Returns:
       residues: list of r values
       f_vals: list of scalar f(x) used
       labels: list of true labels
    """
    f_model.eval()
    if score_model is not None:
        score_model.eval()
    residues = []
    fvals = []
    labs = []
    with torch.enable_grad():
        pbar = tqdm(dataloader, desc='Computing classic residuals')
        for b_idx, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            # compute grad f
            grads, f_scalar, x_req = compute_grad_f(x, f_model, device)
            
            if score_model is not None:
                # Compute score function s(x)
                x_req_score = x_req.clone().detach().requires_grad_(True)
                s = score_model(x_req_score)  # (B, 1, H, W)
                s_flat = s.view(x.size(0), -1)  # (B, D)
                
                # Compute divergence using Hutchinson estimator
                div_s = hutchinson_divergence(x_req_score, s_flat, num_probes=num_probes, device=device)
                
                # Compute s(x)^T grad f(x)
                g_flat = grads.view(x.size(0), -1)  # (B, D)
                s_dot_grad = (s_flat * g_flat).sum(dim=1)  # (B,)
                
                # Classic Stein operator: s(x)^T grad f(x) + f(x) * div(s(x))
                r = s_dot_grad + f_scalar * div_s
            else:
                # fallback: score unknown -> return zeros
                r = torch.zeros_like(f_scalar)
            
            residues.extend(r.detach().cpu().numpy().tolist())
            fvals.extend(f_scalar.detach().cpu().numpy().tolist())
            labs.extend(y.detach().cpu().numpy().tolist())
            if max_batches is not None and b_idx+1 >= max_batches:
                break
    return np.array(residues), np.array(fvals), np.array(labs)


@torch.no_grad()
def compute_stein_residuals_alternative(dataloader: DataLoader, f_model: nn.Module, score_model: Optional[nn.Module],
                                        device: torch.device, max_batches=None, sigmas=None):
    """
    Compute per-sample Stein residuals using alternative Stein operator.
    Uses: r(x) = ||f_c(x) * s(x) + grad_f_c(x)||_2 (L2 norm over spatial dimensions)
    where c = argmax(softmax(f(x))) is the predicted class, and f_c(x) is the softmax probability of class c.
    
    This is an exact computation (no Hutchinson approximation needed).
    If score_model is None -> fallback to simple baseline (use zero score) and warn.
    
    Args:
        sigmas: Optional tensor of noise levels (for UNetScore models). If provided, uses score_at_x helper.
    
    Returns:
       residues: list of r values (scalar per sample)
       f_vals: list of scalar f_c(x) probability values used
       labels: list of true labels
    """
    f_model.eval()
    if score_model is not None:
        score_model.eval()
    residues = []
    fvals = []
    labs = []
    with torch.enable_grad():
        pbar = tqdm(dataloader, desc='Computing alternative residuals')
        for b_idx, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            x_req = x.clone().detach().to(device).requires_grad_(True)
            
            # Get logits and determine argmax class
            logits = f_model(x_req)  # (B, num_classes)
            probs = F.softmax(logits, dim=1)
            pred_classes = torch.argmax(probs, dim=1)  # (B,)
            
            # Get softmax probability of argmax class (scalar per sample) - for consistency with existing approach
            f_c = probs[torch.arange(probs.size(0)), pred_classes]  # (B,)
            
            # Compute gradient of probability c w.r.t. input
            grad_f_c = torch.autograd.grad(f_c.sum(), x_req, create_graph=False)[0]  # (B, C, H, W)
            grad_f_c_flat = grad_f_c.view(x.size(0), -1)  # (B, D)
            
            if score_model is not None:
                # Get score function s(x)
                if sigmas is not None:
                    s = score_at_x(score_model, x_req, sigmas, device, use_sigma_min=True)  # (B, D)
                else:
                    s = score_model(x_req).view(x.size(0), -1)  # (B, D)
                
                # Compute: f_c(x) * s(x) + grad_f_c(x) (element-wise, then L2 norm over spatial dims)
                # f_c is (B,), s is (B, D), so we need to broadcast
                f_c_expanded = f_c.unsqueeze(1)  # (B, 1)
                stein_term = f_c_expanded * s + grad_f_c_flat  # (B, D)
                r = torch.norm(stein_term, dim=1)  # (B,) - L2 norm over spatial dimensions
            else:
                # fallback: score unknown -> just use grad_f_c L2 norm
                r = torch.norm(grad_f_c_flat, dim=1)  # (B,)
            
            residues.extend(r.detach().cpu().numpy().tolist())
            fvals.extend(f_c.detach().cpu().numpy().tolist())
            labs.extend(y.detach().cpu().numpy().tolist())
            if max_batches is not None and b_idx+1 >= max_batches:
                break
    return np.array(residues), np.array(fvals), np.array(labs)


def hutchinson_hessian_diagonal_per_pixel(x_req_grad: torch.Tensor, grads: torch.Tensor, 
                                           num_probes: int, device: torch.device, probe='rademacher'):
    """
    Compute Hessian diagonal elements per pixel using Hutchinson estimator.
    For each pixel (i,j), computes H_{i,j,i,j} = diagonal element of Hessian at that pixel.
    
    Args:
        x_req_grad: input tensor with requires_grad=True (B, C, H, W)
        grads: grad f wrt x, with create_graph=True (B, C, H, W)
        num_probes: number of random probes for estimation (per pixel)
        device: device to use
        probe: 'rademacher' or 'gaussian'
    
    Returns:
        hessian_diag: (B, C, H, W) tensor with diagonal Hessian elements per pixel
    """
    B, C, H, W = x_req_grad.shape
    hessian_diag = torch.zeros(B, C, H, W, device=device)
    
    # For each pixel position, we'll use Hutchinson estimator
    # We can do this more efficiently by processing all pixels, but using unit vectors
    # Actually, for Hutchinson with unit vectors, we need one HVP per pixel which is expensive
    # Instead, we'll use the standard Hutchinson approach but extract per-pixel values
    
    flat_dim = C * H * W
    for probe_idx in range(num_probes):
        if probe == 'rademacher':
            v = (torch.randint(0, 2, (B, flat_dim), device=device).float() * 2.0 - 1.0).requires_grad_(False)
        else:
            v = torch.randn(B, flat_dim, device=device).requires_grad_(False)
        
        v_reshaped = v.view(B, C, H, W)  # (B, C, H, W)
        inner = (grads * v_reshaped).view(B, -1).sum(dim=1)  # (B,)
        
        # Hessian-vector product
        retain = (probe_idx < num_probes - 1)
        Hv = torch.autograd.grad(inner.sum(), x_req_grad, retain_graph=retain)[0]  # (B, C, H, W)
        
        # For Hutchinson estimator of diagonal: E[v * Hv] approximates diagonal
        # We accumulate v * Hv element-wise
        hessian_diag += v_reshaped * Hv
    
    hessian_diag = hessian_diag / float(num_probes)
    return hessian_diag


@torch.no_grad()
def compute_stein_residuals_per_pixel(dataloader: DataLoader, f_model: nn.Module, score_model: Optional[nn.Module],
                                       device: torch.device, num_probes=1, max_batches=None, sigmas=None):
    """
    Compute per-pixel Stein residuals r_{i,j}(x) on dataset in dataloader.
    Uses: r_{i,j}(x) = H_{i,j,i,j} + s_{i,j} * grad_f_{i,j}
    where H_{i,j,i,j} is the diagonal element of the Hessian at pixel (i,j).
    
    If score_model is None -> fallback to simple baseline (use zero score) and warn.
    
    Args:
        sigmas: Optional tensor of noise levels (for UNetScore models). If provided, uses score_at_x helper.
    
    Returns:
       residues: (N, C, H, W) array of per-pixel Stein residuals
       f_vals: (N,) array of scalar f(x) used
       labels: (N,) array of true labels
    """
    f_model.eval()
    if score_model is not None:
        score_model.eval()
    residues_list = []
    fvals = []
    labs = []
    with torch.enable_grad():
        pbar = tqdm(dataloader, desc='Computing per-pixel residuals')
        for b_idx, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            # compute grad f (requires create_graph to compute HVP later)
            grads, f_scalar, x_req = compute_grad_f(x, f_model, device)
            
            # Compute Hessian diagonal per pixel using Hutchinson estimator
            hessian_diag = hutchinson_hessian_diagonal_per_pixel(x_req, grads, num_probes=num_probes, device=device)
            
            if score_model is not None:
                # Get score function s(x) - shape should match (B, C, H, W)
                if sigmas is not None:
                    s_flat = score_at_x(score_model, x_req, sigmas, device, use_sigma_min=True)  # (B, D) where D = C*H*W
                    s = s_flat.view(x.size(0), x.size(1), x.size(2), x.size(3))  # (B, C, H, W)
                else:
                    s = score_model(x_req)  # (B, 1, H, W) typically
                    # Ensure s matches input shape (B, C, H, W)
                    if s.shape[1] == 1 and x.size(1) == 1:
                        # Already correct shape for single channel
                        pass
                    elif s.shape[1] == 1 and x.size(1) > 1:
                        # Broadcast to match input channels
                        s = s.expand(-1, x.size(1), -1, -1)
                    elif s.shape != x.shape:
                        # Reshape if needed (shouldn't happen, but safety check)
                        s = s.view(x.size(0), x.size(1), x.size(2), x.size(3))
                
                # Per-pixel Stein residual: r_{i,j} = H_{i,j,i,j} + s_{i,j} * grad_f_{i,j}
                r_per_pixel = hessian_diag + s * grads  # (B, C, H, W)
            else:
                # fallback: score unknown -> just use Hessian diagonal
                r_per_pixel = hessian_diag  # (B, C, H, W)
            
            residues_list.append(r_per_pixel.detach().cpu().numpy())
            fvals.extend(f_scalar.detach().cpu().numpy().tolist())
            labs.extend(y.detach().cpu().numpy().tolist())
            if max_batches is not None and b_idx+1 >= max_batches:
                break
    
    # Stack all batches: (N, C, H, W)
    residues = np.concatenate(residues_list, axis=0) if residues_list else np.array([])
    return residues, np.array(fvals), np.array(labs)


@torch.no_grad()
def compute_stein_residuals_per_dim(dataloader: DataLoader, f_model: nn.Module, score_model: Optional[nn.Module],
                                    device: torch.device, num_probes=1, max_batches=None, sigmas=None,
                                    aggregation='sum'):
    """
    Compute per-sample Stein residuals for each output dimension (class), then aggregate.
    Uses: r_c(x) = Laplacian(f_c(x)) + s(x)^T grad f_c(x) for each class c
    Then aggregates per-dimension residuals using specified method.
    
    Args:
        sigmas: Optional tensor of noise levels (for UNetScore models). If provided, uses score_at_x helper.
        aggregation: How to aggregate per-dimension residuals. Options: 'sum' (default) or 'l2' (L2 norm)
    
    Returns:
       residues: list of aggregated r values (B,)
       f_vals: list of probability vectors (B, num_classes) - all class probabilities
       labels: list of true labels
       residues_per_dim: (N, num_classes) array of per-dimension residuals (optional, for debugging)
    """
    f_model.eval()
    if score_model is not None:
        score_model.eval()
    residues = []
    fvals_list = []
    labs = []
    residues_per_dim_list = []
    
    with torch.enable_grad():
        pbar = tqdm(dataloader, desc='Computing per-dim residuals')
        for b_idx, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            
            # Compute probabilities first (no gradients needed yet)
            x_req = x.clone().detach().to(device).requires_grad_(True)
            logits = f_model(x_req)
            probs = torch.nn.functional.softmax(logits, dim=1)  # (B, num_classes)
            num_classes = probs.size(1)
            
            # Compute Laplacian for each class separately to minimize memory usage
            # This is similar to the single-class version but repeated for each class
            lap_per_dim = torch.zeros(probs.size(0), num_classes, device=device)
            grads_per_dim_list = []
            
            for c in range(num_classes):
                # Compute gradient for this class only
                f_c = probs[:, c]  # (B,)
                # Always retain graph since we need it for Laplacian and potentially score computation
                grad_c = torch.autograd.grad(f_c.sum(), x_req, create_graph=True, retain_graph=True)[0]  # (B,C,H,W)
                grads_per_dim_list.append(grad_c)
                
                # Compute Laplacian for this class
                # Retain graph if we have more classes to process (for score computation later)
                retain_for_next = (c < num_classes - 1) or (score_model is not None)
                lap_c = hutchinson_laplacian(x_req, grad_c, num_probes=num_probes, device=device, retain_graph_after=retain_for_next)
                lap_per_dim[:, c] = lap_c
            
            # Stack gradients for score computation
            grads_per_dim = torch.stack(grads_per_dim_list, dim=1)  # (B, num_classes, C, H, W)
            
            if score_model is not None:
                # Get score function s(x)
                if sigmas is not None:
                    s = score_at_x(score_model, x_req, sigmas, device, use_sigma_min=True)  # (B, D)
                else:
                    s = score_model(x_req).view(x.size(0), -1)  # (B, D)
                
                # Compute s(x)^T grad f_c(x) for each class c
                B, num_classes = probs.shape
                s_dot_grad_per_dim = torch.zeros(B, num_classes, device=device)
                
                for c in range(num_classes):
                    grad_c = grads_per_dim[:, c, :, :, :]  # (B, C, H, W)
                    grad_c_flat = grad_c.view(B, -1)  # (B, D)
                    s_dot_grad_per_dim[:, c] = (s * grad_c_flat).sum(dim=1)  # (B,)
            else:
                # fallback: score unknown -> use zero
                s_dot_grad_per_dim = torch.zeros_like(lap_per_dim)
            
            # Per-dimension Stein residual: r_c(x) = Laplacian(f_c(x)) + s(x)^T grad f_c(x)
            r_per_dim = lap_per_dim + s_dot_grad_per_dim  # (B, num_classes)
            
            # Aggregate per-dimension residuals
            if aggregation == 'sum':
                r_aggregated = r_per_dim.sum(dim=1)  # (B,)
            elif aggregation == 'l2':
                r_aggregated = torch.norm(r_per_dim, p=2, dim=1)  # (B,) - L2 norm
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}. Must be 'sum' or 'l2'")
            
            residues.extend(r_aggregated.detach().cpu().numpy().tolist())
            fvals_list.append(probs.detach().cpu().numpy())
            residues_per_dim_list.append(r_per_dim.detach().cpu().numpy())
            labs.extend(y.detach().cpu().numpy().tolist())
            
            if max_batches is not None and b_idx+1 >= max_batches:
                break
    
    # Convert fvals_list to a single array (N, num_classes)
    fvals = np.vstack(fvals_list) if fvals_list else np.array([])
    residues_per_dim = np.vstack(residues_per_dim_list) if residues_per_dim_list else np.array([])
    
    return np.array(residues), fvals, np.array(labs), residues_per_dim

