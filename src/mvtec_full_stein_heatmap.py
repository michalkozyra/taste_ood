"""
MVTec-only full Stein heatmap for classification models (ResNet-safe Laplacian).

Computes a per-pixel contribution map for the **full Stein** functional:
    T(x) = Δ f(x) + s(x) · ∇ f(x)

We decompose across input dimensions d (pixel+channel):
    T(x) = Σ_d [ Δ_d f(x) + s_d(x) * ∂_d f(x) ]

Where:
- s(x) is a score field (B,3,H,W) (from diffusion score model)
- f(x) is a scalar test function derived from classifier logits (softmax prob of a chosen class)
- Δ f(x) is approximated *without second-order backprop through the network* using the same
  idea as `softmax_laplacian_approx`: compute gradients of top-K logits and combine them with
  the analytic softmax Hessian to form a Laplacian estimate.

This file is intentionally isolated (not used by existing CIFAR/adversarial benchmarks).
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

# region agent log
def _agent_log(payload: dict) -> None:
    try:
        import json  # noqa

        payload = dict(payload)
        payload["timestamp"] = 0
        log_path = "/Users/michalkozyra/Developer/PhD/stein_shift_detection/.cursor/debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def _agent_run_id() -> str:
    try:
        import os  # noqa

        return str(os.environ.get("AGENT_RUN_ID", "run1"))
    except Exception:
        return "run1"


# endregion agent log


def _nonlinearity(x: torch.Tensor, mode: Literal["raw", "abs", "square", "relu"]) -> torch.Tensor:
    if mode == "raw":
        return x
    if mode == "abs":
        return x.abs()
    if mode == "square":
        return x * x
    if mode == "relu":
        return torch.relu(x)
    raise ValueError(mode)


@torch.no_grad()
def _as_numpy_2d(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def full_stein_resnet_heatmap(
    x_req: torch.Tensor,
    logits: torch.Tensor,
    score: torch.Tensor,
    *,
    mode: Literal["stein_full", "no_lap", "lap_only", "grad_only", "score_only"] = "stein_full",
    class_mode: Literal["fixed", "predicted", "per_class_topk_l2"] = "fixed",
    fixed_class_idx: int = 0,
    topk: int = 5,
    class_topk: int = 5,
    nonlinearity: Literal["raw", "abs", "square", "relu"] = "abs",
) -> np.ndarray:
    """
    Args:
        x_req: (1,3,H,W) input in [0,1] with requires_grad=True (connected to logits)
        logits: (1,K) classifier logits computed from x_req
        score: (1,3,H,W) score field s(x) (from diffusion score model), connected to x_req
        class_mode: choose f(x) as fixed class prob or predicted class prob
        fixed_class_idx: used if class_mode='fixed'
        topk: number of top logits to use in Laplacian approximation
        nonlinearity: map post-processing so the output is a usable anomaly map
    Returns:
        heatmap: (H,W) float32
    """
    if x_req.ndim != 4 or x_req.size(0) != 1 or x_req.size(1) != 3:
        raise ValueError(f"x_req must be (1,3,H,W), got {tuple(x_req.shape)}")
    if logits.ndim != 2 or logits.size(0) != 1:
        raise ValueError(f"logits must be (1,K), got {tuple(logits.shape)}")
    if score.shape != x_req.shape:
        raise ValueError(f"score must match x_req shape, got score={tuple(score.shape)} x_req={tuple(x_req.shape)}")

    device = x_req.device
    B = 1
    K = int(logits.size(1))

    probs = F.softmax(logits, dim=1)  # (1,K)

    # Laplacian term per-dim using ResNet-safe approximation:
    # Build topk gradients of logits: g_i = ∇ z_{c_i}(x) for c_i in top-K logits per sample.
    topk_eff = int(min(max(int(topk), 1), K))
    _, topk_indices = torch.topk(logits, k=topk_eff, dim=1)  # (1, topk)
    topk_indices = topk_indices.to(device)

    grads_list = []
    for i in range(topk_eff):
        cls_i = int(topk_indices[0, i].item())
        z_i = logits[:, cls_i]  # (1,)
        # IMPORTANT: We will compute additional gradients later (e.g. ∇ f_k),
        # so we must keep the computation graph alive throughout this function.
        # If we set retain_graph=False on the last logit-grad, subsequent autograd.grad calls
        # will fail with "Trying to backward through the graph a second time".
        g_i = torch.autograd.grad(z_i.sum(), x_req, retain_graph=True, create_graph=False)[0]  # (1,3,H,W)
        grads_list.append(g_i.reshape(B, -1))  # (1,D)
    grads_stack = torch.stack(grads_list, dim=0)  # (topk,1,D)

    # Analytic softmax Hessian for y_k w.r.t. logits, restricted to top-K subspace.
    # This matches the corrected formulation used in src/gradients.py (includes +p_i p_j term).
    p_topk = probs[torch.arange(B, device=device).unsqueeze(1), topk_indices]  # (1, topk)
    delta_ij = torch.eye(topk_eff, device=device).unsqueeze(0)  # (1, topk, topk)
    p_i = p_topk.unsqueeze(1)  # (1, 1, topk)
    p_j = p_topk.unsqueeze(2)  # (1, topk, 1)

    def _lap_per_dim_for_class(k_idx: int) -> torch.Tensor:
        # Laplacian contribution per input dimension for y_k (softmax prob for class k_idx).
        k_tensor = torch.tensor([int(k_idx)], device=device, dtype=torch.long)  # (1,)
        p_k = probs[0, k_tensor]  # (1,)
        p_k_exp = p_k.view(1, 1, 1)
        delta_ki = (topk_indices == k_tensor.view(1, 1)).float()  # (1, topk)
        delta_kj = delta_ki.unsqueeze(2)  # (1, topk, 1)
        delta_ki_exp = delta_ki.unsqueeze(1)  # (1, 1, topk)
        H = p_k_exp * ((delta_ki_exp - p_i) * (delta_kj - p_j) - delta_ij * p_j + (p_i * p_j))  # (1, topk, topk)
        lap_per_flat = torch.einsum("ibd,bij,jbd->bd", grads_stack, H, grads_stack)  # (1,D)
        return lap_per_flat.reshape_as(x_req)  # (1,3,H,W)

    def _stein_per_pixel_from_terms(*, grad_f_k: torch.Tensor, lap_per_dim_k: torch.Tensor) -> torch.Tensor:
        # Compose ablation-specific per-pixel map (always return (1,H,W)).
        dot_per_dim_k = score * grad_f_k  # (1,3,H,W)
        m = str(mode)
        if m == "stein_full":
            return (lap_per_dim_k + dot_per_dim_k).sum(dim=1)
        if m == "no_lap":
            return dot_per_dim_k.sum(dim=1)
        if m == "lap_only":
            return lap_per_dim_k.sum(dim=1)
        if m == "grad_only":
            return torch.norm(grad_f_k, p=2, dim=1)
        if m == "score_only":
            return torch.norm(score, p=2, dim=1)
        raise ValueError(f"Unsupported mode: {mode}")

    # Choose class index/indices and compute per-pixel map.
    if class_mode == "per_class_topk_l2":
        kk = int(min(max(int(class_topk), 1), topk_eff))
        # Use top-k logits indices (ensures selected classes live in the same top-k subspace used for Laplacian approx).
        ks = [int(topk_indices[0, i].item()) for i in range(kk)]
        per_class_maps = []
        for i, k_idx in enumerate(ks):
            # f_k(x) = softmax prob of class k_idx
            f_k = probs[:, k_idx]  # (1,)
            retain = (i < len(ks) - 1)
            grad_f_k = torch.autograd.grad(f_k.sum(), x_req, create_graph=False, retain_graph=retain)[0]  # (1,3,H,W)
            lap_per_dim_k = _lap_per_dim_for_class(k_idx)
            per_class_maps.append(_stein_per_pixel_from_terms(grad_f_k=grad_f_k, lap_per_dim_k=lap_per_dim_k)[0])  # (H,W)
        stacked = torch.stack(per_class_maps, dim=0)  # (kk,H,W)
        stein_per_pixel = torch.norm(stacked, p=2, dim=0, keepdim=True)  # (1,H,W)
    else:
        if class_mode == "predicted":
            k = int(torch.argmax(probs, dim=1).item())
        else:
            k = int(fixed_class_idx)
        if not (0 <= k < K):
            raise ValueError(f"fixed_class_idx out of range: {k} (K={K})")
        f = probs[:, k]  # (1,)
        grad_f = torch.autograd.grad(f.sum(), x_req, create_graph=False, retain_graph=True)[0]  # (1,3,H,W)
        lap_per_dim = _lap_per_dim_for_class(k)
        stein_per_pixel = _stein_per_pixel_from_terms(grad_f_k=grad_f, lap_per_dim_k=lap_per_dim)  # (1,H,W)

    # Debug: check whether dot term is negligible vs Laplacian (can explain insensitivity to score scaling).
    try:
        n = int(getattr(full_stein_resnet_heatmap, "_agent_log_count", 0))
        if n < 6:
            import os  # local import for hot path safety

            ctx = {
                "mvtec_split": str(os.environ.get("AGENT_MVTEC_SPLIT", "")),
                "mvtec_defect_type": str(os.environ.get("AGENT_MVTEC_DEFECT_TYPE", "")),
                "score_noise_mode": str(os.environ.get("AGENT_SCORE_NOISE_MODE", "")),
                "score_noise_level": float(os.environ.get("AGENT_SCORE_NOISE_LEVEL", "nan")),
            }
            # For per_class_topk_l2 we no longer have single-class dot/lap terms. Log the final map stats only.
            # For other modes, keep the existing diagnostics with one representative class (already computed above).
            dot_pix = None
            lap_pix = None
            gf_pix = None
            if class_mode != "per_class_topk_l2":
                dot_pix = (score * grad_f).sum(dim=1)  # (1,H,W)
                lap_pix = lap_per_dim.sum(dim=1)  # (1,H,W)
                gf_pix = grad_f.abs().sum(dim=1)  # (1,H,W)
            sc_pix = score.abs().sum(dim=1)  # (1,H,W)
            # Pearson corr between |dot| and |grad f| to detect the "noise => grad_f_norm" effect.
            corr = None
            if class_mode != "per_class_topk_l2":
                a = dot_pix.abs().reshape(-1).float()
                b = gf_pix.reshape(-1).float()
                a = a - a.mean()
                b = b - b.mean()
                corr = float((a * b).mean().item() / ((a.std().item() * b.std().item()) + 1e-12))
            _agent_log(
                {
                    "sessionId": "debug-session",
                    "runId": _agent_run_id(),
                    "hypothesisId": "H11_score_noise_collapses_to_gradf_magnitude_under_abs",
                    "location": "src/mvtec_full_stein_heatmap.py:full_stein_resnet_heatmap:term_stats",
                    "message": "term stats (pre-nonlinearity): dot vs lap + linkage to |grad f|",
                    "data": {
                        **ctx,
                        "mode": str(mode),
                        "class_mode": str(class_mode),
                        "k": int(k) if class_mode != "per_class_topk_l2" else None,
                        "topk_eff": int(topk_eff),
                        "class_topk": int(class_topk),
                        "nonlinearity": str(nonlinearity),
                        "score_abs_mean": float(sc_pix.mean().item()),
                        "gradf_abs_mean": float(gf_pix.mean().item()) if gf_pix is not None else None,
                        "dot_abs_mean": float(dot_pix.abs().mean().item()) if dot_pix is not None else None,
                        "dot_std": float(dot_pix.std().item()) if dot_pix is not None else None,
                        "lap_abs_mean": float(lap_pix.abs().mean().item()) if lap_pix is not None else None,
                        "lap_std": float(lap_pix.std().item()) if lap_pix is not None else None,
                        "ratio_abs_mean_lap_over_dot": float((lap_pix.abs().mean() / (dot_pix.abs().mean() + 1e-12)).item()) if (lap_pix is not None and dot_pix is not None) else None,
                        "corr_absdot_vs_absgradf": float(corr) if corr is not None else None,
                    },
                }
            )
            setattr(full_stein_resnet_heatmap, "_agent_log_count", n + 1)
    except Exception:
        pass

    stein_per_pixel = _nonlinearity(stein_per_pixel, mode=nonlinearity)
    return _as_numpy_2d(stein_per_pixel[0])

