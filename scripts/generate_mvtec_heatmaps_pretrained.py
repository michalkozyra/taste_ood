"""
Generate per-pixel anomaly heatmaps for MVTec AD using:
- Pretrained ImageNet classifier (ResNet)
- Pretrained diffusion score model (global prior) at 256×256

This script is intentionally opt-in and does not touch existing benchmark runners.

It saves heatmaps as .npy files in the layout expected by scripts/evaluate_mvtec_localization.py:
  <heatmaps-dir>/<category>/test/<defect_type>/<image_stem>.npy

By default, this script writes a *placeholder* heatmap (diffusion score norm) to validate
the full I/O + resizing + evaluation pipeline. You can replace the heatmap computation by
providing --heatmap-fn as a dotted-path callable that returns a 2D numpy array heatmap.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

import torch

# Ensure repo root is on sys.path so `import src...` works when running as a script.
# This mirrors scripts/benchmark_ood_evaluation.py behavior.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mvtec_ad import ALL_CATEGORIES, discover_mvtec_samples, subsample_mvtec_samples
from src.pretrained_classifier import PretrainedImageNetClassifier
from src.score_models.diffusion_score_256 import DiffusionScore256

def _to_tensor01(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] != 3:
        raise ValueError(f"Expected 3 channels, got shape={arr.shape}")
    t = torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)
    return t


def _resize_pil(img: Image.Image, size: int = 256) -> Image.Image:
    return img.resize((int(size), int(size)), resample=Image.BILINEAR)


def _load_callable(dotted: str) -> Callable:
    if ":" in dotted:
        mod, fn = dotted.split(":", 1)
    else:
        mod, fn = dotted.rsplit(".", 1)
    m = importlib.import_module(mod)
    f = getattr(m, fn)
    if not callable(f):
        raise TypeError(f"{dotted} is not callable")
    return f


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mvtec-root", type=str, required=True)
    p.add_argument("--heatmaps-dir", type=str, required=True)
    p.add_argument("--categories", type=str, default="*", help="Comma-separated categories or '*' for all.")
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Which MVTec split to generate heatmaps for. For train reference heatmaps, use --split train --only-good.",
    )
    p.add_argument(
        "--only-good",
        action="store_true",
        help="If set, only generate heatmaps for defect_type='good' within the selected split.",
    )
    p.add_argument(
        "--mvtec-subsample-frac",
        type=float,
        default=1.0,
        help="If < 1, keep only this fraction of MVTec samples. "
        "For split=train this is per-category; for split=test it's per (category, defect_type). "
        "A minimum of 1 sample per group is kept (if the group is non-empty).",
    )
    p.add_argument("--mvtec-subsample-seed", type=int, default=0, help="Seed for deterministic MVTec subsampling.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--classifier", type=str, default="resnet50", choices=["resnet50", "resnet18"])
    # Explicit selection (zero-shot vs fine-tuned)
    p.add_argument("--classifier-mode", type=str, default="zero_shot", choices=["zero_shot", "finetuned"])
    p.add_argument("--classifier-checkpoint", type=str, default="", help="Path to fine-tuned classifier checkpoint (.pt). Required if --classifier-mode=finetuned.")
    p.add_argument(
        "--diffusion-model-id",
        type=str,
        default="google/ddpm-ema-celebahq-256",
        help="HuggingFace diffusers repo id for a 256×256 DDPM.",
    )
    p.add_argument("--diffusion-mode", type=str, default="zero_shot", choices=["zero_shot", "finetuned"])
    p.add_argument("--diffusion-model-path", type=str, default="", help="HF id or local diffusers dir. Required if --diffusion-mode=finetuned.")
    p.add_argument(
        "--diffusion-timestep",
        type=int,
        default=500,
        help="DDPM timestep used for score extraction. Default=500 (moderate noise).",
    )
    p.add_argument(
        "--diffusion-score-denom",
        type=str,
        default="sigma",
        choices=["sigma_sq", "sigma"],
        help="How to convert epsilon prediction to a score field. "
        "sigma_sq is legacy (-eps/sigma^2) and can blow up at small t; "
        "sigma is recommended (-eps/sigma) for numerical stability.",
    )
    p.add_argument(
        "--diffusion-add-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled (default), generate x_t via scheduler.add_noise(x0, noise, t) and query the UNet on x_t. "
        "If disabled, queries UNet on clean x0 at timestep t (not theory-correct; kept for ablations).",
    )
    p.add_argument(
        "--diffusion-noise-seed",
        type=int,
        default=0,
        help="Deterministic seed for the forward-diffusion noise used when --diffusion-add-noise is enabled.",
    )
    p.add_argument("--score-scale", type=float, default=1.0, help="Scale applied to diffusion score field: s' = a*s + b")
    p.add_argument("--score-bias", type=float, default=0.0, help="Bias applied to diffusion score field: s' = a*s + b")
    p.add_argument(
        "--score-noise-mode",
        type=str,
        default="none",
        choices=["none", "rel_rms", "snr_db"],
        help="Optional additive Gaussian noise on the score field AFTER scale/bias. "
        "rel_rms: sigma = level * rms(score_base). snr_db: sigma = rms(score_base) * 10^(-SNR_dB/20).",
    )
    p.add_argument("--score-noise-level", type=float, default=0.0, help="Noise level (meaning depends on --score-noise-mode).")
    p.add_argument("--score-noise-seed", type=int, default=0, help="Seed for score noise (deterministic across runs).")
    p.add_argument(
        "--score-noise-renorm",
        type=str,
        default="none",
        choices=["none", "match_rms"],
        help="Optional renormalization after adding score noise. "
        "'match_rms' rescales the noisy score so E[s'^2] matches E[score_base^2] per-image, "
        "preventing noise level from trivially increasing dot magnitude.",
    )
    p.add_argument("--size", type=int, default=256)
    p.add_argument(
        "--mock-models",
        action="store_true",
        help="Offline smoke-test mode: do not load torchvision/diffusers weights; use tiny differentiable mock models.",
    )
    p.add_argument("--mock-num-classes", type=int, default=1000, help="Num classes for mock classifier logits.")
    p.add_argument(
        "--heatmap-mode",
        type=str,
        default="score_norm",
        choices=["score_norm", "full_stein_resnet"],
        help="Heatmap computation mode. 'score_norm' is a placeholder; 'full_stein_resnet' computes Δf + s·∇f with ResNet-safe Laplacian.",
    )
    p.add_argument(
        "--stein-class-mode",
        type=str,
        default="fixed",
        choices=["fixed", "predicted"],
        help=(
            "How to choose the classifier-derived test function(s) f(x) for full_stein_resnet heatmaps. "
            "'fixed' and 'predicted' use a single class probability."
        ),
    )
    p.add_argument("--stein-fixed-class-idx", type=int, default=0)
    p.add_argument("--stein-topk", type=int, default=5)
    p.add_argument(
        "--stein-ablation-mode",
        type=str,
        default="stein_full",
        choices=["stein_full", "no_lap", "lap_only", "score_only"],
        help=(
            "Which per-pixel map to output when --heatmap-mode=full_stein_resnet. "
            "stein_full: (Δf + s·∇f) summed over input dims; "
            "no_lap: dot-only (s·∇f); "
            "lap_only: Laplacian-only (Δf); "
            "score_only: ||s|| (L2 over RGB channels per pixel)."
        ),
    )
    # Default to raw (signed) residuals so downstream two-sided percentile tests have access to sign.
    p.add_argument("--stein-map-nonlinearity", type=str, default="raw", choices=["raw", "abs", "square", "relu"])
    p.add_argument("--heatmap-fn", type=str, default="", help="Optional dotted-path callable: fn(x01, logits, score) -> np.ndarray(H,W).")
    p.add_argument("--max-images", type=int, default=0, help="Limit images per category (0=all).")
    p.add_argument(
        "--heatmap-noise-mode",
        type=str,
        default="none",
        choices=["none", "rel_rms", "snr_db"],
        help="Optional additive Gaussian noise applied to the FINAL heatmap (after computation, before saving). "
        "rel_rms: sigma = level * rms(heatmap). snr_db: sigma = rms(heatmap) * 10^(-SNR_dB/20).",
    )
    p.add_argument("--heatmap-noise-level", type=float, default=0.0)
    p.add_argument("--heatmap-noise-seed", type=int, default=0)
    args = p.parse_args(argv)

    # Device resolution (FAIL HARD; no fallbacks)
    req = str(args.device).lower().strip()
    if req not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported --device='{args.device}'. Use one of: cpu|mps|cuda")
    if req == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available in this PyTorch build/runtime.")
    if req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available (torch.backends.mps.is_available() is False).")
    device = torch.device(req)

    # Full-Stein requires input gradients through the classifier.
    # For MPS: allow diffusion on MPS, but keep classifier-side gradients on CPU for stability.
    classifier_device = device
    if device.type == "mps" and args.heatmap_mode == "full_stein_resnet" and (not args.mock_models):
        classifier_device = torch.device("cpu")
        print(
            "[mvtec] note: --device mps with full_stein_resnet => diffusion runs on MPS, "
            "but classifier/gradients/Laplacian run on CPU for stability."
        )
    mvtec_root = Path(args.mvtec_root).expanduser().resolve()
    heatmaps_dir = Path(args.heatmaps_dir).expanduser().resolve()
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    cats = ALL_CATEGORIES if args.categories.strip() == "*" else [c.strip() for c in args.categories.split(",") if c.strip()]
    split = str(args.split).strip()
    # Preflight: validate all categories up-front (FAIL HARD with a single aggregated error)
    preflight_errors: list[str] = []
    for cat in cats:
        try:
            ss = discover_mvtec_samples(mvtec_root, categories=[cat], split=split)
            if args.only_good:
                ss = [s for s in ss if s.defect_type == "good"]
            if len(ss) == 0:
                preflight_errors.append(
                    f"{cat}: discovered 0 samples under '{mvtec_root}/{cat}/{split}' (empty, wrong structure, or filtered by --only-good)"
                )
        except FileNotFoundError as e:
            preflight_errors.append(f"{cat}: {e}")
    if preflight_errors:
        msg = "MVTec root is incomplete or malformed. Preflight failed:\n" + "\n".join(
            f"- {s}" for s in preflight_errors
        )
        raise FileNotFoundError(msg)

    if str(args.classifier_mode) == "finetuned" and (not str(args.classifier_checkpoint).strip()):
        raise ValueError("--classifier-checkpoint is required when --classifier-mode=finetuned")
    if str(args.diffusion_mode) == "finetuned" and (not str(args.diffusion_model_path).strip()):
        raise ValueError("--diffusion-model-path is required when --diffusion-mode=finetuned")

    if args.mock_models:
        print("[mvtec] mock-models enabled (offline). No pretrained weights will be loaded.")
        import torch.nn as nn

        num_classes = int(args.mock_num_classes)
        # Very small differentiable classifier: GAP over (H,W) -> linear to logits
        clf_linear = nn.Linear(3, num_classes, bias=True).to(device)

        def clf_forward_logits(x_req: torch.Tensor) -> torch.Tensor:
            # x_req: (B,3,H,W) in [0,1]
            x_g = x_req.mean(dim=(2, 3))  # (B,3)
            return clf_linear(x_g)  # (B,K)

        def diff_forward(x_req: torch.Tensor) -> torch.Tensor:
            # simple score field: s(x)=x-0.5
            return x_req - 0.5

        clf = None
        diff = None
    else:
        # Classifier selection
        if str(args.classifier_mode) == "zero_shot":
            clf = PretrainedImageNetClassifier(name=args.classifier, device=classifier_device, freeze=False)
            print(f"[mvtec] classifier=zero_shot name={args.classifier}")
        else:
            # Load fine-tuned classifier checkpoint (torch)
            ckpt_path = Path(args.classifier_checkpoint).expanduser().resolve()
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing classifier checkpoint: {ckpt_path}")
            import torchvision.models as tvm  # type: ignore
            payload = torch.load(str(ckpt_path), map_location=classifier_device)
            arch = str(payload.get("arch", args.classifier))
            num_classes = int(payload.get("num_classes", 15))
            if arch == "resnet50":
                model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            elif arch == "resnet18":
                model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            else:
                raise ValueError(f"Unsupported arch in checkpoint: {arch}")
            in_dim = int(model.fc.in_features)
            model.fc = torch.nn.Linear(in_dim, num_classes, bias=True)
            model.load_state_dict(payload["state_dict"], strict=True)
            model = model.to(classifier_device)
            model.eval()
            # Wrap to match PretrainedImageNetClassifier interface used below (forward_logits)
            class _FinetunedWrapper(torch.nn.Module):
                def __init__(self, m: torch.nn.Module, dev: torch.device):
                    super().__init__()
                    self.m = m
                    self.dev = dev
                def forward_logits(self, x01: torch.Tensor) -> torch.Tensor:
                    from src.pretrained_classifier import imagenet_normalize
                    x = x01.to(self.dev, dtype=torch.float32).clamp(0.0, 1.0)
                    x = imagenet_normalize(x)
                    return self.m(x)
            clf = _FinetunedWrapper(model, classifier_device)
            print(f"[mvtec] classifier=finetuned ckpt={ckpt_path} arch={arch} num_classes={num_classes}")

        # Diffusion selection
        if str(args.diffusion_mode) == "zero_shot":
            diff_model_id = str(args.diffusion_model_id)
            print(f"[mvtec] diffusion=zero_shot model_id={diff_model_id}")
        else:
            diff_model_id = str(args.diffusion_model_path)
            print(f"[mvtec] diffusion=finetuned model_id_or_dir={diff_model_id}")
        diff = DiffusionScore256(
            model_id=diff_model_id,
            timestep=(None if int(args.diffusion_timestep) < 0 else int(args.diffusion_timestep)),
            denom_mode=str(args.diffusion_score_denom),
            device=device,
        )

    heatmap_fn: Optional[Callable] = None
    if args.heatmap_fn.strip():
        heatmap_fn = _load_callable(args.heatmap_fn.strip())
        print("[mvtec] using custom heatmap_fn:", args.heatmap_fn.strip())
    else:
        print("[mvtec] heatmap_mode=", args.heatmap_mode)
        if args.heatmap_mode == "score_norm":
            print("[mvtec] using placeholder heatmap: ||score(x)||_2 per pixel")
        else:
            print("[mvtec] using built-in full_stein_resnet heatmap")
            from src.mvtec_full_stein_heatmap import full_stein_resnet_heatmap

    score_scale = float(args.score_scale)
    score_bias = float(args.score_bias)
    if not np.isfinite(score_scale) or not np.isfinite(score_bias):
        raise ValueError(f"Non-finite score scale/bias: scale={score_scale} bias={score_bias}")
    score_noise_mode = str(args.score_noise_mode)
    score_noise_level = float(args.score_noise_level)
    score_noise_renorm = str(args.score_noise_renorm)
    score_noise_seed = int(args.score_noise_seed)
    if score_noise_mode != "none" and (not np.isfinite(score_noise_level)):
        raise ValueError(f"Non-finite --score-noise-level: {score_noise_level}")

    heatmap_noise_mode = str(args.heatmap_noise_mode)
    heatmap_noise_level = float(args.heatmap_noise_level)
    heatmap_noise_seed = int(args.heatmap_noise_seed)
    if heatmap_noise_mode != "none" and (not np.isfinite(heatmap_noise_level)):
        raise ValueError(f"Non-finite --heatmap-noise-level: {heatmap_noise_level}")

    # Optional quality proxy: mean ||s'(x) - s(x)||^2 over generated train/good samples.
    # We compute this only when generating split=train with --only-good (reference set).
    distortion_sum = 0.0
    distortion_count = 0
    pn_over_ps_sum = 0.0

    print(f"[mvtec] generating split={split} only_good={bool(args.only_good)}")
    _agent_logged_score_stats = 0
    noise_gen = torch.Generator(device="cpu").manual_seed(int(score_noise_seed))
    diff_noise_gen = torch.Generator(device="cpu").manual_seed(int(args.diffusion_noise_seed))
    hm_rng = np.random.default_rng(int(heatmap_noise_seed))
    for cat in cats:
        # Fail hard: if a category is missing or malformed, raise immediately.
        samples = discover_mvtec_samples(mvtec_root, categories=[cat], split=split)
        if args.only_good:
            samples = [s for s in samples if s.defect_type == "good"]
        if float(args.mvtec_subsample_frac) < 1.0:
            samples = subsample_mvtec_samples(
                samples,
                frac=float(args.mvtec_subsample_frac),
                seed=int(args.mvtec_subsample_seed),
                min_per_group=1,
            )
        if len(samples) == 0:
            raise FileNotFoundError(
                f"No samples discovered for category '{cat}' under mvtec_root={mvtec_root} split={split} only_good={bool(args.only_good)}. "
                f"Check that '{cat}/{split}' exists and contains images (and 'good' exists if you used --only-good)."
            )
        if args.max_images and int(args.max_images) > 0:
            samples = samples[: int(args.max_images)]

        print(f"[mvtec] category={cat} n={len(samples)}")
        for s in samples:
            img = Image.open(s.image_path).convert("RGB")
            img256 = _resize_pil(img, size=int(args.size))
            x01 = _to_tensor01(img256).unsqueeze(0).to(device=device, dtype=torch.float32).contiguous()  # (1,3,H,W)

            # Classifier forward (logits) + diffusion score forward.
            if args.mock_models:
                x_req = x01.clone().detach().contiguous().requires_grad_(True)
                logits = clf_forward_logits(x_req)
                score_base = diff_forward(x_req)
                score = score_scale * score_base + score_bias
                if split == "train" and bool(args.only_good):
                    # E[||s' - s||^2] proxy
                    d = (score - score_base).detach()
                    distortion_sum += float(torch.mean(d * d).item())
                    distortion_count += 1
            else:
                # Diffusion score does not need gradients w.r.t input for Stein; treat it as an external field.
                with torch.no_grad():
                    # Compute diffusion score field.
                    # IMPORTANT: For epsilon-predicting diffusion models, the theory-correct query is on x_t (noised),
                    # not on clean x0. We enable forward noising by default.
                    if bool(args.diffusion_add_noise):
                        # x0 in [-1,1]
                        x_model = (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)
                        t = int(getattr(diff, "timestep", int(args.diffusion_timestep)))
                        tt = torch.tensor([t], device=x_model.device, dtype=torch.long)
                        eps_noise = (
                            torch.randn(x_model.shape, generator=diff_noise_gen, device="cpu", dtype=torch.float32)
                            .to(x_model.device, dtype=x_model.dtype)
                        )
                        x_t = diff.scheduler.add_noise(x_model, eps_noise, tt)
                        eps = diff.unet(x_t, tt).sample

                        # denom selection must match DiffusionScore256
                        if hasattr(diff.scheduler, "alphas_cumprod"):
                            ac = diff.scheduler.alphas_cumprod.to(device=x_model.device, dtype=x_model.dtype)
                            tt_used = min(int(t), int(ac.numel()) - 1)
                            abar_t = ac[tt_used]
                            sigma = torch.sqrt((1.0 - abar_t).clamp_min(1e-20))
                            if str(args.diffusion_score_denom) == "sigma":
                                denom = sigma
                            else:
                                # Heuristic sigma^2 estimate (consistent with DiffusionScore256/DDPMScoreWrapper style).
                                # We retain this path only for backwards-compatibility experiments.
                                if hasattr(diff.scheduler, "betas"):
                                    betas = diff.scheduler.betas.to(device=x_model.device, dtype=x_model.dtype)
                                    bt = betas[tt_used]
                                    denom = bt * (1.0 - abar_t) / (1.0 - abar_t + 1e-8)
                                else:
                                    denom = sigma * sigma
                        else:
                            # Fallback
                            denom = torch.tensor(1.0, device=x_model.device, dtype=x_model.dtype)

                        score_base = (-eps / (denom + 1e-8)).detach()

                    else:
                        score_base = diff(x01).detach()  # (1,3,H,W) on `device`
                    score = score_scale * score_base + score_bias

                    # Optional additive noise (deterministic)
                    if score_noise_mode != "none" and float(score_noise_level) != 0.0:
                        # Calibrate to the RMS magnitude of score_base (not the scaled score) on this image.
                        ps = torch.mean(score_base.float() * score_base.float()).clamp_min(1e-20)  # E[s^2]
                        signal_rms = torch.sqrt(ps)
                        if score_noise_mode == "rel_rms":
                            sigma = float(score_noise_level) * float(signal_rms.item())
                        elif score_noise_mode == "snr_db":
                            sigma = float(signal_rms.item()) * (10.0 ** (-float(score_noise_level) / 20.0))
                        else:
                            raise ValueError(f"Unsupported score_noise_mode={score_noise_mode}")

                        eps = torch.randn(score_base.shape, generator=noise_gen, device="cpu", dtype=torch.float32).to(score.device)
                        score = score + (float(sigma) * eps)
                        renorm_factor = 1.0
                        if score_noise_renorm == "match_rms":
                            ps_noisy = torch.mean(score.float() * score.float()).clamp_min(1e-20)
                            renorm_factor = float(torch.sqrt(ps / ps_noisy).item())
                            score = score * float(renorm_factor)
                        elif score_noise_renorm != "none":
                            raise ValueError(f"Unsupported score_noise_renorm={score_noise_renorm}")

                        # (debug logging removed)
                    if split == "train" and bool(args.only_good):
                        d = (score - score_base)
                        distortion_sum += float(torch.mean(d * d).item())
                        # pn/ps proxy wrt base (includes scale+bias+noise)
                        try:
                            ps = float(torch.mean(score_base.float() * score_base.float()).item())
                            pn = float(torch.mean((d.float()) * (d.float())).item())
                            pn_over_ps_sum += float(pn / (ps + 1e-20))
                        except Exception:
                            pass
                        distortion_count += 1

                    # (debug logging removed)

                # For full Stein we need ∇f and ResNet-safe Laplacian; compute these on classifier_device.
                if args.heatmap_mode == "full_stein_resnet":
                    x_req = x01.detach().to(classifier_device).contiguous().requires_grad_(True)
                    logits = clf.forward_logits(x_req)
                    score = score.to(classifier_device)
                else:
                    x_req = x01.clone().detach().contiguous().requires_grad_(True)
                    logits = clf.forward_logits(x_req)

            if heatmap_fn is not None:
                hm = heatmap_fn(x_req, logits, score)
                hm = np.asarray(hm, dtype=np.float32)
                if hm.ndim != 2:
                    raise ValueError(f"heatmap_fn must return HxW, got shape={hm.shape}")
            else:
                if args.heatmap_mode == "score_norm":
                    # Placeholder: per-pixel L2 norm of the diffusion score
                    hm = torch.norm(score[0], p=2, dim=0).detach().cpu().numpy().astype(np.float32)  # (H,W)
                else:
                    hm = full_stein_resnet_heatmap(
                        x_req,
                        logits,
                        score,
                        mode=str(args.stein_ablation_mode),
                        class_mode=args.stein_class_mode,
                        fixed_class_idx=int(args.stein_fixed_class_idx),
                        topk=int(args.stein_topk),
                        class_topk=1,
                        nonlinearity=args.stein_map_nonlinearity,
                    )

            # Optional heatmap noise: should degrade toward random as level increases.
            if heatmap_noise_mode != "none" and float(heatmap_noise_level) != 0.0:
                hm_f = np.asarray(hm, dtype=np.float64)
                ps = float(np.mean(hm_f * hm_f))
                rms = float(np.sqrt(max(ps, 1e-20)))
                if heatmap_noise_mode == "rel_rms":
                    sigma = float(heatmap_noise_level) * rms
                elif heatmap_noise_mode == "snr_db":
                    sigma = rms * (10.0 ** (-float(heatmap_noise_level) / 20.0))
                else:
                    raise ValueError(f"Unsupported heatmap_noise_mode={heatmap_noise_mode}")
                eps = hm_rng.standard_normal(size=hm_f.shape)
                hm = (hm_f + sigma * eps).astype(np.float32)
                # (debug logging removed)

            out_path = heatmaps_dir / s.image_path.relative_to(mvtec_root)
            out_path = out_path.with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, hm)

    # Write quality proxy if applicable
    if split == "train" and bool(args.only_good):
        out_json = heatmaps_dir.parent / "score_quality.json"
        payload = {
            "score_scale": float(score_scale),
            "score_bias": float(score_bias),
            "score_noise_mode": str(score_noise_mode),
            "score_noise_level": float(score_noise_level),
            "score_noise_seed": int(score_noise_seed),
            "distortion_mse_mean": (float(distortion_sum) / float(distortion_count) if distortion_count > 0 else float("nan")),
            "distortion_pn_over_ps_mean": (float(pn_over_ps_sum) / float(distortion_count) if distortion_count > 0 else float("nan")),
            "n_images": int(distortion_count),
            "device": str(device),
            "diffusion_timestep": int(args.diffusion_timestep),
            "heatmap_mode": str(args.heatmap_mode),
        }
        import json
        out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print("[mvtec] wrote score quality:", out_json)

    print("[mvtec] done. heatmaps_dir=", heatmaps_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

