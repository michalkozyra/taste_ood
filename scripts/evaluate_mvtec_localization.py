"""
Evaluate per-pixel anomaly heatmaps on MVTec AD using standard metrics:
- Image-level AUROC/AP (image score derived from anomaly map; default=max)
- Pixel-level AUROC/AP
- AUPRO@FPR<=limit (default 0.30)

This script assumes you already have a model (or pipeline) that outputs per-pixel anomaly scores.
To keep benchmarking deterministic and decoupled from model code, this script consumes *saved* heatmaps.

Expected heatmap layout (by default):
  <heatmaps-dir>/<category>/test/<defect_type>/<image_stem>.npy

where each .npy is a 2D array (H,W) or a lower-res 2D array; it will be upsampled to GT mask resolution.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Literal, Optional

import numpy as np
from PIL import Image

# Ensure repo root is on sys.path so `import src...` works when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Avoid Matplotlib cache issues if something imports it indirectly.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mplconfig"))

from src.mvtec_ad import ALL_CATEGORIES, OBJECT_CATEGORIES, TEXTURE_CATEGORIES, discover_mvtec_samples, subsample_mvtec_samples
from src.evaluation.mvtec_metrics import (
    LocalizationMetrics,
    DetectionMetrics,
    TailMode,
    anomaly_mask_from_thresholds,
    compute_image_level_metrics,
    compute_pixel_level_metrics,
    compute_pixel_level_metrics_mean_over_images,
    compute_pro_auc_limits,
    percentile_thresholds,
    percentile_oodness_from_reference,
)


def _apply_map_preprocess_np(x: np.ndarray, mode: str) -> np.ndarray:
    """
    Map preprocessing applied to raw heatmaps BEFORE percentile transforms / thresholding.
    """
    a = np.asarray(x, dtype=np.float64)
    m = str(mode)
    if m == "raw":
        return a
    if m == "abs":
        return np.abs(a)
    if m == "square":
        return a * a
    if m == "relu":
        return np.maximum(a, 0.0)
    raise ValueError(f"Unsupported --map-preprocess='{mode}'")


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


def _load_heatmap(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        d = np.load(path)
        # choose a reasonable default key
        for k in ["heatmap", "map", "scores", "anomaly_map"]:
            if k in d:
                return d[k]
        # fallback: first array
        return d[list(d.keys())[0]]
    raise ValueError(f"Unsupported heatmap file: {path}")


def _resize_to(mask_hw: tuple[int, int], heatmap: np.ndarray) -> np.ndarray:
    """
    Upsample a heatmap to (H,W) using bilinear interpolation via PIL.
    """
    H, W = int(mask_hw[0]), int(mask_hw[1])
    a = np.asarray(heatmap, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got shape={a.shape}")
    if a.shape == (H, W):
        return a.astype(np.float64)
    # Normalize to 0..1 for stable PIL resizing, then rescale back.
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi > lo:
        an = (a - lo) / (hi - lo)
    else:
        an = np.zeros_like(a)
    img = Image.fromarray((an * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    img = img.resize((W, H), resample=Image.BILINEAR)
    out = np.asarray(img).astype(np.float32) / 255.0
    # rescale back
    return (out * (hi - lo) + lo).astype(np.float64)


def _heatmap_path(
    heatmaps_dir: Path,
    *,
    root: Path,
    sample_image_path: Path,
) -> Path:
    rel = sample_image_path.relative_to(root)
    # replace image extension with .npy
    rel2 = rel.with_suffix(".npy")
    return heatmaps_dir / rel2


@dataclass(frozen=True)
class CategoryResults:
    category: str
    image_auroc: float  # AUROC@FPR<=limit (normalized)
    image_auroc_full: float
    image_ap: float  # AP over operating points with FPR<=limit (see mvtec_metrics.py)
    image_ap_full: float
    image_fpr95: float
    pixel_auroc: float  # mean-over-images AUROC@FPR<=limit (normalized)
    pixel_auroc_full: float
    pixel_ap: float  # mean-over-images AP with FPR<=limit
    pixel_ap_full: float
    pixel_metrics_n_valid_images: int
    aupro: float  # AUPRO@FPR<=limit (normalized by limit)
    aupro_full: float  # AUPRO@FPR<=1.0 (normalized by 1.0)
    aupro_fpr_limit: float
    n_test_images: int
    n_anomalous_images: int
    # Optional alpha-thresholded segmentation metrics (percentile-based thresholds from "good" reference pixels)
    alpha: float
    tail: str
    pixel_precision_alpha: float
    pixel_recall_alpha: float
    pixel_f1_alpha: float
    pixel_iou_alpha: float
    pixel_fpr_alpha: float


def _pixel_segmentation_stats(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    pred/gt: HxW uint8 {0,1}.
    Returns: (precision, recall, f1, iou, fpr)
    """
    p = (np.asarray(pred) > 0)
    g = (np.asarray(gt) > 0)
    tp = float(np.sum(p & g))
    fp = float(np.sum(p & (~g)))
    fn = float(np.sum((~p) & g))
    tn = float(np.sum((~p) & (~g)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    return float(precision), float(recall), float(f1), float(iou), float(fpr)


def _fit_reference_thresholds(
    *,
    mvtec_root: Path,
    heatmaps_dir: Path,
    reference_heatmaps_dir: Path,
    category: str,
    alpha: float,
    tail: TailMode,
    reference: Literal["test_good", "train_good"],
    ref_max_images: int,
    ref_pixel_subsample: int,
    rng: np.random.Generator,
    mvtec_subsample_frac: float = 1.0,
    mvtec_subsample_seed: int = 0,
) -> tuple[Optional[float], float]:
    if reference == "train_good":
        ref_samples = discover_mvtec_samples(mvtec_root, categories=[category], split="train")
        ref_samples = [s for s in ref_samples if s.defect_type == "good"]
    elif reference == "test_good":
        ref_samples = discover_mvtec_samples(mvtec_root, categories=[category], split="test")
        ref_samples = [s for s in ref_samples if s.defect_type == "good"]
    else:
        raise ValueError(reference)

    if float(mvtec_subsample_frac) < 1.0:
        ref_samples = subsample_mvtec_samples(
            list(ref_samples),
            frac=float(mvtec_subsample_frac),
            seed=int(mvtec_subsample_seed),
            min_per_group=1,
        )

    if len(ref_samples) == 0:
        raise FileNotFoundError(f"No reference 'good' samples found for category={category} reference={reference}")

    if int(ref_max_images) > 0:
        ref_samples = list(ref_samples)
        rng.shuffle(ref_samples)
        ref_samples = ref_samples[: int(ref_max_images)]

    vals: list[np.ndarray] = []
    for s in ref_samples:
        hm_path = _heatmap_path(reference_heatmaps_dir, root=mvtec_root, sample_image_path=s.image_path)
        if not hm_path.exists():
            hm_path = hm_path.with_suffix(".npz")
        if not hm_path.exists():
            raise FileNotFoundError(
                f"Missing heatmap for reference sample: {hm_path}. "
                f"If you use reference=train_good, you must have heatmaps generated for train/good too."
            )
        hm = _load_heatmap(hm_path)
        if hm.ndim != 2:
            raise ValueError(f"heatmap must be 2D, got shape={hm.shape} at {hm_path}")
        vals.append(hm.reshape(-1).astype(np.float64))

    allv = np.concatenate(vals, axis=0) if vals else np.array([], dtype=np.float64)
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        raise RuntimeError(f"Reference heatmaps produced 0 finite pixels for category={category}")
    if int(ref_pixel_subsample) > 0 and allv.size > int(ref_pixel_subsample):
        idx = rng.choice(allv.size, size=int(ref_pixel_subsample), replace=False)
        allv = allv[idx]

    return percentile_thresholds(allv, alpha=float(alpha), tail=tail)


def _fit_reference_values(
    *,
    mvtec_root: Path,
    heatmaps_dir: Path,
    reference_heatmaps_dir: Path,
    category: str,
    reference: Literal["test_good", "train_good"],
    ref_max_images: int,
    ref_pixel_subsample: int,
    rng: np.random.Generator,
    map_preprocess: str,
    mvtec_subsample_frac: float = 1.0,
    mvtec_subsample_seed: int = 0,
) -> np.ndarray:
    """
    Load a pooled 1D reference score sample from good images, used for percentile-based transforms.
    """
    if reference == "train_good":
        ref_samples = discover_mvtec_samples(mvtec_root, categories=[category], split="train")
        ref_samples = [s for s in ref_samples if s.defect_type == "good"]
    elif reference == "test_good":
        ref_samples = discover_mvtec_samples(mvtec_root, categories=[category], split="test")
        ref_samples = [s for s in ref_samples if s.defect_type == "good"]
    else:
        raise ValueError(reference)

    if float(mvtec_subsample_frac) < 1.0:
        ref_samples = subsample_mvtec_samples(
            list(ref_samples),
            frac=float(mvtec_subsample_frac),
            seed=int(mvtec_subsample_seed),
            min_per_group=1,
        )

    if len(ref_samples) == 0:
        raise FileNotFoundError(f"No reference 'good' samples found for category={category} reference={reference}")

    if int(ref_max_images) > 0:
        ref_samples = list(ref_samples)
        rng.shuffle(ref_samples)
        ref_samples = ref_samples[: int(ref_max_images)]

    vals: list[np.ndarray] = []
    for s in ref_samples:
        hm_path = _heatmap_path(reference_heatmaps_dir, root=mvtec_root, sample_image_path=s.image_path)
        if not hm_path.exists():
            hm_path = hm_path.with_suffix(".npz")
        if not hm_path.exists():
            raise FileNotFoundError(
                f"Missing heatmap for reference sample: {hm_path}. "
                f"If you use reference=train_good, you must have heatmaps generated for train/good too."
            )
        hm = _load_heatmap(hm_path)
        if hm.ndim != 2:
            raise ValueError(f"heatmap must be 2D, got shape={hm.shape} at {hm_path}")
        hm = _apply_map_preprocess_np(hm, map_preprocess)
        vals.append(hm.reshape(-1).astype(np.float64))

    allv = np.concatenate(vals, axis=0) if vals else np.array([], dtype=np.float64)
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        raise RuntimeError(f"Reference heatmaps produced 0 finite pixels for category={category}")
    if int(ref_pixel_subsample) > 0 and allv.size > int(ref_pixel_subsample):
        idx = rng.choice(allv.size, size=int(ref_pixel_subsample), replace=False)
        allv = allv[idx]
    return allv


def evaluate_category(
    *,
    mvtec_root: Path,
    heatmaps_dir: Path,
    category: str,
    image_score_reducer: Literal["max", "mean", "p95"] = "max",
    aupro_fpr_limit: float = 0.30,
    pixel_subsample: Optional[int] = None,
    seed: int = 0,
    # Optional alpha-thresholded evaluation
    alpha: float = 0.0,
    tail: TailMode = "two_sided",
    reference: Literal["test_good", "train_good"] = "test_good",
    reference_heatmaps_dir: Optional[Path] = None,
    ref_max_images: int = 50,
    ref_pixel_subsample: int = 200_000,
    score_transform: Literal["raw", "percentile"] = "percentile",
    map_preprocess: Literal["raw", "abs", "square", "relu"] = "raw",
    mvtec_subsample_frac: float = 1.0,
    mvtec_subsample_seed: int = 0,
) -> CategoryResults:
    samples = discover_mvtec_samples(mvtec_root, categories=[category], split="test")
    if float(mvtec_subsample_frac) < 1.0:
        samples = subsample_mvtec_samples(
            list(samples),
            frac=float(mvtec_subsample_frac),
            seed=int(mvtec_subsample_seed),
            min_per_group=1,
        )
    anomaly_maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    img_labels: list[int] = []

    for s in samples:
        # Load GT mask (or zeros) via the same logic as the dataset.
        img = Image.open(s.image_path).convert("RGB")
        w, h = img.size
        if s.mask_path is None:
            gt = np.zeros((h, w), dtype=np.uint8)
        else:
            m = Image.open(s.mask_path).convert("L")
            gt = (np.asarray(m) > 0).astype(np.uint8)

        hm_path = _heatmap_path(heatmaps_dir, root=mvtec_root, sample_image_path=s.image_path)
        if not hm_path.exists():
            # Try npz fallback
            hm_path = hm_path.with_suffix(".npz")
        if not hm_path.exists():
            raise FileNotFoundError(f"Missing heatmap for {s.image_path} expected at {hm_path}")

        hm = _load_heatmap(hm_path)
        hm = _resize_to(gt.shape, hm)
        hm = _apply_map_preprocess_np(hm, str(map_preprocess))

        anomaly_maps.append(hm)
        masks.append(gt)
        img_labels.append(1 if s.is_anomalous else 0)

    maps_for_metrics = anomaly_maps

    # Optional percentile/p-value transform of scores BEFORE computing AUROC/AP/AUPRO (tail-dependent).
    if str(score_transform) == "percentile":
        rng = np.random.default_rng(int(seed))
        ref_hdir = reference_heatmaps_dir if reference_heatmaps_dir is not None else heatmaps_dir
        # region agent log
        _agent_log(
            {
                "sessionId": "debug-session",
                "runId": _agent_run_id(),
                "hypothesisId": "H12_reference_recalibration_hides_degradation",
                "location": "scripts/evaluate_mvtec_localization.py:evaluate_category:ref_dirs",
                "message": "reference vs eval heatmaps dir",
                "data": {"category": str(category), "heatmaps_dir": str(heatmaps_dir), "reference_heatmaps_dir": str(ref_hdir)},
            }
        )
        # endregion agent log
        ref_vals = _fit_reference_values(
            mvtec_root=mvtec_root,
            heatmaps_dir=heatmaps_dir,
            reference_heatmaps_dir=ref_hdir,
            category=category,
            reference=reference,
            ref_max_images=int(ref_max_images),
            ref_pixel_subsample=int(ref_pixel_subsample),
            rng=rng,
            map_preprocess=str(map_preprocess),
            mvtec_subsample_frac=float(mvtec_subsample_frac),
            mvtec_subsample_seed=int(mvtec_subsample_seed),
        )
        maps_for_metrics = [
            percentile_oodness_from_reference(m, ref_scores=ref_vals, tail=tail) for m in anomaly_maps
        ]

    # Log distribution evidence (one per category)
    try:
        raw_all = np.concatenate([np.asarray(m).reshape(-1) for m in anomaly_maps], axis=0)
        raw_all = raw_all[np.isfinite(raw_all)]
        met_all = np.concatenate([np.asarray(m).reshape(-1) for m in maps_for_metrics], axis=0)
        met_all = met_all[np.isfinite(met_all)]
        _agent_log(
            {
                "sessionId": "debug-session",
                "runId": _agent_run_id(),
                "hypothesisId": "H6_abs_removes_sign_and_can_hurt_two_sided",
                "location": "scripts/evaluate_mvtec_localization.py:evaluate_category:map_stats",
                "message": "map preprocess/transform stats",
                "data": {
                    "category": str(category),
                    "tail": str(tail),
                    "score_transform": str(score_transform),
                    "map_preprocess": str(map_preprocess),
                    "raw_min": float(np.min(raw_all)) if raw_all.size else float("nan"),
                    "raw_max": float(np.max(raw_all)) if raw_all.size else float("nan"),
                    "raw_frac_neg": float(np.mean(raw_all < 0.0)) if raw_all.size else float("nan"),
                    "metrics_min": float(np.min(met_all)) if met_all.size else float("nan"),
                    "metrics_max": float(np.max(met_all)) if met_all.size else float("nan"),
                    "metrics_frac_neg": float(np.mean(met_all < 0.0)) if met_all.size else float("nan"),
                },
            }
        )
    except Exception:
        pass

    aupro, aupro_full = compute_pro_auc_limits(
        anomaly_maps=maps_for_metrics,
        masks=masks,
        fpr_limit=float(aupro_fpr_limit),
        seed=int(seed),
    )

    # Optional alpha-thresholded segmentation metrics (percentile-based on good reference pixels)
    if float(alpha) > 0.0:
        rng = np.random.default_rng(int(seed))
        ref_hdir_thr = reference_heatmaps_dir if reference_heatmaps_dir is not None else heatmaps_dir
        thr_low, thr_high = _fit_reference_thresholds(
            mvtec_root=mvtec_root,
            heatmaps_dir=heatmaps_dir,
            reference_heatmaps_dir=ref_hdir_thr,
            category=category,
            alpha=float(alpha),
            tail=tail,
            reference=reference,
            ref_max_images=int(ref_max_images),
            ref_pixel_subsample=int(ref_pixel_subsample),
            rng=rng,
            mvtec_subsample_frac=float(mvtec_subsample_frac),
            mvtec_subsample_seed=int(mvtec_subsample_seed),
        )
        stats = []
        for hm, gt in zip(anomaly_maps, masks):
            pred = anomaly_mask_from_thresholds(hm, thr_low=thr_low, thr_high=float(thr_high), tail=tail)
            stats.append(_pixel_segmentation_stats(pred, gt))
        # Mean across images
        prec = float(np.mean([s[0] for s in stats])) if stats else float("nan")
        rec = float(np.mean([s[1] for s in stats])) if stats else float("nan")
        f1 = float(np.mean([s[2] for s in stats])) if stats else float("nan")
        iou = float(np.mean([s[3] for s in stats])) if stats else float("nan")
        fpr = float(np.mean([s[4] for s in stats])) if stats else float("nan")
        tail_s = str(tail)
        alpha_s = float(alpha)
    else:
        alpha_s = 0.0
        tail_s = str(tail)
        prec = rec = f1 = iou = fpr = float("nan")

    # Image-level metrics: compute both <=limit and full
    det = compute_image_level_metrics(
        anomaly_maps=maps_for_metrics,
        image_labels=np.asarray(img_labels, dtype=np.int64),
        reducer=str(image_score_reducer),  # type: ignore[arg-type]
        fpr_limit=float(aupro_fpr_limit),
    )

    # Pixel metrics are mean-over-images (unweighted), not pooled over all pixels.
    # (Pooled metrics can be very pessimistic and are not directly comparable to per-image viz.)
    px_auroc, px_ap, px_auroc_full, px_ap_full, px_n_valid = compute_pixel_level_metrics_mean_over_images(
        anomaly_maps=maps_for_metrics,
        masks=masks,
        fpr_limit=float(aupro_fpr_limit),
        normalize=True,
    )

    # Debug evidence: compare with pooled metrics as well (do not export to CSV by default).
    try:
        pooled_auroc_030, pooled_ap_030, pooled_auroc_full, pooled_ap_full = compute_pixel_level_metrics(
            anomaly_maps=maps_for_metrics,
            masks=masks,
            pixel_subsample=pixel_subsample,
            seed=int(seed),
            fpr_limit=float(aupro_fpr_limit),
        )
        _agent_log(
            {
                "sessionId": "debug-session",
                "runId": _agent_run_id(),
                "hypothesisId": "H16_metrics_limit_vs_full",
                "location": "scripts/evaluate_mvtec_localization.py:evaluate_category:pooled_vs_mean",
                "message": "pixel metrics mean-over-images vs pooled",
                "data": {
                    "category": str(category),
                    "tail": str(tail),
                    "score_transform": str(score_transform),
                    "map_preprocess": str(map_preprocess),
                    "fpr_limit": float(aupro_fpr_limit),
                    "mean_auroc_limit": float(px_auroc),
                    "mean_ap_limit": float(px_ap),
                    "mean_auroc_full": float(px_auroc_full),
                    "mean_ap_full": float(px_ap_full),
                    "mean_n_valid_images": int(px_n_valid),
                    "pooled_auroc_limit": float(pooled_auroc_030),
                    "pooled_ap_limit": float(pooled_ap_030),
                    "pooled_auroc_full": float(pooled_auroc_full),
                    "pooled_ap_full": float(pooled_ap_full),
                },
            }
        )
    except Exception:
        pass

    return CategoryResults(
        category=category,
        image_auroc=float(det.auroc),
        image_auroc_full=float(det.auroc_full),
        image_ap=float(det.ap),
        image_ap_full=float(det.ap_full),
        image_fpr95=float(det.fpr95),
        pixel_auroc=float(px_auroc),
        pixel_auroc_full=float(px_auroc_full),
        pixel_ap=float(px_ap),
        pixel_ap_full=float(px_ap_full),
        pixel_metrics_n_valid_images=int(px_n_valid),
        aupro=float(aupro),
        aupro_full=float(aupro_full),
        aupro_fpr_limit=float(aupro_fpr_limit),
        n_test_images=int(len(samples)),
        n_anomalous_images=int(sum(img_labels)),
        alpha=float(alpha_s),
        tail=str(tail_s),
        pixel_precision_alpha=float(prec),
        pixel_recall_alpha=float(rec),
        pixel_f1_alpha=float(f1),
        pixel_iou_alpha=float(iou),
        pixel_fpr_alpha=float(fpr),
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mvtec-root", type=str, required=True, help="Path to mvtec_anomaly_detection root folder.")
    p.add_argument("--heatmaps-dir", type=str, required=True, help="Root folder containing saved heatmaps.")
    p.add_argument("--out-dir", type=str, required=True, help="Where to write CSV outputs.")
    p.add_argument("--categories", type=str, default="*", help="Comma-separated categories or '*' for all 15.")
    p.add_argument(
        "--mvtec-subsample-frac",
        type=float,
        default=1.0,
        help="If < 1, evaluate only this fraction of MVTec samples. "
        "Train reference is subsampled per-category; test is subsampled per (category, defect_type). "
        "A minimum of 1 sample per group is kept (if the group is non-empty).",
    )
    p.add_argument("--mvtec-subsample-seed", type=int, default=0, help="Seed for deterministic MVTec subsampling.")
    p.add_argument("--image-score-reducer", type=str, default="max", choices=["max", "mean", "p95"])
    p.add_argument("--aupro-fpr-limit", type=float, default=0.30)
    p.add_argument("--pixel-subsample", type=int, default=0, help="Optional subsample of pixels per image for pixel metrics (0=all).")
    p.add_argument("--seed", type=int, default=0)
    # Optional alpha-thresholded metrics (percentile test)
    p.add_argument("--alpha", type=float, default=0.0, help="If >0, compute alpha-thresholded pixel segmentation stats using percentile test.")
    p.add_argument("--tail", type=str, default="two_sided", choices=["upper", "two_sided"], help="Tail for percentile test at alpha.")
    p.add_argument("--reference", type=str, default="test_good", choices=["test_good", "train_good"], help="Reference split for percentile thresholds.")
    p.add_argument("--ref-max-images", type=int, default=50)
    p.add_argument("--ref-pixel-subsample", type=int, default=200_000)
    p.add_argument(
        "--reference-heatmaps-dir",
        type=str,
        default="",
        help="Optional separate heatmaps dir used ONLY for fitting the reference-good distribution (percentile/threshold). "
        "If empty, uses --heatmaps-dir. Useful for score-degradation sweeps where you want a fixed clean reference.",
    )
    p.add_argument(
        "--score-transform",
        type=str,
        default="percentile",
        choices=["raw", "percentile"],
        help="If 'percentile', transform anomaly maps to oodness=-log(p) using reference-good pixels BEFORE AUROC/AP/AUPRO. Tail-dependent.",
    )
    p.add_argument(
        "--map-preprocess",
        type=str,
        default="raw",
        choices=["raw", "abs", "square", "relu"],
        help="Preprocess raw heatmaps BEFORE thresholding/percentile transforms. Use 'raw' for signed maps (recommended for two_sided).",
    )

    args = p.parse_args(argv)
    mvtec_root = Path(args.mvtec_root).expanduser().resolve()
    heatmaps_dir = Path(args.heatmaps_dir).expanduser().resolve()
    reference_heatmaps_dir = (
        Path(args.reference_heatmaps_dir).expanduser().resolve() if str(args.reference_heatmaps_dir).strip() else None
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.categories.strip() == "*":
        cats = ALL_CATEGORIES
    else:
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]

    pixel_subsample = None if int(args.pixel_subsample) <= 0 else int(args.pixel_subsample)

    per_cat: list[CategoryResults] = []
    for cat in cats:
        print(f"[mvtec] evaluating category={cat}")
        r = evaluate_category(
            mvtec_root=mvtec_root,
            heatmaps_dir=heatmaps_dir,
            category=cat,
            image_score_reducer=args.image_score_reducer,
            aupro_fpr_limit=float(args.aupro_fpr_limit),
            pixel_subsample=pixel_subsample,
            seed=int(args.seed),
            alpha=float(args.alpha),
            tail=str(args.tail),  # type: ignore[arg-type]
            reference=str(args.reference),  # type: ignore[arg-type]
            reference_heatmaps_dir=reference_heatmaps_dir,
            ref_max_images=int(args.ref_max_images),
            ref_pixel_subsample=int(args.ref_pixel_subsample),
            score_transform=str(args.score_transform),  # type: ignore[arg-type]
            map_preprocess=str(args.map_preprocess),  # type: ignore[arg-type]
            mvtec_subsample_frac=float(args.mvtec_subsample_frac),
            mvtec_subsample_seed=int(args.mvtec_subsample_seed),
        )
        per_cat.append(r)

    # Write per-category table
    per_cat_csv = out_dir / "mvtec_ad_per_category.csv"
    with per_cat_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_cat[0].__dict__.keys()))
        w.writeheader()
        for r in per_cat:
            w.writerow(r.__dict__)
    print("Wrote:", per_cat_csv)

    # Summary: overall mean + object/texture means
    def _mean(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return float(np.mean(xs)) if xs else float("nan")

    def _summ(rows: list[CategoryResults]):
        return {
            "n_categories": int(len(rows)),
            "image_auroc_mean": _mean([r.image_auroc for r in rows]),
            "image_auroc_full_mean": _mean([r.image_auroc_full for r in rows]),
            "image_ap_mean": _mean([r.image_ap for r in rows]),
            "image_ap_full_mean": _mean([r.image_ap_full for r in rows]),
            "image_fpr95_mean": _mean([r.image_fpr95 for r in rows]),
            "pixel_auroc_mean": _mean([r.pixel_auroc for r in rows]),
            "pixel_auroc_full_mean": _mean([r.pixel_auroc_full for r in rows]),
            "pixel_ap_mean": _mean([r.pixel_ap for r in rows]),
            "pixel_ap_full_mean": _mean([r.pixel_ap_full for r in rows]),
            "aupro_mean": _mean([r.aupro for r in rows]),
            "aupro_full_mean": _mean([r.aupro_full for r in rows]),
            "aupro_fpr_limit": float(args.aupro_fpr_limit),
            # Alpha-thresholded segmentation means (only meaningful if --alpha>0)
            "alpha": float(args.alpha),
            "tail": str(args.tail),
            "pixel_precision_alpha_mean": _mean([r.pixel_precision_alpha for r in rows]),
            "pixel_recall_alpha_mean": _mean([r.pixel_recall_alpha for r in rows]),
            "pixel_f1_alpha_mean": _mean([r.pixel_f1_alpha for r in rows]),
            "pixel_iou_alpha_mean": _mean([r.pixel_iou_alpha for r in rows]),
            "pixel_fpr_alpha_mean": _mean([r.pixel_fpr_alpha for r in rows]),
        }

    all_rows = per_cat
    obj_rows = [r for r in per_cat if r.category in OBJECT_CATEGORIES]
    tex_rows = [r for r in per_cat if r.category in TEXTURE_CATEGORIES]

    summary = {
        "split": "test",
        "image_score_reducer": str(args.image_score_reducer),
        "pixel_subsample": (0 if pixel_subsample is None else int(pixel_subsample)),
        "all": _summ(all_rows),
        "objects": _summ(obj_rows) if obj_rows else None,
        "textures": _summ(tex_rows) if tex_rows else None,
    }

    summary_csv = out_dir / "mvtec_ad_summary.csv"
    # Flatten into a simple CSV for papers
    rows = []
    for group in ["all", "objects", "textures"]:
        d = summary.get(group)
        if d is None:
            continue
        rows.append({"group": group, **d})
    pd = __import__("pandas")
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print("Wrote:", summary_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

