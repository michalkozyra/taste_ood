"""
Visualize per-pixel anomaly masks from saved MVTec heatmaps.

Given:
  - MVTec AD root: <mvtec-root>/<category>/{train,test}/...
  - Heatmaps saved by scripts/generate_mvtec_heatmaps_pretrained.py:
      <heatmaps-dir>/<category>/test/<defect_type>/<image_stem>.npy

This script:
  - For each category, builds a per-pixel threshold from a reference set of "good" heatmaps
    using a percentile tail test (upper or two-sided).
  - Samples a random subset of test images and overlays the predicted anomaly mask.
  - If a GT mask exists (non-good samples), overlays it for qualitative comparison.

Fail-fast: any missing files or malformed directories raise immediately.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Literal, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path so `import src...` works when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Avoid Matplotlib cache issues (this script uses matplotlib directly).
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mplconfig"))

from src.mvtec_ad import ALL_CATEGORIES, discover_mvtec_samples, subsample_mvtec_samples
from src.evaluation.mvtec_metrics import (
    TailMode,
    anomaly_mask_from_thresholds,
    percentile_thresholds,
    _roc_curve,
    auc_from_roc,
    auc_upto_fpr_limit,
    fpr_at_tpr,
    average_precision,
)


ReferenceMode = Literal["test_good", "train_good"]

def _apply_map_preprocess_np(x: np.ndarray, mode: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    m = str(mode)
    if m == "raw":
        return a.astype(np.float32)
    if m == "abs":
        return np.abs(a).astype(np.float32)
    if m == "square":
        return (a * a).astype(np.float32)
    if m == "relu":
        return np.maximum(a, 0.0).astype(np.float32)
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

        return str(os.environ.get("AGENT_RUN_ID", "viz"))
    except Exception:
        return "viz"


# endregion agent log


def _pixel_metrics_for_image(hm: np.ndarray, gt_mask01: np.ndarray) -> tuple[float, float, float, float]:
    """
    Compute per-image pixel AUROC, AUROC@FPR<=0.30 (normalized), FPR@95, and AP.
    Returns (auroc_full, auroc_30_norm, fpr95, ap). If GT is degenerate (all-0 or all-1), returns (nan, nan, nan, nan).
    """
    a = np.asarray(hm, dtype=np.float64).reshape(-1)
    g = (np.asarray(gt_mask01) > 0).astype(np.int64).reshape(-1)
    if a.size == 0 or g.size == 0 or a.size != g.size:
        return float("nan"), float("nan"), float("nan"), float("nan")
    P = int(np.sum(g == 1))
    N = int(np.sum(g == 0))
    if P == 0 or N == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    fpr, tpr, _ = _roc_curve(a, g)
    auroc_full = float(auc_from_roc(fpr, tpr))
    auroc_30 = float(auc_upto_fpr_limit(fpr, tpr, fpr_limit=0.30, normalize=True))
    fpr95 = float(fpr_at_tpr(fpr, tpr, target_tpr=0.95))
    ap = float(average_precision(a, g))
    return auroc_full, auroc_30, fpr95, ap


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mvtec-root", type=str, required=True)
    p.add_argument("--heatmaps-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument(
        "--reference-heatmaps-dir",
        type=str,
        default="",
        help="Optional separate heatmaps dir used ONLY for fitting the reference-good distribution (threshold). "
        "If empty, uses --heatmaps-dir. Useful for score-degradation sweeps with fixed clean reference.",
    )

    p.add_argument("--categories", type=str, default="*", help="Comma-separated categories or '*' for all.")
    p.add_argument(
        "--mvtec-subsample-frac",
        type=float,
        default=1.0,
        help="If < 1, visualize only this fraction of MVTec samples. "
        "Train reference is subsampled per-category; test is subsampled per (category, defect_type). "
        "A minimum of 1 sample per group is kept (if the group is non-empty).",
    )
    p.add_argument("--mvtec-subsample-seed", type=int, default=0, help="Seed for deterministic MVTec subsampling.")

    p.add_argument("--alpha", type=float, default=0.01, help="Pixel-level upper-tail alpha (p<=alpha => anomaly).")
    p.add_argument(
        "--tail",
        type=str,
        default="two_sided",
        choices=["upper", "two_sided"],
        help="Percentile tail mode for pixel-level test (upper or two-sided).",
    )
    p.add_argument(
        "--reference",
        type=str,
        default="test_good",
        choices=["test_good", "train_good"],
        help="Which split to use for fitting the per-category pixel threshold.",
    )
    p.add_argument("--ref-max-images", type=int, default=50, help="Max reference good images per category.")
    p.add_argument(
        "--ref-pixel-subsample",
        type=int,
        default=200_000,
        help="Max pixels to use from reference heatmaps per category (for speed/memory).",
    )
    p.add_argument(
        "--map-preprocess",
        type=str,
        default="raw",
        choices=["raw", "abs", "square", "relu"],
        help="Preprocess raw heatmaps BEFORE thresholding. Use 'raw' for signed maps (recommended for two_sided).",
    )

    p.add_argument("--n-per-category", type=int, default=12, help="How many test images to visualize per category.")
    p.add_argument(
        "--sample-mode",
        type=str,
        default="mixed",
        choices=["mixed", "good_only", "anomalous_only"],
        help="Which test samples to draw for visualization.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--overlay-alpha", type=float, default=0.35, help="Alpha for predicted-mask overlay.")
    p.add_argument(
        "--grid-k",
        type=int,
        default=0,
        help="If >0, additionally generate a single PNG per category with k rows × 3 columns (image, pred overlay, pred+GT overlay), omitting the heatmap subplot.",
    )
    p.add_argument(
        "--grid-count",
        type=int,
        default=3,
        help="How many grid PNGs to generate per category when --grid-k>0 (each uses a different random subset).",
    )
    p.add_argument(
        "--grid-only",
        action="store_true",
        help="If set and --grid-k>0, do not write per-image 2x2 PNGs; only write the k×3 grid PNGs.",
    )
    p.add_argument(
        "--grid-mix-categories",
        action="store_true",
        help="If set and --grid-k>0, generate mixed-category grids under <out-dir>/_mixed_grids "
        "(instead of per-category grids). The first row is always a test/good sample and all other rows "
        "are guaranteed non-good.",
    )
    p.add_argument(
        "--target-pxac-percentile",
        type=float,
        default=-1.0,
        help="If set (>=0) and using --grid-mix-categories, rank all anomalous test images by per-image "
        "pxAC (we use pxAUROC30 by default) and only sample anomalous rows from percentile band "
        "[p-range, p+range]. Accepts 0..1 or 0..100.",
    )
    p.add_argument(
        "--target-range",
        type=float,
        default=0.0,
        help="Percentile half-width around --target-pxac-percentile. Accepts 0..1 or 0..100 (same scale as percentile).",
    )
    p.add_argument(
        "--pxac-metric",
        type=str,
        default="pxAUROC30",
        choices=["pxAUROC30", "pxAUROC", "pxAP", "pxAUROC30_plus_pxAUROC", "pxAP_plus_pxAUROC"],
        help="Which per-image pixel metric to use as pxAC when doing targeted sampling (anomalous rows only).",
    )

    return p.parse_args()


def _to_unit_percentile(p: float) -> float:
    """
    Normalize a percentile input. Accepts either [0,1] or [0,100].
    """
    pp = float(p)
    if pp < 0:
        return pp
    if pp > 1.0:
        return pp / 100.0
    return pp


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """
    Compute percentile ranks in [0,1] for a 1D array.
    rank[i] ~= fraction of values <= values[i] (using stable sorting).
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(v.size)
    if n == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = (np.arange(n, dtype=np.float64) + 1.0) / float(n)
    return ranks


def _heatmap_path(*, heatmaps_dir: Path, mvtec_root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(mvtec_root)
    return (heatmaps_dir / rel).with_suffix(".npy")


def _load_heatmap_resize_to(hm_path: Path, *, target_hw: tuple[int, int]) -> np.ndarray:
    hm = np.load(hm_path).astype(np.float32)
    if hm.ndim != 2:
        raise ValueError(f"Heatmap must be HxW float array. Got shape={hm.shape} at {hm_path}")
    th, tw = target_hw
    if hm.shape == (th, tw):
        return hm
    # Use PIL float mode for value-preserving resize
    im = Image.fromarray(hm, mode="F").resize((tw, th), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def _load_gt_mask_resize_to(mask_path: Optional[Path], *, target_hw: tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    if mask_path is None:
        return np.zeros((th, tw), dtype=np.uint8)
    m = Image.open(mask_path).convert("L").resize((tw, th), resample=Image.NEAREST)
    return (np.asarray(m, dtype=np.uint8) > 0).astype(np.uint8)


def _fit_threshold_from_reference(
    *,
    mvtec_root: Path,
    heatmaps_dir: Path,
    category: str,
    alpha: float,
    reference: ReferenceMode,
    ref_max_images: int,
    ref_pixel_subsample: int,
    rng: np.random.Generator,
    tail: TailMode,
    map_preprocess: str,
    mvtec_subsample_frac: float = 1.0,
    mvtec_subsample_seed: int = 0,
) -> tuple[Optional[float], float]:
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"--alpha must be in (0,1). Got {alpha}")

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

    if ref_max_images > 0:
        ref_samples = list(ref_samples)
        rng.shuffle(ref_samples)
        ref_samples = ref_samples[: int(ref_max_images)]

    vals: list[np.ndarray] = []
    for s in ref_samples:
        hm_path = _heatmap_path(heatmaps_dir=heatmaps_dir, mvtec_root=mvtec_root, image_path=s.image_path)
        if not hm_path.exists():
            raise FileNotFoundError(
                f"Missing heatmap for reference sample: {hm_path}. "
                f"Did you generate heatmaps for split='{s.split}'? (Expected test heatmaps by default.)"
            )
        hm = np.load(hm_path).astype(np.float32)
        if hm.ndim != 2:
            raise ValueError(f"Heatmap must be HxW. Got {hm.shape} at {hm_path}")
        hm = _apply_map_preprocess_np(hm, str(map_preprocess))
        vals.append(hm.reshape(-1))

    allv = np.concatenate(vals, axis=0) if vals else np.array([], dtype=np.float32)
    if allv.size == 0:
        raise RuntimeError(f"Reference heatmaps produced 0 pixels for category={category}")

    if ref_pixel_subsample > 0 and allv.size > int(ref_pixel_subsample):
        idx = rng.choice(allv.size, size=int(ref_pixel_subsample), replace=False)
        allv = allv[idx]

    thr_low, thr_high = percentile_thresholds(allv, alpha=float(alpha), tail=tail)
    return thr_low, float(thr_high)


def _overlay_mask_rgb(img_rgb: np.ndarray, mask01: np.ndarray, *, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    if img_rgb.dtype != np.uint8:
        raise ValueError("img_rgb must be uint8")
    if mask01.dtype != np.uint8:
        mask01 = mask01.astype(np.uint8)
    if mask01.ndim != 2:
        raise ValueError("mask01 must be HxW")
    if img_rgb.shape[:2] != mask01.shape:
        raise ValueError(f"shape mismatch: img={img_rgb.shape} mask={mask01.shape}")

    out = img_rgb.astype(np.float32)
    m = (mask01 > 0).astype(np.float32)[..., None]
    col = np.array(color, dtype=np.float32)[None, None, :]
    out = out * (1.0 - alpha * m) + col * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_viz(
    *,
    out_path: Path,
    img: Image.Image,
    hm: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    title: str,
    overlay_alpha: float,
    dpi: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    hm_vis = hm.astype(np.float32)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    axs[0, 0].imshow(img_rgb)
    axs[0, 0].set_title("image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(hm_vis, cmap="magma")
    axs[0, 1].set_title("heatmap")
    axs[0, 1].axis("off")

    overlay_pred = _overlay_mask_rgb(img_rgb, pred_mask, color=(255, 0, 0), alpha=float(overlay_alpha))
    axs[1, 0].imshow(overlay_pred)
    axs[1, 0].set_title("pred mask (red)")
    axs[1, 0].axis("off")

    overlay_both = overlay_pred
    if gt_mask is not None and np.any(gt_mask > 0):
        overlay_both = _overlay_mask_rgb(overlay_both, gt_mask.astype(np.uint8), color=(0, 255, 0), alpha=0.35)
    axs[1, 1].imshow(overlay_both)
    axs[1, 1].set_title("pred (red) + GT (green)")
    axs[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _save_grid_viz(
    *,
    out_path: Path,
    rows: list[dict],
    title: str,
    dpi: int,
) -> None:
    """
    Save a k×3 grid (k=len(rows)) with columns:
      1) image
      2) pred mask overlay
      3) pred+GT overlay
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    k = int(len(rows))
    if k <= 0:
        raise ValueError("rows must be non-empty")

    # Page-friendly sizing: aim for ~letter/A4 aspect while still readable at k=6.
    # Width ~11in; height grows linearly with k.
    fig_w = 11.0
    fig_h = max(6.0, 2.0 * k + 1.2)
    fig, axs = plt.subplots(k, 3, figsize=(fig_w, fig_h), constrained_layout=True)
    # Intentionally omit a big header title (requested). Keep column titles instead.

    # Handle k=1 case (axs shape differs)
    if k == 1:
        axs = np.asarray([axs])  # (1,3)

    col_titles = ["image", "pred mask (red)", "pred (red) + GT (green)"]
    for j in range(3):
        axs[0, j].set_title(col_titles[j])

    for i, r in enumerate(rows):
        img_rgb = r["img_rgb"]
        overlay_pred = r["overlay_pred"]
        overlay_both = r["overlay_both"]
        row_label = r.get("row_label", "")

        axs[i, 0].imshow(img_rgb)
        axs[i, 1].imshow(overlay_pred)
        axs[i, 2].imshow(overlay_both)
        for j in range(3):
            axs[i, j].axis("off")
            # Keep image content visually centered in each subplot.
            axs[i, j].set_aspect("equal")
        if row_label:
            # Place caption under the middle panel so it's centered across the row.
            axs[i, 1].text(
                0.5,
                -0.08,
                row_label,
                transform=axs[i, 1].transAxes,
                fontsize=9,
                ha="center",
                va="top",
                clip_on=False,
            )

    # Fine-tune padding so margins aren't lopsided and captions have space.
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.10)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)

def main() -> int:
    args = _parse_args()
    mvtec_root = Path(args.mvtec_root).expanduser().resolve()
    heatmaps_dir = Path(args.heatmaps_dir).expanduser().resolve()
    ref_heatmaps_dir = Path(args.reference_heatmaps_dir).expanduser().resolve() if str(args.reference_heatmaps_dir).strip() else heatmaps_dir
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cats = ALL_CATEGORIES if str(args.categories).strip() == "*" else [c.strip() for c in str(args.categories).split(",") if c.strip()]
    rng = np.random.default_rng(int(args.seed))

    reference: ReferenceMode = str(args.reference)  # type: ignore[assignment]

    # Optional: mixed-category grids (first row good, remaining rows anomalous).
    if int(args.grid_k) > 0 and bool(args.grid_mix_categories):
        tail: TailMode = str(args.tail)  # type: ignore[assignment]
        # Precompute per-category thresholds once.
        thr_by_cat: dict[str, tuple[Optional[float], float]] = {}
        for cat in cats:
            thr_by_cat[str(cat)] = _fit_threshold_from_reference(
                mvtec_root=mvtec_root,
                heatmaps_dir=ref_heatmaps_dir,
                category=str(cat),
                alpha=float(args.alpha),
                reference=reference,
                ref_max_images=int(args.ref_max_images),
                ref_pixel_subsample=int(args.ref_pixel_subsample),
                rng=rng,
                tail=tail,
                map_preprocess=str(args.map_preprocess),
                mvtec_subsample_frac=float(args.mvtec_subsample_frac),
                mvtec_subsample_seed=int(args.mvtec_subsample_seed),
            )

        # Build global pools from test split across categories.
        all_test = discover_mvtec_samples(mvtec_root, categories=list(cats), split="test")
        if float(args.mvtec_subsample_frac) < 1.0:
            all_test = subsample_mvtec_samples(
                list(all_test),
                frac=float(args.mvtec_subsample_frac),
                seed=int(args.mvtec_subsample_seed),
                min_per_group=1,
            )
        good_pool = [s for s in all_test if s.defect_type == "good"]
        anom_pool = [s for s in all_test if s.defect_type != "good"]
        if len(good_pool) == 0:
            raise FileNotFoundError("No test/good samples found across selected categories.")
        if len(anom_pool) == 0 and int(args.grid_k) > 1:
            raise FileNotFoundError("No test anomalous samples found across selected categories.")

        # Optional: targeted sampling band based on pxAC percentile among anomalous images.
        target_p = _to_unit_percentile(float(args.target_pxac_percentile))
        target_r = _to_unit_percentile(float(args.target_range))
        if target_p >= 0.0:
            if not (0.0 <= target_p <= 1.0):
                raise ValueError(f"--target-pxac-percentile must be in [0,1] or [0,100]. Got {args.target_pxac_percentile}")
            if target_r < 0.0 or target_r > 1.0:
                raise ValueError(f"--target-range must be in [0,1] or [0,100]. Got {args.target_range}")

            # Compute per-image metrics for all anomalous test images (can be slow; cached to CSV).
            cache_dir = out_dir / "_mixed_grids_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / (
                f"anom_px_metrics__tail_{str(tail)}__alpha_{float(args.alpha):.3g}"
                f"__pre_{str(args.map_preprocess)}__ref_{str(reference)}.csv"
            )

            rows = []
            if cache_path.exists():
                try:
                    import csv  # noqa

                    with open(cache_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            rows.append(r)
                except Exception:
                    rows = []

            pxac_vals = []
            anom_samples_for_selection = []
            if rows:
                # Use cached rows (must match current selection categories implicitly by file contents).
                by_path = {r["image_path"]: r for r in rows if "image_path" in r}
                for s in anom_pool:
                    rr = by_path.get(str(s.image_path))
                    if rr is None:
                        continue
                    try:
                        pxauroc30 = float(rr.get("pxAUROC30", "nan"))
                        pxauroc = float(rr.get("pxAUROC", "nan"))
                        pxap = float(rr.get("pxAP", "nan"))
                    except Exception:
                        continue
                    m = str(args.pxac_metric)
                    if m == "pxAUROC30":
                        val = pxauroc30
                    elif m == "pxAUROC":
                        val = pxauroc
                    elif m == "pxAP":
                        val = pxap
                    elif m == "pxAUROC30_plus_pxAUROC":
                        val = pxauroc30 + pxauroc
                    elif m == "pxAP_plus_pxAUROC":
                        val = pxap + pxauroc
                    else:
                        raise ValueError(f"Unsupported --pxac-metric='{m}'")
                    if np.isfinite(val):
                        pxac_vals.append(val)
                        anom_samples_for_selection.append(s)
            else:
                # Compute fresh and write cache.
                import csv  # noqa

                with open(cache_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "category",
                            "defect_type",
                            "image_path",
                            "pxAUROC30",
                            "pxAUROC",
                            "pxFPR95",
                            "pxAP",
                        ],
                    )
                    writer.writeheader()
                    for s in anom_pool:
                        img = Image.open(s.image_path).convert("RGB")
                        w, h = img.size
                        hm_path = _heatmap_path(heatmaps_dir=heatmaps_dir, mvtec_root=mvtec_root, image_path=s.image_path)
                        if not hm_path.exists():
                            raise FileNotFoundError(f"Missing heatmap: {hm_path}")
                        hm = _load_heatmap_resize_to(hm_path, target_hw=(h, w))
                        hm = _apply_map_preprocess_np(hm, str(args.map_preprocess))
                        gt = _load_gt_mask_resize_to(s.mask_path, target_hw=(h, w))

                        px_auroc, px_auroc30, px_fpr95, px_ap = _pixel_metrics_for_image(hm, gt)
                        writer.writerow(
                            {
                                "category": str(s.category),
                                "defect_type": str(s.defect_type),
                                "image_path": str(s.image_path),
                                "pxAUROC30": f"{px_auroc30:.6g}",
                                "pxAUROC": f"{px_auroc:.6g}",
                                "pxFPR95": f"{px_fpr95:.6g}",
                                "pxAP": f"{px_ap:.6g}",
                            }
                        )
                        m = str(args.pxac_metric)
                        if m == "pxAUROC30":
                            val = float(px_auroc30)
                        elif m == "pxAUROC":
                            val = float(px_auroc)
                        elif m == "pxAP":
                            val = float(px_ap)
                        elif m == "pxAUROC30_plus_pxAUROC":
                            val = float(px_auroc30) + float(px_auroc)
                        elif m == "pxAP_plus_pxAUROC":
                            val = float(px_ap) + float(px_auroc)
                        else:
                            raise ValueError(f"Unsupported --pxac-metric='{m}'")
                        if np.isfinite(val):
                            pxac_vals.append(val)
                            anom_samples_for_selection.append(s)

            if len(pxac_vals) == 0:
                raise RuntimeError("Failed to compute any finite pxAC values for anomalous images.")

            pxac_vals_arr = np.asarray(pxac_vals, dtype=np.float64)
            ranks = _percentile_ranks(pxac_vals_arr)
            lo = max(0.0, float(target_p - target_r))
            hi = min(1.0, float(target_p + target_r))
            band_mask = (ranks >= lo) & (ranks <= hi)
            band = [s for s, m in zip(anom_samples_for_selection, band_mask) if bool(m)]
            if len(band) == 0:
                raise RuntimeError(
                    f"No anomalous samples in pxAC percentile band [{lo:.3f}, {hi:.3f}] "
                    f"(metric={args.pxac_metric}). Try widening --target-range."
                )
            anom_pool = band

        k = int(args.grid_k)
        n_grids = max(int(args.grid_count), 1)
        for gi in range(n_grids):
            rrng = np.random.default_rng(int(args.seed) + 20_000 + gi)
            good_s = good_pool[int(rrng.integers(0, len(good_pool)))]
            if k > 1:
                if len(anom_pool) < (k - 1):
                    raise ValueError(f"Not enough anomalous samples to draw k-1={k-1} without replacement.")
                idxs = rrng.choice(len(anom_pool), size=(k - 1), replace=False)
                sel = [good_s] + [anom_pool[int(i)] for i in idxs]
            else:
                sel = [good_s]

            grid_rows: list[dict] = []
            for s in sel:
                img = Image.open(s.image_path).convert("RGB")
                w, h = img.size
                hm_path = _heatmap_path(heatmaps_dir=heatmaps_dir, mvtec_root=mvtec_root, image_path=s.image_path)
                if not hm_path.exists():
                    raise FileNotFoundError(f"Missing heatmap: {hm_path}")
                hm = _load_heatmap_resize_to(hm_path, target_hw=(h, w))
                hm = _apply_map_preprocess_np(hm, str(args.map_preprocess))

                thr_low, thr_high = thr_by_cat[str(s.category)]
                pred = anomaly_mask_from_thresholds(hm, thr_low=thr_low, thr_high=float(thr_high), tail=tail)
                gt = _load_gt_mask_resize_to(s.mask_path, target_hw=(h, w))

                img_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
                overlay_pred = _overlay_mask_rgb(img_rgb, pred, color=(255, 0, 0), alpha=float(args.overlay_alpha))
                overlay_both = overlay_pred
                if gt is not None and np.any(gt > 0):
                    overlay_both = _overlay_mask_rgb(overlay_both, gt.astype(np.uint8), color=(0, 255, 0), alpha=0.35)

                frac_flagged = float(np.mean(pred > 0))
                if s.defect_type == "good":
                    # Pixel ROC/AP are undefined for a degenerate GT mask (all zeros), so report a diagnostic instead.
                    row_label = f"{s.category} | good | flagged={frac_flagged:.3%} @ alpha={float(args.alpha):.3g}"
                else:
                    px_auroc, px_auroc30, px_fpr95, px_ap = _pixel_metrics_for_image(hm, gt)
                    row_label = (
                        f"{s.category} | {s.defect_type} | pxAP={px_ap:.3f} | pxAUROC={px_auroc:.3f} | "
                        f"pxAUROC30={px_auroc30:.3f} | pxFPR95={px_fpr95:.3f} | flagged={frac_flagged:.3%}"
                    )

                grid_rows.append(
                    {
                        "img_rgb": img_rgb,
                        "overlay_pred": overlay_pred,
                        "overlay_both": overlay_both,
                        "row_label": row_label,
                    }
                )

            grid_dir = out_dir / "_mixed_grids"
            grid_path = grid_dir / f"mixed__grid_{gi:02d}__k_{k}__alpha_{float(args.alpha):.3g}__tail_{str(tail)}.png"
            _save_grid_viz(out_path=grid_path, rows=grid_rows, title="", dpi=int(args.dpi))

        # In mixed-grid mode, we do not generate the per-category grids.
        if bool(args.grid_only):
            return 0

    for cat in cats:
        tail: TailMode = str(args.tail)  # type: ignore[assignment]
        thr_low, thr_high = _fit_threshold_from_reference(
            mvtec_root=mvtec_root,
            heatmaps_dir=ref_heatmaps_dir,
            category=cat,
            alpha=float(args.alpha),
            reference=reference,
            ref_max_images=int(args.ref_max_images),
            ref_pixel_subsample=int(args.ref_pixel_subsample),
            rng=rng,
            tail=tail,
            map_preprocess=str(args.map_preprocess),
            mvtec_subsample_frac=float(args.mvtec_subsample_frac),
            mvtec_subsample_seed=int(args.mvtec_subsample_seed),
        )

        test_samples = discover_mvtec_samples(mvtec_root, categories=[cat], split="test")
        if str(args.sample_mode) == "good_only":
            test_samples = [s for s in test_samples if s.defect_type == "good"]
        elif str(args.sample_mode) == "anomalous_only":
            test_samples = [s for s in test_samples if s.defect_type != "good"]
        if float(args.mvtec_subsample_frac) < 1.0:
            test_samples = subsample_mvtec_samples(
                list(test_samples),
                frac=float(args.mvtec_subsample_frac),
                seed=int(args.mvtec_subsample_seed),
                min_per_group=1,
            )

        if len(test_samples) == 0:
            raise FileNotFoundError(f"No test samples found for category={cat} sample_mode={args.sample_mode}")

        test_samples = list(test_samples)
        rng.shuffle(test_samples)
        test_samples = test_samples[: int(args.n_per_category)]

        # Optional: build grid PNG(s) first, since they reuse the same loaded data.
        if int(args.grid_k) > 0:
            k = int(args.grid_k)
            n_grids = max(int(args.grid_count), 1)
            for gi in range(n_grids):
                rrng = np.random.default_rng(int(args.seed) + 10_000 + gi)
                pool = list(test_samples)
                rrng.shuffle(pool)
                sel = pool[:k]
                if len(sel) == 0:
                    continue

                grid_rows: list[dict] = []
                for s in sel:
                    img = Image.open(s.image_path).convert("RGB")
                    w, h = img.size
                    hm_path = _heatmap_path(heatmaps_dir=heatmaps_dir, mvtec_root=mvtec_root, image_path=s.image_path)
                    if not hm_path.exists():
                        raise FileNotFoundError(f"Missing heatmap: {hm_path}")
                    hm = _load_heatmap_resize_to(hm_path, target_hw=(h, w))
                    hm = _apply_map_preprocess_np(hm, str(args.map_preprocess))
                    pred = anomaly_mask_from_thresholds(hm, thr_low=thr_low, thr_high=float(thr_high), tail=tail)
                    gt = _load_gt_mask_resize_to(s.mask_path, target_hw=(h, w))

                    img_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
                    overlay_pred = _overlay_mask_rgb(img_rgb, pred, color=(255, 0, 0), alpha=float(args.overlay_alpha))
                    overlay_both = overlay_pred
                    if gt is not None and np.any(gt > 0):
                        overlay_both = _overlay_mask_rgb(overlay_both, gt.astype(np.uint8), color=(0, 255, 0), alpha=0.35)

                    px_auroc, px_auroc30, px_fpr95, px_ap = _pixel_metrics_for_image(hm, gt)
                    # AP is more informative under strong class imbalance (few anomalous pixels).
                    row_label = (
                        f"{s.defect_type} | pxAP={px_ap:.3f} | pxAUROC={px_auroc:.3f} | "
                        f"pxAUROC30={px_auroc30:.3f} | pxFPR95={px_fpr95:.3f}"
                    )

                    grid_rows.append(
                        {
                            "img_rgb": img_rgb,
                            "overlay_pred": overlay_pred,
                            "overlay_both": overlay_both,
                            "row_label": row_label,
                        }
                    )

                grid_dir = out_dir / cat / "_grids"
                grid_path = grid_dir / f"{cat}__grid_{gi:02d}__k_{k}__alpha_{float(args.alpha):.3g}__tail_{str(tail)}.png"
                _save_grid_viz(out_path=grid_path, rows=grid_rows, title="", dpi=int(args.dpi))

            if bool(args.grid_only):
                continue

        for s in test_samples:
            img = Image.open(s.image_path).convert("RGB")
            w, h = img.size

            hm_path = _heatmap_path(heatmaps_dir=heatmaps_dir, mvtec_root=mvtec_root, image_path=s.image_path)
            if not hm_path.exists():
                raise FileNotFoundError(f"Missing heatmap: {hm_path}")

            hm = _load_heatmap_resize_to(hm_path, target_hw=(h, w))
            hm = _apply_map_preprocess_np(hm, str(args.map_preprocess))
            pred = anomaly_mask_from_thresholds(hm, thr_low=thr_low, thr_high=float(thr_high), tail=tail)
            gt = _load_gt_mask_resize_to(s.mask_path, target_hw=(h, w))

            rel = str(s.image_path.relative_to(mvtec_root)).replace("/", "__")
            out_path = out_dir / cat / s.split / s.defect_type / f"{Path(rel).stem}__alpha_{args.alpha:.3g}.png"

            px_auroc, px_auroc30, px_fpr95, _px_ap = _pixel_metrics_for_image(hm, gt)
            _agent_log(
                {
                    "sessionId": "debug-session",
                    "runId": _agent_run_id(),
                    "hypothesisId": "H8_viz_pixel_metrics_correct",
                    "location": "scripts/visualize_mvtec_pixel_anomalies.py:main:per_image_metrics",
                    "message": "per-image pixel metrics",
                    "data": {
                        "category": cat,
                        "defect_type": s.defect_type,
                        "image": str(s.image_path),
                        "tail": str(tail),
                        "alpha": float(args.alpha),
                        "map_preprocess": str(args.map_preprocess),
                        "gt_pos": int(np.sum(gt > 0)),
                        "gt_neg": int(np.sum(gt == 0)),
                        "px_auroc": float(px_auroc),
                        "px_auroc30": float(px_auroc30),
                        "px_fpr95": float(px_fpr95),
                    },
                }
            )

            if str(tail) == "upper":
                title = (
                    f"{cat} | {s.split}/{s.defect_type} | tail=upper | alpha={float(args.alpha):.3g} | "
                    f"thr_high={float(thr_high):.4g} | pxAUROC={px_auroc:.3f} pxAUROC30={px_auroc30:.3f} pxFPR95={px_fpr95:.3f}"
                )
            else:
                title = (
                    f"{cat} | {s.split}/{s.defect_type} | tail=two_sided | alpha={float(args.alpha):.3g} | "
                    f"thr_low={float(thr_low):.4g} thr_high={float(thr_high):.4g} | "
                    f"pxAUROC={px_auroc:.3f} pxAUROC30={px_auroc30:.3f} pxFPR95={px_fpr95:.3f}"
                )
            _save_viz(
                out_path=out_path,
                img=img,
                hm=hm,
                pred_mask=pred,
                gt_mask=gt,
                title=title,
                overlay_alpha=float(args.overlay_alpha),
                dpi=int(args.dpi),
            )

    print("Wrote visualizations to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

