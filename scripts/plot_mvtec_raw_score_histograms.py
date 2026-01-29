"""
Plot raw-score histograms for a single MVTec category to diagnose tail behavior.

This script is meant for quick debugging: it loads saved heatmaps and visualizes the
distribution of *raw* anomaly-map values for:
  - reference good pixels (train_good or test_good)
  - test good pixels
  - test anomalous pixels

It overlays percentile thresholds derived from the reference good distribution:
  - upper tail: q_{1-alpha}
  - two-sided: q_{alpha/2}, q_{1-alpha/2}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from PIL import Image

from src.mvtec_ad import discover_mvtec_samples
from src.evaluation.mvtec_metrics import percentile_thresholds


# region agent log
def _agent_log(payload: dict) -> None:
    try:
        payload = dict(payload)
        payload["timestamp"] = 0
        log_path = "/Users/michalkozyra/Developer/PhD/stein_shift_detection/.cursor/debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def _run_id() -> str:
    import os

    return str(os.environ.get("AGENT_RUN_ID", "raw_hist"))


# endregion agent log


def _heatmap_path(heatmaps_dir: Path, *, mvtec_root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(mvtec_root)
    return (heatmaps_dir / rel).with_suffix(".npy")


def _load_heatmap_resize_to(hm_path: Path, *, target_hw: tuple[int, int]) -> np.ndarray:
    hm = np.load(hm_path).astype(np.float32)
    if hm.ndim != 2:
        raise ValueError(f"Heatmap must be HxW float array. Got shape={hm.shape} at {hm_path}")
    th, tw = target_hw
    if hm.shape == (th, tw):
        return hm.astype(np.float64)
    im = Image.fromarray(hm, mode="F").resize((tw, th), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.float32).astype(np.float64)


def _collect_pixels(
    *,
    mvtec_root: Path,
    heatmaps_dir: Path,
    category: str,
    split: Literal["train", "test"],
    only_good: bool,
    pixel_subsample: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pixels: 1D float64
      labels: 1D int {0,1} where 1=anomalous (only meaningful for split='test')
    """
    rng = np.random.default_rng(int(seed))
    samples = discover_mvtec_samples(mvtec_root, categories=[category], split=split)
    if only_good:
        samples = [s for s in samples if s.defect_type == "good"]
    if len(samples) == 0:
        raise FileNotFoundError(f"No samples for category={category} split={split} only_good={only_good}")

    vals = []
    labs = []
    for s in samples:
        hm_path = _heatmap_path(heatmaps_dir, mvtec_root=mvtec_root, image_path=s.image_path)
        if not hm_path.exists():
            raise FileNotFoundError(f"Missing heatmap: {hm_path}")
        img = Image.open(s.image_path).convert("RGB")
        w, h = img.size
        hm = _load_heatmap_resize_to(hm_path, target_hw=(h, w))
        vals.append(hm.reshape(-1))
        labs.append(np.full(hm.size, 1 if getattr(s, "is_anomalous", False) else 0, dtype=np.int8))

    x = np.concatenate(vals, axis=0).astype(np.float64)
    y = np.concatenate(labs, axis=0).astype(np.int64)
    x = x[np.isfinite(x)]
    if pixel_subsample > 0 and x.size > int(pixel_subsample):
        idx = rng.choice(x.size, size=int(pixel_subsample), replace=False)
        x = x[idx]
        y = y[idx]
    return x, y


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mvtec-root", type=str, required=True)
    p.add_argument("--heatmaps-dir", type=str, required=True)
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--reference", type=str, default="train_good", choices=["train_good", "test_good"])
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--pixel-subsample", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-path", type=str, default="")
    args = p.parse_args(argv)

    mvtec_root = Path(args.mvtec_root).expanduser().resolve()
    heatmaps_dir = Path(args.heatmaps_dir).expanduser().resolve()
    cat = str(args.category)

    # Reference pixels
    if str(args.reference) == "train_good":
        ref_split = "train"
        ref_only_good = True
    else:
        ref_split = "test"
        ref_only_good = True

    ref_x, _ = _collect_pixels(
        mvtec_root=mvtec_root,
        heatmaps_dir=heatmaps_dir,
        category=cat,
        split=ref_split,  # type: ignore[arg-type]
        only_good=ref_only_good,
        pixel_subsample=int(args.pixel_subsample),
        seed=int(args.seed),
    )

    thr_upper = percentile_thresholds(ref_x, alpha=float(args.alpha), tail="upper")[1]
    thr_low, thr_high = percentile_thresholds(ref_x, alpha=float(args.alpha), tail="two_sided")

    # Test pixels
    test_x, test_y = _collect_pixels(
        mvtec_root=mvtec_root,
        heatmaps_dir=heatmaps_dir,
        category=cat,
        split="test",
        only_good=False,
        pixel_subsample=int(args.pixel_subsample),
        seed=int(args.seed) + 1,
    )

    test_good = test_x[test_y == 0]
    test_anom = test_x[test_y == 1]

    _agent_log(
        {
            "sessionId": "debug-session",
            "runId": _run_id(),
            "hypothesisId": "H5_two_sided_weaker_due_to_lower_tail_triggering_on_good",
            "location": "scripts/plot_mvtec_raw_score_histograms.py:main",
            "message": "raw hist stats",
            "data": {
                "category": cat,
                "reference": str(args.reference),
                "alpha": float(args.alpha),
                "ref_n": int(ref_x.size),
                "ref_min": float(np.min(ref_x)),
                "ref_max": float(np.max(ref_x)),
                "thr_upper_q_1m_a": float(thr_upper),
                "thr_two_low_q_a2": float(thr_low if thr_low is not None else float("nan")),
                "thr_two_high_q_1m_a2": float(thr_high),
                "test_good_n": int(test_good.size),
                "test_anom_n": int(test_anom.size),
                "test_good_min": float(np.min(test_good)) if test_good.size else float("nan"),
                "test_good_max": float(np.max(test_good)) if test_good.size else float("nan"),
                "test_anom_min": float(np.min(test_anom)) if test_anom.size else float("nan"),
                "test_anom_max": float(np.max(test_anom)) if test_anom.size else float("nan"),
            },
        }
    )

    # Plot
    import matplotlib.pyplot as plt

    if str(args.out_path).strip():
        out_path = Path(args.out_path).expanduser().resolve()
    else:
        out_path = (heatmaps_dir.parent / f"raw_hist__{cat}__alpha_{float(args.alpha):.3g}.png").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Choose a robust plotting range to avoid a few extreme pixels dominating the bins
    lo = float(np.quantile(ref_x, 0.001))
    hi = float(np.quantile(ref_x, 0.999))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = float(np.min(ref_x)), float(np.max(ref_x))

    bins = 200
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for ax, logy in [(axs[0], False), (axs[1], True)]:
        ax.hist(ref_x, bins=bins, range=(lo, hi), density=True, alpha=0.4, label=f"ref({args.reference}) pixels")
        if test_good.size:
            ax.hist(test_good, bins=bins, range=(lo, hi), density=True, alpha=0.4, label="test good pixels")
        if test_anom.size:
            ax.hist(test_anom, bins=bins, range=(lo, hi), density=True, alpha=0.4, label="test anomalous pixels")

        ax.axvline(thr_upper, color="k", linestyle="--", linewidth=1.5, label="upper q(1-a)")
        if thr_low is not None:
            ax.axvline(float(thr_low), color="k", linestyle=":", linewidth=1.5, label="two-sided q(a/2)")
        ax.axvline(float(thr_high), color="k", linestyle="-.", linewidth=1.5, label="two-sided q(1-a/2)")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.2)
        if logy:
            ax.set_yscale("log")
            ax.set_ylabel("density (log)")

    axs[0].set_title(
        f"MVTec raw heatmap scores | category={cat} | alpha={float(args.alpha):.3g} | ref={args.reference}\n"
        f"upper thr={thr_upper:.3e} | two thr_low={float(thr_low) if thr_low is not None else float('nan'):.3e} thr_high={thr_high:.3e}"
    )
    axs[-1].set_xlabel("raw heatmap value")
    axs[0].legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

