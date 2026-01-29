"""
Run a quick timestep sweep for the MVTec pipeline.

This script calls `scripts/run_mvtec_full_pipeline.py` once per timestep and ensures each run writes to a
separate output directory via `--suffix`.

Default sweep timesteps:
  0, 10, 20, 50, 100, 200, 500, 999
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def _as_list_csv(s: str) -> str:
    ss = str(s).strip()
    if ss == "*":
        return "*"
    parts = [p.strip() for p in ss.split(",") if p.strip()]
    return ",".join(parts) if parts else "*"


def _run(cmd: List[str], *, env: dict, dry_run: bool) -> None:
    printable = " ".join(shlex.quote(c) for c in cmd)
    print("\n[run]", printable)
    if dry_run:
        return
    subprocess.run(cmd, env=env, check=True)


def _parse_timesteps(s: str) -> list[int]:
    ss = str(s).strip()
    if not ss:
        raise ValueError("Empty --timesteps")
    out: list[int] = []
    for p in ss.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        raise ValueError("No timesteps parsed from --timesteps")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()

    # Core paths (same as pipeline wrapper)
    p.add_argument("--mvtec-root", type=str, required=True)
    p.add_argument("--heatmaps-dir", type=str, required=True)
    p.add_argument("--eval-out-dir", type=str, required=True)
    p.add_argument("--viz-out-dir", type=str, required=True)

    # Sweep control
    p.add_argument(
        "--timesteps",
        type=str,
        default="0,10,20,50,100,200,500,999",
        help="Comma-separated list of DDPM timesteps to run.",
    )
    p.add_argument(
        "--suffix-template",
        type=str,
        default="__t{t:03d}",
        help="Suffix template for output dirs. Must contain '{t}'. Example: '__t{t:03d}'.",
    )

    # Pass-through knobs (match the terminal snippet defaults unless overridden)
    p.add_argument("--categories", type=str, default="*", help="Comma-separated categories or '*' for all.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--classifier", type=str, default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--classifier-mode", type=str, default="zero_shot", choices=["zero_shot", "finetuned"])
    p.add_argument("--classifier-checkpoint", type=str, default="")

    p.add_argument("--diffusion-mode", type=str, default="zero_shot", choices=["zero_shot", "finetuned"])
    p.add_argument("--diffusion-model-id", type=str, default="google/ddpm-ema-celebahq-256")
    p.add_argument("--diffusion-model-path", type=str, default="")
    p.add_argument("--diffusion-score-denom", type=str, default="sigma", choices=["sigma_sq", "sigma"])
    p.add_argument("--diffusion-add-noise", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--diffusion-noise-seed", type=int, default=0)

    p.add_argument("--heatmap-mode", type=str, default="full_stein_resnet", choices=["score_norm", "full_stein_resnet"])
    p.add_argument("--stein-class-mode", type=str, default="fixed", choices=["fixed", "predicted"])
    p.add_argument("--stein-fixed-class-idx", type=int, default=0)
    p.add_argument("--stein-topk", type=int, default=5)
    p.add_argument("--stein-map-nonlinearity", type=str, default="abs", choices=["raw", "abs", "square", "relu"])

    p.add_argument("--eval-score-transform", type=str, default="raw", choices=["raw", "percentile"])
    p.add_argument("--eval-map-preprocess", type=str, default="raw", choices=["raw", "abs", "square", "relu"])
    p.add_argument("--eval-alpha", type=float, default=0.0)

    p.add_argument("--viz-tail", type=str, default="upper", choices=["upper", "two_sided"])
    p.add_argument("--viz-alpha", type=float, default=0.01)
    p.add_argument("--viz-reference", type=str, default="train_good", choices=["test_good", "train_good"])
    p.add_argument("--viz-map-preprocess", type=str, default="raw", choices=["raw", "abs", "square", "relu"])

    p.add_argument("--mvtec-subsample-frac", type=float, default=1.0)
    p.add_argument("--mvtec-subsample-seed", type=int, default=0)

    # Optional: skip parts (useful if you already generated heatmaps)
    p.add_argument("--skip-generate-train", action="store_true")
    p.add_argument("--skip-generate-test", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-viz", action="store_true")

    # Control
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")

    args = p.parse_args(argv)

    timesteps = _parse_timesteps(str(args.timesteps))
    suffix_tmpl = str(args.suffix_template)
    if "{t" not in suffix_tmpl and "{t}" not in suffix_tmpl:
        raise ValueError("--suffix-template must contain '{t}' (optionally formatted like '{t:03d}')")

    pipeline_script = str((REPO_ROOT / "scripts" / "run_mvtec_full_pipeline.py").resolve())
    py = sys.executable

    # Environment: ensure scripts can import src and matplotlib can write cache.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLCONFIGDIR", str((REPO_ROOT / ".mplconfig").resolve()))

    cats = _as_list_csv(args.categories)

    for t in timesteps:
        suffix = suffix_tmpl.format(t=int(t))
        cmd = [
            py,
            pipeline_script,
            "--mvtec-root",
            str(Path(args.mvtec_root).expanduser().resolve()),
            "--heatmaps-dir",
            str(Path(args.heatmaps_dir).expanduser().resolve()),
            "--eval-out-dir",
            str(Path(args.eval_out_dir).expanduser().resolve()),
            "--viz-out-dir",
            str(Path(args.viz_out_dir).expanduser().resolve()),
            "--suffix",
            str(suffix),
            "--categories",
            str(cats),
            "--device",
            str(args.device),
            "--classifier",
            str(args.classifier),
            "--classifier-mode",
            str(args.classifier_mode),
            "--classifier-checkpoint",
            str(args.classifier_checkpoint),
            "--diffusion-mode",
            str(args.diffusion_mode),
            "--diffusion-model-id",
            str(args.diffusion_model_id),
            "--diffusion-model-path",
            str(args.diffusion_model_path),
            "--diffusion-timestep",
            str(int(t)),
            "--diffusion-score-denom",
            str(args.diffusion_score_denom),
            ("--diffusion-add-noise" if bool(args.diffusion_add_noise) else "--no-diffusion-add-noise"),
            "--diffusion-noise-seed",
            str(int(args.diffusion_noise_seed)),
            "--heatmap-mode",
            str(args.heatmap_mode),
            "--stein-class-mode",
            str(args.stein_class_mode),
            "--stein-fixed-class-idx",
            str(int(args.stein_fixed_class_idx)),
            "--stein-topk",
            str(int(args.stein_topk)),
            "--stein-map-nonlinearity",
            str(args.stein_map_nonlinearity),
            "--eval-score-transform",
            str(args.eval_score_transform),
            "--eval-map-preprocess",
            str(args.eval_map_preprocess),
            "--eval-alpha",
            str(float(args.eval_alpha)),
            "--viz-tail",
            str(args.viz_tail),
            "--viz-alpha",
            str(float(args.viz_alpha)),
            "--viz-reference",
            str(args.viz_reference),
            "--viz-map-preprocess",
            str(args.viz_map_preprocess),
            "--mvtec-subsample-frac",
            str(float(args.mvtec_subsample_frac)),
            "--mvtec-subsample-seed",
            str(int(args.mvtec_subsample_seed)),
        ]

        if bool(args.skip_generate_train):
            cmd.append("--skip-generate-train")
        if bool(args.skip_generate_test):
            cmd.append("--skip-generate-test")
        if bool(args.skip_eval):
            cmd.append("--skip-eval")
        if bool(args.skip_viz):
            cmd.append("--skip-viz")
        if bool(args.dry_run):
            cmd.append("--dry-run")

        _run(cmd, env=env, dry_run=bool(args.dry_run))

    print("\n[done] timestep sweep finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

