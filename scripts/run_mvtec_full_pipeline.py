"""
Run the full MVTec pipeline end-to-end:
  1) generate train-good heatmaps (reference)
  2) generate test heatmaps
  3) evaluate localization metrics
  4) generate visualizations

This is a thin wrapper around:
  - scripts/generate_mvtec_heatmaps_pretrained.py
  - scripts/evaluate_mvtec_localization.py
  - scripts/visualize_mvtec_pixel_anomalies.py

It exists to make runs reproducible and to keep all knobs in one place.
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
    # Keep '*' as-is; otherwise normalize a comma-separated list.
    ss = str(s).strip()
    if ss == "*":
        return "*"
    parts = [p.strip() for p in ss.split(",") if p.strip()]
    return ",".join(parts) if parts else "*"


def _bool_flag(name: str, value: bool) -> List[str]:
    # Uses argparse.BooleanOptionalAction pattern: --foo / --no-foo
    return [f"--{name}" if value else f"--no-{name}"]


def _run(cmd: List[str], *, env: dict, dry_run: bool) -> None:
    printable = " ".join(shlex.quote(c) for c in cmd)
    print("\n[run]", printable)
    if dry_run:
        return
    subprocess.run(cmd, env=env, check=True)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()

    # Core paths
    p.add_argument(
        "--mvtec-root",
        type=str,
        required=True,
        help="Path to mvtec_anomaly_detection root folder.",
    )
    p.add_argument(
        "--heatmaps-dir",
        type=str,
        required=True,
        help="Where to write heatmaps (train-good and test) in MVTec layout.",
    )
    p.add_argument(
        "--eval-out-dir",
        type=str,
        required=True,
        help="Where to write evaluation CSVs.",
    )
    p.add_argument(
        "--viz-out-dir",
        type=str,
        required=True,
        help="Where to write visualization PNGs.",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default="",
        help="If set, append this string to --heatmaps-dir/--eval-out-dir/--viz-out-dir.",
    )

    # Subset selection
    p.add_argument("--categories", type=str, default="*", help="Comma-separated categories or '*' for all.")
    p.add_argument("--max-images", type=int, default=0, help="Optional cap per category for heatmap generation (0=all).")
    p.add_argument(
        "--mvtec-subsample-frac",
        type=float,
        default=1.0,
        help="If < 1, run the whole pipeline on a subset of MVTec samples. "
        "Generator/eval/viz will use the same deterministic subsample so directories stay consistent. "
        "Train reference is subsampled per-category; test is subsampled per (category, defect_type).",
    )
    p.add_argument("--mvtec-subsample-seed", type=int, default=0, help="Seed for deterministic MVTec subsampling.")

    # Model/device knobs (passed through to generator)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--classifier", type=str, default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--classifier-mode", type=str, default="zero_shot", choices=["zero_shot", "finetuned"])
    p.add_argument("--classifier-checkpoint", type=str, default="")

    p.add_argument("--diffusion-mode", type=str, default="zero_shot", choices=["zero_shot", "finetuned"])
    p.add_argument("--diffusion-model-id", type=str, default="google/ddpm-ema-celebahq-256")
    p.add_argument("--diffusion-model-path", type=str, default="")

    # Diffusion score extraction controls (passed through to generator)
    p.add_argument(
        "--diffusion-timestep",
        type=int,
        default=-1,
        help="DDPM timestep. Use -1 to reproduce the legacy 'unspecified timestep' behavior (scheduler.timesteps[0]).",
    )
    p.add_argument("--diffusion-score-denom", type=str, default="sigma_sq", choices=["sigma_sq", "sigma"])
    p.add_argument("--diffusion-add-noise", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--diffusion-noise-seed", type=int, default=0)

    # Stein heatmap knobs
    p.add_argument("--heatmap-mode", type=str, default="full_stein_resnet", choices=["score_norm", "full_stein_resnet"])
    p.add_argument("--stein-class-mode", type=str, default="fixed", choices=["fixed", "predicted"])
    p.add_argument("--stein-fixed-class-idx", type=int, default=0)
    p.add_argument("--stein-topk", type=int, default=5)
    p.add_argument("--stein-map-nonlinearity", type=str, default="abs", choices=["raw", "abs", "square", "relu"])
    p.add_argument("--stein-ablation-mode", type=str, default="stein_full", choices=["stein_full", "no_lap", "lap_only", "score_only"])

    # Optional score corruption (passed through)
    p.add_argument("--score-scale", type=float, default=1.0)
    p.add_argument("--score-bias", type=float, default=0.0)
    p.add_argument("--score-noise-mode", type=str, default="none", choices=["none", "rel_rms", "snr_db"])
    p.add_argument("--score-noise-level", type=float, default=0.0)
    p.add_argument("--score-noise-seed", type=int, default=0)
    p.add_argument("--score-noise-renorm", type=str, default="none", choices=["none", "match_rms"])

    # Evaluation knobs
    p.add_argument("--eval-score-transform", type=str, default="raw", choices=["raw", "percentile"])
    p.add_argument("--eval-map-preprocess", type=str, default="raw", choices=["raw", "abs", "square", "relu"])
    p.add_argument("--eval-alpha", type=float, default=0.0)
    p.add_argument("--eval-tail", type=str, default="upper", choices=["upper", "two_sided"])
    p.add_argument("--eval-reference", type=str, default="train_good", choices=["test_good", "train_good"])
    p.add_argument("--eval-ref-max-images", type=int, default=50)
    p.add_argument("--eval-ref-pixel-subsample", type=int, default=200_000)
    p.add_argument(
        "--eval-reference-heatmaps-dir",
        type=str,
        default="",
        help="If set, used ONLY to fit the reference-good distribution for percentile/thresholds.",
    )

    # Visualization knobs
    p.add_argument("--viz-n-per-category", type=int, default=12)
    p.add_argument("--viz-sample-mode", type=str, default="mixed", choices=["mixed", "anomalous_only", "uniform"])
    p.add_argument("--viz-seed", type=int, default=0)
    p.add_argument("--viz-tail", type=str, default="upper", choices=["upper", "two_sided"])
    p.add_argument("--viz-alpha", type=float, default=0.01)
    p.add_argument("--viz-reference", type=str, default="train_good", choices=["test_good", "train_good"])
    p.add_argument("--viz-ref-max-images", type=int, default=50)
    p.add_argument("--viz-ref-pixel-subsample", type=int, default=200_000)
    p.add_argument(
        "--viz-reference-heatmaps-dir",
        type=str,
        default="",
        help="If set, used ONLY to fit the reference-good thresholds for viz overlays.",
    )
    p.add_argument("--viz-map-preprocess", type=str, default="raw", choices=["raw", "abs", "square", "relu"])

    # Control
    p.add_argument("--skip-generate-train", action="store_true")
    p.add_argument("--skip-generate-test", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-viz", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")

    args = p.parse_args(argv)

    mvtec_root = str(Path(args.mvtec_root).expanduser().resolve())
    suffix = str(args.suffix)
    heatmaps_dir = str(Path(f"{args.heatmaps_dir}{suffix}").expanduser().resolve())
    eval_out_dir = str(Path(f"{args.eval_out_dir}{suffix}").expanduser().resolve())
    viz_out_dir = str(Path(f"{args.viz_out_dir}{suffix}").expanduser().resolve())

    cats = _as_list_csv(args.categories)
    max_images = int(args.max_images)

    py = sys.executable  # whichever python you used to run this script

    # Environment: ensure scripts can import src and matplotlib can write cache.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLCONFIGDIR", str((REPO_ROOT / ".mplconfig").resolve()))

    gen_script = str((REPO_ROOT / "scripts" / "generate_mvtec_heatmaps_pretrained.py").resolve())
    eval_script = str((REPO_ROOT / "scripts" / "evaluate_mvtec_localization.py").resolve())
    viz_script = str((REPO_ROOT / "scripts" / "visualize_mvtec_pixel_anomalies.py").resolve())

    common_gen = [
        py,
        gen_script,
        "--mvtec-root",
        mvtec_root,
        "--heatmaps-dir",
        heatmaps_dir,
        "--categories",
        cats,
        "--mvtec-subsample-frac",
        str(float(args.mvtec_subsample_frac)),
        "--mvtec-subsample-seed",
        str(int(args.mvtec_subsample_seed)),
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
        str(int(args.diffusion_timestep)),
        "--diffusion-score-denom",
        str(args.diffusion_score_denom),
        *_bool_flag("diffusion-add-noise", bool(args.diffusion_add_noise)),
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
        "--stein-ablation-mode",
        str(args.stein_ablation_mode),
        "--stein-map-nonlinearity",
        str(args.stein_map_nonlinearity),
        "--score-scale",
        str(float(args.score_scale)),
        "--score-bias",
        str(float(args.score_bias)),
        "--score-noise-mode",
        str(args.score_noise_mode),
        "--score-noise-level",
        str(float(args.score_noise_level)),
        "--score-noise-seed",
        str(int(args.score_noise_seed)),
        "--score-noise-renorm",
        str(args.score_noise_renorm),
    ]
    if max_images > 0:
        common_gen += ["--max-images", str(max_images)]

    # 1) Generate train/good reference heatmaps
    if not bool(args.skip_generate_train):
        cmd = common_gen + ["--split", "train", "--only-good"]
        _run(cmd, env=env, dry_run=bool(args.dry_run))

    # 2) Generate test heatmaps
    if not bool(args.skip_generate_test):
        cmd = common_gen + ["--split", "test"]
        _run(cmd, env=env, dry_run=bool(args.dry_run))

    # 3) Evaluate
    if not bool(args.skip_eval):
        cmd = [
            py,
            eval_script,
            "--mvtec-root",
            mvtec_root,
            "--heatmaps-dir",
            heatmaps_dir,
            "--out-dir",
            eval_out_dir,
            "--categories",
            cats,
            "--mvtec-subsample-frac",
            str(float(args.mvtec_subsample_frac)),
            "--mvtec-subsample-seed",
            str(int(args.mvtec_subsample_seed)),
            "--score-transform",
            str(args.eval_score_transform),
            "--map-preprocess",
            str(args.eval_map_preprocess),
            "--alpha",
            str(float(args.eval_alpha)),
            "--tail",
            str(args.eval_tail),
            "--reference",
            str(args.eval_reference),
            "--ref-max-images",
            str(int(args.eval_ref_max_images)),
            "--ref-pixel-subsample",
            str(int(args.eval_ref_pixel_subsample)),
            "--seed",
            str(int(args.viz_seed)),
        ]
        if str(args.eval_reference_heatmaps_dir).strip():
            cmd += ["--reference-heatmaps-dir", str(Path(args.eval_reference_heatmaps_dir).expanduser().resolve())]
        _run(cmd, env=env, dry_run=bool(args.dry_run))

    # 4) Visualize
    if not bool(args.skip_viz):
        cmd = [
            py,
            viz_script,
            "--mvtec-root",
            mvtec_root,
            "--heatmaps-dir",
            heatmaps_dir,
            "--out-dir",
            viz_out_dir,
            "--categories",
            cats,
            "--mvtec-subsample-frac",
            str(float(args.mvtec_subsample_frac)),
            "--mvtec-subsample-seed",
            str(int(args.mvtec_subsample_seed)),
            "--tail",
            str(args.viz_tail),
            "--alpha",
            str(float(args.viz_alpha)),
            "--reference",
            str(args.viz_reference),
            "--ref-max-images",
            str(int(args.viz_ref_max_images)),
            "--ref-pixel-subsample",
            str(int(args.viz_ref_pixel_subsample)),
            "--n-per-category",
            str(int(args.viz_n_per_category)),
            "--sample-mode",
            str(args.viz_sample_mode),
            "--seed",
            str(int(args.viz_seed)),
            "--map-preprocess",
            str(args.viz_map_preprocess),
        ]
        if str(args.viz_reference_heatmaps_dir).strip():
            cmd += ["--reference-heatmaps-dir", str(Path(args.viz_reference_heatmaps_dir).expanduser().resolve())]
        _run(cmd, env=env, dry_run=bool(args.dry_run))

    print("\n[done] pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

