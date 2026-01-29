"""
Sweep 5 diffusion timesteps spanning the scheduler range (min -> max) and evaluate using the same
percentile-based setup as the noise sweeps:

- score_transform = percentile (oodness = -log(p))
- tail = two_sided (for signed/raw residual maps)
- map_preprocess = raw
- reference = train_good

For each selected timestep t:
  1) Generate heatmaps for train/good (reference) and test/* (evaluation)
  2) Run evaluation (two_sided + percentile) at alpha=0.01
  3) Run visualization (two_sided) at alpha=0.01

Outputs are written under:
  results/<run_name>/timestep_<t>/{heatmaps,eval,viz}

This script is intentionally a thin orchestrator around:
  - scripts/generate_mvtec_heatmaps_pretrained.py
  - scripts/evaluate_mvtec_localization.py
  - scripts/visualize_mvtec_pixel_anomalies.py
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("[sweep] running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _compute_5_scheduler_timesteps(*, model_id_or_dir: str) -> list[int]:
    """
    Return 5 timesteps sampled evenly (by index) from the sorted scheduler timesteps,
    spanning min -> max.
    Uses diffusers scheduler-only load (no UNet weights).
    """
    try:
        from diffusers import DDPMScheduler  # type: ignore
    except Exception as e:
        raise ModuleNotFoundError(
            "diffusers is required to auto-compute scheduler timesteps. "
            "Either install diffusers in your environment, or pass --timesteps explicitly."
        ) from e

    import numpy as np

    # Load scheduler config only.
    # HF repos vary: some store scheduler_config at repo root, others under "scheduler/".
    try:
        scheduler = DDPMScheduler.from_pretrained(model_id_or_dir)
    except Exception:
        scheduler = DDPMScheduler.from_pretrained(model_id_or_dir, subfolder="scheduler")
    n_train = int(getattr(scheduler.config, "num_train_timesteps", 1000))
    # Populate full-range timesteps
    scheduler.set_timesteps(num_inference_steps=n_train)
    ts = scheduler.timesteps
    if ts is None or len(ts) == 0:
        raise RuntimeError("scheduler.timesteps is empty after set_timesteps()")

    ts_list = [int(x.item()) for x in ts]
    ts_sorted = np.array(sorted(set(ts_list)), dtype=np.int64)
    if ts_sorted.size < 5:
        raise RuntimeError(f"Expected at least 5 unique timesteps, got {ts_sorted.size}")

    idxs = np.linspace(0, ts_sorted.size - 1, 5)
    idxs = np.round(idxs).astype(int).tolist()
    out = [int(ts_sorted[i]) for i in idxs]
    if len(set(out)) != 5:
        raise RuntimeError(f"Expected 5 unique timesteps, got {out}")
    return out


def _argv_passthrough(args: argparse.Namespace, keys: Iterable[str]) -> list[str]:
    out: list[str] = []
    for k in keys:
        v = getattr(args, k)
        if isinstance(v, bool):
            if v:
                out.append(f"--{k.replace('_', '-')}")
        elif v is None:
            continue
        else:
            s = str(v)
            if s == "":
                continue
            out.extend([f"--{k.replace('_', '-')}", s])
    return out


def _resolve_python(repo_root: Path, explicit: str) -> str:
    """
    Resolve which Python interpreter to use for subprocesses.

    Priority:
      1) --python (explicit)
      2) $PYTHON (explicit)
      3) active venv via $VIRTUAL_ENV/bin/python
      4) common repo-local venv paths: .venv/bin/python, venv/bin/python
      5) sys.executable
    """
    if str(explicit).strip():
        # Do NOT resolve symlinks: venv python is often a symlink and must be invoked via the link path.
        return str(Path(explicit).expanduser().absolute())

    env_py = os.environ.get("PYTHON", "").strip()
    if env_py:
        return str(Path(env_py).expanduser().absolute())

    venv = os.environ.get("VIRTUAL_ENV", "").strip()
    if venv:
        cand = Path(venv) / "bin" / "python"
        if cand.exists():
            return str(cand)

    for rel in [
        Path(".venv/bin/python"),
        Path(".venv/bin/python3"),
        Path("venv/bin/python"),
        Path("venv/bin/python3"),
    ]:
        # Do NOT resolve symlinks (same reason as above).
        cand = (repo_root / rel)
        if cand.exists():
            return str(cand)

    return sys.executable


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mvtec-root", type=str, required=True)
    p.add_argument("--categories", type=str, default="*")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--timesteps",
        type=str,
        default="",
        help="Optional override: comma-separated list of 5 timesteps to sweep (bypasses diffusers scheduler loading).",
    )
    p.add_argument(
        "--python",
        type=str,
        default="",
        help="Optional interpreter for subprocesses. If omitted, uses $VIRTUAL_ENV/bin/python when available, else falls back to sys.executable.",
    )

    # Model selection pass-through (generator)
    p.add_argument("--classifier", type=str, default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--classifier-mode", type=str, default="zero_shot", choices=["zero_shot", "finetuned"])
    p.add_argument("--classifier-checkpoint", type=str, default="")

    p.add_argument("--diffusion-mode", type=str, default="zero_shot", choices=["zero_shot", "finetuned"])
    p.add_argument("--diffusion-model-id", type=str, default="google/ddpm-ema-celebahq-256")
    p.add_argument("--diffusion-model-path", type=str, default="")
    p.add_argument(
        "--diffusion-score-denom",
        type=str,
        default="sigma",
        choices=["sigma_sq", "sigma"],
        help="Score conversion denom passed to generator. Use sigma to avoid blow-ups at small timesteps.",
    )
    p.add_argument("--mock-models", action="store_true", help="Pass through to generator for offline smoke-testing.")
    p.add_argument("--mock-num-classes", type=int, default=1000, help="Pass through to generator when --mock-models.")

    p.add_argument("--size", type=int, default=256)
    p.add_argument("--heatmap-mode", type=str, default="full_stein_resnet", choices=["score_norm", "full_stein_resnet"])
    p.add_argument("--stein-class-mode", type=str, default="fixed", choices=["fixed", "predicted"])
    p.add_argument("--stein-fixed-class-idx", type=int, default=0)
    p.add_argument("--stein-topk", type=int, default=5)
    p.add_argument("--stein-map-nonlinearity", type=str, default="abs", choices=["raw", "abs", "square", "relu"])
    p.add_argument("--max-images", type=int, default=0)

    # Eval/viz settings
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--reference", type=str, default="train_good", choices=["train_good", "test_good"])
    p.add_argument("--ref-max-images", type=int, default=50)
    p.add_argument("--ref-pixel-subsample", type=int, default=200_000)
    p.add_argument("--viz-n-per-category", type=int, default=12)
    p.add_argument("--viz-sample-mode", type=str, default="mixed", choices=["mixed", "good_only", "anomalous_only"])
    p.add_argument("--seed", type=int, default=0)

    # Output
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--out-root", type=str, default="")

    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    mvtec_root = Path(args.mvtec_root).expanduser().resolve()

    # Determine which diffusion identifier to use
    if str(args.diffusion_mode) == "finetuned":
        model_id_or_dir = str(args.diffusion_model_path).strip()
        if not model_id_or_dir:
            raise ValueError("--diffusion-model-path is required when --diffusion-mode=finetuned")
    else:
        model_id_or_dir = str(args.diffusion_model_id).strip()

    if str(args.timesteps).strip():
        parts = [p.strip() for p in str(args.timesteps).split(",") if p.strip()]
        timesteps = [int(x) for x in parts]
        if len(timesteps) != 5:
            raise ValueError(f"--timesteps must contain exactly 5 comma-separated ints. Got {timesteps}")
    else:
        timesteps = _compute_5_scheduler_timesteps(model_id_or_dir=model_id_or_dir)

    # Output root
    if str(args.out_root).strip():
        out_root = Path(args.out_root).expanduser().resolve()
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = str(args.run_name).strip() or f"mvtec_sweep_timesteps_{ts}"
        out_root = (repo_root / "results" / name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "mvtec_root": str(mvtec_root),
        "device": str(args.device),
        "alpha": float(args.alpha),
        "reference": str(args.reference),
        "timesteps": timesteps,
        "heatmap_mode": str(args.heatmap_mode),
        "diffusion_mode": str(args.diffusion_mode),
        "diffusion_model_id_or_dir": model_id_or_dir,
        "timesteps_override": (str(args.timesteps).strip() or None),
        "classifier_mode": str(args.classifier_mode),
        "classifier": str(args.classifier),
        "time": time.time(),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("[sweep] out_root=", out_root)
    print("[sweep] timesteps=", timesteps)

    gen_script = str((repo_root / "scripts" / "generate_mvtec_heatmaps_pretrained.py").resolve())
    eval_script = str((repo_root / "scripts" / "evaluate_mvtec_localization.py").resolve())
    viz_script = str((repo_root / "scripts" / "visualize_mvtec_pixel_anomalies.py").resolve())
    py = _resolve_python(repo_root, explicit=str(args.python))
    print("[sweep] python=", py)

    # Ensure repo root is importable for subprocesses (so `import src...` works).
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "").strip()
    repo_pp = str(repo_root)
    env["PYTHONPATH"] = (repo_pp if not existing_pp else f"{repo_pp}:{existing_pp}")

    # Generator pass-through args
    gen_keys = [
        "categories",
        "device",
        "mock_models",
        "mock_num_classes",
        "classifier",
        "classifier_mode",
        "classifier_checkpoint",
        "diffusion_model_id",
        "diffusion_mode",
        "diffusion_model_path",
        "diffusion_score_denom",
        "size",
        "heatmap_mode",
        "stein_class_mode",
        "stein_fixed_class_idx",
        "stein_topk",
        "stein_map_nonlinearity",
        "max_images",
    ]
    gen_common = _argv_passthrough(args, gen_keys)

    # Eval args
    eval_common = [
        "--mvtec-root",
        str(mvtec_root),
        "--categories",
        str(args.categories),
        "--alpha",
        str(float(args.alpha)),
        "--reference",
        str(args.reference),
        "--ref-max-images",
        str(int(args.ref_max_images)),
        "--ref-pixel-subsample",
        str(int(args.ref_pixel_subsample)),
        "--score-transform",
        "percentile",
        "--map-preprocess",
        "raw",
        "--seed",
        str(int(args.seed)),
    ]

    # Viz args
    viz_common = [
        "--mvtec-root",
        str(mvtec_root),
        "--categories",
        str(args.categories),
        "--alpha",
        str(float(args.alpha)),
        "--reference",
        str(args.reference),
        "--ref-max-images",
        str(int(args.ref_max_images)),
        "--ref-pixel-subsample",
        str(int(args.ref_pixel_subsample)),
        "--map-preprocess",
        "raw",
        "--n-per-category",
        str(int(args.viz_n_per_category)),
        "--sample-mode",
        str(args.viz_sample_mode),
        "--seed",
        str(int(args.seed)),
    ]

    for t in timesteps:
        t_dir = out_root / f"timestep_{t}"
        heatmaps_dir = t_dir / "heatmaps"
        heatmaps_dir.mkdir(parents=True, exist_ok=True)

        # 1) Generate train/good reference heatmaps
        _run(
            [
                py,
                gen_script,
                "--mvtec-root",
                str(mvtec_root),
                "--heatmaps-dir",
                str(heatmaps_dir),
                "--split",
                "train",
                "--only-good",
                "--diffusion-timestep",
                str(int(t)),
                *gen_common,
            ],
            cwd=repo_root,
            env=env,
        )

        # 2) Generate test heatmaps
        _run(
            [
                py,
                gen_script,
                "--mvtec-root",
                str(mvtec_root),
                "--heatmaps-dir",
                str(heatmaps_dir),
                "--split",
                "test",
                "--diffusion-timestep",
                str(int(t)),
                *gen_common,
            ],
            cwd=repo_root,
            env=env,
        )

        # 3) Eval (two_sided + percentile)
        eval_dir = t_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        _run(
            [
                py,
                eval_script,
                "--mvtec-root",
                str(mvtec_root),
                "--heatmaps-dir",
                str(heatmaps_dir),
                "--out-dir",
                str(eval_dir),
                "--tail",
                "two_sided",
                *eval_common,
            ],
            cwd=repo_root,
            env=env,
        )

        # 4) Viz (two_sided)
        viz_dir = t_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        _run(
            [
                py,
                viz_script,
                "--mvtec-root",
                str(mvtec_root),
                "--heatmaps-dir",
                str(heatmaps_dir),
                "--out-dir",
                str(viz_dir),
                "--tail",
                "two_sided",
                *viz_common,
            ],
            cwd=repo_root,
            env=env,
        )

    print("[sweep] done:", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

