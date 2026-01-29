"""
Pretrained diffusion score wrapper for 256×256 images.

This is intentionally **MVTec-only** and opt-in:
- It does not modify any existing CIFAR/adversarial pipelines.
- It does not reuse src/ddpm_score.py (which is CIFAR-10 specific).

Contract:
- Input x is expected to be float tensor in [0, 1], shape (B, 3, H, W).
- The wrapper returns per-pixel score s(x) with same shape as x.

Note: Diffusion checkpoints differ in preprocessing conventions. We keep this configurable and
try to follow common diffusers practice: model input in [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DiffusionScoreConfig:
    model_id: str
    timestep: Optional[int] = None
    denom_mode: str = "sigma_sq"  # 'sigma_sq' (legacy) or 'sigma' (recommended)


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


class DiffusionScore256(nn.Module):
    """
    Generic score wrapper for diffusers DDPMPipeline-style models.

    For a DDPM model predicting epsilon at timestep t:
      score(x_t, t) ≈ -eps_theta(x_t, t) / sigma_t^2
    """

    def __init__(
        self,
        *,
        model_id: str,
        timestep: Optional[int] = None,
        denom_mode: str = "sigma_sq",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.model_id = str(model_id)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        # Lazy import so the rest of the repo doesn't require diffusers unless you use this.
        from diffusers import DDPMPipeline  # type: ignore

        print(f"[mvtec] loading diffusion pipeline: {self.model_id}")
        pipe = DDPMPipeline.from_pretrained(self.model_id)
        pipe = pipe.to(self.device)

        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.unet.eval()
        self.prediction_type = str(getattr(getattr(self.scheduler, "config", None), "prediction_type", "unknown"))
        self.variance_type = str(getattr(getattr(self.scheduler, "config", None), "variance_type", "unknown"))
        self.denom_mode = str(denom_mode)
        if self.denom_mode not in {"sigma_sq", "sigma"}:
            raise ValueError(f"Unsupported denom_mode='{self.denom_mode}'. Use 'sigma' or 'sigma_sq'.")

        if timestep is None:
            # Use smallest-noise timestep if available
            if hasattr(self.scheduler, "timesteps") and len(self.scheduler.timesteps) > 0:
                self.timestep = int(self.scheduler.timesteps[0].item())
            else:
                self.timestep = 0
        else:
            self.timestep = int(timestep)

        print(f"[mvtec] diffusion loaded. timestep={self.timestep} device={self.device} dtype={self.dtype}")
        # region agent log
        _agent_log(
            {
                "sessionId": "debug-session",
                "runId": _agent_run_id(),
                "hypothesisId": "H15_score_scale_blowup_due_to_denom_choice",
                "location": "src/score_models/diffusion_score_256.py:DiffusionScore256:init:scheduler_config",
                "message": "scheduler config summary",
                "data": {
                    "model_id": str(self.model_id),
                    "prediction_type": str(self.prediction_type),
                    "variance_type": str(self.variance_type),
                    "timestep": int(self.timestep),
                    "denom_mode": str(self.denom_mode),
                },
            }
        )
        # endregion agent log

    @staticmethod
    def _to_model_range(x01: torch.Tensor) -> torch.Tensor:
        # [0,1] -> [-1,1]
        return x01 * 2.0 - 1.0

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x01: (B,3,H,W) float in [0,1]
        Returns:
            score: (B,3,H,W)
        """
        if x01.ndim != 4 or x01.size(1) != 3:
            raise ValueError(f"Expected x01 shape (B,3,H,W), got {tuple(x01.shape)}")
        x = x01.to(device=self.device, dtype=self.dtype)
        x = torch.clamp(x, 0.0, 1.0)
        x_model = self._to_model_range(x)

        B = x_model.size(0)
        t = torch.full((B,), self.timestep, device=x_model.device, dtype=torch.long)

        # Do NOT wrap in no_grad: Stein needs gradients through the score model.
        eps = self.unet(x_model, t).sample

        # Estimate sigma_t^2 from scheduler if available
        sigma_sq: torch.Tensor
        denom: torch.Tensor
        if hasattr(self.scheduler, "betas") and hasattr(self.scheduler, "alphas_cumprod"):
            betas = self.scheduler.betas.to(device=x_model.device, dtype=x_model.dtype)
            alphas_cumprod = self.scheduler.alphas_cumprod.to(device=x_model.device, dtype=x_model.dtype)
            tt = min(self.timestep, int(betas.numel()) - 1)
            beta_t = betas[tt]
            abar_t = alphas_cumprod[tt]
            # NOTE: For epsilon-predicting DDPMs, the score conversion is often scaled by sigma_t (not sigma_t^2).
            # We expose denom_mode to avoid huge blow-ups at small t when using sigma_t^2.
            # heuristic; stable and monotone in t
            sigma_sq = beta_t * (1.0 - abar_t) / (1.0 - abar_t + 1e-8)
            sigma = torch.sqrt((1.0 - abar_t).clamp_min(1e-20))
            if self.denom_mode == "sigma":
                denom = sigma
            else:
                denom = sigma_sq
            # region agent log
            try:
                n = int(getattr(self, "_agent_logged_timestep_stats", 0))
                if n < 2:
                    # basic eps stats
                    e = eps.detach().float()
                    eps_mean = float(e.mean().item())
                    eps_std = float(e.std().item())
                    eps_l2 = float(torch.mean(e * e).sqrt().item())
                    _agent_log(
                        {
                            "sessionId": "debug-session",
                            "runId": _agent_run_id(),
                            "hypothesisId": "H14_timestep_noise_ordering",
                            "location": "src/score_models/diffusion_score_256.py:DiffusionScore256:forward:timestep_stats",
                            "message": "diffusion timestep stats (alpha_bar and noise level proxy)",
                            "data": {
                                "timestep": int(self.timestep),
                                "tt_used": int(tt),
                                "beta_t": float(beta_t.detach().float().item()),
                                "alpha_bar_t": float(abar_t.detach().float().item()),
                                "one_minus_alpha_bar_t": float((1.0 - abar_t).detach().float().item()),
                                "sigma_sq_used": float(sigma_sq.detach().float().item()),
                                "sigma_used": float(sigma.detach().float().item()),
                                "denom_mode": str(self.denom_mode),
                                "denom_used": float(denom.detach().float().item()),
                                "prediction_type": str(self.prediction_type),
                                "eps_mean": float(eps_mean),
                                "eps_std": float(eps_std),
                                "eps_l2": float(eps_l2),
                            },
                        }
                    )
                    setattr(self, "_agent_logged_timestep_stats", n + 1)
            except Exception:
                pass
            # endregion agent log
        else:
            sigma_sq = torch.tensor(1e-4, device=x_model.device, dtype=x_model.dtype)
            denom = sigma_sq

        score = -eps / (denom + 1e-8)
        return score

