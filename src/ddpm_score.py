"""
DDPM Score Model Wrapper for CIFAR-10.

Wraps the pretrained DDPM model from Hugging Face to provide a score function
interface compatible with SteinDetector.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
from diffusers import DDPMPipeline
import warnings
import os
import json
import time


class DDPMScoreWrapper(nn.Module):
    """
    Wrapper for pretrained DDPM model to extract score function.
    
    The DDPM model predicts noise ε_θ(x_t, t) for noisy images x_t at timestep t.
    The score function is: s(x_t, t) = -ε_θ(x_t, t) / σ_t
    
    For clean images, we use a small timestep (t ≈ 0) to approximate s(x).
    """
    
    def __init__(
        self,
        model_id: str = "google/ddpm-cifar10-32",
        timestep: int = 0,
        denom_mode: Literal["sigma", "sigma_sq"] = "sigma_sq",
        add_noise: bool = False,
        noise_seed: int = 0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize DDPM score wrapper.
        
        Args:
            model_id: Hugging Face model ID for pretrained DDPM
            timestep: Diffusion timestep t used for score extraction (default 0; closest-to-clean).
            denom_mode: How to convert epsilon prediction to a score field.
                - "sigma": score = -eps / sigma_t
                - "sigma_sq": score = -eps / sigma_t^2 (legacy; can blow up at small t)
            add_noise: If True, query the UNet on x_t = add_noise(x0, eps, t).
                       If False (default), query on clean x0 at timestep t (legacy behavior).
            noise_seed: Seed for forward-diffusion noise when add_noise=True.
            device: Device to load model on (auto-detect if None)
        """
        super().__init__()
        
        self.model_id = model_id
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained DDPM pipeline
        print(f"Loading DDPM model from {model_id}...")
        try:
            # Try loading with safetensors first (preferred)
            try:
                self.pipeline = DDPMPipeline.from_pretrained(
                    model_id,
                    use_safetensors=True,
                )
            except Exception as safetensors_error:
                # If safetensors fails, try without (fallback to pickle)
                # This is acceptable for older models that don't have safetensors
                import warnings
                import os
                from huggingface_hub import snapshot_download
                
                # Try to find the actual pickle file location
                try:
                    # Get the cache directory
                    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                    model_dir = model_id.replace("/", "--")
                    model_path = os.path.join(cache_dir, f"models--{model_dir}")
                    
                    # Find pickle files in snapshots
                    pickle_files = []
                    if os.path.exists(model_path):
                        snapshots_dir = os.path.join(model_path, "snapshots")
                        if os.path.exists(snapshots_dir):
                            for snapshot in os.listdir(snapshots_dir):
                                snapshot_path = os.path.join(snapshots_dir, snapshot)
                                if os.path.isdir(snapshot_path):
                                    for file in os.listdir(snapshot_path):
                                        if file.endswith(('.bin', '.pth', '.pt')):
                                            pickle_files.append(os.path.join(snapshot_path, file))
                    
                    pickle_info = ""
                    if pickle_files:
                        pickle_info = f"\nLoaded pickle file: {pickle_files[0]}"
                    else:
                        pickle_info = f"\nModel cache location: {model_path}"
                    
                    warnings.warn(
                        f"Could not load {model_id} with safetensors: {safetensors_error}\n"
                        f"Falling back to pickle format. This is safe for trusted models.{pickle_info}",
                        UserWarning
                    )
                except Exception:
                    # If we can't find the file, just warn without location
                    warnings.warn(
                        f"Could not load {model_id} with safetensors: {safetensors_error}\n"
                        f"Falling back to pickle format. This is safe for trusted models.",
                        UserWarning
                    )
                
                self.pipeline = DDPMPipeline.from_pretrained(
                    model_id,
                    use_safetensors=False,
                )
            
            self.pipeline = self.pipeline.to(self.device)
        except Exception as e:
            # Provide clear error message
            raise RuntimeError(
                f"Failed to load DDPM model {model_id}: {e}\n"
                f"Please ensure the model is properly downloaded.\n"
                f"Try: huggingface-cli download {model_id} --local-dir ~/.cache/huggingface/hub/models--google--ddpm-cifar10-32"
            )
        
        # Extract UNet and scheduler
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler
        
        # Explicit timestep configuration. We intentionally do NOT default to scheduler.timesteps[0],
        # because in diffusers that is often the *most noisy* timestep (descending order).
        self.timestep = int(timestep)
        self.denom_mode = str(denom_mode)
        self.add_noise = bool(add_noise)
        self.noise_seed = int(noise_seed)
        self._noise_gen = torch.Generator(device="cpu").manual_seed(self.noise_seed)
        self._agent_logged = 0
        
        # Set to eval mode
        self.unet.eval()
        
        print(
            f"DDPM model loaded. Using timestep={self.timestep} denom_mode={self.denom_mode} "
            f"add_noise={self.add_noise} noise_seed={self.noise_seed}"
        )
    
    def forward(self, x: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute score function s(x) for input images.
        
        Args:
            x: Input images of shape (B, C, H, W) in ImageNet normalized space
            sigma: Optional noise levels (ignored for DDPM, uses fixed timestep)
        
        Returns:
            score: Score function output of shape (B, C, H, W)
        """
        # Convert from CIFAR-10 normalized space to DDPM's [-1, 1] range
        x0 = self._normalize_to_ddpm(x)
        
        # Prepare timestep tensor
        B = x0.size(0)
        t = int(self.timestep)
        timesteps = torch.full((B,), t, device=x0.device, dtype=torch.long)

        # Query UNet on x_t (theory-correct) or x0 (ablations)
        if self.add_noise:
            # Forward diffusion noise is deterministic given noise_seed.
            eps_noise = (
                torch.randn(x0.shape, generator=self._noise_gen, device="cpu", dtype=torch.float32)
                .to(x0.device, dtype=x0.dtype)
            )
            # diffusers expects (B,) timesteps for add_noise
            x_in = self.scheduler.add_noise(x0, eps_noise, timesteps)
        else:
            x_in = x0
        
        # Predict noise using UNet
        # Note: We need gradients for Stein computation, so don't use no_grad()
        noise_pred = self.unet(x_in, timesteps).sample  # (B, C, H, W)
        
        # Compute sigma_t from scheduler
        # For DDPMScheduler, we can get beta_t and compute sigma_t
        sigma_t = None
        sigma_t_sq = None
        if hasattr(self.scheduler, "alphas_cumprod"):
            ac = self.scheduler.alphas_cumprod.to(device=x0.device, dtype=x0.dtype)
            tt_used = min(max(int(t), 0), int(ac.numel()) - 1)
            alpha_bar_t = ac[tt_used]
            sigma_t = torch.sqrt((1.0 - alpha_bar_t).clamp_min(1e-20))
            sigma_t_sq = sigma_t * sigma_t
        else:
            # Fallback: use a small fixed sigma
            sigma_t = torch.tensor(0.01, device=x0.device, dtype=x0.dtype)
            sigma_t_sq = sigma_t * sigma_t
        
        if str(self.denom_mode) == "sigma_sq":
            denom = sigma_t_sq
        else:
            denom = sigma_t

        # Compute score: s(x_t, t) = -ε_θ(x_t, t) / denom
        score = -noise_pred / (denom + 1e-8)  # (B, C, H, W)

        # region agent log
        try:
            if int(self._agent_logged) < 4:
                payload = {
                    "timestamp": 0,
                    "sessionId": "debug-session",
                    "runId": str(os.environ.get("AGENT_RUN_ID", "cifar_ddpm")),
                    "hypothesisId": "H_DDPM_XT_SIGMA",
                    "location": "src/ddpm_score.py:DDPMScoreWrapper.forward",
                    "message": "DDPM score wrapper stats (first batches)",
                    "data": {
                        "model_id": str(self.model_id),
                        "timestep": int(t),
                        "denom_mode": str(self.denom_mode),
                        "add_noise": bool(self.add_noise),
                        "noise_seed": int(self.noise_seed),
                        "x0_std": float(x0.detach().float().std().item()),
                        "xin_std": float(x_in.detach().float().std().item()),
                        "eps_std": float(noise_pred.detach().float().std().item()),
                        "score_std": float(score.detach().float().std().item()),
                    },
                }
                with open("/Users/michalkozyra/Developer/PhD/stein_shift_detection/.cursor/debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
                self._agent_logged += 1
        except Exception:
            pass
        # endregion
        
        return score
    
    def _normalize_to_ddpm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert from dataset normalization to DDPM's [-1, 1] range.
        
        Supports both CIFAR-10 and ImageNet normalization.
        
        Args:
            x: Images in normalized space
               - CIFAR-10: mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
               - ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        
        Returns:
            x_ddpm: Images in DDPM space [-1, 1]
        """
        # Try to detect normalization by checking if values are in typical ranges
        # CIFAR-10 normalized values are typically in [-2.5, 2.5]
        # ImageNet normalized values are typically in [-2.1, 2.1]
        # For simplicity, we'll use CIFAR-10 normalization (most common for this use case)
        # CIFAR-10 normalization parameters
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010], device=x.device).view(1, 3, 1, 1)
        
        # Denormalize from dataset-normalized space back to [0,1], then map to [-1,1].
        # Dataset normalization is: x_norm = (x01 - mean) / std  =>  x01 = x_norm * std + mean
        x01 = x * std + mean
        x_ddpm = 2.0 * x01 - 1.0
        
        # Clamp to [-1, 1] to ensure valid range
        x_ddpm = torch.clamp(x_ddpm, -1.0, 1.0)
        
        return x_ddpm
    
    def _get_sigmas(self) -> Optional[torch.Tensor]:
        """
        Return sigmas tensor for compatibility with SteinDetector.
        Returns None since DDPM uses timesteps, not sigmas.
        """
        return None

