"""
Classifier-geometry diagnostic detectors.

These are *not* Stein operators. They help answer:
  "Is the classifier test function geometry (f, ∇f) separating ID vs OOD by itself?"
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_ood.api import Detector

from ..utils import get_device


class GradFNormDetector(Detector):
    """
    Diagnostic detector: score(x) = || ∇_x f(x) ||_2

    For classification, f(x) is a scalar derived from the classifier:
      - predicted_class_prob: softmax(logits)[argmax] (data-dependent)
      - fixed_class_prob:     softmax(logits)[fixed_class_idx] (fixed test function)

    Higher score means "sharper sensitivity" of f to the input.
    """

    def __init__(
        self,
        model: nn.Module,
        classification_scalar_mode: Literal["predicted_class_prob", "fixed_class_prob"] = "predicted_class_prob",
        fixed_class_idx: int = 0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model = model
        self.classification_scalar_mode = classification_scalar_mode
        self.fixed_class_idx = int(fixed_class_idx)
        self.device = device if device is not None else get_device()

    def fit(self, *args, **kwargs):  # noqa: D401
        """No-op: this detector does not require fitting."""
        return self

    @torch.no_grad()
    def _forward_probs(self, x: Tensor) -> Tensor:
        self.model.eval()
        logits = self.model(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        x_req = x.clone().detach().requires_grad_(True)
        self.model.eval()

        with torch.enable_grad():
            probs = self._forward_probs(x_req)  # (B, K)
            if self.classification_scalar_mode == "predicted_class_prob":
                k = torch.argmax(probs, dim=1)
            elif self.classification_scalar_mode == "fixed_class_prob":
                k = torch.full(
                    (probs.size(0),),
                    fill_value=self.fixed_class_idx,
                    device=probs.device,
                    dtype=torch.long,
                )
                k = torch.clamp(k, 0, probs.size(1) - 1)
            else:
                raise ValueError(f"Unknown classification_scalar_mode={self.classification_scalar_mode!r}")

            f_vals = probs[torch.arange(probs.size(0), device=probs.device), k]  # (B,)
            grads = torch.autograd.grad(f_vals.sum(), x_req, create_graph=False)[0]

        grads = grads.view(grads.size(0), -1)
        return torch.norm(grads, p=2, dim=1)

    def fit_features(self, x: Tensor, y: Tensor):  # noqa: D401
        """Not applicable (requires full input); use fit() or treat as no-op."""
        return self

    def predict_features(self, x: Tensor) -> Tensor:
        """Not applicable for feature-based API; use predict(x) on inputs."""
        raise NotImplementedError("GradFNormDetector operates on input tensors; use predict().")

