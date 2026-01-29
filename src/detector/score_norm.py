"""
Score-model diagnostic detectors.

These are *not* Stein operators. They help answer: "is the score model itself separating ID vs OOD?"
"""

from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor

from pytorch_ood.api import Detector

from ..eval_functions import score_at_x
from ..utils import get_device


class ScoreNormDetector(Detector):
    """
    Simple diagnostic detector: score(x) = || s(x) ||_2 where s(x) = grad log p(x).

    Higher score means "more extreme" under the score model; empirically this can correlate with OOD,
    but there is no guarantee. Use it as a probe for score-model quality / mismatch.
    """

    def __init__(
        self,
        score_model: Optional[nn.Module] = None,
        score_function: Optional[Callable[[Tensor], Tensor]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if score_model is None and score_function is None:
            raise ValueError("Either score_model or score_function must be provided.")
        self.score_model = score_model
        self.score_function = score_function
        self.device = device if device is not None else get_device()

    def fit(self, *args, **kwargs):  # noqa: D401
        """No-op: this detector does not require fitting."""
        return self

    def predict(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        if self.score_function is not None:
            s = self.score_function(x)
        else:
            self.score_model.eval()
            # If the model exposes a sigma schedule, use the annealed helper (matches UNetScore usage)
            sigmas = getattr(self.score_model, "sigmas", None)
            if sigmas is not None:
                s = score_at_x(self.score_model, x, sigmas, self.device, use_sigma_min=True)
            else:
                s = self.score_model(x)

        s = s.view(x.size(0), -1)
        return torch.norm(s, p=2, dim=1)

    def fit_features(self, x: Tensor, y: Tensor):  # noqa: D401
        """Not applicable (requires full input); use fit() or treat as no-op."""
        return self

    def predict_features(self, x: Tensor) -> Tensor:
        """Not applicable for feature-based API; use predict(x) on inputs."""
        raise NotImplementedError("ScoreNormDetector operates on input tensors; use predict().")

