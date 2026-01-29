"""
Pretrained ImageNet classifier loader (MVTec-only helper).

IMPORTANT: This file is placed at `src/pretrained_classifier.py` rather than `src/models/...`
to avoid shadowing the existing `src/models.py` module (creating a `src/models/` package would
change import resolution in the rest of the codebase).

Contract:
- Input x is float tensor of shape (B,3,H,W) in [0,1].
- We internally apply ImageNet normalization (mean/std) and run the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(x01: torch.Tensor) -> torch.Tensor:
    if x01.ndim != 4 or x01.size(1) != 3:
        raise ValueError(f"Expected x shape (B,3,H,W), got {tuple(x01.shape)}")
    mean = torch.tensor(IMAGENET_MEAN, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - mean) / std


@dataclass(frozen=True)
class PretrainedClassifierConfig:
    name: str = "resnet50"
    device: Optional[str] = None
    freeze: bool = True


class PretrainedImageNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        name: str = "resnet50",
        device: Optional[torch.device] = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.name = str(name)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy import to keep dependencies local.
        import torchvision.models as tvm  # type: ignore

        # We use ImageNet weights via torchvision.
        if self.name == "resnet50":
            self.model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            self._feature_dim = 2048
        elif self.name == "resnet18":
            self.model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            self._feature_dim = 512
        else:
            raise ValueError(f"Unsupported classifier name={self.name}")

        self.model = self.model.to(self.device)
        self.model.eval()
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

        # Build a feature extractor (penultimate layer) without depending on torchvision feature_extraction API.
        # For ResNet: features are produced by everything up to avgpool, then flattened.
        self._backbone = nn.Sequential(*list(self.model.children())[:-1]).to(self.device)
        self._backbone.eval()
        if freeze:
            for p in self._backbone.parameters():
                p.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        return int(self._feature_dim)

    def forward_logits(self, x01: torch.Tensor) -> torch.Tensor:
        x = x01.to(self.device, dtype=torch.float32)
        x = torch.clamp(x, 0.0, 1.0)
        x = imagenet_normalize(x)
        return self.model(x)

    def forward_features(self, x01: torch.Tensor) -> torch.Tensor:
        x = x01.to(self.device, dtype=torch.float32)
        x = torch.clamp(x, 0.0, 1.0)
        x = imagenet_normalize(x)
        f = self._backbone(x)  # (B, C, 1, 1)
        return f.flatten(1)

