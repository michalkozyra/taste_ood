"""
MVTec AD dataset loader (classic, 15 categories).

Placed at `src/mvtec_ad.py` (NOT under `src/data/`) to avoid conflicting with the existing
`src/data.py` module in this repo.

Assumptions / conventions (matching common MVTec AD releases):
- Images live under:
    <root>/<category>/train/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/<defect_type>/*.png
    <root>/<category>/ground_truth/<defect_type>/*_mask.png
- For 'good' images, there is no mask; we return an all-zero mask.
- Ground-truth masks are binary PNGs (0 background, 255 anomaly), but we treat mask > 0 as anomaly.

This loader is intentionally minimal and evaluation-oriented: it focuses on correct pairing and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
import zlib


OBJECT_CATEGORIES = {
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
}

TEXTURE_CATEGORIES = {"carpet", "grid", "leather", "tile", "wood"}

ALL_CATEGORIES = sorted(OBJECT_CATEGORIES | TEXTURE_CATEGORIES)


@dataclass(frozen=True)
class MVTecSample:
    category: str
    split: str  # 'train' or 'test'
    defect_type: str  # 'good' or specific defect type
    image_path: Path
    mask_path: Optional[Path]  # None for good

    @property
    def is_anomalous(self) -> bool:
        return self.defect_type != "good"


@dataclass(frozen=True)
class MVTecTrainGoodSample:
    """
    Train-only sample used for supervised fine-tuning on MVTec category labels.

    We use ONLY train/good images and label = category index (15-way).
    """

    category: str
    category_idx: int
    image_path: Path


def build_mvtec_category_mapping(categories: Optional[list[str]] = None) -> Tuple[dict[str, int], list[str]]:
    """
    Build a stable category_to_idx mapping (15 categories by default).
    """
    cats = categories or ALL_CATEGORIES
    cats_sorted = sorted([str(c) for c in cats])
    category_to_idx = {c: i for i, c in enumerate(cats_sorted)}
    idx_to_category = list(cats_sorted)
    return category_to_idx, idx_to_category


def discover_mvtec_train_good_samples(
    root: Path, *, categories: Optional[list[str]] = None
) -> list[MVTecTrainGoodSample]:
    """
    Enumerate train/good images across categories, labeled by category index (15-way).

    Fail-fast:
      - If any category is missing train/good, this will raise.
    """
    root = Path(root).expanduser().resolve()
    cats = categories or ALL_CATEGORIES
    category_to_idx, _ = build_mvtec_category_mapping(list(cats))

    out: list[MVTecTrainGoodSample] = []
    for cat in cats:
        cat = str(cat)
        good_dir = root / cat / "train" / "good"
        if not good_dir.exists():
            raise FileNotFoundError(f"Missing directory: {good_dir}")
        imgs = sorted([p for p in good_dir.iterdir() if p.is_file() and _is_image_file(p)])
        if len(imgs) == 0:
            raise FileNotFoundError(f"No train/good images found under: {good_dir}")
        for p in imgs:
            out.append(MVTecTrainGoodSample(category=cat, category_idx=int(category_to_idx[cat]), image_path=p))
    return out


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _find_mask_path(*, root: Path, category: str, defect_type: str, image_path: Path) -> Path:
    """
    MVTec convention: mask filename is image stem + '_mask.png' under ground_truth/<defect_type>/.
    """
    gt_dir = root / category / "ground_truth" / defect_type
    cand = gt_dir / f"{image_path.stem}_mask.png"
    if cand.exists():
        return cand
    # Some variants use identical filename with _mask already present.
    if image_path.stem.endswith("_mask"):
        cand2 = gt_dir / f"{image_path.stem}.png"
        if cand2.exists():
            return cand2
    raise FileNotFoundError(f"Missing mask for image: {image_path} expected at: {cand}")


def discover_mvtec_samples(root: Path, *, categories: Optional[list[str]] = None, split: str = "test") -> list[MVTecSample]:
    root = Path(root).expanduser().resolve()
    if categories is None:
        categories = ALL_CATEGORIES
    split = str(split)
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    samples: list[MVTecSample] = []
    for cat in categories:
        cat_dir = root / cat / split
        if not cat_dir.exists():
            raise FileNotFoundError(f"Missing category directory: {cat_dir}")

        for defect_dir in sorted([p for p in cat_dir.iterdir() if p.is_dir()]):
            defect_type = defect_dir.name
            for img_path in sorted([p for p in defect_dir.iterdir() if p.is_file() and _is_image_file(p)]):
                if split == "train":
                    # In classic MVTec AD: train contains only 'good'. Keep robust though.
                    mask_path = None
                else:
                    mask_path = None if defect_type == "good" else _find_mask_path(
                        root=root, category=cat, defect_type=defect_type, image_path=img_path
                    )
                samples.append(
                    MVTecSample(
                        category=cat,
                        split=split,
                        defect_type=defect_type,
                        image_path=img_path,
                        mask_path=mask_path,
                    )
                )
    return samples


def subsample_mvtec_samples(
    samples: list[MVTecSample],
    *,
    frac: float,
    seed: int = 0,
    min_per_group: int = 1,
) -> list[MVTecSample]:
    """
    Deterministically subsample MVTec samples.

    Behavior (matches "quick sweep" usage):
    - For train split: subsample per-category (10% of train images per category).
    - For test split: subsample per (category, defect_type) group (10% of each test subcategory).
    - Always keep at least `min_per_group` samples per group (if the group is non-empty).

    The selection is deterministic given `seed` and the group key, and does not depend on input order.
    """
    if frac >= 1.0:
        return list(samples)
    if frac <= 0.0:
        raise ValueError(f"frac must be in (0,1]. Got {frac}")
    if min_per_group < 1:
        raise ValueError(f"min_per_group must be >= 1. Got {min_per_group}")

    # Grouping rule: train -> (category); test -> (category, defect_type)
    groups: dict[tuple[str, ...], list[MVTecSample]] = {}
    for s in samples:
        if str(s.split) == "train":
            key = (str(s.category),)
        else:
            key = (str(s.category), str(s.defect_type))
        groups.setdefault(key, []).append(s)

    out: list[MVTecSample] = []
    for key, items in sorted(groups.items(), key=lambda kv: kv[0]):
        if len(items) == 0:
            continue
        # Deterministic per-group RNG.
        key_bytes = ("|".join(key)).encode("utf-8")
        key_hash = int(zlib.adler32(key_bytes)) & 0xFFFFFFFF
        rng = np.random.default_rng(int(seed) ^ key_hash)

        n = len(items)
        n_keep = int(np.floor(float(frac) * float(n)))
        if n_keep < int(min_per_group):
            n_keep = int(min_per_group)
        if n_keep > n:
            n_keep = n

        # Sample without replacement, then sort for stable downstream iteration.
        idx = rng.choice(n, size=n_keep, replace=False)
        chosen = [items[int(i)] for i in idx.tolist()]
        chosen_sorted = sorted(chosen, key=lambda s: str(s.image_path))
        out.extend(chosen_sorted)

    return out


class MVTecAD:
    """
    Simple, evaluation-oriented MVTec AD dataset.

    Returns:
      - image: PIL.Image (RGB)
      - mask: np.ndarray[H,W] uint8 (0/1)
      - is_anomalous: bool
      - category: str
      - defect_type: str
      - rel_id: str (stable identifier within root; used for matching predictions)
    """

    def __init__(
        self,
        root: str | Path,
        *,
        categories: Optional[list[str]] = None,
        split: str = "test",
        image_transform: Optional[Callable[[Image.Image], object]] = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.categories = categories or ALL_CATEGORIES
        self.split = split
        self.image_transform = image_transform
        self.samples = discover_mvtec_samples(self.root, categories=self.categories, split=self.split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        img_out = self.image_transform(img) if self.image_transform is not None else img

        if s.mask_path is None:
            # Use image size for dummy mask.
            w, h = img.size
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            m = Image.open(s.mask_path).convert("L")
            mask = (np.asarray(m) > 0).astype(np.uint8)

        rel_id = str(s.image_path.relative_to(self.root))
        return img_out, mask, s.is_anomalous, s.category, s.defect_type, rel_id

