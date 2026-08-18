"""Dataset discovery and deterministic, stratified splitting.

Two on-disk layouts are supported and auto-detected:

1. **Class sub-folders**:  ``<root>/<class_name>/*.{bmp,png,jpg,...}``
2. **Flat NEU-CLS layout**: ``<root>/Cr_1.bmp`` etc., where the two-letter
   filename prefix identifies the class.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from idefect import NEU_PREFIX_TO_CLASS
from idefect.config import PreprocessingConfig
from idefect.data.preprocessing import load_grayscale, preprocess_image

IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

Sample = Tuple[str, int]  # (image_path, label_index)


@dataclass
class DatasetSplit:
    """A deterministic train/val/test split with a fixed class ordering."""

    train: List[Sample]
    val: List[Sample]
    test: List[Sample]
    class_names: List[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def summary(self) -> str:
        return (
            f"{self.num_classes} classes | "
            f"train={len(self.train)} val={len(self.val)} test={len(self.test)}"
        )


def _iter_images(directory: Path):
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _has_class_subfolders(root: Path) -> bool:
    for child in root.iterdir():
        if child.is_dir():
            if any(_iter_images(child)):
                return True
    return False


def load_samples(root: str | Path) -> Tuple[List[Sample], List[str]]:
    """Scan ``root`` and return ``(samples, class_names)``.

    ``samples`` is a list of ``(path, label_index)`` where ``label_index`` is the
    position of the class in the returned, alphabetically-sorted ``class_names``.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            "Download NEU (see data/README.md) or generate synthetic data with "
            "scripts/generate_dummy_data.py."
        )

    pairs: List[Tuple[str, str]] = []  # (path, class_name)

    if _has_class_subfolders(root):
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            for img_path in _iter_images(child):
                pairs.append((str(img_path), child.name))
    else:
        # Flat NEU-CLS layout: infer class from the filename prefix.
        for img_path in _iter_images(root):
            cls = _class_from_filename(img_path.name)
            if cls is None:
                continue
            pairs.append((str(img_path), cls))

    if not pairs:
        raise RuntimeError(
            f"No images found under {root}. Expected class sub-folders or the "
            "flat NEU-CLS layout (Cr_*, In_*, Pa_*, PS_*, RS_*, Sc_*)."
        )

    class_names = sorted({cls for _, cls in pairs})
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    samples = [(path, class_to_idx[cls]) for path, cls in pairs]
    return samples, class_names


def _class_from_filename(name: str) -> Optional[str]:
    """Map a flat NEU-CLS filename (e.g. ``PS_12.bmp``) to a class name."""
    stem = name.split("_")[0]
    # Try the exact two-letter code first, then a case-insensitive match.
    if stem in NEU_PREFIX_TO_CLASS:
        return NEU_PREFIX_TO_CLASS[stem]
    for prefix, cls in NEU_PREFIX_TO_CLASS.items():
        if stem.lower() == prefix.lower():
            return cls
    return None


def make_splits(
    samples: Sequence[Sample],
    class_names: Sequence[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> DatasetSplit:
    """Deterministic **stratified** split.

    The split is reproducible for a given ``seed`` and keeps class proportions
    roughly balanced across train/val/test. Implemented with a per-class shuffle
    so there is no hard dependency on scikit-learn's splitter here.
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    rng = random.Random(seed)
    by_class: dict[int, List[Sample]] = {i: [] for i in range(len(class_names))}
    for path, label in samples:
        by_class[label].append((path, label))

    train: List[Sample] = []
    val: List[Sample] = []
    test: List[Sample] = []

    for label in sorted(by_class):
        items = sorted(by_class[label])  # deterministic base order
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # Guard against rounding overflow.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    # Shuffle the concatenated splits so classes are interleaved.
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return DatasetSplit(
        train=train,
        val=val,
        test=test,
        class_names=list(class_names),
    )


def load_split_arrays(
    samples: Sequence[Sample],
    cfg: PreprocessingConfig,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load & preprocess a list of samples into ``(X, y)`` NumPy arrays.

    ``X`` has shape ``(N, image_size, image_size)`` (float32) and ``y`` shape
    ``(N,)`` (int64). Used by the classical (HOG) branch.
    """
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for path, label in samples:
        img = preprocess_image(load_grayscale(path), cfg, image_size)
        xs.append(img)
        ys.append(label)
    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# PyTorch dataset (used by the CNN branch). torch is imported lazily so the
# classical-only path and the unit tests do not require it.
# ---------------------------------------------------------------------------
def _augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light on-the-fly augmentation: random flips and a small rotation."""
    if rng.random() < 0.5:
        img = np.fliplr(img)
    if rng.random() < 0.5:
        img = np.flipud(img)
    k = int(rng.integers(0, 4))  # 0/90/180/270 deg rotations are label-safe
    if k:
        img = np.rot90(img, k)
    return np.ascontiguousarray(img)


try:  # pragma: no cover - exercised only when torch is present
    from torch.utils.data import Dataset as _TorchDataset
    import torch

    class NEUImageDataset(_TorchDataset):
        """PyTorch dataset yielding ``(image_tensor, label)`` pairs."""

        def __init__(
            self,
            samples: Sequence[Sample],
            cfg: PreprocessingConfig,
            image_size: int,
            augment: bool = False,
            seed: int = 42,
        ):
            self.samples = list(samples)
            self.cfg = cfg
            self.image_size = image_size
            self.augment = augment
            self._rng = np.random.default_rng(seed)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            path, label = self.samples[idx]
            img = preprocess_image(
                load_grayscale(path), self.cfg, self.image_size
            )
            if self.augment:
                img = _augment(img, self._rng)
            tensor = torch.from_numpy(img).unsqueeze(0).float()  # (1, H, W)
            return tensor, int(label)

except ImportError:  # pragma: no cover

    class NEUImageDataset:  # type: ignore
        """Placeholder raised if instantiated without torch installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for NEUImageDataset (the CNN branch). "
                "Install it with `pip install torch torchvision`."
            )
