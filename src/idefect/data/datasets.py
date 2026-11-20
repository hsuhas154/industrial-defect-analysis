"""Dataset discovery and deterministic, stratified splitting.

Two on-disk layouts are supported and auto-detected:

1. **Class sub-folders**:  ``<root>/<class_name>/*.{bmp,png,jpg,...}``
2. **Flat NEU-CLS layout**: ``<root>/Cr_1.bmp`` etc., where the two-letter
   filename prefix identifies the class.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
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


def pixel_hash(img: np.ndarray) -> str:
    """Hash of decoded pixels (not file bytes): the same image re-saved with
    different metadata still hashes identically."""
    h = hashlib.md5()
    h.update(str(img.shape).encode())
    h.update(np.ascontiguousarray(img).tobytes())
    return h.hexdigest()


def find_duplicate_groups(samples: Sequence[Sample]) -> List[List[str]]:
    """Groups of image paths whose pixels are exactly identical (whole image).

    This is the cheap exact check used by the dataset audit. The split itself
    uses the full near-duplicate detector in :mod:`idefect.data.duplicates`,
    which also finds flipped, rotated, overlapping and re-processed copies.
    """
    by_hash: dict[str, List[str]] = defaultdict(list)
    for path, _ in samples:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            by_hash[pixel_hash(img)].append(str(path))
    return [sorted(g) for g in by_hash.values() if len(g) > 1]


def make_splits(
    samples: Sequence[Sample],
    class_names: Sequence[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    groups: Optional[Sequence[Sequence[str]]] = None,
) -> DatasetSplit:
    """Deterministic, stratified, group-aware split.

    ``groups`` lists images that must stay together (duplicates, overlapping
    crops, re-processed copies). Every group is one indivisible unit, so no
    group can span two splits, even if its members carry different labels.

    Units are stratified by their majority label. Within each class the units
    are shuffled with ``seed`` and then placed largest first into whichever
    split is furthest below its target size, which keeps the 70/15/15 (or
    other) proportions close even when some groups are large. Without groups
    every image is its own unit and the split is exactly proportional.
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    group_of = {}
    for gi, g in enumerate(groups or []):
        for p in g:
            group_of[str(p)] = gi

    # Build units globally (a group can mix labels), then stratify.
    units_by_key: dict = {}
    for path, label in sorted(samples):
        gi = group_of.get(str(path))
        key = ("g", gi) if gi is not None else ("s", str(path))
        units_by_key.setdefault(key, []).append((path, label))
    by_class: dict[int, List[List[Sample]]] = {i: [] for i in range(len(class_names))}
    for unit in units_by_key.values():
        labels = [y for _, y in unit]
        majority = max(sorted(set(labels)), key=labels.count)
        by_class[majority].append(unit)

    rng = random.Random(seed)
    fracs = (train_frac, val_frac, test_frac)
    parts: tuple = ([], [], [])
    for label in sorted(by_class):
        units = sorted(by_class[label])
        rng.shuffle(units)
        units.sort(key=len, reverse=True)  # stable: equal sizes keep shuffled order
        n = sum(len(u) for u in units)
        targets = [n * f for f in fracs]
        filled = [0, 0, 0]
        for unit in units:
            deficits = [targets[i] - filled[i] for i in range(3)]
            dest = max(range(3), key=lambda i: (deficits[i], -i))
            parts[dest].extend(unit)
            filled[dest] += len(unit)

    train, val, test = (list(p) for p in parts)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return DatasetSplit(train=train, val=val, test=test, class_names=list(class_names))


def duplicate_report(cfg, samples: Sequence[Sample], verbose: bool = False, cache_only: bool = False):
    """Run (or load from cache) the near-duplicate detector for ``samples``."""
    from idefect.data.duplicates import detect_duplicates

    return detect_duplicates([p for p, _ in samples], cfg.data.duplicates, use_cache=True,
                             verbose=verbose, cache_only=cache_only)


def build_split(cfg, verbose: bool = False,
                cache_only: bool = False) -> Tuple[List[Sample], List[str], DatasetSplit, List[List[str]]]:
    """Scan the dataset in ``cfg.data.root`` and build the shared split.

    Every branch (classical, CNN, dataset report) calls this, so all of them
    see exactly the same train/val/test partition. Near-duplicate groups (see
    :mod:`idefect.data.duplicates`) are kept inside one split. The detector's
    result is cached, so only the first call pays for it; with ``cache_only``
    a missing cache raises ``DuplicateCacheMissing`` instead of running the
    detection. Returns ``(samples, class_names, split, duplicate_groups)``.
    """
    if getattr(cfg.data, "manifest", ""):
        from idefect.data.stats import verify_manifest

        m = verify_manifest(cfg.data.root, cfg.data.manifest)
        if m.get("checked") and (m["missing"] or m["modified"] or m["unexpected"]):
            raise RuntimeError(
                f"Dataset in {cfg.data.root} does not match {cfg.data.manifest}: "
                f"missing={m['missing'][:5]} modified={m['modified'][:5]} "
                f"unexpected={m['unexpected'][:5]}. Restore the original files "
                "(see data/README.md) or set data.manifest to '' for a different dataset."
            )
    samples, class_names = load_samples(cfg.data.root)
    groups: List[List[str]] = []
    if cfg.data.duplicates.enabled:
        groups = duplicate_report(cfg, samples, verbose=verbose, cache_only=cache_only).groups
    split = make_splits(
        samples, class_names,
        cfg.data.train_frac, cfg.data.val_frac, cfg.data.test_frac,
        cfg.data.split_seed, groups=groups,
    )
    return samples, class_names, split, groups


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
def augment_image(img: np.ndarray, flip_lr: bool, flip_ud: bool, k_rot90: int) -> np.ndarray:
    """Apply a deterministic flip / 90-degree rotation combination.

    NEU defects have no canonical orientation, so horizontal and vertical flips
    and 0/90/180/270 degree rotations are label-preserving. The random choice of
    parameters is made by the caller so this function stays pure and testable.
    """
    if flip_lr:
        img = np.fliplr(img)
    if flip_ud:
        img = np.flipud(img)
    if k_rot90 % 4:
        img = np.rot90(img, k_rot90 % 4)
    return np.ascontiguousarray(img)


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` ``worker_init_fn`` that seeds NumPy and ``random``.

    PyTorch already gives every worker a distinct torch seed that changes each
    epoch (derived from the loader's generator). This propagates that seed to
    NumPy and Python's ``random`` so any library code using them is also
    reproducible and not duplicated across workers.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


try:  # pragma: no cover - exercised only when torch is present
    from torch.utils.data import Dataset as _TorchDataset
    import torch

    class NEUImageDataset(_TorchDataset):
        """PyTorch dataset yielding ``(image_tensor, label)`` pairs.

        Augmentation parameters are drawn from torch's RNG inside
        ``__getitem__``. Inside ``DataLoader`` workers PyTorch reseeds that RNG
        per worker and per epoch, so every epoch sees fresh augmentations while
        the whole run stays reproducible for a fixed global seed. (Holding a
        NumPy generator on the dataset instead would be copied into every
        worker and replay the identical sequence each epoch.)

        With ``cache=True`` the deterministic preprocessing is done once and
        kept in memory (about 30 MB for NEU at 128 px), so only augmentation
        runs per epoch.
        """

        def __init__(
            self,
            samples: Sequence[Sample],
            cfg: PreprocessingConfig,
            image_size: int,
            augment: bool = False,
            cache: bool = False,
        ):
            self.samples = list(samples)
            self.cfg = cfg
            self.image_size = image_size
            self.augment = augment
            self._cache: Optional[List[np.ndarray]] = None
            if cache:
                self._cache = [self._load(i) for i in range(len(self.samples))]

        def _load(self, idx: int) -> np.ndarray:
            path, _ = self.samples[idx]
            return preprocess_image(load_grayscale(path), self.cfg, self.image_size)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            label = self.samples[idx][1]
            img = self._cache[idx] if self._cache is not None else self._load(idx)
            if self.augment:
                r = torch.randint(0, 4, (3,))
                img = augment_image(
                    img,
                    flip_lr=bool(r[0] % 2),
                    flip_ud=bool(r[1] % 2),
                    k_rot90=int(r[2]),
                )
            tensor = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).float()
            return tensor, int(label)

except ImportError:  # pragma: no cover

    class NEUImageDataset:  # type: ignore
        """Placeholder raised if instantiated without torch installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for NEUImageDataset (the CNN branch). "
                "Install it with `pip install torch torchvision`."
            )
