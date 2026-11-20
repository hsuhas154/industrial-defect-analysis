"""Feature registry: build any combination of handcrafted descriptors.

A feature set is named by joining descriptor names with "+", for example
"hog", "lbp+glcm" or "hog+lbp+glcm". Every extractor is a pure function of a
single preprocessed image with no fitted state, so extracting features for
the whole dataset up front cannot leak information between splits; anything
learned from data (scaling, PCA, the classifier) is fitted later inside the
cross-validation pipeline on training images only.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from idefect.config import Config
from idefect.features.glcm import extract_glcm, glcm_feature_names
from idefect.features.hog import extract_hog
from idefect.features.lbp import extract_lbp, lbp_feature_names


def _extractors(cfg: Config) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "hog": lambda im: extract_hog(im, cfg.hog),
        "lbp": lambda im: extract_lbp(im, cfg.lbp),
        "glcm": lambda im: extract_glcm(im, cfg.glcm),
    }


DESCRIPTORS = ("hog", "lbp", "glcm")


def parse_feature_set(name: str) -> List[str]:
    parts = [p.strip() for p in name.split("+") if p.strip()]
    bad = [p for p in parts if p not in DESCRIPTORS]
    if bad or not parts:
        raise ValueError(f"unknown descriptor(s) {bad} in feature set '{name}'; "
                         f"choose from {DESCRIPTORS}")
    return parts


def extract_one(img: np.ndarray, feature_set: str, cfg: Config) -> np.ndarray:
    ex = _extractors(cfg)
    return np.concatenate([ex[p](img) for p in parse_feature_set(feature_set)]).astype(np.float32)


def extract_many(images: np.ndarray, feature_set: str, cfg: Config, n_jobs: int = -1) -> np.ndarray:
    """Features for a stack of preprocessed images ``(N, H, W)``, in parallel."""
    from joblib import Parallel, delayed

    chunks = np.array_split(np.arange(len(images)), max(1, min(len(images), 64)))

    def run(idx):
        return np.stack([extract_one(images[i], feature_set, cfg) for i in idx])

    parts = Parallel(n_jobs=n_jobs)(delayed(run)(c) for c in chunks if len(c))
    return np.concatenate(parts).astype(np.float32)


def hog_feature_names(cfg: Config, hog_dim: int | None = None) -> List[str]:
    """Readable HOG names such as ``hog blk3,4 cell1,0 40deg``.

    scikit-image orders the HOG vector as (block row, block column, cell row
    within the block, cell column within the block, orientation bin). Falls
    back to plain indices if the dimensions do not match the configuration.
    """
    h = cfg.hog
    n_cells = [cfg.data.image_size // int(c) for c in h.pixels_per_cell]
    n_blocks = [n_cells[i] - int(h.cells_per_block[i]) + 1 for i in range(2)]
    cpb = [int(c) for c in h.cells_per_block]
    expected = n_blocks[0] * n_blocks[1] * cpb[0] * cpb[1] * h.orientations
    if hog_dim is not None and hog_dim != expected:
        return [f"hog_{i}" for i in range(hog_dim)]
    names = []
    for br in range(n_blocks[0]):
        for bc in range(n_blocks[1]):
            for cr in range(cpb[0]):
                for cc in range(cpb[1]):
                    for o in range(h.orientations):
                        deg = int(round(o * 180.0 / h.orientations))
                        names.append(f"hog blk{br},{bc} cell{cr},{cc} {deg}deg")
    return names


def feature_names(feature_set: str, cfg: Config, hog_dim: int | None = None) -> List[str]:
    names: List[str] = []
    for p in parse_feature_set(feature_set):
        if p == "hog":
            names += hog_feature_names(cfg, hog_dim)
        elif p == "lbp":
            names += lbp_feature_names(cfg.lbp)
        elif p == "glcm":
            names += glcm_feature_names(cfg.glcm)
    return names


def descriptor_slices(feature_set: str, dims: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    out, s = {}, 0
    for p in parse_feature_set(feature_set):
        out[p] = (s, s + dims[p])
        s += dims[p]
    return out
