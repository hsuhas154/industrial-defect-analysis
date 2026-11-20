"""Tests for HOG feature extraction."""

import numpy as np

from idefect.config import HOGConfig
from idefect.features.hog import extract_hog, extract_hog_batch


def test_hog_returns_1d_vector():
    img = np.random.rand(128, 128).astype(np.float32)
    cfg = HOGConfig()
    feats = extract_hog(img, cfg)
    assert feats.ndim == 1
    assert feats.shape[0] > 0
    assert feats.dtype == np.float32


def test_hog_dim_is_stable_across_images():
    cfg = HOGConfig()
    a = extract_hog(np.random.rand(128, 128).astype(np.float32), cfg)
    b = extract_hog(np.random.rand(128, 128).astype(np.float32), cfg)
    assert a.shape == b.shape


def test_hog_batch_shape():
    cfg = HOGConfig()
    images = np.random.rand(5, 128, 128).astype(np.float32)
    feats = extract_hog_batch(images, cfg)
    assert feats.shape[0] == 5
    assert feats.ndim == 2


def test_hog_feature_names_match_dimension_and_layout():
    from idefect.config import Config
    from idefect.features.registry import hog_feature_names

    cfg = Config()
    feats = extract_hog(np.random.rand(cfg.data.image_size, cfg.data.image_size).astype(np.float32), cfg.hog)
    names = hog_feature_names(cfg, feats.size)
    assert len(names) == feats.size == 1764
    assert names[0] == "hog blk0,0 cell0,0 0deg"
    assert names[cfg.hog.orientations] == "hog blk0,0 cell0,1 0deg"
    assert hog_feature_names(cfg, 7) == [f"hog_{i}" for i in range(7)]  # mismatch falls back
