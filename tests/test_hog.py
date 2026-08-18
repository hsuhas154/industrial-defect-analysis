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
