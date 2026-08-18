"""Tests for the preprocessing pipeline."""

import numpy as np

from idefect.config import PreprocessingConfig
from idefect.data.preprocessing import preprocess_image


def test_preprocess_shape_and_range():
    img = (np.random.rand(200, 200) * 255).astype(np.uint8)
    cfg = PreprocessingConfig(normalize=True, clahe=True, denoise="none")
    out = preprocess_image(img, cfg, image_size=128)
    assert out.shape == (128, 128)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_no_normalize_keeps_0_255_scale():
    img = np.full((50, 50), 200, dtype=np.uint8)
    cfg = PreprocessingConfig(normalize=False, clahe=False, denoise="none")
    out = preprocess_image(img, cfg, image_size=32)
    assert out.max() > 1.0  # still on the 0-255 scale


def test_preprocess_denoise_options_run():
    img = (np.random.rand(80, 80) * 255).astype(np.uint8)
    for method in ("none", "gaussian", "median"):
        cfg = PreprocessingConfig(denoise=method, denoise_ksize=3)
        out = preprocess_image(img, cfg, image_size=64)
        assert out.shape == (64, 64)


def test_preprocess_accepts_color_input():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    cfg = PreprocessingConfig()
    out = preprocess_image(img, cfg, image_size=48)
    assert out.shape == (48, 48)
