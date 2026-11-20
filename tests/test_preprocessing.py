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


def test_enhance_sobel_and_tophat():
    import pytest

    from idefect.config import Config, _validate as validate
    from idefect.data.preprocessing import enhance

    img = np.full((40, 40), 120, np.uint8)
    img[:, 20:] = 200                      # a vertical step edge
    edges = enhance(img, "sobel")
    assert edges.dtype == np.uint8 and edges.shape == img.shape
    assert edges[:, 18:22].mean() > 10 * max(1.0, edges[:, :10].mean())

    flat = np.full((40, 40), 120, np.uint8)
    flat[10, 10], flat[30, 30] = 200, 40  # a small bright and a small dark detail
    out = enhance(flat, "tophat", ksize=9)
    assert out[10, 10] > 200 - 1 and out[30, 30] < 40 + 1
    assert out[10, 10] >= flat[10, 10] and out[30, 30] <= flat[30, 30]
    assert out[20, 20] == 120              # plain background is unchanged
    assert np.array_equal(enhance(flat, "none"), flat)

    cfg = Config()
    cfg.preprocessing.enhance = "canny"
    with pytest.raises(ValueError):
        validate(cfg)
