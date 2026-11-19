"""Tests for the classical CV operations (edges, morphology, histograms)."""

import numpy as np
import pytest

from idefect.data import cv_ops


def _square(size=64):
    img = np.zeros((size, size), np.uint8)
    img[16:48, 16:48] = 200
    return img


def test_canny_finds_square_edges():
    edges = cv_ops.canny_edges(_square(), 50, 150)
    assert edges.dtype == np.uint8 and edges.shape == (64, 64)
    assert edges[16, 30] == 255 or edges[15, 30] == 255
    assert edges[32, 32] == 0  # flat interior has no edges


def test_sobel_and_laplacian_are_uint8_and_peak_on_edges():
    img = _square()
    for fn in (cv_ops.sobel_magnitude, cv_ops.laplacian):
        out = fn(img)
        assert out.dtype == np.uint8
        assert out[32, 32] == 0
        assert out.max() == 255


def test_morphology_open_removes_small_speck():
    img = np.zeros((32, 32), np.uint8)
    img[10, 10] = 255  # single-pixel speck
    assert cv_ops.morphology(img, "open", 3).max() == 0
    assert cv_ops.morphology(img, "tophat", 3)[10, 10] == 255


def test_morphology_rejects_unknown_op():
    with pytest.raises(ValueError):
        cv_ops.morphology(_square(), "nonsense")


def test_float_input_is_accepted():
    out = cv_ops.sobel_magnitude(_square().astype(np.float32) / 255.0)
    assert out.dtype == np.uint8


def test_histogram_sums_to_one_and_otsu_splits_square():
    img = _square()
    h = cv_ops.intensity_histogram(img)
    assert h.shape == (256,) and abs(h.sum() - 1.0) < 1e-6
    t, mask = cv_ops.otsu_threshold(img)
    assert 0 <= t < 200
    assert mask[32, 32] == 255 and mask[0, 0] == 0
