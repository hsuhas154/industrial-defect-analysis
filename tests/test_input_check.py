"""Inference input checks: each warning fires when it should, and only then."""

import numpy as np

from idefect.inference.input_check import check_input, colourfulness

# Reference shaped like the one stored in every model artifact; values close to
# the NEU training images (brightness 34 to 246, contrast 6 to 74).
REF = {
    "image": {
        "mean": {"median": 125.0, "mad_sigma": 45.0, "min": 34.0, "max": 246.0},
        "std": {"median": 24.0, "mad_sigma": 13.0, "min": 6.0, "max": 74.0},
    },
    "confidence_floor": 0.6,
}


def _gray(mean, std, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, std, (200, 200)), 0, 255).astype(np.uint8)


def _has(warnings, text):
    return any(text in w for w in warnings)


def test_typical_input_gives_no_warning():
    assert check_input(_gray(125, 24), REF, confidence=0.95, model_kind="custom_cnn") == []


def test_brightness_and_contrast_outside_training_range():
    bright_flat = np.full((200, 200), 252, np.uint8)
    w = check_input(bright_flat, REF, confidence=0.95)
    assert _has(w, "Unusual brightness") and _has(w, "Unusual contrast")
    # 252 is only 2.8 robust deviations above the median: the range check must catch it.
    assert (252 - 125) / 45 < 4
    assert not _has(check_input(_gray(30, 24), REF, 0.95), "Unusual contrast")
    assert _has(check_input(_gray(20, 10), REF, 0.95), "Unusual brightness")


def test_far_from_median_still_fires_without_a_stored_range():
    ref = {"image": {"mean": {"median": 125.0, "mad_sigma": 10.0},
                     "std": {"median": 24.0, "mad_sigma": 13.0}}}
    assert _has(check_input(_gray(200, 24), ref, None), "Unusual brightness")


def test_knn_warning_names_k():
    w1 = check_input(_gray(125, 24), REF, 1.0, model_kind="lbp+glcm+knn", n_neighbors=1)
    assert _has(w1, "k = 1, so they are always 0% or 100%")
    w5 = check_input(_gray(125, 24), REF, 0.8, model_kind="lbp+glcm+knn", n_neighbors=5)
    assert _has(w5, "k = 5") and not _has(w5, "always 0%")
    assert not _has(check_input(_gray(125, 24), REF, 0.95, model_kind="lbp+glcm+svm"), "k-NN")


def test_n_neighbors_of_pipeline_and_other_models():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    from idefect.inference.predict import n_neighbors_of

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=3))])
    assert n_neighbors_of(pipe) == 3
    assert n_neighbors_of(KNeighborsClassifier(n_neighbors=1)) == 1
    assert n_neighbors_of(DecisionTreeClassifier()) is None


def test_colour_low_confidence_and_tree_warnings():
    rgb = np.zeros((64, 64, 3), np.uint8)
    rgb[..., 0] = 200
    assert colourfulness(rgb) > 8
    assert _has(check_input(rgb, REF, 0.95), "in colour")
    assert not _has(check_input(np.dstack([_gray(125, 24)] * 3), REF, 0.95), "in colour")
    assert _has(check_input(_gray(125, 24), REF, confidence=0.4), "Low confidence")
    assert _has(check_input(_gray(125, 24), REF, 0.95, model_kind="lbp+glcm+decision_tree"),
                "Decision tree probabilities")
