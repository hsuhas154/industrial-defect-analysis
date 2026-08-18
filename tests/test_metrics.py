"""Tests for the classification metric bundle."""

from idefect.evaluation.classification import compute_metrics


CLASSES = ["a", "b", "c"]


def test_perfect_prediction():
    y = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y, y, CLASSES)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    # Confusion matrix is purely diagonal.
    cm = m["confusion_matrix"]
    for i in range(3):
        for j in range(3):
            if i != j:
                assert cm[i][j] == 0


def test_all_wrong():
    y_true = [0, 0, 0]
    y_pred = [1, 1, 1]
    m = compute_metrics(y_true, y_pred, CLASSES)
    assert m["accuracy"] == 0.0


def test_metrics_are_json_serialisable():
    import json

    m = compute_metrics([0, 1, 2], [0, 1, 1], CLASSES)
    # Should not raise.
    json.dumps(m)
    assert set(m["per_class"].keys()) == set(CLASSES)
    assert m["n_samples"] == 3
