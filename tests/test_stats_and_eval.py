"""Tests for dataset auditing, ROC-AUC, prediction export and comparison."""

import json
import shutil

import numpy as np

from idefect.data.datasets import load_samples, make_splits
from idefect.data.stats import audit_samples, audit_split, problems
from idefect.evaluation.classification import compute_metrics, load_predictions, save_predictions
from idefect.evaluation.compare import collect_metrics, markdown_table, update_readme
from idefect.evaluation.error_analysis import confused_pairs, misclassified


def test_audit_clean_dataset_has_no_issues(dummy_dataset):
    samples, names = load_samples(dummy_dataset)
    report = audit_samples(samples, names)
    split = make_splits(samples, names, 0.5, 0.25, 0.25, seed=0)
    sreport = audit_split(split, report["duplicate_groups"])
    assert report["n_images"] == 48
    assert report["unreadable_files"] == []
    assert sreport["is_disjoint"]
    assert problems(report, sreport) == []


def test_audit_detects_duplicates_and_leakage(dummy_dataset, tmp_path):
    root = tmp_path / "ds"
    shutil.copytree(dummy_dataset, root)
    cls_dir = sorted(p for p in root.iterdir() if p.is_dir())[0]
    src = sorted(cls_dir.iterdir())[0]
    shutil.copy(src, cls_dir / ("zz_copy" + src.suffix))
    samples, names = load_samples(root)
    report = audit_samples(samples, names)
    assert report["n_duplicate_images"] == 1
    assert any("duplicate" in i for i in problems(report))


def test_roc_auc_perfect_and_absent():
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y]
    m = compute_metrics(y, y, ["a", "b", "c"], y_proba=proba)
    assert m["roc_auc_ovr_macro"] == 1.0
    assert m["per_class"]["a"]["roc_auc"] == 1.0
    # A class missing from y_true makes AUC undefined: reported as None.
    m2 = compute_metrics([0, 1], [0, 1], ["a", "b", "c"], y_proba=np.eye(3)[[0, 1]])
    assert m2["roc_auc_ovr_macro"] is None
    json.dumps(m2)


def test_predictions_roundtrip_and_error_helpers(tmp_path):
    names = ["a", "b"]
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
    path = save_predictions(tmp_path / "m_test_predictions.csv", ["x", "y", "z"],
                            [0, 1, 0], [0, 1, 1], names, proba)
    rows = load_predictions(path)
    assert [r["correct"] for r in rows] == ["1", "1", "0"]
    assert misclassified(rows)[0]["path"] == "z"
    assert confused_pairs(rows) == [{"true": "a", "pred": "b", "count": 1}]


def test_compare_table_and_readme_block(tmp_path):
    for name, acc in (("hog+svm", 0.9), ("custom_cnn", 0.95)):
        m = compute_metrics([0, 1], [0, 1], ["a", "b"])
        m.update(model=name, accuracy=acc, fit_time_sec=1.0)
        (tmp_path / f"{name}_metrics.json").write_text(json.dumps(m))
    rows = collect_metrics(tmp_path)
    table = markdown_table(rows)
    assert "HOG + SVM (RBF)" in table and "Custom CNN" in table
    readme = tmp_path / "README.md"
    readme.write_text("top\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nbottom\n")
    assert update_readme(readme, table)
    text = readme.read_text()
    assert "old" not in text and "Custom CNN" in text and text.endswith("bottom\n")
