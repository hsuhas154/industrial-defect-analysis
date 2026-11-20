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
    for name, disp, acc in (("hog+svm", "HOG + SVM", 0.9), ("custom_cnn", "Custom CNN", 0.95)):
        m = compute_metrics([0, 1], [0, 1], ["a", "b"])
        m.update(model=name, display_name=disp, accuracy=acc, fit_time_sec=1.0)
        (tmp_path / f"{name}_metrics.json").write_text(json.dumps(m))
    rows = collect_metrics(tmp_path)
    table = markdown_table(rows)
    assert "HOG + SVM" in table and "Custom CNN" in table
    assert table.index("HOG + SVM") < table.index("Custom CNN")  # classical rows first
    readme = tmp_path / "README.md"
    readme.write_text("top\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nbottom\n")
    assert update_readme(readme, table)
    text = readme.read_text()
    assert "old" not in text and "Custom CNN" in text and text.endswith("bottom\n")


def test_manifest_detects_missing_and_modified_files(tmp_path):
    import hashlib

    import pytest

    from idefect.config import Config
    from idefect.data.datasets import build_split
    from idefect.data.stats import verify_manifest

    root = tmp_path / "ds"
    root.mkdir()
    for i in range(3):
        (root / f"Cr_{i}.bmp").write_bytes(bytes([i]) * 64)
    manifest = tmp_path / "m.sha256"
    manifest.write_text("".join(
        f"{hashlib.sha256((root / f'Cr_{i}.bmp').read_bytes()).hexdigest()}  Cr_{i}.bmp\n"
        for i in range(3)))
    ok = verify_manifest(root, manifest)
    assert ok["checked"] and not (ok["missing"] or ok["modified"] or ok["unexpected"])
    (root / "Cr_0.bmp").unlink()
    (root / "Cr_1.bmp").write_bytes(b"edited by an image viewer")
    bad = verify_manifest(root, manifest)
    assert bad["missing"] == ["Cr_0.bmp"] and bad["modified"] == ["Cr_1.bmp"]
    cfg = Config()
    cfg.data.root, cfg.data.manifest = str(root), str(manifest)
    with pytest.raises(RuntimeError):
        build_split(cfg)


def test_tidy_layout_with_and_without_title(tmp_path):
    from idefect.visualization.plots import plt, tidy

    for title in (None, "overall title"):
        fig, ax = plt.subplots(2, 2)
        if title:
            fig.suptitle(title)
        tidy(fig)  # must not recurse or raise, with or without a suptitle
        fig.savefig(tmp_path / f"fig_{bool(title)}.png")
        plt.close(fig)


def test_plots_forces_agg_outside_notebooks_only():
    # Scripts get Agg; inside a Jupyter kernel the inline backend must survive
    # the import, otherwise plt.show() in a notebook displays nothing.
    import os
    import subprocess
    import sys
    from pathlib import Path

    src = str(Path(__file__).resolve().parents[1] / "src")
    probe = ("import sys, types\n"
             "if sys.argv[1] == 'kernel':\n"
             "    sys.modules['ipykernel'] = types.ModuleType('ipykernel')\n"
             "import matplotlib\n"
             "import idefect.visualization.plots\n"
             "print(matplotlib.get_backend().lower())\n")
    path = os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)
    env = {**os.environ, "MPLBACKEND": "pdf", "PYTHONPATH": path}
    run = lambda mode: subprocess.run([sys.executable, "-c", probe, mode], env=env, capture_output=True,
                                      text=True, timeout=120, check=True).stdout.strip()
    assert run("script") == "agg"
    assert run("kernel") == "pdf"


def test_gallery_and_roc_figures_render(dummy_dataset, tmp_path):
    from pathlib import Path

    from idefect.evaluation.error_analysis import plot_misclassified_gallery
    from idefect.visualization.classical import plot_roc_pr

    imgs = sorted(str(p) for p in Path(dummy_dataset).rglob("*.png"))[:8]
    rows = [{"path": p, "true": "a", "pred": "b", "correct": "0", "confidence": "0.9"} for p in imgs]
    out = plot_misclassified_gallery(rows, tmp_path / "gallery.png", "title", cols=3)
    assert out is not None and out.exists()

    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 10)
    proba = rng.dirichlet(np.ones(3), size=30)
    assert plot_roc_pr(y, proba, ["a", "b", "c"], tmp_path / "roc.png", "roc").exists()
