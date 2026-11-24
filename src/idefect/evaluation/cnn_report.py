"""CNN-branch figures from saved artifacts: Grad-CAM galleries.

For each trained CNN (the best-validation seed kept as ``<stem>.pt``):

* ``gradcam_<stem>.png``: per class, the first correctly classified test
  images and then the model's misclassified test images, each with its
  Grad-CAM heat map over the image the model actually sees;
* ``gradcam_comparison.png``: the same test images (one per class, plus the
  image every classical model also misclassifies, Sc_44) for every CNN, side
  by side.

The test images are only visualized here; nothing is chosen or tuned on them.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


def _load(path: Path):
    import torch

    from idefect.evaluation.gradcam import GradCAM
    from idefect.inference.predict import Predictor

    pred = Predictor(path)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred.model.to(dev).eval()
    return pred, GradCAM(pred.model), dev


def _cams(pred, cam, dev, paths: List[str]):
    import torch

    from idefect.data.preprocessing import load_and_preprocess

    imgs = np.stack([load_and_preprocess(p, pred._prep_cfg, pred.image_size) for p in paths])
    out_cams, out_probs = [], []
    for i in range(0, len(imgs), 32):
        x = torch.from_numpy(imgs[i:i + 32])[:, None].float().to(dev)
        c, p = cam(x)
        out_cams.append(c)
        out_probs.append(p)
    return imgs, np.concatenate(out_cams), np.concatenate(out_probs)


def _predictions(results_dir: Path, stem: str) -> List[Dict]:
    with open(results_dir / f"{stem}_test_predictions.csv") as fh:
        return list(csv.DictReader(fh))


def make_cnn_figures(cfg, per_class: int = 2, max_errors: int = 6) -> Dict[str, Path]:
    from idefect.models.registry import ARCH_ORDER
    from idefect.visualization import cnn as V

    results_dir = Path(cfg.output.results_dir)
    fig_dir = Path(cfg.output.figures_dir) / "cnn"
    out: Dict[str, Path] = {}
    def order(p: Path):
        # scratch models first, then each ImageNet backbone as frozen, partial, full
        if p.stem in ARCH_ORDER:
            return (0, ARCH_ORDER.index(p.stem), 0)
        arch, _, ft = p.stem.rpartition("_")
        modes = ("frozen", "partial", "full")
        return (1, ARCH_ORDER.index(arch) if arch in ARCH_ORDER else 99, modes.index(ft) if ft in modes else 9)

    arts = sorted(results_dir.glob("*.pt"), key=order)
    comparison_paths: List[str] = []
    comparison: Dict[str, tuple] = {}
    for art in arts:
        stem = art.stem
        rows = _predictions(results_dir, stem)
        if not comparison_paths:
            classes = list(dict.fromkeys(r["true"] for r in rows))
            comparison_paths = [next(r["path"] for r in rows if r["true"] == c) for c in classes]
            sc44 = [r["path"] for r in rows if Path(r["path"]).name == "Sc_44.bmp"]
            comparison_paths += sc44
        pred, cam, dev = _load(art)
        chosen = []
        for c in pred.class_names:
            chosen += [r for r in rows if r["true"] == c and r["correct"] == "1"][:per_class]
        chosen += [r for r in rows if r["correct"] == "0"][:max_errors]
        imgs, cams, probs = _cams(pred, cam, dev, [r["path"] for r in chosen])
        titles = [f"{Path(r['path']).stem}\ntrue {r['true']}\npred {pred.class_names[int(p.argmax())]} "
                  f"({p.max():.2f})" for r, p in zip(chosen, probs)]
        out[f"gradcam_{stem}"] = V.plot_gradcam_grid(imgs, cams, titles, fig_dir / f"gradcam_{stem}.png",
                                                     f"Grad-CAM: {pred.kind} (test images; errors last)",
                                                     cols=2 * per_class)
        cimgs, ccams, cprobs = _cams(pred, cam, dev, comparison_paths)
        comparison[stem] = (ccams, [pred.class_names[int(p.argmax())] for p in cprobs])
        pred.model.cpu()
    if comparison:
        from idefect.data.preprocessing import load_and_preprocess

        base = [load_and_preprocess(p, cfg.preprocessing, cfg.data.image_size) for p in comparison_paths]
        out["gradcam_comparison"] = V.plot_gradcam_comparison(
            base, [Path(p).stem for p in comparison_paths], comparison, fig_dir / "gradcam_comparison.png")
    return out


def make_efficiency_figures(cfg) -> Dict[str, Path]:
    """Accuracy against cost, from ``efficiency.json`` (written by the benchmark)."""
    import json

    from idefect.visualization import cnn as V

    results_dir = Path(cfg.output.results_dir)
    f = results_dir / "efficiency.json"
    if not f.exists():
        return {}
    rows = json.loads(f.read_text())["rows"]
    fig_dir = Path(cfg.output.figures_dir) / "cnn"
    return {
        "accuracy_vs_cpu_latency": V.plot_accuracy_vs_cost(
            rows, "cpu_latency_ms", "single-image CPU latency (ms, log scale; preprocessing excluded)",
            fig_dir / "accuracy_vs_cpu_latency.png", "Accuracy against CPU latency"),
        "accuracy_vs_params": V.plot_accuracy_vs_cost(
            [r for r in rows if r["branch"] == "cnn"], "params_total", "parameters (log scale)",
            fig_dir / "accuracy_vs_params.png", "CNN accuracy against size"),
    }


def make_study_figures(cfg) -> Dict[str, Path]:
    """Sensitivity (resolution, CLAHE) and data-curve figures from the saved studies."""
    import json

    from idefect.visualization import cnn as V

    d = Path(cfg.output.results_dir) / "cnn_studies"
    fig_dir = Path(cfg.output.figures_dir) / "cnn"
    out: Dict[str, Path] = {}
    if (d / "cnn_sensitivity.json").exists():
        s = json.loads((d / "cnn_sensitivity.json").read_text())["summary"]
        out["cnn_resolution"] = V.plot_sensitivity(s, "resolution", ["64 px", "128 px", "200 px"],
                                                   fig_dir / "cnn_resolution.png",
                                                   "CNN input resolution (validation split)")
        out["cnn_clahe"] = V.plot_sensitivity(s, "preprocessing", ["no CLAHE", "CLAHE"],
                                              fig_dir / "cnn_clahe.png", "Does CLAHE help a CNN? (validation split)")
    if (d / "data_curve.json").exists():
        c = json.loads((d / "data_curve.json").read_text())
        n_by_fraction = {r["fraction"]: r["n_train"] for r in c["rows"]}
        out["data_curve"] = V.plot_data_curve(c["summary"], n_by_fraction, fig_dir / "data_curve.png")
    return out
