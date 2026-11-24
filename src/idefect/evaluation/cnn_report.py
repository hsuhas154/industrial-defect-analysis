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
    arts = sorted(results_dir.glob("*.pt"),
                  key=lambda p: (next((i for i, a in enumerate(ARCH_ORDER) if p.stem.startswith(a)), 99), p.stem))
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
