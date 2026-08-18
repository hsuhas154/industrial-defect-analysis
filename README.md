# Automated Industrial Surface Defect Analysis

**Classical Computer Vision + Machine Learning vs. CNN-Based Deep Learning for steel-surface defect classification.**

A modular, reproducible computer-vision project that classifies hot-rolled
steel-strip surface defects on the [NEU Surface Defect Database](https://faculty.neu.edu.cn/songkc/en/zdylm/263265).
The project studies defects through a deliberate progression, from classical
image processing and handcrafted features, to classical machine learning, to
CNN-based deep learning, and compares the two families of methods on the same
data splits.

> **Status: Phase 1 — Resume Prototype.** This repository currently implements a
> working, defensible end-to-end baseline: two parallel pipelines
> (`HOG → Decision Tree / SVM` and a `custom CNN`), fair evaluation on a shared
> deterministic split, an inference CLI, and a Streamlit demo. The full
> comparative study (Random Forest, XGBoost, LBP/GLCM, VGG/ResNet/DenseNet/
> MobileNet, transfer learning, and U-Net localization) is planned — see the
> [Roadmap](#roadmap).

---

## The central idea

Two pipelines answer one question: *do carefully engineered classical features
compete with a representation learned directly from pixels?*

```
NEU → Preprocessing → HOG → { Decision Tree, SVM } → Classification
NEU → Preprocessing → Custom CNN                    → Classification
```

Both branches consume the **same deterministic, stratified train/val/test
split**, so the comparison is fair.

## What's implemented in Phase 1

| Phase 1 deliverable                              | Where |
|--------------------------------------------------|-------|
| Dataset loading and validation                   | `src/idefect/data/datasets.py` |
| Deterministic, stratified train/val/test split   | `src/idefect/data/datasets.py` |
| Preprocessing & normalization (resize, CLAHE, denoise) | `src/idefect/data/preprocessing.py` |
| HOG feature extraction                           | `src/idefect/features/hog.py` |
| Decision Tree baseline                           | `src/idefect/classical_ml/decision_tree.py` |
| SVM baseline                                      | `src/idefect/classical_ml/svm.py` |
| Small custom CNN (from scratch, PyTorch)         | `src/idefect/models/custom_cnn.py` |
| Accuracy, precision, recall, F1, confusion matrix | `src/idefect/evaluation/classification.py` |
| Simple inference script                          | `scripts/predict.py` |
| Streamlit image-upload demo                      | `app/streamlit_app.py` |
| Unit tests                                        | `tests/` |
| Clean Git repository with README                 | this file |

## Repository structure

```
industrial-defect-analysis/
├── README.md               ├── LICENSE          ├── requirements.txt
├── pyproject.toml          ├── .gitignore
├── configs/
│   └── neu_baseline.yaml   # single source of truth for every run
├── data/
│   └── README.md           # how to obtain NEU (images are NOT committed)
├── src/idefect/
│   ├── config.py           # typed YAML config
│   ├── seed.py             # reproducibility
│   ├── data/               # datasets, splits, preprocessing, synthetic data
│   ├── features/           # HOG
│   ├── classical_ml/       # Decision Tree, SVM
│   ├── models/             # custom CNN
│   ├── training/           # train_classical_ml, train_cnn
│   ├── evaluation/         # classification metrics
│   ├── visualization/      # confusion matrices, curves, HOG plots
│   └── inference/          # unified Predictor (classical + CNN)
├── app/streamlit_app.py    # interactive demo
├── scripts/                # CLI entry points
├── tests/                  # pytest suite (runs on synthetic data)
├── experiments/            # results / figures / logs (git-ignored outputs)
└── docs/                   # methodology.md, dataset_notes.md
```

## Setup (Linux / WSL 2)

```bash
git clone <your-repo-url> industrial-defect-analysis
cd industrial-defect-analysis

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
# Optional (editable install so `import idefect` works anywhere):
pip install -e .
```

**GPU note (NVIDIA RTX 5070 / Blackwell).** The default `torch` wheel works on
most GPUs; the Phase 1 CNN is small enough to train on CPU in a couple of
minutes too. If `python -c "import torch; print(torch.cuda.is_available())"`
prints `False`, install a CUDA build matching your driver, e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Quickstart

**1. Get data.** Download NEU (see [`data/README.md`](data/README.md)) into
`data/raw/NEU-CLS/`. Don't have it yet? Generate a synthetic stand-in so you can
watch the whole pipeline run end-to-end:

```bash
python scripts/generate_dummy_data.py --out data/raw/NEU-CLS --per-class 40
```

**2. Train the classical branch (HOG → Decision Tree & SVM):**

```bash
python scripts/train_classical.py --config configs/neu_baseline.yaml
```

**3. Train the custom CNN:**

```bash
python scripts/train_cnn.py --config configs/neu_baseline.yaml
```

Each run writes metrics (JSON) to `experiments/results/` and figures
(confusion matrices, training curves, HOG visualization) to
`experiments/figures/`.

**4. Predict on a single image:**

```bash
python scripts/predict.py path/to/image.bmp --model experiments/results/custom_cnn.pt
python scripts/predict.py path/to/image.bmp --model experiments/results/hog+svm.joblib
```

**5. Launch the demo:**

```bash
streamlit run app/streamlit_app.py
```

## Configuration

Everything is driven by [`configs/neu_baseline.yaml`](configs/neu_baseline.yaml):
the split fractions, preprocessing toggles (CLAHE, denoise), HOG parameters,
Decision Tree / SVM hyperparameters, and CNN training settings (epochs, batch
size, learning rate, device). Point any script at a different config with
`--config`.

## Tests

The suite runs on the synthetic dataset, so it needs no downloads:

```bash
pip install pytest
pytest
```

It covers preprocessing shapes/ranges, deterministic and stratified splitting,
HOG feature dimensions, and the metric computations.

## Reported metrics

Per the project's evaluation policy, **accuracy is never reported alone**. Each
model reports accuracy, macro precision/recall/F1, per-class precision/recall/F1,
and a confusion matrix, all on the held-out test split.

*(Quantitative results on the real NEU dataset will be filled in here once you
run the pipeline on your machine. Results from the synthetic stand-in are not
meaningful and should not be reported.)*

## Roadmap

Planned for the post-resume, end-to-end phase (target: October 2026):

- Classical CV: LBP and GLCM texture features; edge/morphology experiments.
- Classical ML: Random Forest and XGBoost; hyperparameter comparison.
- Deep learning: VGG-style, ResNet, DenseNet, MobileNet; transfer learning.
- Localization: U-Net encoder-decoder for pixel-level defect masks (IoU / Dice).
- Datasets: DAGM 2007 (generalization), KolektorSDD (localization).
- Engineering: Docker packaging, optional FastAPI backend, experiment tracking,
  and a concise technical report.

See [`docs/methodology.md`](docs/methodology.md) for the design rationale.

## Datasets & licensing

No raw dataset images are committed. The **code** is MIT-licensed (see
[`LICENSE`](LICENSE)); datasets retain their own licenses and terms. See
[`data/README.md`](data/README.md) and [`docs/dataset_notes.md`](docs/dataset_notes.md)
for sources, expected layouts, and license notes (NEU and, for later phases:
DAGM 2007, KolektorSDD, MVTec AD).

## Acknowledgements

NEU Surface Defect Database: Northeastern University (K. Song et al.).
Please cite and attribute the dataset per its official source when using it.
