# Methodology (Phase 1)

Phase 1 is deliberately modest. Its purpose is to stand up a **working,
defensible end-to-end prototype** that demonstrates the project's central idea:
*classical computer vision + handcrafted features* versus a *learned CNN
representation*, evaluated fairly on the same data splits.

## Two parallel pipelines

```
NEU → Preprocessing → HOG → {Decision Tree, SVM} → Classification
NEU → Preprocessing → Custom CNN                  → Classification
```

Both branches consume the **same deterministic train/validation/test split** so
that the comparison between handcrafted features and learned features is fair.

## Stage A — Data ingestion & preprocessing

`idefect.data.datasets` scans the dataset directory (class sub-folders *or* the
flat NEU-CLS layout with two-letter prefixes) and produces a reproducible,
**stratified** split driven by a fixed seed.

`idefect.data.preprocessing` provides the classical CV preprocessing used by the
handcrafted branch and (optionally) the CNN branch:

- grayscale handling and resizing to a fixed square size;
- intensity normalization to `[0, 1]`;
- CLAHE contrast enhancement (contrast-limited adaptive histogram equalisation);
- optional Gaussian / median denoising.

The project explicitly investigates whether the *same* preprocessing helps a
classical model but is unnecessary for a CNN — a question we can revisit by
toggling the config flags.

## Stage B — Handcrafted features (HOG)

`idefect.features.hog` wraps Histogram of Oriented Gradients. HOG encodes local
edge and gradient structure, which is a strong prior for the crack-, scratch-
and scale-type defects in NEU. Cell size, block size, orientation bins and
normalization are all configurable.

*Why HOG for industrial defects?* Many surface defects are fundamentally
**gradient/edge phenomena** — cracks, scratches, and boundaries of patches.
HOG summarises the distribution of gradient orientations in local cells, giving
a compact, illumination-robust descriptor that a classical classifier can
separate.

## Stage C — Classical machine learning

`idefect.classical_ml` builds two baselines on top of HOG features:

- **Decision Tree** — an interpretable, non-linear baseline. Useful as a
  conceptual contrast to margin-based and ensemble methods, and a reminder of
  why a single tree tends to overfit.
- **SVM (RBF kernel)** — margin-based classification on scaled features. A
  `StandardScaler` is applied first because SVMs are sensitive to feature
  scaling.

(Random Forest and XGBoost are intentionally deferred to the post-resume phase.)

## Stage D — Custom CNN

`idefect.models.custom_cnn` is a small CNN implemented from scratch in PyTorch:
a few `Conv → BatchNorm → ReLU → MaxPool` blocks, global average pooling, and a
linear classifier head. It learns a representation **directly from pixels** with
no handcrafted feature extraction, which is exactly the contrast Phase 1 is
designed to show. It is intentionally small so it trains quickly on a laptop GPU
(or even CPU).

## Evaluation

`idefect.evaluation.classification` reports, for every model, on the **held-out
test split**:

- accuracy (never reported alone);
- macro precision / recall / F1;
- per-class precision / recall / F1;
- a confusion matrix.

`idefect.visualization.plots` renders confusion matrices, CNN training curves,
sample grids, and HOG visualizations into `experiments/figures/`.

## Reproducibility

- A single YAML config (`configs/neu_baseline.yaml`) drives every run.
- All randomness (Python / NumPy / PyTorch) is seeded via `idefect.seed`.
- The data split is deterministic for a given seed.
- Metrics are written to `experiments/results/` as JSON.
