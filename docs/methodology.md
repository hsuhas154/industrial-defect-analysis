# Methodology

The project compares two families of methods on one industrial dataset:
*classical computer vision with handcrafted features and classical machine
learning*, and *CNN-based deep learning*. Each component exists to answer a
specific question. This document explains the design and the reasoning behind
it; results are in [`experiments.md`](experiments.md).

## Two parallel branches

```
NEU → preprocessing → HOG → { Decision Tree, SVM } → classification
NEU → preprocessing → custom CNN                   → classification
```

Both branches use the **same deterministic train/validation/test split**, the
same preprocessing configuration and the same metrics, so differences in the
results come from the representation and the classifier, not from the data.

## Stage A: data ingestion, auditing and preprocessing

**Discovery.** `idefect.data.datasets` accepts class sub-folders or the flat
NEU-CLS layout, where the two-letter filename prefix (`Cr`, `In`, `Pa`, `PS`,
`RS`, `Sc`) gives the class.

**Auditing.** `idefect.data.stats` reads every image once and checks sizes,
channel counts, data types, unreadable files, class balance and **exact
duplicates** (hash of decoded pixels). NEU-CLS contains one duplicate pair.

**Splitting.** A per-class shuffle with a fixed seed gives a stratified 70/15/15
split that is identical across runs and across both branches. Duplicate groups
are treated as one unit, so a duplicated image can never sit in both train and
test (which would leak information and inflate test scores).

**Preprocessing** (`idefect.data.preprocessing`, all configurable):

* resize to a square input (area interpolation when shrinking, which averages
  pixels instead of dropping them and so avoids aliasing);
* **CLAHE**, contrast-limited adaptive histogram equalisation: histogram
  equalisation applied per tile (8x8 grid) with a clip limit so noise in flat
  regions is not amplified. NEU images have uneven illumination (most darken
  towards one edge), which a single global adjustment cannot correct;
* optional Gaussian or median denoising. Median filtering removes impulse noise
  while preserving edges better than a Gaussian, but both erode fine structure
  such as the crazing network, so denoising is off by default and treated as an
  experiment rather than an assumption;
* scaling to [0, 1].

**Classical CV operations** (`idefect.data.cv_ops`) are used for analysis and as
candidate inputs: Sobel gradient magnitude and the Laplacian (first and second
derivative edges), Canny (gradient, non-maximum suppression, hysteresis
thresholds set from the median intensity), morphology (opening and closing to
remove small specks or fill gaps; top-hat and black-hat to isolate small bright
and dark details such as inclusions and pits), Otsu thresholding and intensity
histograms.

## Stage B: handcrafted features (HOG)

Histogram of Oriented Gradients (`idefect.features.hog`):

1. compute image gradients and their orientation;
2. in each 16x16-pixel cell, build a 9-bin histogram of orientations weighted by
   gradient magnitude;
3. normalize groups of 2x2 cells (blocks, overlapping) with L2-Hys, which makes
   the descriptor robust to local illumination and contrast changes;
4. concatenate: on a 128x128 input this gives 7 x 7 blocks x 4 cells x 9 bins =
   **1,764 features**.

*Why HOG for surface defects?* Many defects are fundamentally gradient
phenomena: scratches are long, consistently oriented edges, crazing is a dense
network of short edges in many orientations, rolled-in scale produces oriented
streaks. HOG encodes exactly this local edge structure. Its known weakness is
intensity texture without strong edges (soft blobs, fine pitting), which is why
texture descriptors (LBP, GLCM) are the next experiment.

## Stage C: classical machine learning

* **Decision Tree** (CART): recursively chooses the single feature and threshold
  that most reduce Gini impurity, `G = 1 - sum_k p_k^2`. It is interpretable and
  non-linear, but each split looks at one feature, so on high-dimensional,
  correlated features a deep tree fits noise (high variance). It is kept as the
  reference point for that overfitting behaviour and for the ensembles
  (Random Forest reduces variance by bagging, XGBoost reduces bias by boosting).
* **SVM with RBF kernel**: maximizes the margin between classes, with `C`
  trading margin width against training errors. The RBF kernel
  `k(x, x') = exp(-gamma ||x - x'||^2)` gives a non-linear boundary without
  computing an explicit feature map. Because it is distance-based, features are
  standardized first (`StandardScaler`); otherwise high-variance bins dominate.
  Multi-class is handled one-versus-one; probabilities come from Platt scaling
  (used for ROC-AUC and the demo's confidences).

## Stage D: custom CNN

`idefect.models.custom_cnn` is implemented from scratch in PyTorch:

```
input 1x128x128
→ [Conv 3x3 (32), BatchNorm, ReLU, MaxPool 2x2]   → 32x64x64
→ [Conv 3x3 (64), BatchNorm, ReLU, MaxPool 2x2]   → 64x32x32
→ [Conv 3x3 (128), BatchNorm, ReLU, MaxPool 2x2]  → 128x16x16
→ global average pooling → dropout 0.3 → linear (6 classes)
```

* **Convolution and parameter sharing:** the same 3x3 filter is applied at
  every position, so the network detects a pattern anywhere with few parameters
  (93,670 in total).
* **Receptive field:** each conv-pool stage roughly doubles it; after three
  stages a unit sees about 22x22 input pixels, enough for local defect texture,
  and global average pooling then aggregates evidence from the whole image
  (translation invariant, no large fully connected layer to overfit).
* **BatchNorm** normalizes activations per channel, stabilizing and speeding up
  training. Its running statistics need some optimizer steps to settle, which
  explains near-chance validation scores in the first epochs.
* **Loss:** cross-entropy, `L = -sum_k y_k log p_k`, with softmax outputs.
* **Optimization:** Adam with weight decay and a cosine learning-rate schedule.
* **Augmentation:** horizontal and vertical flips and 90-degree rotations. NEU
  defects have no canonical orientation (scratches occur both vertically and
  horizontally), so these are label-preserving and effectively multiply the
  data by eight. Random parameters are drawn from PyTorch's generator inside
  each data-loader worker, which is reseeded per worker and per epoch, so every
  epoch sees new augmentations and the run remains reproducible.
* **Checkpoint selection:** the epoch with the best validation macro-F1 is kept;
  the test split is used once, at the end.

## Evaluation

`idefect.evaluation` reports, on the held-out test split:

* accuracy (never on its own), macro precision, recall and F1, which weight
  every class equally;
* per-class precision, recall, F1 and one-vs-rest ROC-AUC (threshold-free:
  measures how well the model *ranks* the true class);
* the confusion matrix (raw and row-normalized);
* per-image predictions with probabilities, used for error analysis
  (most-confident mistakes, most-confused class pairs).

A comparison table and chart are generated from the saved metrics, so reported
numbers cannot drift from the actual runs.

## Reproducibility

* One YAML config drives every run; it is validated on load and its hash is
  stored with each result and in the run log.
* Python, NumPy and PyTorch are seeded and cuDNN runs deterministically; a rerun
  reproduces the same metrics.
* Library versions and the GPU are recorded in every metrics file.
* Trained artifacts embed their preprocessing and feature configuration, so
  inference reproduces the training pipeline exactly.
