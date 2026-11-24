# Methodology

The project compares two families of methods on one industrial dataset:
*classical computer vision with handcrafted features and classical machine
learning*, and *CNN-based deep learning*. Each component exists to answer a
specific question. This document explains the design and the reasoning behind
it; results are in [`experiments.md`](experiments.md).

## Two parallel branches

```
NEU → preprocessing → { HOG, LBP, GLCM } → { DT, LogReg, k-NN, SVM, RF, XGBoost } → classification
NEU → preprocessing → custom CNN                                                → classification
```

Both branches use the **same deterministic, group-aware train/validation/test
split**, the same preprocessing configuration and the same metrics, so
differences in the results come from the representation and the classifier,
not from the data.

## Stage A: data ingestion, integrity, leakage control and preprocessing

**Discovery.** `idefect.data.datasets` accepts class sub-folders or the flat
NEU-CLS layout, where the two-letter filename prefix (`Cr`, `In`, `Pa`, `PS`,
`RS`, `Sc`) gives the class.

**Integrity.** `data/neu_cls_manifest.sha256` stores the SHA-256 of every
image. Every run verifies it and refuses to continue if a file is missing,
altered or unexpected (an image viewer that re-saves a file would otherwise
change results silently).

**Auditing.** `idefect.data.stats` reads every image once and checks sizes,
channel counts, data types, unreadable files and class balance.

**Near-duplicate detection** (`idefect.data.duplicates`). Leakage happens when
the same visual content sits on both sides of a split, and NEU-CLS turns out to
be built largely from overlapping crops: most images share an exact region with
at least one other image, often flipped or rotated, and some re-processed
copies exist too. Simple similarity scores cannot separate copies from
look-alikes here, because different crops along one long scratch correlate at
0.998 while a JPEG copy can fall to 0.86. The detector therefore decides on two
pieces of evidence that a different image cannot fake:

* **E1, pixel identity.** At the best alignment over the 8 flips/rotations and
  all shifts, at least 99 percent of the overlapping pixels agree within 2 grey
  levels. Real overlaps agree on 100 percent of pixels; all other pairs never
  exceed 74 percent. To find *every* exact overlap, short horizontal pixel runs
  of each image (and of each flipped/rotated version) are hashed; matching runs
  vote for an alignment, so any shared textured region a few pixels high is
  found.
* **E2, shared background grain.** For re-processed copies (brightness,
  contrast, resampling, mild compression) the fine grain of the steel surface
  away from strong structures is the fingerprint. The overlap is band-passed,
  lines, edges and saturated pixels are masked with a wide margin, and the
  median per-tile normalized cross-correlation is computed over tiles that are
  mostly background. Candidates come from each image's 10 best partners in a
  GPU search of masked cross-correlation over all pairs, shifts and
  transforms. Thresholds were calibrated on NEU: pairs from different classes
  that share no pixels never exceed 0.32 and no pair judged distinct exceeds
  0.385, while pixel-identical pairs (copies by construction) score 1.0 and
  planted re-processed copies down to JPEG quality 50 score 0.54 or more; the
  rule is same at 0.50 or more, human review in [0.40, 0.50), distinct below.

Pairs without enough background to judge and without pixel identity are kept
apart and listed. Human decisions live in `configs/duplicate_review.yaml` and
override the detector. Groups are connected components of the "same" relation
(union-find), so chains of overlapping crops stay together. The method is
covered by tests with planted copies (exact, rotated, flipped, brightened,
JPEG, resampled, overlapping crops) and planted look-alikes (same bright line,
different surface).

**Splitting.** Every group is one indivisible unit, even when its members have
different labels (some inclusion and scratch crops share exact regions).
Units are stratified by majority label, shuffled with a fixed seed and placed
largest first into whichever split is furthest below its target, which keeps
70/15/15 per class while no group spans two splits. The split seed is separate
from the training seed, so all models and all training seeds share one split.

**Preprocessing** (`idefect.data.preprocessing`, all configurable):

* resize to a square input (area interpolation when shrinking, which averages
  pixels instead of dropping them and so avoids aliasing);
* **CLAHE**, contrast-limited adaptive histogram equalisation: histogram
  equalisation applied per tile (8x8 grid) with a clip limit so noise in flat
  regions is not amplified. NEU images have uneven illumination (most darken
  towards one edge), which a single global adjustment cannot correct;
* optional Gaussian or median denoising. Median filtering removes impulse noise
  while preserving edges better than a Gaussian, but both erode fine structure
  such as the crazing network, so denoising is off by default and evaluated in
  the preprocessing ablation;
* optional edge or morphology enhanced input, also off by default and
  evaluated in the ablation: the Sobel gradient magnitude in place of the
  image, or morphological contrast enhancement (image + top-hat - black-hat);
* scaling to [0, 1].

**Classical CV operations** (`idefect.data.cv_ops`): Sobel gradient
magnitude and the Laplacian (first and second derivative edges), Canny
(gradient, non-maximum suppression, hysteresis thresholds set from the median
intensity), morphology (opening and closing to remove small specks or fill
gaps; top-hat and black-hat to isolate small bright and dark details such as
inclusions and pits), Otsu thresholding and intensity histograms. They are used
for analysis (notebook 01), and the Sobel magnitude and top-hat/black-hat
enhancement are also tested as model input in the preprocessing ablation.

## Stage B: handcrafted features

Every extractor is a pure function of one preprocessed image with no fitted
state, so features can be computed for the whole dataset up front without
leaking anything; anything learned from data is fitted later, per
cross-validation fold.

**HOG** (`idefect.features.hog`): image gradients, a 9-bin orientation
histogram per 16x16 cell weighted by gradient magnitude, L2-Hys normalization
over overlapping 2x2-cell blocks (robust to local illumination), concatenated:
7 x 7 blocks x 4 cells x 9 bins = **1,764 features** at 128x128. HOG encodes
edge structure: scratches are long oriented edges, crazing a network of short
edges, rolled-in scale oriented streaks. Its weakness is intensity texture
without strong edges (soft blobs, fine pitting).

**LBP** (`idefect.features.lbp`): each pixel's P neighbours on a circle of
radius R are thresholded against the centre, giving a P-bit pattern. With the
rotation-invariant uniform mapping, patterns with at most two 0/1 transitions
(spots, edges, corners, flat areas) are reduced to their number of set bits,
and all other patterns share one bin: P + 2 codes. The normalized histogram of
codes at (8, 1), (16, 2) and (24, 3) gives **54 features**. LBP captures
micro-texture, ignores monotonic brightness changes and is rotation invariant.

**GLCM** (`idefect.features.glcm`): after quantization to 32 grey levels, the
grey-level co-occurrence matrix counts how often level i neighbours level j at
distance d and angle theta (0, 45, 90, 135 degrees). Six statistics are taken:
contrast `sum P(i,j)(i-j)^2`, dissimilarity `sum P(i,j)|i-j|`, homogeneity
`sum P(i,j)/(1+(i-j)^2)`, energy `sqrt(ASM)`, ASM `sum P(i,j)^2` and correlation.
For d in {1, 2, 4, 8}, the mean over angles (rotation invariant) and the range
over angles (directionality, high for streaks and scratches) give **48
features**.

**Feature sets.** Descriptors are first evaluated alone, and concatenated only
afterwards (LBP + GLCM, HOG + LBP + GLCM), as the project plan requires.

## Stage C: classical machine learning

* **Decision Tree** (CART): recursively chooses the single feature and threshold
  that most reduce Gini impurity `1 - sum_k p_k^2` (or entropy). Interpretable
  and non-linear, but one feature per split, so a deep tree on many correlated
  features fits noise (high variance); depth, leaf size and cost-complexity
  pruning are tuned and the depth curve shows the overfitting directly.
* **Logistic regression** (multinomial, L2): a linear baseline; C is tuned.
* **k-NN**: an instance-based baseline on standardized features; k, distance
  weighting and metric are tuned.
* **SVM**: maximizes the margin between classes, C trading margin width against
  training errors. The RBF kernel `exp(-gamma ||x - x'||^2)` gives non-linear
  boundaries without an explicit feature map; a linear kernel is also searched.
  Features are standardized because the method is distance based.
* **Random Forest**: many decorrelated trees (bootstrap samples, random feature
  subsets per split) averaged, which reduces variance relative to a single
  tree.
* **XGBoost**: gradient-boosted trees fitted sequentially to the residual
  gradient of the loss with shrinkage and regularization, which reduces bias;
  trained on the GPU (`device="cuda"`).

**Tuning protocol (leakage-free).** For every (feature set, classifier) pair,
hyperparameters are chosen by grid search with 5-fold *stratified group*
cross-validation on the training split: near-duplicate groups never straddle a
fold, and scalers live inside the pipeline so they are refitted on each fold's
training part. The chosen configuration is refitted on the whole training
split. Each classifier's final feature set is the one with the best mean CV
macro-F1. The validation split is a second, independent check; the test split
is evaluated once and never used for any decision. Grid searches run in parallel on all CPU threads (`classical_ml.n_jobs`), XGBoost
on the GPU; search and fit times are recorded. Parallelism changes only speed and
memory, never results: every fit is independent and seeded.

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
* **Training length:** chosen by a separate study (several epoch budgets x
  several seeds) on validation macro-F1 only.
* **Seeds:** the CNN is trained once per seed (five seeds) on the same split.
  The model kept as the artifact is the seed with the best *validation*
  macro-F1; test macro-F1 is reported for it and as mean and standard deviation
  over all seeds, so a lucky seed cannot pass as the result.

## Stage E: CNN architecture study and transfer learning

Four classic architectures are implemented from scratch in PyTorch
(`idefect.models`), each built for 128x128 grayscale input, and each ending in
global average pooling, dropout and one linear layer like the custom CNN. Each
answers one question about what makes a CNN work:

* **VGG-style** (`vgg_style.py`, 1.17 M parameters): does depth through
  repetition help? Four stages of two 3x3 convolutions (two stacked 3x3 layers
  see a 5x5 region with fewer parameters and one more non-linearity than one
  5x5 layer), max pooling between stages, widths 32 to 256. VGG's original
  4096-unit fully connected head is replaced by global average pooling, since
  on 1,260 images it could only memorize.
* **ResNet-18** (`resnet.py`, 2.80 M): do residual connections help?
  Each block learns a correction added to its input, `y = ReLU(F(x) + x)`, so
  gradients reach early layers through the identity path and a block with
  nothing to add can stay close to the identity (its last BatchNorm starts at
  zero). The ImageNet 7x7 stem and stem max pooling are dropped, because at
  128 px they would discard the fine texture that matters here.
* **DenseNet-BC** (`densenet.py`, 0.77 M): does feature reuse help? Every layer
  receives the concatenated maps of all earlier layers in its block and adds
  16 new ones; 1x1 bottlenecks and compressing transitions keep it narrow.
* **MobileNetV2** (`mobilenet.py`, 2.23 M): what does efficiency cost?
  Depthwise separable convolutions (a per-channel 3x3 filter, then a 1x1
  channel mix) cost about 8 to 9 times fewer multiplications than standard
  3x3 convolutions; inverted residual blocks expand, filter, and project back
  linearly.

**Transfer learning** (`idefect.models.transfer`): ResNet-18, DenseNet-121 and
MobileNetV2 with ImageNet weights (torchvision). A grayscale image is repeated
into three channels and normalized with the ImageNet mean and standard
deviation inside the model. Three fine-tuning depths are compared: **frozen**
(only a new linear head is trained; the backbone's BatchNorm statistics stay
fixed), **partial** (the last stage is unfrozen too) and **full** (every layer
trains). Unfrozen pretrained layers learn ten times slower than the new head.

**One recipe for every network.** All CNNs, scratch and pretrained, use the
custom CNN's training recipe unchanged: the same split, augmentation, Adam
(learning rate 1e-3, weight decay 1e-4), cosine schedule, 150 epochs, batch 64,
checkpoint by validation macro-F1, five seeds and full precision. Differences
in results therefore come from the architecture or the pretraining, not from
tuning effort spent on one network and not another. The trainer also supports
bfloat16 mixed precision and TensorBoard logging, both off for reported runs.

**Grad-CAM** (`idefect.evaluation.gradcam`): the last convolutional feature
maps are weighted by the average gradient of the predicted class score and
summed, `CAM = ReLU(sum_k w_k A_k)` with `w_k = mean dy/dA_k`, then upsampled.
Bright regions are those that raised the class score. A unit test checks it on
a model whose evidence is known by construction.

**Efficiency** (`idefect.evaluation.efficiency`): parameters, size on disk,
preprocessing time, single-image latency (median of 50 timed calls after
warm-up) and batch-64 throughput on CPU and GPU, and peak GPU memory for
inference and for one training step, measured for every trained model with
nothing else on the GPU.

**Sensitivity studies** (`idefect.training.cnn_studies`): input resolution
(64, 128, 200 px), CLAHE on or off, and a data-requirement curve (10 to 100
percent of the training split, class-stratified subsets, for CNNs and
classical models). Like the classical ablations, they report validation
macro-F1 only, over three seeds; the test split is never used.

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
