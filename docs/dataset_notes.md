# Dataset notes

## NEU Surface Defect Database

- 1,800 grayscale images, 200×200 px, six defect classes, 300 images per class.
- Classes: rolled-in scale, patches, crazing, pitted surface, inclusion,
  scratches.
- Official source: https://faculty.neu.edu.cn/songkc/en/zdylm/263265

### Class abbreviations (flat NEU-CLS layout)

| Prefix | Class            |
|--------|------------------|
| `Cr`   | crazing          |
| `In`   | inclusion        |
| `Pa`   | patches          |
| `PS`   | pitted_surface   |
| `RS`   | rolled-in_scale  |
| `Sc`   | scratches        |

### Data quality

* All 1,800 images are readable, 200x200, single-channel 8-bit, 300 per class.
* **NEU-CLS is built largely from overlapping crops.** The near-duplicate
  detector (`src/idefect/data/duplicates.py`) finds that 1,583 of the 1,800
  images belong to 248 groups of related images:
  * 3,238 pairs share an exact region of pixels (100 percent pixel agreement
    over the overlap), often after a flip or a 90-degree rotation: crops cut
    from the same strip image;
  * 245 pairs are re-processed copies (same surface grain, different
    brightness or contrast);
  * one pair is byte-identical (`Pa_101.bmp`, `Pa_105.bmp`);
  * some groups even cross classes: inclusion and scratch crops share exact
    regions (for example `In_272.bmp` and `Sc_69.bmp`), where the shared part
    is plain background and each image's defect lies outside it;
  * groups chain along a strip; the largest has 105 inclusion images.
* A plain stratified random split therefore puts close relatives of most test
  images into the training set. This project keeps every group inside one
  split (see `docs/methodology.md`); the leakage study in
  `docs/experiments.md` measures how much a random split inflates scores.
* Borderline pairs were decided by visual inspection and recorded in
  `configs/duplicate_review.yaml`.
* Most images show an illumination gradient (darker towards one edge), which
  is why local contrast normalisation (CLAHE) is part of preprocessing.
* `data/neu_cls_manifest.sha256` fingerprints every file; runs stop if any
  image is missing or altered (an image viewer that re-saves a file would
  otherwise change results silently).

### License / redistribution

The official page does not provide a clear standardized open-source license.
This repository therefore does not redistribute the raw images; it documents the
source and provides download + preprocessing instructions instead. Preserve the
original attribution when using the data.

## KolektorSDD (Phase 4: localization)

Source: https://www.vicos.si/resources/kolektorsdd/ (Tabernik et al. 2019,
fine annotations), CC BY-NC-SA 4.0: attribution, non-commercial use only,
share-alike. Images are not redistributed here.

### Contents (measured)

- 50 parts `kos01` to `kos50` (each an electrical commutator), 8 images per
  part except `kos21` with 7: 399 grayscale JPG images, 500 px wide, 1,235
  to 1,285 px high (45 distinct sizes; the website's "1240 to 1270" is
  slightly off), so every image is resized before training.
- 52 defective images: one per part, two in `kos38` and `kos39`.
- Masks: the 347 normal images have all-zero grayscale BMPs; the 52 defective
  images have palette BMPs. 51 use palette index 1 = white; `kos20/Part4`
  has its own palette (index 1 = grey 85 for 2 stray pixels, index 2 = white
  for the 3,690 defect pixels). Rule: a pixel is defective when its value is
  above zero, which gives the same mask whether a reader returns palette
  indices or grey levels (checked for all 52).
- Official split (`splits/split.pyb`, used in the JIM 2019 paper): 3 folds
  by part, test folds of 17/17/16 parts (135/136/128 images, 18/18/16
  defective). It is a Python 2 pickle; it is read by a data-only decoder
  that refuses any opcode other than lists, tuples, strings and memo.

### Shared content and leakage in the official folds

`scripts/phase4_dataset_report.py` found three relations between parts and
nothing else:

| Parts | Kind | Evidence |
|---|---|---|
| kos01, kos26 | byte-identical copy (all 8 images) | identical files |
| kos27, kos34 | byte-identical copy (all 8 images) | identical files |
| kos38, kos39 | the same commutator captured twice | every image has exactly one partner: Part i of kos38 matches Part (i - 2) mod 8 of kos39, fine score 0.839 to 0.931 |

- The twin defective images were annotated independently. Their masks differ:
  IoU 0.632 (`kos01/Part5`: 1,143 versus 1,706 px) and 0.660 (`kos27/Part0`:
  4,087 versus 2,988 px). This is a direct measure of how much two
  annotations of the same crack disagree, and an upper bound worth keeping in
  mind when reading U-Net IoU scores.
- `kos38` and `kos39` are not file copies: after alignment only 45 to 53
  percent of pixels agree within 2 grey levels and no image row is identical,
  as expected from a second capture. The two defective images of each part
  are the same two defects (`kos38/Part0, Part1` = `kos39/Part6, Part7`).
  The best pair of different parts scores 0.116 at fine scale, so the gap to
  0.839 is wide.
- Every official fold leaks: fold 0 puts kos27/kos34 and kos38/kos39 on both
  sides of train/test, fold 1 kos01/kos26 and kos38/kos39, fold 2 kos01/kos26
  and kos27/kos34. A model can therefore be tested on a defect it was
  trained on.

### Corrected folds (`data/splits/kolektorsdd_folds.json`)

Policy `drop_copies` (config `kolektorsdd.duplicate_policy`): of each
byte-identical pair only the lower part number is kept (kos26 and kos34 are
dropped), and related parts (kos38, kos39) are kept together in the test fold
of the lower part number. Keeping both captures was preferred over dropping
one: they are genuine, separately captured and annotated images, and once
they sit on the same side of every split they cannot leak. Validation parts
are drawn per fold from the training parts (20 percent, seeded, whole groups
only).

| Fold | Train parts / images / defective | Val | Test |
|---|---|---|---|
| 0 | 24 / 192 / 24 | 6 / 48 / 6 | 18 / 143 / 20 |
| 1 | 26 / 208 / 28 | 6 / 47 / 6 | 16 / 128 / 16 |
| 2 | 27 / 216 / 29 | 7 / 55 / 7 | 14 / 112 / 14 |

No relation crosses train, validation or test in any fold (checked in code
every time the split is built), and the 48 kept parts appear in exactly one
test fold.

### How shared content is checked

Three layers, run per dataset (per DAGM class):

1. Identical files and identical decoded pixels.
2. Pixel identity (E1 of the NEU detector) under any of the 8 flips and
   rotations and any shift, for images of any shape, with a texture guard:
   at least 1,024 textured pixels of the overlap must agree too, so flat or
   black areas cannot pass as shared content.
3. Re-processed or re-captured copies: masked normalized cross-correlation of
   band-passed images over every pair, transform and shift (FFT, GPU), first
   on images reduced 5x (KolektorSDD) or 8x (DAGM), then at 2x for each
   image's 10 best partners and every pair above 0.9. The 2x score decides:
   0.5 or more is the same content, 0.3 to 0.5 is grouped and shown to a
   person, whose decision is recorded in `configs/phase4_review.yaml`. The
   NEU grain test is not used here: it was calibrated on 200 px steel images
   and compares non-square images on a squashed grid.

Calibration on the data itself: 40 images per dataset (per DAGM class) are
copied 9 or 10 ways (JPEG quality 50 and 75, brightness and contrast, gamma,
blur, noise, a 10 percent crop, crop plus JPEG, flip plus brightness, and for
square images rotation plus JPEG) and pushed through the same pipeline. Every
copy must become a candidate and reach 0.5; otherwise the run stops. On
KolektorSDD all 360 copies were caught; the lowest fine score was 0.937
(10 percent crop).

## DAGM 2007 (Phase 4: detection and generalization)

Source: Zenodo record 10.5281/zenodo.8086136 (the DAGM 2007 competition
"Weakly Supervised Learning for Industrial Optical Inspection"), CC BY 4.0.
Images are not redistributed here.

### Contents (measured)

- 10 classes of synthetic textures, 512 x 512 grayscale PNG, each class with
  its own texture and defect type (`experiments/figures/phase4_dagm_samples.png`).
- Classes 1 to 6 have 1,150 images (575 Train, 575 Test), Classes 7 to 10
  have 2,300 (1,150 Train, 1,150 Test): 16,100 images, 2,100 defective.
- Labels are weak: each defective image comes with a label PNG holding a
  coarse ellipse around the defect, not a pixel mask. `Labels.txt` is parsed
  strictly (header, five fields per line, every defective image with its own
  label file, every normal image without one) and checked against the folder
  in both directions.

### Shared content

The three layers described above ran on every class: 14,539,450 pairs in all, 123,417
of them re-scored at fine scale.

- No identical files or pixels, no pixel-identical overlaps under any flip,
  rotation or shift, and no pair at or above 0.5 (the same content) in any
  class.
- Nine pairs fall in the review band, all just above 0.3 (0.3001 to 0.3167):
  two in Class 4, seven in Class 10, four of them across the official
  Train/Test boundary. They were checked by eye
  (`experiments/figures/phase4_dagm_review_pairs.png`) and are recorded as
  distinct in `configs/phase4_review.yaml`. Class 4 draws a starburst centred
  at the same place in every image and Class 10 a fan of curves rising from
  the bottom edge, so two images align at zero shift and score about 0.3; at
  native resolution the individual rays and lines differ. The scores continue
  smoothly from the best pairs below 0.3 (0.2997 in Class 4, 0.2963 in Class
  10), while planted re-processed copies of these classes score 0.959 or
  more. A decision can only move a pair out of the review band: the report
  stops if one names a pair with identical pixels or a score of 0.5 or more,
  and flags a decision whose pair the check no longer reports.
- Calibration: all 4,000 planted copies (40 per class, 10 variants each)
  became candidates and reached 0.5. The hardest variant in every class was
  the 10 percent crop; its lowest score was 0.777 in Class 1, 0.865 in Class
  2 and at least 0.947 in the other classes.

| Class | Images | Defective (Train / Test) | Pairs re-scored | Best distinct pair | Lowest planted copy |
|---|---|---|---|---|---|
| 1 | 1,150 | 79 / 71 | 9,861 | 0.2452 | 0.7771 |
| 2 | 1,150 | 66 / 84 | 7,177 | 0.1067 | 0.8650 |
| 3 | 1,150 | 66 / 84 | 9,537 | 0.1479 | 0.9657 |
| 4 | 1,150 | 82 / 68 | 8,861 | 0.3039 (decided) | 0.9658 |
| 5 | 1,150 | 70 / 80 | 9,060 | 0.0985 | 0.9470 |
| 6 | 1,150 | 83 / 67 | 7,974 | 0.1179 | 0.9539 |
| 7 | 2,300 | 150 / 150 | 22,009 | 0.2226 | 0.9596 |
| 8 | 2,300 | 150 / 150 | 18,145 | 0.2216 | 0.9729 |
| 9 | 2,300 | 150 / 150 | 15,519 | 0.1423 | 0.9746 |
| 10 | 2,300 | 150 / 150 | 15,274 | 0.3167 (decided) | 0.9592 |

### Split (`data/splits/dagm_split.json`)

The official Test side is the test set. A validation set of 20 percent is
drawn from each class's official Train side, stratified by the defect flag
(seed 42); the rest is the training set. No shared-content groups remain, so
no image is tied to another.

| Side | Images | Defective |
|---|---|---|
| Train | 6,440 | 837 |
| Validation | 1,610 | 209 |
| Test (official) | 8,050 | 1,054 |

The review decisions are stored in the split file, and the file is recomputed
and compared on every run.

### Running the check

`scripts/phase4_dagm_check.py` runs several classes at once, each in its own
process, with the pixel-identity step split over 12 helper processes per
class. On a laptop RTX 5070 (8 GB) and a 16-core Ryzen it took 1 h 08 min.
The coarse GPU screen sets the pace: about 10 minutes for each 2,300-image
class. The same run recomputed Classes 1 and 2, which an earlier run (one
class at a time) had cached, and both came out identical.

That earlier run stopped responding in Class 3: for two hours the process
used one CPU core at 100 percent while the GPU did nothing, and the same step
finished in two minutes in a fresh process. The parallel runner therefore
treats five minutes without progress as a hang: it writes the stack of every
thread to the class log, ends that process and its helpers, and retries the
class in a new process. It also reserves RAM and GPU memory before each large
step so that parallel classes cannot run the machine out of memory.

## Other datasets

| Dataset      | Purpose                        | License          | Link |
|--------------|--------------------------------|------------------|------|
| MVTec AD     | Optional anomaly benchmark (not used) | CC BY-NC-SA 4.0 (non-commercial) | https://www.mvtec.com/research-teaching/datasets/mvtec-ad |

License references:
- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
- CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
