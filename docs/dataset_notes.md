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

## Datasets reserved for later phases

| Dataset      | Purpose                        | License          | Link |
|--------------|--------------------------------|------------------|------|
| DAGM 2007    | Cross-texture generalization   | CC BY 4.0 (Heidelberg mirror) | https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html |
| KolektorSDD  | Defect localization/segmentation | CC BY-NC-SA 4.0 | https://www.vicos.si/resources/kolektorsdd/ |
| MVTec AD     | Optional anomaly benchmark     | CC BY-NC-SA 4.0 (non-commercial) | https://www.mvtec.com/research-teaching/datasets/mvtec-ad |

License references:
- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
- CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
