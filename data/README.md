# Datasets

Raw datasets are **not** committed to this repository. This directory only
documents how to obtain and lay out the data. Downloaded images are ignored by
`.gitignore`.

## NEU Surface Defect Database (primary dataset)

- **Content:** 1,800 grayscale images of hot-rolled steel-strip surface
  defects, 300 samples for each of six classes, 200×200 pixels, 8-bit BMP.
- **Six classes:** rolled-in scale, patches, crazing, pitted surface,
  inclusion, and scratches.
- **Official source:** https://faculty.neu.edu.cn/songkc/en/zdylm/263265

### Download

The official page links two archives. This project uses **NEU-CLS** (the
classification set, images only); NEU-DET adds bounding boxes and is not needed.

1. Open the official page and follow the NEU-CLS link (Google Drive; a Baidu
   Pan mirror is also listed).
2. The download is a RAR archive (about 48 MB) containing a single `NEU-CLS/`
   folder with 1,800 `.bmp` files and a stray `Thumbs.db`.
3. Extract it and copy the images into `data/raw/NEU-CLS/`, for example:

```bash
mkdir -p data/raw
bsdtar -xf NEU-CLS.rar -C data/raw      # or: unrar x NEU-CLS.rar data/raw/
ls data/raw/NEU-CLS/*.bmp | wc -l       # expect 1800
```

4. Verify with the audit, which checks counts, sizes, unreadable files and
   duplicates:

```bash
python scripts/dataset_report.py
```

Version used for the reported results: archive SHA-256
`a85bcc6bfa1a40e76f28079e135694aaf5f6a4ee01203f7eaacff0ebafaa96f8`.

### Known data issues: duplicates and overlapping crops

`Pa_101.bmp` and `Pa_105.bmp` are byte-identical, and far more importantly
most NEU-CLS images share exact regions with other images (overlapping crops,
often flipped or rotated) or are re-processed copies of each other. The
pipeline detects these relations and keeps each group inside one split; see
`docs/dataset_notes.md` for the numbers and `docs/methodology.md` for the
method.

### Integrity

`data/neu_cls_manifest.sha256` lists the SHA-256 of every image. Check it
with:

```bash
(cd data/raw/NEU-CLS && sha256sum -c ../../neu_cls_manifest.sha256 --quiet) && echo intact
```

Every run performs the same check and stops if a file is missing or changed.
Do not open, rotate or save images inside `data/raw` with an image editor or
viewer; copy them elsewhere first.

### License caution

The official page establishes public availability and describes the dataset,
but it does not provide a clear standardized open-source license statement.
This project therefore preserves the original attribution and source
information and does **not** redistribute the raw dataset in Git. Please
download the data yourself from the official source and respect its terms.

### Expected layout

After downloading, place the images under `data/raw/NEU-CLS/`. The loader
(`idefect.data.datasets`) auto-detects either of these two layouts:

**Layout A, class sub-folders (recommended):**

```
data/raw/NEU-CLS/
├── crazing/          Cr_1.bmp, Cr_2.bmp, ...
├── inclusion/        In_1.bmp, ...
├── patches/          Pa_1.bmp, ...
├── pitted_surface/   PS_1.bmp, ...
├── rolled-in_scale/  RS_1.bmp, ...
└── scratches/        Sc_1.bmp, ...
```

**Layout B, flat NEU-CLS folder (as distributed):**

```
data/raw/NEU-CLS/
├── Cr_1.bmp   Cr_2.bmp   ...   (crazing)
├── In_1.bmp   ...              (inclusion)
├── Pa_1.bmp   ...              (patches)
├── PS_1.bmp   ...              (pitted_surface)
├── RS_1.bmp   ...              (rolled-in_scale)
└── Sc_1.bmp   ...              (scratches)
```

The two-letter filename prefixes map to classes as follows:

| Prefix | Class            |
|--------|------------------|
| `Cr`   | crazing          |
| `In`   | inclusion        |
| `Pa`   | patches          |
| `PS`   | pitted_surface   |
| `RS`   | rolled-in_scale  |
| `Sc`   | scratches        |

## Don't have the data yet? Generate a synthetic stand-in

To exercise the full pipeline (and the tests) before downloading NEU, generate
a small synthetic dataset with the same six-class structure:

```bash
python scripts/generate_dummy_data.py --out data/raw/NEU-CLS --per-class 40
```

This produces class sub-folders of procedurally generated grayscale textures.
It is **only** for smoke-testing the code path; it is not real defect data and
must not be used to report results.

## Phase 4 datasets: KolektorSDD and DAGM 2007

Both are used from Phase 4 on: KolektorSDD for pixel-level defect localization
(U-Net) and DAGM 2007 for normal versus defective detection and for testing how
far methods developed on NEU generalize.

### Download

Put the archives exactly here (`data/raw/` is git-ignored):

```
data/raw/KolektorSDD/KolektorSDD.zip                    (101,831,129 bytes)
data/raw/KolektorSDD/KolektorSDD-training-splits.zip
data/raw/DAGM/zips/Class1.zip ... Class10.zip
```

- **KolektorSDD** (fine annotations, CC BY-NC-SA 4.0, non-commercial):
  https://www.vicos.si/resources/kolektorsdd/ links
  `https://data.vicos.si/datasets/KSDD/KolektorSDD.zip` and the official
  3-fold split `https://data.vicos.si/datasets/KSDD/KolektorSDD-training-splits.zip`.
  The box-annotation and dilated-mask variants on the same page are not used.
- **DAGM 2007** (CC BY 4.0): Zenodo record 10.5281/zenodo.8086136, one zip
  per class, `https://zenodo.org/records/8086136/files/Class1.zip?download=1`
  and so on up to `Class10.zip`.

### Verify, extract and fingerprint

```bash
python scripts/prepare_phase4_data.py
```

The script stops at the first problem and never overwrites a file:

1. Every archive must match the SHA-256 in `data/phase4_archives.sha256`;
   the DAGM zips must also match the MD5 that Zenodo publishes.
2. Every member name is validated before anything is written (no absolute
   paths, `..`, symlinks, encrypted or duplicate members, nothing outside the
   expected folder); `Thumbs.db` files inside the DAGM zips are skipped.
3. Every file is written to a temporary name, its size and CRC-32 are checked
   against the archive, and only then is it moved into place. An existing file
   that differs from the archive stops the run (it was changed after
   extraction).
4. The extracted trees are checked against the committed manifests
   `data/kolektorsdd_manifest.sha256` (805 files) and
   `data/dagm_manifest.sha256` (18,220 files): nothing missing, modified or
   unexpected. Independently:

```bash
(cd data/raw/KolektorSDD && sha256sum -c --quiet ../../kolektorsdd_manifest.sha256) && echo intact
(cd data/raw/DAGM && sha256sum -c --quiet ../../dagm_manifest.sha256) && echo intact
```

A second run re-verifies everything and writes nothing.

### Layout after extraction

```
data/raw/KolektorSDD/
├── kos01 ... kos50/        PartN.jpg + PartN_label.bmp (399 images, 52 defective)
└── splits/                 split.pyb (official 3 folds), split_small_*.pyb, README
data/raw/DAGM/
├── zips/                   the downloads
└── Class1 ... Class10/
    ├── Train/              NNNN.PNG, Label/Labels.txt, Label/NNNN_label.PNG
    └── Test/               same layout (16,100 images, 2,100 defective)
```

### Leakage-safe splits

```bash
python scripts/phase4_dataset_report.py
```

checks shared content within each dataset (identical files, pixel-identical
overlaps under any flip, rotation or shift, and re-processed or re-captured
copies, calibrated with planted copies) and writes the splits used by every
Phase 4 experiment to `data/splits/`: `kolektorsdd_folds.json` (official folds
corrected for the leaks they contain, plus validation parts) and
`dagm_split.json` (official Train/Test plus a validation set carved from
Train). Once written, a split file is recomputed and compared on every run, so
it cannot change silently. Pairs in the review band (fine score 0.3 to 0.5)
are grouped until a person decides them; decisions go into
`configs/phase4_review.yaml` (a side-by-side figure of every such pair is
written to `experiments/figures/phase4_dagm_review_pairs.png`). The findings
are in `docs/dataset_notes.md`.

The DAGM check is the slow part (16,100 images, every pair compared). Run it
first with

```bash
python scripts/phase4_dagm_check.py
```

which computes several classes at once in separate processes and caches each
class's result where the report finds it. It reserves RAM and GPU memory
before each memory-heavy step so the machine cannot run out, retries a class
in a fresh process if it stops making progress for 5 minutes (writing the
Python stack to its log first), prints a status block every minute and
recomputes the already-cached classes to prove that it gives exactly the same
result. `python scripts/phase4_dagm_check.py --status` shows the latest status
from a second terminal; logs are in `experiments/logs/phase4_dagm/latest/`.

The split file `split.pyb` is a pickle; it is decoded by a small interpreter
that accepts only lists, tuples and strings and is never passed to `pickle`.

## Other datasets

- **MVTec AD** (optional anomaly benchmark): https://www.mvtec.com/research-teaching/datasets/mvtec-ad (license: CC BY-NC-SA 4.0, non-commercial).
