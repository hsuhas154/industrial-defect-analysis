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

### Known data issue: one duplicate pair

`Pa_101.bmp` and `Pa_105.bmp` are byte-identical. The split keeps exact
duplicates in the same partition (`data.group_duplicates: true` in the config),
so the duplicate cannot appear in both training and test data.

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

## Other datasets (later phases)

These are documented for later phases and are not used by the baseline:

- **DAGM 2007** (generalization): https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html (license: CC BY 4.0 via the Heidelberg mirror).
- **KolektorSDD** (localization): https://www.vicos.si/resources/kolektorsdd/ (license: CC BY-NC-SA 4.0).
- **MVTec AD** (optional anomaly benchmark): https://www.mvtec.com/research-teaching/datasets/mvtec-ad (license: CC BY-NC-SA 4.0, non-commercial).
