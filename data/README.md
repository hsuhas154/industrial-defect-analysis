# Datasets

Raw datasets are **not** committed to this repository. This directory only
documents how to obtain and lay out the data. Downloaded images are ignored by
`.gitignore`.

## NEU Surface Defect Database (Phase 1 primary dataset)

- **Content:** 1,800 grayscale images of hot-rolled steel-strip surface
  defects — 300 samples for each of six classes, originally 200×200 pixels.
- **Six classes:** rolled-in scale, patches, crazing, pitted surface,
  inclusion, and scratches.
- **Official source:** https://faculty.neu.edu.cn/songkc/en/zdylm/263265

### License caution

The official page establishes public availability and describes the dataset,
but it does not provide a clear standardized open-source license statement.
This project therefore preserves the original attribution and source
information and does **not** redistribute the raw dataset in Git. Please
download the data yourself from the official source and respect its terms.

### Expected layout

After downloading, place the images under `data/raw/NEU-CLS/`. The loader
(`idefect.data.datasets`) auto-detects either of these two layouts:

**Layout A — class sub-folders (recommended):**

```
data/raw/NEU-CLS/
├── crazing/          Cr_1.bmp, Cr_2.bmp, ...
├── inclusion/        In_1.bmp, ...
├── patches/          Pa_1.bmp, ...
├── pitted_surface/   PS_1.bmp, ...
├── rolled-in_scale/  RS_1.bmp, ...
└── scratches/        Sc_1.bmp, ...
```

**Layout B — flat NEU-CLS folder (as distributed):**

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
It is **only** for smoke-testing the code path — it is not real defect data and
must not be used to report results.

## Other datasets (post-Phase-1)

These are documented for the later phases and are not used in Phase 1:

- **DAGM 2007** (generalization): https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html — Heidelberg mirror is CC BY 4.0.
- **KolektorSDD** (localization): https://www.vicos.si/resources/kolektorsdd/ — CC BY-NC-SA 4.0.
- **MVTec AD** (optional anomaly benchmark): https://www.mvtec.com/research-teaching/datasets/mvtec-ad — CC BY-NC-SA 4.0 (non-commercial).
