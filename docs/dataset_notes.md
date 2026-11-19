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

* All 1,800 images are readable, 200x200, single-channel 8-bit.
* Exact duplicate pair: `Pa_101.bmp` and `Pa_105.bmp` (identical pixels). Kept in
  the same split by the duplicate-aware splitter.
* Most images show an illumination gradient (darker towards one edge), which is
  why local contrast normalisation (CLAHE) is part of preprocessing.

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
