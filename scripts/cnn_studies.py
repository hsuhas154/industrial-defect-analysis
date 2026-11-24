#!/usr/bin/env python3
"""CNN sensitivity studies (resolution, CLAHE) and the data-requirement curve.

Validation macro-F1 only; the test split is never used. Models are given as
arch or arch:finetune, classical models as feature_set:model.

Usage:
    python scripts/cnn_studies.py sensitivity --models custom_cnn resnet18_imagenet:full
    python scripts/cnn_studies.py curve --models custom_cnn --classical lbp+glcm:logreg
"""

import _bootstrap  # noqa: F401

from idefect.scripts_entry import cnn_studies_main

if __name__ == "__main__":
    cnn_studies_main()
