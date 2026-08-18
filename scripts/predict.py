#!/usr/bin/env python3
"""Predict the defect class of a single image from the command line.

Usage:
    python scripts/predict.py path/to/image.bmp --model experiments/results/custom_cnn.pt
    python scripts/predict.py path/to/image.bmp --model experiments/results/hog+svm.joblib
"""

import _bootstrap  # noqa: F401

from idefect.scripts_entry import predict_main

if __name__ == "__main__":
    predict_main()
