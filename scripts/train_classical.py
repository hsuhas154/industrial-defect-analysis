#!/usr/bin/env python3
"""Train the classical branch (HOG + Decision Tree / SVM).

Usage:
    python scripts/train_classical.py --config configs/neu_baseline.yaml
"""

import _bootstrap  # noqa: F401

from idefect.scripts_entry import train_classical_main

if __name__ == "__main__":
    train_classical_main()
