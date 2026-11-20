#!/usr/bin/env python3
"""Choose the CNN training length on validation macro-F1 (several seeds).

Usage:
    python scripts/cnn_epoch_study.py --config configs/neu_baseline.yaml
"""

import _bootstrap  # noqa: F401

from idefect.scripts_entry import epoch_study_main

if __name__ == "__main__":
    epoch_study_main()
