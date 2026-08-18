#!/usr/bin/env python3
"""Train the custom CNN branch.

Usage:
    python scripts/train_cnn.py --config configs/neu_baseline.yaml
"""

import _bootstrap  # noqa: F401

from idefect.scripts_entry import train_cnn_main

if __name__ == "__main__":
    train_cnn_main()
