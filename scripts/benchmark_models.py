#!/usr/bin/env python3
"""Efficiency benchmark of every trained model: parameters, size, latency,
throughput and GPU memory. Run it with nothing else on the GPU.

Usage:
    python scripts/benchmark_models.py --config configs/neu_baseline.yaml
"""

import _bootstrap  # noqa: F401

from idefect.scripts_entry import benchmark_main

if __name__ == "__main__":
    benchmark_main()
