#!/usr/bin/env bash
# Reproduce every baseline result from scratch on the NEU dataset:
# dataset audit and EDA figures, classical branch, custom CNN, error analysis,
# and the comparison table (which also refreshes the README results block).
#
# Usage:
#   bash scripts/run_baseline.sh                 # uses configs/neu_baseline.yaml
#   CONFIG=configs/other.yaml bash scripts/run_baseline.sh
#   PYTHON=/path/to/python bash scripts/run_baseline.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/neu_baseline.yaml}"

echo "==> dataset report"
"$PYTHON" scripts/dataset_report.py --config "$CONFIG"
echo "==> classical branch (HOG + Decision Tree / SVM)"
"$PYTHON" scripts/train_classical.py --config "$CONFIG"
echo "==> custom CNN"
"$PYTHON" scripts/train_cnn.py --config "$CONFIG"
echo "==> error analysis"
"$PYTHON" scripts/error_analysis.py --config "$CONFIG"
echo "==> comparison table"
"$PYTHON" scripts/compare_results.py --config "$CONFIG" --update-readme
echo "==> publish key figures for the README (docs/figures)"
"$PYTHON" scripts/publish_figures.py --config "$CONFIG"
