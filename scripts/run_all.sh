#!/usr/bin/env bash
# Reproduce every result in this repository, step by step, on the NEU dataset.
#
#   1. dataset report: integrity (manifest), statistics, near-duplicate audit, EDA figures
#   2. classical branch: features x classifiers, tuned with group-aware CV
#   3. custom CNN: one run per seed on the same split
#   4. error analysis for every final model
#   5. classical ablations: preprocessing, resolution, HOG parameters
#   6. leakage study: random split versus group-aware split
#   7. classical figures, comparison table (refreshes the README), README figures
#
# The CNN training length was chosen beforehand with scripts/cnn_epoch_study.py
# (validation only); rerun it separately if you change the model.
#
# Usage:
#   bash scripts/run_all.sh                       # uses configs/neu_baseline.yaml
#   CONFIG=configs/other.yaml bash scripts/run_all.sh
#   PYTHON=/path/to/python bash scripts/run_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/neu_baseline.yaml}"
step() { echo; echo "==> $*"; }

step "1/7 dataset report";               "$PYTHON" scripts/dataset_report.py   --config "$CONFIG"
step "2/7 classical branch";             "$PYTHON" scripts/train_classical.py  --config "$CONFIG"
step "3/7 custom CNN (all seeds)";       "$PYTHON" scripts/train_cnn.py        --config "$CONFIG"
step "4/7 error analysis";               "$PYTHON" scripts/error_analysis.py   --config "$CONFIG"
step "5/7 classical ablations";          "$PYTHON" scripts/run_ablations.py    --config "$CONFIG"
step "6/7 leakage study";                "$PYTHON" scripts/leakage_study.py    --config "$CONFIG"
step "7/7 figures and comparison table"
"$PYTHON" scripts/classical_report.py --config "$CONFIG"
"$PYTHON" scripts/compare_results.py  --config "$CONFIG" --update-readme
"$PYTHON" scripts/publish_figures.py  --config "$CONFIG"
echo; echo "done"
