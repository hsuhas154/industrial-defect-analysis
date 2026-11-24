#!/usr/bin/env bash
# Reproduce every result in this repository, step by step, on the NEU dataset.
#
#   1. dataset report: integrity (manifest), statistics, near-duplicate audit, EDA figures
#   2. classical branch: features x classifiers, tuned with group-aware CV
#   3. custom CNN: one run per seed on the same split
#   4. classical ablations: preprocessing, resolution, HOG parameters
#   5. leakage study: random split versus group-aware split
#   6. CNN architectures from scratch (VGG, ResNet, DenseNet, MobileNet)
#   7. ImageNet transfer learning (frozen, partial, full fine-tuning)
#   8. error analysis for every final model
#   9. efficiency benchmark (run with nothing else on the GPU), CNN resolution
#      and CLAHE studies, data-requirement curve
#  10. figures (classical, Grad-CAM, studies), comparison table (refreshes the
#      README), README figures
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

step "1/10 dataset report";              "$PYTHON" scripts/dataset_report.py   --config "$CONFIG"
step "2/10 classical branch";            "$PYTHON" scripts/train_classical.py  --config "$CONFIG"
step "3/10 custom CNN (all seeds)";      "$PYTHON" scripts/train_cnn.py        --config "$CONFIG"
step "4/10 classical ablations";         "$PYTHON" scripts/run_ablations.py    --config "$CONFIG"
step "5/10 leakage study";               "$PYTHON" scripts/leakage_study.py    --config "$CONFIG"
step "6/10 CNNs from scratch";           "$PYTHON" scripts/train_cnn.py        --config "$CONFIG" \
    --arch vgg resnet densenet mobilenet
step "7/10 ImageNet transfer learning";  "$PYTHON" scripts/train_cnn.py        --config "$CONFIG" \
    --arch resnet18_imagenet densenet121_imagenet mobilenet_v2_imagenet --finetune frozen partial full
step "8/10 error analysis";              "$PYTHON" scripts/error_analysis.py   --config "$CONFIG"
step "9/10 efficiency benchmark and CNN studies"
"$PYTHON" scripts/benchmark_models.py --config "$CONFIG"
"$PYTHON" scripts/cnn_studies.py sensitivity --config "$CONFIG" --models custom_cnn vgg resnet18_imagenet:full
"$PYTHON" scripts/cnn_studies.py curve       --config "$CONFIG" --models custom_cnn vgg resnet18_imagenet:full
step "10/10 figures and comparison table"
"$PYTHON" scripts/classical_report.py --config "$CONFIG"
"$PYTHON" scripts/cnn_report.py       --config "$CONFIG"
"$PYTHON" scripts/compare_results.py  --config "$CONFIG" --update-readme
"$PYTHON" scripts/publish_figures.py  --config "$CONFIG"
echo; echo "done"
