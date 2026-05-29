#!/usr/bin/env bash
set -e
export PYTHONPATH=src

python scripts/00_check_data.py --config configs/default.yaml
python scripts/01_prepare_manifest.py --config configs/default.yaml
python scripts/02_extract_signmusketeers_dinov2_features.py --config configs/default.yaml

python scripts/03_train_t5.py --config configs/default.yaml --run_name paper_like_pretrained_dinov2 --mode paper
python scripts/04_evaluate_t5.py --config configs/default.yaml --checkpoint checkpoints/paper_like_pretrained_dinov2/best.pt --split test --out_csv outputs/pred_paper_like_pretrained_dinov2.csv

python scripts/03_train_t5.py --config configs/default.yaml --run_name proposed_confidence_aware --mode confidence
python scripts/04_evaluate_t5.py --config configs/default.yaml --checkpoint checkpoints/proposed_confidence_aware/best.pt --split test --out_csv outputs/pred_proposed_confidence_aware.csv
