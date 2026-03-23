#!/usr/bin/env bash
set -e

mkdir -p data/processed logs

echo "[1/3] Generate dataset (3 langs, 3 ciphers, key=str always)"
python src/data/generate_cipher_dataset_tri.py \
  --out_dir data/processed \
  --n_train 58800 \
  --n_val 1200 \
  --seed 42

echo "[2/3] Validate (key must be string)"
python scripts/validate_instruct.py --path data/processed/train_cipher_train.jsonl
python scripts/validate_instruct.py --path data/processed/val_cipher.jsonl

echo "[3/3] Stats"
python scripts/stats_dataset.py --path data/processed/train_cipher_train.jsonl

