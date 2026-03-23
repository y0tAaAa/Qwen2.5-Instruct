#!/bin/bash
set -e

mkdir -p data/processed

python -m src.data.generate_cipher_dataset_variants \
  --num_samples 60000 \
  --out data/processed/train_cipher_variants.jsonl

python -m src.data.split_train_val \
  --input data/processed/train_cipher_variants.jsonl \
  --train_out data/processed/train_cipher_train.jsonl \
  --val_out data/processed/val_cipher.jsonl \
  --val_ratio 0.02

