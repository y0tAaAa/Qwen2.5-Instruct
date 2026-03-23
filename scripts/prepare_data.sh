#!/usr/bin/env bash
set -e

# 1) Синтетический датасет (Caesar/Vigenere/Substitution, en/sk/uk, reasoning + self_score)
python src/data/make_cipher_dataset_multi.py

# 2) Конвертация старого датасета (если файла нет — просто выведет "no old dataset, skip")
if [ -f data/raw/train_with_reasoning.jsonl ]; then
  python src/data/convert_train_with_reasoning.py
else
  echo "data/raw/train_with_reasoning.jsonl not found, skip convert."
fi

# 3) Склейка + train/val split
python src/data/split_train_val.py

