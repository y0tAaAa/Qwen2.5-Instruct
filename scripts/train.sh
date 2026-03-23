#!/usr/bin/env bash
set -e

# Дообучаем существующий адаптер y0ta/qwen-cipher-lora
# с reasoning + self_score
python src/model/train_lora.py

