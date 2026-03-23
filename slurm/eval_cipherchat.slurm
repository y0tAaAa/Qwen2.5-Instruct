#!/bin/bash
#SBATCH --job-name=cipher_eval
#SBATCH --partition=dgx
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=08:00:00

set -e
mkdir -p logs

export PYTHONUNBUFFERED=1

python -m src.model.eval_model \
  --base Qwen/Qwen2.5-7B-Instruct \
  --lora_path checkpoints/qwen-cipherchat-variants \
  --eval_path data/processed/val_cipher.jsonl \
  --max_samples 500 \
  --max_new_tokens 256 \
  --log_every 50

