#!/bin/bash
#SBATCH --job-name=cipher_train
#SBATCH --partition=dgx
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=24:00:00

set -e
mkdir -p logs

# если conda activate ломается, используй источник:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cypher

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

python -m src.model.train_lora_cipherchat \
  --base Qwen/Qwen2.5-7B-Instruct \
  --train_path data/processed/train_cipher_train.jsonl \
  --val_path data/processed/val_cipher.jsonl \
  --out_dir checkpoints/qwen-cipherchat-variants \
  --max_len 512 \
  --epochs 1 \
  --bsz 1 \
  --grad_accum 8 \
  --lr 2e-4 \
  --save_steps 500 \
  --log_steps 25

