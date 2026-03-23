#!/bin/bash
#SBATCH --job-name=cypher_eval
#SBATCH --partition=dgx
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --cpus-per-task=10
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher

cd ~/cipherchat

python src/model/eval_model.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path checkpoints/qwen-cipher-lora-selfscore \
  --eval_path data/processed/val_cipher.jsonl \
  --max_new_tokens 256 \
  --max_samples 500 \
  --log_every 50

