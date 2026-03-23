#!/bin/bash
#SBATCH --job-name=cypher_eval
#SBATCH --partition=dgx
#SBATCH --output=logs/eval_tri_%j.out
#SBATCH --error=logs/eval_tri_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=shard:1
#SBATCH --time=08:00:00

set -euo pipefail

echo "Starting eval job..."
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true
echo

BASE="Qwen/Qwen2.5-7B-Instruct"
LORA_PATH="checkpoints/qwen-cipherchat-tri"
EVAL_PATH="data/processed/val_cipher.jsonl"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python -u src/model/eval_model.py \
  --base "$BASE" \
  --lora_path "$LORA_PATH" \
  --eval_path "$EVAL_PATH" \
  --max_samples 1000 \
  --max_new_tokens 256 \
  --log_every 50

echo "Finished at: $(date)"

