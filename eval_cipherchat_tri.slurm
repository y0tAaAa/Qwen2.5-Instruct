#!/bin/bash
#SBATCH --job-name=cypher_eval_tri
#SBATCH --partition=dgx
#SBATCH --output=logs/eval_tri_%j.out
#SBATCH --error=logs/eval_tri_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=shard:1
#SBATCH --time=12:00:00

set -euo pipefail

echo "Starting eval job..."
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "PWD:    $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo

nvidia-smi || true
echo

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

BASE="Qwen/Qwen2.5-7B-Instruct"
LORA="checkpoints/qwen-cipherchat-tri"
EVAL="data/processed/val_cipher.jsonl"

echo "Running eval:"
echo "  BASE=$BASE"
echo "  LORA=$LORA"
echo "  EVAL=$EVAL"
echo

python -u src/model/eval_model.py \
  --base "$BASE" \
  --lora_path "$LORA" \
  --eval_path "$EVAL" \
  --max_samples 1000 \
  --max_new_tokens 256 \
  --log_every 50

echo
echo "Finished at: $(date)"

