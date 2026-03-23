#!/bin/bash
#SBATCH --job-name=cipherchat_train_tri
#SBATCH --partition=dgx
#SBATCH --output=logs/train_tri_%j.out
#SBATCH --error=logs/train_tri_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00

set -e
mkdir -p logs checkpoints

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

# conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher

# кеши HF (TRANSFORMERS_CACHE deprecated — убираем)
export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets

python -u src/model/train_lora_cipherchat.py \
  --base Qwen/Qwen2.5-7B-Instruct \
  --train_path data/processed/train_cipher_train.jsonl \
  --val_path data/processed/val_cipher.jsonl \
  --out_dir checkpoints/qwen-cipherchat-tri \
  --max_len 1024 \
  --lr 2e-4 \
  --epochs 1 \
  --bsz 1 \
  --grad_accum 8 \
  --save_steps 500 \
  --log_steps 20

echo "Finished at:"
date

