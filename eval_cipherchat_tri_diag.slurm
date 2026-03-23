#!/bin/bash
#SBATCH --job-name=cypher_eval_diag
#SBATCH --partition=dgx
#SBATCH --output=logs/eval_tri_diag_%j.out
#SBATCH --error=logs/eval_tri_diag_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=shard:1
#SBATCH --time=12:00:00

set -euo pipefail
set -x

echo "== HEADER =="
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

echo "== PYTHON INFO =="
which python
python --version
python -c "import sys; print('executable:', sys.executable)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devs', torch.cuda.device_count())"
echo

BASE="Qwen/Qwen2.5-7B-Instruct"
LORA="checkpoints/qwen-cipherchat-tri"
EVAL="data/processed/val_cipher.jsonl"

echo "== PATH CHECKS =="
ls -lah src/model/eval_model.py
head -n 40 src/model/eval_model.py
ls -lah "$LORA" | head
wc -l "$EVAL"
head -n 2 "$EVAL"
echo

echo "== RUN EVAL =="
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python -X faulthandler -u src/model/eval_model.py \
  --base "$BASE" \
  --lora_path "$LORA" \
  --eval_path "$EVAL" \
  --max_samples 50 \
  --max_new_tokens 64 \
  --log_every 10

echo "== DONE =="
date

