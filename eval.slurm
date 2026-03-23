#!/bin/bash
#SBATCH --job-name=cipher_eval4
#SBATCH --partition=dgx
#SBATCH --output=logs/eval4_%j.out
#SBATCH --error=logs/eval4_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00

set -euo pipefail
mkdir -p logs data/processed

echo "Job: $SLURM_JOB_ID | $(hostname) | $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher

export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false

BASE="Qwen/Qwen2.5-7B-Instruct"

COMPUTE_ADAPTER="checkpoints/qwen-cipherchat-compute-v2"
DETECT_ADAPTER="checkpoints/qwen-cipherchat-detect-v2"

# Easy val (из вашего split)
COMPUTE_EASY="data/processed/train_compute.clean.val.jsonl"
DETECT_EASY="data/processed/train_detect.clean.val.jsonl"

# Hard val (если нет — сгенерируем)
COMPUTE_HARD="data/processed/val_hard_compute.jsonl"
DETECT_HARD="data/processed/val_hard_detect.jsonl"

echo "[0/2] Ensure hard-val exists..."
if [ ! -f "$COMPUTE_HARD" ]; then
  python -u src/data/generate_val_hard.py --style compute --num 2000 --out "$COMPUTE_HARD" --seed 777 --hard_key_holdout
fi
if [ ! -f "$DETECT_HARD" ]; then
  python -u src/data/generate_val_hard.py --style detect --num 2000 --out "$DETECT_HARD" --seed 888 --hard_key_holdout
fi

echo "[1/2] Run 4 evals..."

python -u src/eval/eval_cipherchat.py \
  --base "$BASE" \
  --adapter "$COMPUTE_ADAPTER" \
  --data "$COMPUTE_EASY" \
  --task compute \
  --report_out "logs/eval_compute_easy.json"

python -u src/eval/eval_cipherchat.py \
  --base "$BASE" \
  --adapter "$COMPUTE_ADAPTER" \
  --data "$COMPUTE_HARD" \
  --task compute \
  --report_out "logs/eval_compute_hard.json"

python -u src/eval/eval_cipherchat.py \
  --base "$BASE" \
  --adapter "$DETECT_ADAPTER" \
  --data "$DETECT_EASY" \
  --task detect \
  --report_out "logs/eval_detect_easy.json"

python -u src/eval/eval_cipherchat.py \
  --base "$BASE" \
  --adapter "$DETECT_ADAPTER" \
  --data "$DETECT_HARD" \
  --task detect \
  --report_out "logs/eval_detect_hard.json"

echo "[2/2] Done: $(date)"
echo "Reports:"
echo " - logs/eval_compute_easy.json"
echo " - logs/eval_compute_hard.json"
echo " - logs/eval_detect_easy.json"
echo " - logs/eval_detect_hard.json"
