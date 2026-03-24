#!/bin/bash
#SBATCH --job-name=eval_both
#SBATCH --partition=dgx
#SBATCH --output=logs/eval_both_%j.out
#SBATCH --error=logs/eval_both_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00

set -euo pipefail
mkdir -p logs eval_results

echo "========================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Host     : $(hostname)"
echo "Start    : $(date)"
echo "========================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher

export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false

DATA="data/eval300.json"
OUT_DIR="eval_results"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ── 3B evaluation (~3.5 hours) ──────────────────────────────
echo "========================================================"
echo "[1/2] Evaluating cipher-instruct-3B-20260320_235526"
echo "Start : $(date)"
echo "========================================================"

python -u eval_3b.py \
    --data    "$DATA"    \
    --out_dir "$OUT_DIR"

echo ""
echo "[1/2] 3B evaluation done: $(date)"
echo ""

# ── 14B evaluation (~5.5 hours) ─────────────────────────────
echo "========================================================"
echo "[2/2] Evaluating cipher-instruct-14B-20260321_143806"
echo "Start : $(date)"
echo "========================================================"

python -u eval_14b.py \
    --data    "$DATA"    \
    --out_dir "$OUT_DIR"

echo ""
echo "[2/2] 14B evaluation done: $(date)"
echo ""

echo "========================================================"
echo "Both evaluations complete: $(date)"
echo "Results in: $OUT_DIR/"
ls -lh "$OUT_DIR"/eval_3b_*.json "$OUT_DIR"/eval_14b_*.json 2>/dev/null || true
echo "========================================================"
