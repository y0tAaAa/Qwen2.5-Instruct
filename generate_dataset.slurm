#!/bin/bash
#SBATCH --job-name=cipher_gen_dataset
#SBATCH --partition=dgx
#SBATCH --output=logs/gen_dataset_%j.out
#SBATCH --error=logs/gen_dataset_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
mkdir -p logs data

echo "========================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Host     : $(hostname)"
echo "Start    : $(date)"
echo "========================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher

export HF_HOME=$HOME/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

# ── параметры датасета ──────────────────────────────────────
NUM_TRAIN=14000
NUM_VAL=1500
SEED=42

OUT_TRAIN="data/train.json"
OUT_VAL="data/val.json"
# ───────────────────────────────────────────────────────────

echo "[1/1] Generating dataset..."
echo "  num_train : $NUM_TRAIN"
echo "  num_val   : $NUM_VAL"
echo "  seed      : $SEED"
echo "  out_train : $OUT_TRAIN"
echo "  out_val   : $OUT_VAL"
echo ""

python -u generate_dataset.py \
    --num_train "$NUM_TRAIN" \
    --num_val   "$NUM_VAL"   \
    --out_train "$OUT_TRAIN" \
    --out_val   "$OUT_VAL"   \
    --seed      "$SEED"

echo ""
echo "========================================================"
echo "Dataset generation done: $(date)"
echo "Train : $OUT_TRAIN  ($(wc -c < "$OUT_TRAIN") bytes)"
echo "Val   : $OUT_VAL    ($(wc -c < "$OUT_VAL") bytes)"
echo "========================================================"
