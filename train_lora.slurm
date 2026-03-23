#!/bin/bash
#SBATCH --job-name=cipherchat_train
#SBATCH --partition=dgx
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs checkpoints

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

# ── параметры обучения ──────────────────────────────────────
BASE="Qwen/Qwen2.5-7B-Instruct"

TRAIN_FILE="data/train.json"
VAL_FILE="data/val.json"

OUTPUT_DIR="checkpoints/cipherchat-7b-$(date +%Y%m%d_%H%M%S)"

LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

EPOCHS=2
BATCH_SIZE=2
GRAD_ACCUM=8
LR=2e-4
MAX_SEQ_LEN=1536
MAX_TRAIN=12000
SEED=42
# ───────────────────────────────────────────────────────────

# Проверка что датасет существует
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Train file not found: $TRAIN_FILE"
    echo "Run generate_dataset.slurm first."
    exit 1
fi
if [ ! -f "$VAL_FILE" ]; then
    echo "ERROR: Val file not found: $VAL_FILE"
    echo "Run generate_dataset.slurm first."
    exit 1
fi

echo "Base model  : $BASE"
echo "Train file  : $TRAIN_FILE"
echo "Val file    : $VAL_FILE"
echo "Output dir  : $OUTPUT_DIR"
echo "LoRA r/alpha: $LORA_R / $LORA_ALPHA"
echo "Epochs      : $EPOCHS"
echo "Eff. batch  : $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "LR          : $LR"
echo "Max seq len : $MAX_SEQ_LEN"
echo ""

# Вывод информации о GPU
nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader 2>/dev/null || true
echo ""

echo "[1/1] Training CipherChat..."

python -u train_lora.py \
    --base        "$BASE"        \
    --train_file  "$TRAIN_FILE"  \
    --val_file    "$VAL_FILE"    \
    --output_dir  "$OUTPUT_DIR"  \
    --lora_r      "$LORA_R"      \
    --lora_alpha  "$LORA_ALPHA"  \
    --lora_dropout "$LORA_DROPOUT" \
    --epochs      "$EPOCHS"      \
    --batch_size  "$BATCH_SIZE"  \
    --grad_accum  "$GRAD_ACCUM"  \
    --lr          "$LR"          \
    --max_seq_len "$MAX_SEQ_LEN" \
    --max_train   "$MAX_TRAIN"   \
    --seed        "$SEED"

echo ""
echo "========================================================"
echo "Training done: $(date)"
echo "Adapter saved: $OUTPUT_DIR/final_adapter"
echo "========================================================"
