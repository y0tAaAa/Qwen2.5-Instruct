#!/bin/bash
#SBATCH --job-name=cipher_finetune_v2
#SBATCH --partition=dgx
#SBATCH --output=logs/finetune_v2_%j.out
#SBATCH --error=logs/finetune_v2_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs data checkpoints

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

# ── параметры ───────────────────────────────────────────────
BASE="Qwen/Qwen2.5-7B-Instruct"

# Продолжаем с существующего адаптера
PREV_ADAPTER="checkpoints/cipherchat-7b-20260306_153334/final_adapter"

TRAIN_FILE="data/train_v2.json"
VAL_FILE="data/val_v2.json"

OUTPUT_DIR="checkpoints/cipherchat-7b-v2-$(date +%Y%m%d_%H%M%S)"

# LoRA — те же параметры что при первом обучении
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

# 3 эпохи, пониженный LR (дообучение — не с нуля)
EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LR=5e-5        # ← ниже чем при первом обучении (2e-4)
MAX_SEQ_LEN=1536
MAX_TRAIN=12000
SEED=7
# ───────────────────────────────────────────────────────────

if [ ! -d "$PREV_ADAPTER" ]; then
    echo "ERROR: Previous adapter not found: $PREV_ADAPTER"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ── Шаг 1: генерация датасета v2 ────────────────────────────
echo "[1/2] Generating focused dataset v2..."
echo "  Transposition=40% | UK=40% | detect=50%"
echo ""

python -u generate_dataset_v2.py \
    --num_train "$MAX_TRAIN" \
    --num_val   1500          \
    --out_train "$TRAIN_FILE" \
    --out_val   "$VAL_FILE"   \
    --seed      "$SEED"

echo ""
echo "Dataset ready."
echo ""

# ── Шаг 2: дообучение с текущего адаптера ───────────────────
echo "[2/2] Fine-tuning from existing adapter..."
echo "  Base        : $BASE"
echo "  From adapter: $PREV_ADAPTER"
echo "  Output      : $OUTPUT_DIR"
echo "  LR          : $LR (reduced for fine-tuning)"
echo "  Epochs      : $EPOCHS"
echo "  Eff. batch  : $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo ""

python -u train_lora.py \
    --base                 "$BASE"         \
    --resume_from_adapter  "$PREV_ADAPTER" \
    --train_file           "$TRAIN_FILE"   \
    --val_file             "$VAL_FILE"     \
    --output_dir           "$OUTPUT_DIR"   \
    --lora_r               "$LORA_R"       \
    --lora_alpha           "$LORA_ALPHA"   \
    --lora_dropout         "$LORA_DROPOUT" \
    --epochs               "$EPOCHS"       \
    --batch_size           "$BATCH_SIZE"   \
    --grad_accum           "$GRAD_ACCUM"   \
    --lr                   "$LR"           \
    --max_seq_len          "$MAX_SEQ_LEN"  \
    --max_train            "$MAX_TRAIN"    \
    --seed                 "$SEED"

echo ""
echo "========================================================"
echo "Done: $(date)"
echo "New adapter: $OUTPUT_DIR/final_adapter"
echo ""
echo "Next step — run eval:"
echo "  Edit eval300.slurm: ADAPTER=\"$OUTPUT_DIR/final_adapter\""
echo "  sbatch eval300.slurm"
echo "========================================================"
