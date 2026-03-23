#!/bin/bash
#SBATCH --job-name=cipherchat_v3
#SBATCH --partition=dgx
#SBATCH --output=logs/v3_%j.out
#SBATCH --error=logs/v3_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=12:00:00

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

# ── параметры ────────────────────────────────────────────────
BASE="Qwen/Qwen2.5-7B-Instruct"
V2_ADAPTER="checkpoints/cipherchat-7b-v2-20260309_173304/final_adapter"

TRAIN_FILE="data/train_v3.json"
VAL_FILE="data/val_v3.json"

OUTPUT_DIR="checkpoints/cipherchat-7b-v3-$(date +%Y%m%d_%H%M%S)"

LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

# Ключевые изменения vs V2:
#   LR:    5e-5 → 1e-4  (восстанавливаем JSON формат)
#   Epochs: 3  → 2      (не перегреваем)
EPOCHS=2
BATCH_SIZE=2
GRAD_ACCUM=8
LR=1e-4
MAX_SEQ_LEN=1536
MAX_TRAIN=12000
SEED=42
# ────────────────────────────────────────────────────────────

echo "Base model  : $BASE"
echo "V2 adapter  : $V2_ADAPTER"
echo "Output dir  : $OUTPUT_DIR"
echo "LR          : $LR  (V2 было 5e-5 → теперь 1e-4)"
echo "Epochs      : $EPOCHS  (V2 было 3 → теперь 2)"
echo "Eff. batch  : $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo ""

# Проверка V2 адаптера
if [ ! -d "$V2_ADAPTER" ]; then
    echo "ERROR: V2 adapter not found: $V2_ADAPTER"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ── Шаг 1: генерация датасета v3 ─────────────────────────────
if [ -f "$TRAIN_FILE" ] && [ -f "$VAL_FILE" ]; then
    echo "[1/2] V3 dataset already exists — skipping."
    echo "  $TRAIN_FILE"
    echo "  $VAL_FILE"
else
    echo "[1/2] Generating V3 dataset..."
    echo "  Vigenere=35%(↑) | Transposition=35%(↓) | Caesar=30%"
    echo "  UK=35%(↓) | SK=32.5% | EN=32.5%"
    echo "  compute=55% | detect=45%"
    echo "  JSON-reinforce: 15% short examples"

    python -u generate_dataset_v3.py \
        --num_train "$MAX_TRAIN"  \
        --num_val   1500          \
        --out_train "$TRAIN_FILE" \
        --out_val   "$VAL_FILE"   \
        --seed      7
fi
echo ""

# ── Шаг 2: обучение от V2 адаптера ───────────────────────────
echo "[2/2] Training V3 (resuming from V2 adapter, LR=1e-4, 2 epochs)..."

python -u train_lora_v3.py \
    --base                  "$BASE"         \
    --resume_from_adapter   "$V2_ADAPTER"   \
    --train_file            "$TRAIN_FILE"   \
    --val_file              "$VAL_FILE"     \
    --output_dir            "$OUTPUT_DIR"   \
    --lora_r                "$LORA_R"       \
    --lora_alpha            "$LORA_ALPHA"   \
    --lora_dropout          "$LORA_DROPOUT" \
    --epochs                "$EPOCHS"       \
    --batch_size            "$BATCH_SIZE"   \
    --grad_accum            "$GRAD_ACCUM"   \
    --lr                    "$LR"           \
    --max_seq_len           "$MAX_SEQ_LEN"  \
    --max_train             "$MAX_TRAIN"    \
    --seed                  "$SEED"

echo ""
echo "========================================================"
echo "Done: $(date)"
echo "V3 adapter: $OUTPUT_DIR/final_adapter"
echo ""
echo "Следующий шаг — eval:"
echo "  Edit eval300.slurm:"
echo "    ADAPTER=\"$OUTPUT_DIR/final_adapter\""
echo "    TRAIN_FILE=\"data/train_v3.json\""
echo "    VAL_FILE=\"data/val_v3.json\""
echo "  sbatch eval300.slurm"
echo "========================================================"
