#!/bin/bash
#SBATCH --job-name=cipherchat_v4
#SBATCH --partition=dgx
#SBATCH --output=logs/v4_%j.out
#SBATCH --error=logs/v4_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=16:00:00

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

# V4: обучение С НУЛЯ — никакого resume
# Датасет v3 (Vigenere↑, UK умеренно, JSON-reinforce 15%)
TRAIN_FILE="data/train_v3.json"
VAL_FILE="data/val_v3.json"

OUTPUT_DIR="checkpoints/cipherchat-7b-v4-$(date +%Y%m%d_%H%M%S)"

LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

# V1 гиперпараметры — они дали лучший Valid JSON (96%)
# + 3 эпохи как в V2 (датасет v3 богаче чем v1)
EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LR=2e-4
MAX_SEQ_LEN=1536
MAX_TRAIN=12000
SEED=42
# ────────────────────────────────────────────────────────────

echo "Strategy : FROM SCRATCH (no adapter resume)"
echo "Base     : $BASE"
echo "Dataset  : $TRAIN_FILE  (v3 — Vigenere↑, JSON-reinforce 15%)"
echo "Output   : $OUTPUT_DIR"
echo "LR       : $LR  (V1 значение — V2/V3 были 5e-5/1e-4)"
echo "Epochs   : $EPOCHS"
echo "Eff batch: $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo ""

# Проверяем датасет
if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    echo "Dataset not found — generating v3..."
    python -u generate_dataset_v3.py \
        --num_train "$MAX_TRAIN" \
        --num_val   1500          \
        --out_train "$TRAIN_FILE" \
        --out_val   "$VAL_FILE"   \
        --seed      7
    echo ""
else
    echo "Dataset found — skipping generation."
    echo "  train: $TRAIN_FILE"
    echo "  val  : $VAL_FILE"
fi
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

echo "Training V4 from scratch..."
python -u train_lora.py \
    --base         "$BASE"         \
    --train_file   "$TRAIN_FILE"   \
    --val_file     "$VAL_FILE"     \
    --output_dir   "$OUTPUT_DIR"   \
    --lora_r       "$LORA_R"       \
    --lora_alpha   "$LORA_ALPHA"   \
    --lora_dropout "$LORA_DROPOUT" \
    --epochs       "$EPOCHS"       \
    --batch_size   "$BATCH_SIZE"   \
    --grad_accum   "$GRAD_ACCUM"   \
    --lr           "$LR"           \
    --max_seq_len  "$MAX_SEQ_LEN"  \
    --max_train    "$MAX_TRAIN"    \
    --seed         "$SEED"

echo ""
echo "========================================================"
echo "Done: $(date)"
echo "V4 adapter: $OUTPUT_DIR/final_adapter"
echo ""
echo "Eval:"
echo "  Edit eval300.slurm:"
echo "    ADAPTER=\"$OUTPUT_DIR/final_adapter\""
echo "    TRAIN_FILE=\"data/train_v3.json\""
echo "    VAL_FILE=\"data/val_v3.json\""
echo "  sbatch eval300.slurm"
echo "========================================================"
