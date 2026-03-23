#!/bin/bash
#SBATCH --job-name=cipher_scale
#SBATCH --partition=dgx
#SBATCH --output=logs/scale_%x_%j.out
#SBATCH --error=logs/scale_%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=12:00:00

# ─────────────────────────────────────────────────────────────
# Универсальный скрипт для scaling эксперимента.
# Запуск:
#   sbatch --job-name=instruct_0.5b scale.slurm instruct 0.5B
#   sbatch --job-name=instruct_1b   scale.slurm instruct 1B
#   sbatch --job-name=instruct_3b   scale.slurm instruct 3B
#   sbatch --job-name=instruct_14b  scale.slurm instruct 14B
#   sbatch --job-name=coder_0.5b    scale.slurm coder 0.5B
#   sbatch --job-name=coder_1b      scale.slurm coder 1B
#   sbatch --job-name=coder_3b      scale.slurm coder 3B
#   sbatch --job-name=coder_14b     scale.slurm coder 14B
#
# Аргументы:
#   $1 — тип модели: instruct | coder
#   $2 — размер:     0.5B | 1B | 3B | 14B
# ─────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs data checkpoints

MODEL_TYPE="${1:-instruct}"
MODEL_SIZE="${2:-7B}"

echo "========================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Host     : $(hostname)"
echo "Start    : $(date)"
echo "Type     : $MODEL_TYPE  |  Size: $MODEL_SIZE"
echo "========================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher

export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false

# ── Маппинг размера → HuggingFace ID ─────────────────────────
case "$MODEL_TYPE-$MODEL_SIZE" in
    instruct-0.5B) BASE="Qwen/Qwen2.5-0.5B-Instruct" ;;
    instruct-1B)   BASE="Qwen/Qwen2.5-1.5B-Instruct" ;;
    instruct-3B)   BASE="Qwen/Qwen2.5-3B-Instruct"   ;;
    instruct-7B)   BASE="Qwen/Qwen2.5-7B-Instruct"   ;;
    instruct-14B)  BASE="Qwen/Qwen2.5-14B-Instruct"  ;;
    coder-0.5B)    BASE="Qwen/Qwen2.5-Coder-0.5B-Instruct" ;;
    coder-1B)      BASE="Qwen/Qwen2.5-Coder-1.5B-Instruct" ;;
    coder-3B)      BASE="Qwen/Qwen2.5-Coder-3B-Instruct"   ;;
    coder-7B)      BASE="Qwen/Qwen2.5-Coder-7B-Instruct"   ;;
    coder-14B)     BASE="Qwen/Qwen2.5-Coder-14B-Instruct"  ;;
    *) echo "ERROR: Unknown type-size combination: $MODEL_TYPE-$MODEL_SIZE"; exit 1 ;;
esac

# ── Датасет — тот же v1 что дал лучший результат ─────────────
# Для Coder используем датасет с code полем
if [ "$MODEL_TYPE" = "coder" ]; then
    TRAIN_FILE="data/train_coder.json"
    VAL_FILE="data/val_coder.json"
    TRAIN_SCRIPT="train_lora_coder.py"
    MAX_SEQ_LEN=2048
else
    TRAIN_FILE="data/train.json"
    VAL_FILE="data/val.json"
    TRAIN_SCRIPT="train_lora.py"
    MAX_SEQ_LEN=1536
fi

OUTPUT_DIR="checkpoints/cipher-${MODEL_TYPE}-${MODEL_SIZE}-$(date +%Y%m%d_%H%M%S)"

# ── Гиперпараметры — одинаковые для всех (V1 стратегия) ──────
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
EPOCHS=2
BATCH_SIZE=2
GRAD_ACCUM=8
LR=2e-4
MAX_TRAIN=12000
SEED=42
# ────────────────────────────────────────────────────────────

echo "Base model : $BASE"
echo "Dataset    : $TRAIN_FILE"
echo "Output     : $OUTPUT_DIR"
echo "LR         : $LR | Epochs: $EPOCHS | Batch: $((BATCH_SIZE*GRAD_ACCUM))"
echo "Max seq    : $MAX_SEQ_LEN"
echo ""

# Проверяем датасет
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Train file not found: $TRAIN_FILE"
    if [ "$MODEL_TYPE" = "coder" ]; then
        echo "  Run: python generate_dataset_coder.py"
    else
        echo "  Run: python generate_dataset.py"
    fi
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

python -u $TRAIN_SCRIPT \
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
echo "Adapter: $OUTPUT_DIR/final_adapter"
echo "========================================================"
