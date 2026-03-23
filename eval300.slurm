#!/bin/bash
#SBATCH --job-name=cipherchat_eval300
#SBATCH --partition=dgx
#SBATCH --output=logs/eval300_%j.out
#SBATCH --error=logs/eval300_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=06:00:00

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

# ── параметры ───────────────────────────────────────────────
BASE="Qwen/Qwen2.5-7B-Instruct"
ADAPTER="checkpoints/cipherchat-7b-v3-20260314_225425/final_adapter"

# Для eval300 используем val_v2.json (unseen из v2 датасета)
# и train_v2.json (seen из v2 датасета)
TRAIN_FILE="data/train_v3.json"
VAL_FILE="data/val_v3.json"
EVAL_FILE="data/eval300_v2.json"

N_SEEN=150       # примеров из train (seen)
N_UNSEEN=150     # примеров из val (unseen)

OUT_DIR="eval_results"

# Результат v1 для сравнения (если есть)
V1_EVAL="eval_results/eval300_20260308_201750.json"
MAX_NEW_TOKENS=512
# ───────────────────────────────────────────────────────────

# Проверки
if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: Adapter not found: $ADAPTER"
    exit 1
fi
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Train file not found: $TRAIN_FILE"
    exit 1
fi
if [ ! -f "$VAL_FILE" ]; then
    echo "ERROR: Val file not found: $VAL_FILE"
    exit 1
fi

echo "Base model  : $BASE"
echo "Adapter     : $ADAPTER"
echo "Seen        : $N_SEEN samples from $TRAIN_FILE"
echo "Unseen      : $N_UNSEEN samples from $VAL_FILE"
echo "Eval file   : $EVAL_FILE"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ── Шаг 1: подготовить eval300.json ─────────────────────────
echo "[1/2] Preparing eval300 dataset..."

python -u prepare_eval300.py \
    --train_file "$TRAIN_FILE" \
    --val_file   "$VAL_FILE"   \
    --out        "$EVAL_FILE"  \
    --n_seen     "$N_SEEN"     \
    --n_unseen   "$N_UNSEEN"

echo ""
N_EVAL=$(python -c "import json; print(len(json.load(open('$EVAL_FILE'))))")
echo "eval300 ready: $N_EVAL examples"
echo ""

# ── Шаг 2: запустить eval ───────────────────────────────────
echo "[2/2] Running evaluation (base vs fine-tuned)..."

python -u eval_cipherchat.py \
    --base           "$BASE"           \
    --adapter        "$ADAPTER"        \
    --data           "$EVAL_FILE"      \
    --out_dir        "$OUT_DIR"        \
    --max_new_tokens "$MAX_NEW_TOKENS"

# ── Шаг 3: сравнение v1 vs v2 ───────────────────────────────
echo ""
echo "[3/3] Comparing V1 vs V2..."

# Найти последний сгенерированный eval файл v2
V2_EVAL=$(ls -t "$OUT_DIR"/eval300_*.json | head -1)

if [ -f "$V1_EVAL" ] && [ -f "$V2_EVAL" ]; then
    python -u compare_evals.py \
        --v1 "$V1_EVAL" \
        --v2 "$V2_EVAL"
    echo ""
    echo "Comparison saved → $OUT_DIR/comparison_v1_vs_v2.txt"
else
    echo "SKIP compare: V1=$V1_EVAL (exists=$([ -f "$V1_EVAL" ] && echo yes || echo no))"
    echo "              V2=$V2_EVAL"
fi

echo ""
echo "========================================================"
echo "Evaluation done: $(date)"
echo "Results in     : $OUT_DIR/"
echo "========================================================"
