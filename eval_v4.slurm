#!/bin/bash
#SBATCH --job-name=cipherchat_eval_v4
#SBATCH --partition=dgx
#SBATCH --output=logs/eval_v4_%j.out
#SBATCH --error=logs/eval_v4_%j.err
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

# ── параметры — ОБНОВИТЬ после окончания finetune_v4.slurm ──
BASE="Qwen/Qwen2.5-7B-Instruct"
V4_ADAPTER="checkpoints/cipherchat-7b-v4-20260315_173141/final_adapter"

TRAIN_FILE="data/train_v3.json"
VAL_FILE="data/val_v3.json"
EVAL_FILE="data/eval300_v3.json"

N_SEEN=150
N_UNSEEN=150
OUT_DIR="eval_results"
MAX_NEW_TOKENS=512

# Предыдущие eval файлы для сравнения
V1_EVAL="eval_results/eval300_20260308_201750.json"
V2_EVAL="eval_results/eval300_20260312_215731.json"
V3_EVAL="eval_results/eval300_20260315_130425.json"
# ────────────────────────────────────────────────────────────

# Проверки
if [[ "$V4_ADAPTER" == *"XXXXXXXX"* ]]; then
    echo "ERROR: Обнови V4_ADAPTER в eval_v4.slurm!"
    echo "  Найди адаптер: ls checkpoints/cipherchat-7b-v4-*/final_adapter"
    exit 1
fi
if [ ! -d "$V4_ADAPTER" ]; then
    echo "ERROR: V4 adapter not found: $V4_ADAPTER"
    exit 1
fi

echo "Base     : $BASE"
echo "V4 adapter: $V4_ADAPTER"
echo "Train    : $TRAIN_FILE"
echo "Val      : $VAL_FILE"
echo "Eval file: $EVAL_FILE"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ── Шаг 1: eval300 датасет ───────────────────────────────────
echo "[1/3] Preparing eval300 (v3 dataset)..."

python -u prepare_eval300.py \
    --train_file "$TRAIN_FILE" \
    --val_file   "$VAL_FILE"   \
    --out        "$EVAL_FILE"  \
    --n_seen     "$N_SEEN"     \
    --n_unseen   "$N_UNSEEN"

N_EVAL=$(python -c "import json; print(len(json.load(open('$EVAL_FILE'))))")
echo "eval300 ready: $N_EVAL examples"
echo ""

# ── Шаг 2: eval V4 ───────────────────────────────────────────
echo "[2/3] Running evaluation (base vs V4)..."

python -u eval_cipherchat.py \
    --base           "$BASE"           \
    --adapter        "$V4_ADAPTER"     \
    --data           "$EVAL_FILE"      \
    --out_dir        "$OUT_DIR"        \
    --max_new_tokens "$MAX_NEW_TOKENS"

# Найти новый eval файл
V4_EVAL=$(ls -t "$OUT_DIR"/eval300_*.json | head -1)
echo ""
echo "V4 eval saved: $V4_EVAL"

# ── Шаг 3: сравнение всех версий ─────────────────────────────
echo ""
echo "[3/3] Comparing V1 vs V2 vs V3 vs V4..."

# Собираем существующие файлы
EVAL_FILES=()
EVAL_NAMES=()

for pair in "$V1_EVAL:V1" "$V2_EVAL:V2" "$V3_EVAL:V3" "$V4_EVAL:V4"; do
    path="${pair%%:*}"
    name="${pair##*:}"
    if [ -f "$path" ]; then
        EVAL_FILES+=("$path")
        EVAL_NAMES+=("$name")
    else
        echo "SKIP $name: $path not found"
    fi
done

if [ ${#EVAL_FILES[@]} -ge 2 ]; then
    python -u compare_all.py \
        --evals "${EVAL_FILES[@]}" \
        --names "${EVAL_NAMES[@]}" \
        --out   "$OUT_DIR/comparison_all.txt"
    echo ""
    echo "Comparison saved → $OUT_DIR/comparison_all.txt"
else
    echo "Not enough eval files to compare (need >= 2)"
fi

echo ""
echo "========================================================"
echo "Evaluation done: $(date)"
echo "Results: $OUT_DIR/"
echo "========================================================"
