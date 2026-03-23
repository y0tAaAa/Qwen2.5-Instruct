#!/bin/bash
#SBATCH --job-name=cipherchat_compute
#SBATCH --partition=dgx
#SBATCH --output=logs/compute_%j.out
#SBATCH --error=logs/compute_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs checkpoints data/processed

echo "Job: $SLURM_JOB_ID | $(hostname) | $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher

export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false

# -------- INPUT (твой исходный файл) --------
RAW="data/processed/train_compute.jsonl"

# -------- SANITIZED --------
CLEAN="data/processed/train_compute.clean.jsonl"

# -------- SPLIT OUTPUT --------
TRAIN_OUT="data/processed/train_compute.clean.train.jsonl"
VAL_OUT="data/processed/train_compute.clean.val.jsonl"
VAL_RATIO=0.02
SPLIT_SEED=42

echo "[1/3] Sanitize JSONL..."
python -u src/data/sanitize_cipher_jsonl.py \
  --inp "$RAW" \
  --out "$CLEAN" \
  --skip_bad

echo "[2/3] Split train/val..."
python - << PY
import os, random
inp="$CLEAN"; tr="$TRAIN_OUT"; va="$VAL_OUT"
val_ratio=float("$VAL_RATIO"); seed=int("$SPLIT_SEED")

n=0
with open(inp,"r",encoding="utf-8") as f:
    for _ in f: n+=1
val_n=max(1,int(n*val_ratio))
rnd=random.Random(seed)
val_idx=set(rnd.sample(range(n), val_n))

os.makedirs(os.path.dirname(tr), exist_ok=True)
with open(inp,"r",encoding="utf-8") as fin, \
     open(tr,"w",encoding="utf-8") as ftr, \
     open(va,"w",encoding="utf-8") as fva:
    for i,line in enumerate(fin):
        (fva if i in val_idx else ftr).write(line)

print(f"[SPLIT] total={n} train={n-val_n} val={val_n}")
print("train->", tr)
print("val  ->", va)
PY

echo "[3/3] Train COMPUTE..."
BASE="Qwen/Qwen2.5-7B-Instruct"
OUT_DIR="checkpoints/qwen-cipherchat-compute-v2"

python -u src/model/train_lora_compute_v2.py \
  --base "$BASE" \
  --train_path "$TRAIN_OUT" \
  --val_path "$VAL_OUT" \
  --out_dir "$OUT_DIR" \
  --max_len 1024 \
  --lr 2e-4 \
  --epochs 1 \
  --bsz 1 \
  --grad_accum 8 \
  --save_steps 500 \
  --log_steps 20

echo "Done: $(date)"
