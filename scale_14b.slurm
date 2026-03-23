#!/bin/bash
#SBATCH --job-name=instruct_14b
#SBATCH --partition=dgx
#SBATCH --output=logs/scale_instruct_14b_%j.out
#SBATCH --error=logs/scale_instruct_14b_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs checkpoints

echo "========================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Host     : $(hostname)"
echo "Start    : $(date)"
echo "Size     : 14B  |  Time limit: 24h"
echo "========================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher

export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false

BASE="Qwen/Qwen2.5-14B-Instruct"
TRAIN_FILE="$HOME/cipherchat/data/train.json"
VAL_FILE="$HOME/cipherchat/data/val.json"
OUTPUT_DIR="$HOME/cipherchat/checkpoints/cipher-instruct-14B-$(date +%Y%m%d_%H%M%S)"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

python -u $HOME/cipherchat/train_lora.py \
    --base         "$BASE"         \
    --train_file   "$TRAIN_FILE"   \
    --val_file     "$VAL_FILE"     \
    --output_dir   "$OUTPUT_DIR"   \
    --lora_r       64              \
    --lora_alpha   128             \
    --lora_dropout 0.05            \
    --epochs       2               \
    --batch_size   2               \
    --grad_accum   8               \
    --lr           2e-4            \
    --max_seq_len  1536            \
    --max_train    12000           \
    --seed         42

echo "========================================================"
echo "Done: $(date)"
echo "Adapter: $OUTPUT_DIR/final_adapter"
echo "========================================================"
