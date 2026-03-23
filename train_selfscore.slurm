#!/bin/bash
#SBATCH --job-name=selfscore_ft
#SBATCH --partition=dgx
#SBATCH --output=logs/train_selfscore_%j.out
#SBATCH --error=logs/train_selfscore_%j.err
#SBATCH --cpus-per-task=10
#SBATCH --gres=shard:3
#SBATCH --time=24:00:00

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher

cd ~/cipherchat

python train_selfscore.py \
  --train_file data/processed/train_selfscore.jsonl \
  --output_dir checkpoints/qwen-cipher-lora-selfscore \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path checkpoints/qwen-cipher-lora-finetuned \
  --num_epochs 1 \
  --batch_size 1 \
  --grad_accum 8 \
  --learning_rate 2e-4 \
  --max_length 2048 \
  --logging_steps 20 \
  --save_steps 1000

