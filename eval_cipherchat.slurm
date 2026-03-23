#!/bin/bash
#SBATCH --job-name=eval_cipherchat
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

echo "Starting evaluation job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher

cd ~/cipherchat

export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

python src/model/eval_model.py \
  --base Qwen/Qwen2.5-7B-Instruct \
  --lora_path checkpoints/qwen-cipherchat-variants \
  --eval_path data/processed/val_cipher.jsonl \
  --max_samples 1000 \
  --max_new_tokens 256 \
  --log_every 50

echo "Finished at $(date)"
#!/bin/bash
#SBATCH --job-name=eval_cipherchat
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

echo "Starting evaluation job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher

cd ~/cipherchat

export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

python src/model/eval_model.py \
  --base Qwen/Qwen2.5-7B-Instruct \
  --lora_path checkpoints/qwen-cipherchat-variants \
  --eval_path data/processed/val_cipher.jsonl \
  --max_samples 1000 \
  --max_new_tokens 256 \
  --log_every 50

echo "Finished at $(date)"
#!/bin/bash
#SBATCH --job-name=eval_cipherchat
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

echo "Starting evaluation job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cypher

cd ~/cipherchat

export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

python src/model/eval_model.py \
  --base Qwen/Qwen2.5-7B-Instruct \
  --lora_path checkpoints/qwen-cipherchat-variants \
  --eval_path data/processed/val_cipher.jsonl \
  --max_samples 1000 \
  --max_new_tokens 256 \
  --log_every 50

echo "Finished at $(date)"

