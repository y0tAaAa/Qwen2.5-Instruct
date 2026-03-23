#!/bin/bash
#SBATCH --job-name=cypher2_train          # название задачи
#SBATCH --partition=dgx                   # очередь (partition)
#SBATCH --output=logs/train_%j.out        # stdout лог (%j = job id)
#SBATCH --error=logs/train_%j.err         # stderr лог
#SBATCH --cpus-per-task=10                # число CPU
#SBATCH --gres=shard:3                    # число GPU shard-ов (подправь при нужде)
#SBATCH --time=48:00:00                   # лимит времени (часы:минуты:секунды)

## уведомления на почту (по желанию)
## #SBATCH --mail-type=ALL
## #SBATCH --mail-user=tvuj.mail@example.com

# --- подготовка окружения ---
mkdir -p logs

# если используешь conda:
source ~/.bashrc
conda activate cypher    # или название твоего env

# переход в папку проекта
cd ~/cipherchat

# при необходимости можно один раз сгенерить данные:
# bash scripts/prepare_data.sh

# --- запуск обучения ---
python3 src/model/train_lora.py --config config/train_lora_cipher.json

