"""
train_lora.py
-------------
Обучение CipherChat с правильными подходами:

  ✅ QLoRA (4-bit NF4 + bfloat16) — экономия VRAM, работает на одном GPU
  ✅ LoRA r=64 / alpha=128 — достаточная ёмкость для алгоритмических паттернов
  ✅ Completion-only loss — loss только по ответу ассистента
  ✅ Только верифицированные примеры (round-trip OK из датасета)
  ✅ Единый пайплайн для всех шифров и языков
  ✅ paged_adamw_8bit + cosine scheduler + warmup + grad clipping
  ✅ load_best_model_at_end по eval_loss

Запуск:
  python train_lora.py \
      --base Qwen/Qwen2.5-7B-Instruct \
      --train_file train.json \
      --val_file val.json \
      --output_dir ./checkpoints/cipherchat-7b
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")

    logger = logging.getLogger("cipherchat_train")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_path}")
    return logger


# ─────────────────────────────────────────────
# COMPLETION-ONLY COLLATOR
# ─────────────────────────────────────────────
# Критически важно: loss считается ТОЛЬКО по ответу ассистента.
# Всё до <|im_start|>assistant\n маскируется меткой -100.

class CompletionOnlyCollator:
    """
    Маскирует всё кроме assistant-части в labels.
    Работает с любым chat template, который использует
    маркер <|im_start|>assistant\n (Qwen2.5).
    """

    RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.response_ids = tokenizer.encode(
            self.RESPONSE_TEMPLATE, add_special_tokens=False
        )
        self.ignore = -100

    def __call__(self, examples):
        # Стандартный padding от tokenizer
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            padding=True,
        )
        labels = batch["input_ids"].clone()

        for i in range(len(labels)):
            ids = batch["input_ids"][i].tolist()
            pos = self._last_template_pos(ids)

            if pos is not None:
                # Маскируем всё ДО конца шаблона (включая сам шаблон)
                labels[i, : pos + len(self.response_ids)] = self.ignore
            else:
                # Шаблон не найден — маскируем весь пример (не обучаемся на нём)
                labels[i, :] = self.ignore

            # Маскируем padding
            labels[i, batch["attention_mask"][i] == 0] = self.ignore

        batch["labels"] = labels
        return batch

    def _last_template_pos(self, input_ids: list) -> int | None:
        """Ищет последнее вхождение response_ids в input_ids."""
        tlen = len(self.response_ids)
        last = None
        for j in range(len(input_ids) - tlen + 1):
            if input_ids[j: j + tlen] == self.response_ids:
                last = j
        return last


# ─────────────────────────────────────────────
# ЗАГРУЗКА И ФОРМАТИРОВАНИЕ ДАТАСЕТА
# ─────────────────────────────────────────────

def load_dataset_from_json(
    path: str,
    tokenizer,
    max_seq_length: int,
    logger: logging.Logger,
) -> Dataset:
    """
    Читает JSON файл с полем "messages" (system/user/assistant),
    применяет chat template и токенизирует.
    """
    logger.info(f"Loading dataset: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"  Raw examples: {len(data)}")

    formatted = []
    skipped = 0
    for item in data:
        messages = item["messages"]
        # Применяем chat template (формат Qwen2.5)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)

    # Логируем пример для проверки формата
    logger.debug(f"  Example (first 800 chars):\n{dataset[0]['text'][:800]}")

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_length,
        )

    dataset = dataset.map(tokenize_fn, remove_columns=["text"])

    # Статистика длин
    lengths = [len(ex["input_ids"]) for ex in dataset]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    over_limit = sum(1 for l in lengths if l >= max_seq_length)
    logger.info(f"  Tokenized: {len(dataset)} samples | "
                f"avg_len={avg_len:.0f} | max_len={max_len} | "
                f"truncated={over_limit} ({100*over_limit/len(dataset):.1f}%)")

    return dataset


# ─────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train CipherChat with QLoRA")
    ap.add_argument("--base",        default="Qwen/Qwen2.5-7B-Instruct",
                    help="Base model HuggingFace ID")
    ap.add_argument("--train_file",  default="train.json")
    ap.add_argument("--val_file",    default="val.json")
    ap.add_argument("--output_dir",  default=None,
                    help="Output directory (auto-named if not set)")

    # LoRA
    ap.add_argument("--lora_r",      type=int,   default=64)
    ap.add_argument("--lora_alpha",  type=int,   default=128)
    ap.add_argument("--lora_dropout",type=float, default=0.05)

    # Обучение
    ap.add_argument("--epochs",      type=int,   default=2)
    ap.add_argument("--batch_size",  type=int,   default=2)
    ap.add_argument("--grad_accum",  type=int,   default=8,
                    help="Effective batch = batch_size * grad_accum")
    ap.add_argument("--lr",          type=float, default=2e-4)
    ap.add_argument("--max_seq_len", type=int,   default=1536)
    ap.add_argument("--max_train",   type=int,   default=12000,
                    help="Max train samples (0=all)")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--eval_steps",  type=int,   default=100)
    ap.add_argument("--save_steps",  type=int,   default=200)
    ap.add_argument("--resume_from_adapter", default=None,
                    help="Path to existing LoRA adapter to continue training from")
    ap.add_argument("--resume_from_checkpoint", default=None,
                    help="Path to HF Trainer checkpoint-XXXX folder to resume from")
    args = ap.parse_args()

    # Автоимя директории
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.base.split("/")[-1]
        args.output_dir = f"./checkpoints/cipherchat-{model_short}-{ts}"

    logger = setup_logging(args.output_dir)

    logger.info("=" * 65)
    logger.info("CIPHERCHAT TRAINING START")
    logger.info("=" * 65)
    logger.info(f"Base model:      {args.base}")
    logger.info(f"Resume adapter:  {args.resume_from_adapter or 'None (train from scratch)'}")
    logger.info(f"Output dir:      {args.output_dir}")
    logger.info(f"LoRA r/alpha:    {args.lora_r}/{args.lora_alpha}")
    logger.info(f"Epochs:          {args.epochs}")
    logger.info(f"Effective batch: {args.batch_size} × {args.grad_accum} = "
                f"{args.batch_size * args.grad_accum}")
    logger.info(f"Learning rate:   {args.lr}")
    logger.info(f"Max seq length:  {args.max_seq_len}")
    logger.info("=" * 65)

    # ──────────────────────────────────────────
    # 1. ТОКЕНИЗАТОР
    # ──────────────────────────────────────────
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("  pad_token set to eos_token")

    # ──────────────────────────────────────────
    # 2. ДАТАСЕТЫ
    # ──────────────────────────────────────────
    logger.info("Loading datasets...")
    train_dataset = load_dataset_from_json(
        args.train_file, tokenizer, args.max_seq_len, logger
    )
    val_dataset = load_dataset_from_json(
        args.val_file, tokenizer, args.max_seq_len, logger
    )

    # Ограничение размера тренировочного сета
    if args.max_train > 0 and len(train_dataset) > args.max_train:
        train_dataset = train_dataset.select(range(args.max_train))
        logger.info(f"  Train limited to {args.max_train} samples")

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples:   {len(val_dataset)}")

    # ──────────────────────────────────────────
    # 3. МОДЕЛЬ С QLoRA
    # ──────────────────────────────────────────
    logger.info("Loading model with 4-bit quantization (QLoRA)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 — лучше для весов с нормальным распределением
        bnb_4bit_compute_dtype=torch.bfloat16,  # Вычисления в bfloat16
        bnb_4bit_use_double_quant=True,      # Двойная квантизация: ещё -0.4 бит/параметр
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    logger.info("  Base model loaded.")

    # Подготовка к kbit-обучению (нормализация весов, freeze base params)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # ──────────────────────────────────────────
    # 4. LoRA АДАПТЕРЫ
    # ──────────────────────────────────────────
    # r=64 / alpha=128: высокая ёмкость нужна для алгоритмических паттернов
    # (точные арифметические вычисления над символами)
    if args.resume_from_adapter and os.path.isdir(args.resume_from_adapter):
        # Продолжаем с существующего адаптера — загружаем веса и делаем trainable
        logger.info(f"  Loading existing adapter: {args.resume_from_adapter}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            args.resume_from_adapter,
            is_trainable=True,
        )
        logger.info("  Adapter loaded and set to trainable.")
    else:
        # Новый адаптер с нуля
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        logger.info("  New LoRA adapter created.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable params: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.4f}%)")

    # ──────────────────────────────────────────
    # 5. COLLATOR (completion-only loss)
    # ──────────────────────────────────────────
    collator = CompletionOnlyCollator(tokenizer=tokenizer)
    logger.info(f"  CompletionOnlyCollator: template = "
                f"{repr(CompletionOnlyCollator.RESPONSE_TEMPLATE)}")

    # ──────────────────────────────────────────
    # 6. АРГУМЕНТЫ ОБУЧЕНИЯ
    # ──────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,

        # Batch
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # Оптимизатор
        optim="paged_adamw_8bit",      # Экономит VRAM за счёт paging
        learning_rate=args.lr,
        lr_scheduler_type="cosine",    # Плавное убывание LR
        warmup_ratio=0.05,             # 5% шагов — warm-up
        weight_decay=0.01,             # L2 регуляризация
        max_grad_norm=0.3,             # Клиппинг градиентов (важно для QLoRA)

        # Точность
        bf16=True,

        # Gradient checkpointing (экономия памяти, +20% время)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Оценка и сохранение
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Логирование
        logging_steps=20,
        logging_first_step=True,
        report_to="none",

        # Воспроизводимость
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # ──────────────────────────────────────────
    # 7. TRAINER
    # ──────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    # ──────────────────────────────────────────
    # 8. ОБУЧЕНИЕ
    # ──────────────────────────────────────────
    logger.info("🚀 Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Training complete.")
    logger.info(f"  Runtime:        {train_result.metrics.get('train_runtime', 0):.1f}s")
    logger.info(f"  Samples/sec:    {train_result.metrics.get('train_samples_per_second', 0):.2f}")
    logger.info(f"  Final loss:     {train_result.metrics.get('train_loss', 0):.6f}")

    # Сохранение метрик
    metrics_path = os.path.join(args.output_dir, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    # ──────────────────────────────────────────
    # 9. СОХРАНЕНИЕ АДАПТЕРА
    # ──────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    logger.info(f"✅ Adapter saved → {final_path}")
    logger.info("=" * 65)
    logger.info("DONE")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
