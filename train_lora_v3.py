"""
train_lora_v3.py
-----------------
V3 обучение — восстановление после деградации V2.

Стратегия:
  - Продолжаем от V2 адаптера (больше обучен, плохо форматирует)
  - LR = 1e-4  (между V1=2e-4 и V2=5e-5 — достаточно чтобы восстановить формат)
  - 2 эпохи    (не перегреваем)
  - Датасет v3 (Vigenere↑, UK чуть ↓, JSON-reinforce 15%)

Запуск:
  python train_lora_v3.py \
      --resume_from_adapter checkpoints/cipherchat-7b-v2-20260309_173304/final_adapter \
      --train_file data/train_v3.json \
      --val_file   data/val_v3.json \
      --output_dir checkpoints/cipherchat-7b-v3
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
    logger = logging.getLogger("cipherchat_v3")
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
    return logger


# ─────────────────────────────────────────────
# COMPLETION-ONLY COLLATOR
# ─────────────────────────────────────────────

class CompletionOnlyCollator:
    RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.response_ids = tokenizer.encode(
            self.RESPONSE_TEMPLATE, add_special_tokens=False)
        self.ignore = -100

    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        for i in range(len(labels)):
            ids = batch["input_ids"][i].tolist()
            pos = self._last_template_pos(ids)
            if pos is not None:
                labels[i, : pos + len(self.response_ids)] = self.ignore
            else:
                labels[i, :] = self.ignore
            labels[i, batch["attention_mask"][i] == 0] = self.ignore
        batch["labels"] = labels
        return batch

    def _last_template_pos(self, input_ids):
        tlen = len(self.response_ids)
        last = None
        for j in range(len(input_ids) - tlen + 1):
            if input_ids[j: j + tlen] == self.response_ids:
                last = j
        return last


# ─────────────────────────────────────────────
# ДАТАСЕТ
# ─────────────────────────────────────────────

def load_dataset_from_json(path, tokenizer, max_seq_length, logger):
    logger.info(f"Loading dataset: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"  Raw examples: {len(data)}")

    formatted = []
    for item in data:
        text = tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False)
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)

    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, max_length=max_seq_length)

    dataset = dataset.map(tokenize_fn, remove_columns=["text"])

    lengths = [len(ex["input_ids"]) for ex in dataset]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    over = sum(1 for l in lengths if l >= max_seq_length)
    logger.info(f"  Tokenized: {len(dataset)} | avg={avg_len:.0f} | "
                f"max={max_len} | truncated={over} ({100*over/len(dataset):.1f}%)")
    return dataset


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",        default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train_file",  default="data/train_v3.json")
    ap.add_argument("--val_file",    default="data/val_v3.json")
    ap.add_argument("--output_dir",  default=None)

    # LoRA
    ap.add_argument("--lora_r",       type=int,   default=64)
    ap.add_argument("--lora_alpha",   type=int,   default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Обучение — ключевые изменения vs v2
    ap.add_argument("--epochs",      type=int,   default=2)      # 3→2
    ap.add_argument("--batch_size",  type=int,   default=2)
    ap.add_argument("--grad_accum",  type=int,   default=8)
    ap.add_argument("--lr",          type=float, default=1e-4)   # 5e-5→1e-4
    ap.add_argument("--max_seq_len", type=int,   default=1536)
    ap.add_argument("--max_train",   type=int,   default=12000)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--resume_from_adapter", default=None,
                    help="Path to V2 adapter (обязательно для v3)")
    args = ap.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./checkpoints/cipherchat-7b-v3-{ts}"

    logger = setup_logging(args.output_dir)

    logger.info("=" * 65)
    logger.info("CIPHERCHAT V3 TRAINING")
    logger.info("=" * 65)
    logger.info(f"Base model:      {args.base}")
    logger.info(f"Resume from V2:  {args.resume_from_adapter or 'NONE — set --resume_from_adapter!'}")
    logger.info(f"Output dir:      {args.output_dir}")
    logger.info(f"LR:              {args.lr}  (↑ vs V2=5e-5, восстанавливаем JSON формат)")
    logger.info(f"Epochs:          {args.epochs}  (↓ vs V2=3, не перегреваем)")
    logger.info(f"Eff. batch:      {args.batch_size} x {args.grad_accum} = "
                f"{args.batch_size * args.grad_accum}")
    logger.info("=" * 65)

    if not args.resume_from_adapter:
        logger.warning("WARNING: --resume_from_adapter not set! Training from scratch.")

    # ── 1. ТОКЕНИЗАТОР ───────────────────────────────────────
    tokenizer_path = args.resume_from_adapter or args.base
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 2. ДАТАСЕТЫ ──────────────────────────────────────────
    train_dataset = load_dataset_from_json(
        args.train_file, tokenizer, args.max_seq_len, logger)
    val_dataset = load_dataset_from_json(
        args.val_file, tokenizer, args.max_seq_len, logger)

    if args.max_train > 0 and len(train_dataset) > args.max_train:
        train_dataset = train_dataset.select(range(args.max_train))

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ── 3. МОДЕЛЬ (QLoRA) ────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # ── 4. LoRA — продолжаем с V2 ────────────────────────────
    if args.resume_from_adapter and os.path.isdir(args.resume_from_adapter):
        logger.info(f"Loading V2 adapter: {args.resume_from_adapter}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, args.resume_from_adapter, is_trainable=True)
        logger.info("V2 adapter loaded → trainable")
    else:
        logger.warning("No adapter found — creating new LoRA from scratch")
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
        )
        model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    # ── 6. PRECISION ──────────────────────────────────────────
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    logger.info(f"Precision: bf16={use_bf16} | fp16={use_fp16}")

    # ── 5. COLLATOR ──────────────────────────────────────────
    collator = CompletionOnlyCollator(tokenizer)

    # ── 6. TRAINING ARGS ─────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=0.3,

        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=20,
        logging_first_step=True,
        report_to="none",

        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # ── 7. TRAINER ───────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    # ── 8. ОБУЧЕНИЕ ──────────────────────────────────────────
    logger.info("Starting V3 training...")
    train_result = trainer.train()

    logger.info(f"Runtime:    {train_result.metrics.get('train_runtime', 0):.1f}s")
    logger.info(f"Final loss: {train_result.metrics.get('train_loss', 0):.6f}")

    with open(os.path.join(args.output_dir, "train_metrics.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    # ── 9. СОХРАНЕНИЕ ────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"✅ Adapter saved -> {final_path}")


if __name__ == "__main__":
    main()
