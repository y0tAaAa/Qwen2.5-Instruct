#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import argparse
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import PeftModel


class SelfScoreDataset(Dataset):
    """
    Датасет для дообучения self_score.

    Ожидает jsonl вида:
    {
      "instruction": "...",
      "json": {
         ...,
         "self_score": "<grade_1|...|grade_5>"
      }
    }

    Мы просто склеиваем prompt + target_json и учим next-token LM.
    """

    def __init__(self, path: str, tokenizer, max_len: int = 2048):
        self.samples: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        if not os.path.exists(path):
            raise FileNotFoundError(f"Train file '{path}' not found")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                prompt = data["instruction"]
                target_json = json.dumps(data["json"], ensure_ascii=False)
                full_text = prompt + target_json

                enc = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=self.max_len,
                    add_special_tokens=True,
                )
                ids = enc["input_ids"]
                self.samples.append({"input_ids": ids})

        print(f"[DATA] Loaded {len(self.samples)} samples from {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]["input_ids"]
        t = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": t, "labels": t.clone()}


def train_selfscore_model():
    parser = argparse.ArgumentParser(description="Further fine-tune LoRA on self_score supervision.")
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed/train_selfscore.jsonl",
        help="Training jsonl with instruction + json (incl. self_score).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/qwen-cipher-lora-selfscore",
        help="Where to store the new LoRA checkpoint.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF id базовой модели (та же, что и при основном обучении).",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Путь к уже дообученному LoRA (например, checkpoints/qwen-cipher-lora-finetuned).",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Сколько эпох дообучать self_score.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="per_device_train_batch_size.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="gradient_accumulation_steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate для дообучения LoRA.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Максимальная длина последовательности.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # 1) Токенайзер берём из LoRA-чекпоинта (там уже есть <grade_*> и т.п.)
    print(f"[INFO] Loading tokenizer from {args.lora_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.lora_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    print(f"[INFO] Tokenizer vocab size = {vocab_size}")

    # 2) Базовая модель — ОРИГИНАЛЬНЫЙ Qwen, а не lora_path
    print(f"[INFO] Loading base model {args.base_model} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3) Ресайз эмбеддингов базы под размер токенайзера LoRA
    print(f"[INFO] Resizing token embeddings to {vocab_size} ...")
    base_model.resize_token_embeddings(vocab_size)

    # 4) Натягиваем существующий LoRA-адаптер
    print(f"[INFO] Loading LoRA adapter from {args.lora_path} (trainable) ...")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_path,
        is_trainable=True,
    )

    model.print_trainable_parameters()

    # 5) Датасет с self_score
    print("[INFO] Loading SelfScoreDataset ...")
    train_dataset = SelfScoreDataset(
        path=args.train_file,
        tokenizer=tokenizer,
        max_len=args.max_length,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "train_selfscore_config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print("[INFO] Start self_score fine-tuning ...")
    trainer.train()

    print("[INFO] Saving LoRA + tokenizer ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[INFO] Done.")


if __name__ == "__main__":
    train_selfscore_model()

