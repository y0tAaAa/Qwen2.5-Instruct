#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import random
import argparse
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from peft import PeftModel

# Спец-токены для самооценки
GRADE_TOKENS = ["<grade_1>", "<grade_2>", "<grade_3>", "<grade_4>", "<grade_5>"]


# ==================== CONFIG ====================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Читает JSON-конфиг с гиперпараметрами тренировки.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file '{config_path}' not found. "
            f"Создай его в config/train_lora_cipher.json или укажи путь через --config."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    defaults = {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "lora_init": "y0ta/qwen-cipher-lora",
        "train_path": "data/processed/train_cipher_train.jsonl",
        "val_path": "data/processed/val_cipher.jsonl",
        "output_dir": "checkpoints/qwen-cipher-lora-finetuned",
        "max_length": 2048,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_epochs": 2,
        "learning_rate": 2e-4,
        "logging_steps": 20,
        "save_steps": 500,
        "save_total_limit": 3,
        "eval_every_n_steps": 200,
        "eval_samples": 32,
        "eval_max_new_tokens": 512,
        "seed": 42,
    }

    for k, v in defaults.items():
        cfg.setdefault(k, v)

    return cfg


# ==================== DATASETS ====================

class CipherDataset(Dataset):
    """
    Ожидает jsonl:
    {"instruction": "...", "json": {...}}
    где instruction уже содержит:
    "### Instruction:\\n...\\n\\n### Response (JSON only):\\n"
    """

    def __init__(self, path: str, tokenizer, max_len: int = 2048):
        self.samples: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj["instruction"]
                target_json = json.dumps(obj["json"], ensure_ascii=False)
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


class CipherEvalSet:
    """
    Для онлайн-оценки: храним промпт и таргет-json (dict).
    """

    def __init__(self, path: str):
        self.data: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.data.append(
                    {"prompt": obj["instruction"], "target": obj["json"]}
                )
        if not self.data:
            raise ValueError(f"No eval data in {path}")
        print(f"[EVAL] Loaded {len(self.data)} eval samples from {path}")

    def sample(self, n: int) -> List[Dict[str, Any]]:
        n = min(n, len(self.data))
        return random.sample(self.data, n)


# ==================== SCORING & GENERATION ====================

def parse_self_score_token(v: Any) -> Optional[int]:
    """
    Превращаем "<grade_3>" → 3, если формат правильный.
    """
    if isinstance(v, str) and v.startswith("<grade_") and v.endswith(">"):
        inner = v[len("<grade_"):-1]
        try:
            x = int(inner)
            if 1 <= x <= 5:
                return x
        except ValueError:
            return None
    return None


def grade_answer(target: Dict[str, Any], pred_text: str):
    """
    Возвращаем:
      real_grade (1–5) — насколько ответ совпадает с эталоном
      pred_self (1–5 или None) — что модель сама про себя написала в self_score
    """
    try:
        pred = json.loads(pred_text)
    except Exception:
        return 1, None

    fields = ["cipher_type", "key", "cipher_text", "plaintext"]
    matches = 0
    for k in fields:
        if k in target and k in pred and str(target[k]) == str(pred[k]):
            matches += 1

    if matches == 4:
        real = 5
    elif matches == 3:
        real = 4
    elif matches == 2:
        real = 3
    elif matches == 1:
        real = 2
    else:
        real = 1

    pred_self = None
    if "self_score" in pred:
        pred_self = parse_self_score_token(pred["self_score"])

    return real, pred_self


def generate_json(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = out[0, enc["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return text.strip()


# ==================== CALLBACK ====================

class OnlineEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        eval_set: CipherEvalSet,
        every_n_steps: int = 200,
        n_samples: int = 32,
        max_new_tokens: int = 512,
    ):
        self.tokenizer = tokenizer
        self.eval_set = eval_set
        self.every_n_steps = every_n_steps
        self.n_samples = n_samples
        self.max_new_tokens = max_new_tokens

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0:
            return
        if state.global_step % self.every_n_steps != 0:
            return

        model = kwargs["model"]
        model.eval()

        samples = self.eval_set.sample(self.n_samples)
        grades: List[int] = []
        self_scores: List[int] = []

        for ex in samples:
            prompt = ex["prompt"]
            target = ex["target"]
            pred_text = generate_json(
                model, self.tokenizer, prompt, max_new_tokens=self.max_new_tokens
            )
            real, self_s = grade_answer(target, pred_text)
            grades.append(real)
            if self_s is not None:
                self_scores.append(self_s)

        if not grades:
            return

        avg_grade = sum(grades) / len(grades)
        perfect = sum(1 for g in grades if g == 5) / len(grades) * 100.0
        avg_self = sum(self_scores) / len(self_scores) if self_scores else 0.0

        print(
            f"\n[ONLINE EVAL] step {state.global_step}: "
            f"avg_grade={avg_grade:.2f} (1–5), perfect_5={perfect:.1f}% "
            f"on {len(grades)} samples, avg_self_score={avg_self:.2f}\n"
        )


# ==================== MAIN TRAIN ====================

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune y0ta/qwen-cipher-lora with reasoning + self_score using JSON config."
    )
    p.add_argument(
        "--config",
        type=str,
        default="config/train_lora_cipher.json",
        help="Path to JSON config with training hyperparameters.",
    )
    return p.parse_args()


def main():
    cli_args = parse_args()
    cfg = load_config(cli_args.config)

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    BASE_MODEL = cfg["base_model"]
    LORA_INIT = cfg["lora_init"]

    train_path = cfg["train_path"]
    val_path = cfg["val_path"]
    output_dir = cfg["output_dir"]

    max_len = cfg["max_length"]
    batch_size = cfg["batch_size"]
    grad_accum = cfg["gradient_accumulation_steps"]
    num_epochs = cfg["num_epochs"]
    lr = cfg["learning_rate"]
    logging_steps = cfg["logging_steps"]
    save_steps = cfg["save_steps"]
    save_total_limit = cfg["save_total_limit"]
    eval_every_n_steps = cfg["eval_every_n_steps"]
    eval_samples = cfg["eval_samples"]
    eval_max_new_tokens = cfg["eval_max_new_tokens"]

    print("[INFO] Config:", cfg)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Добавляем спец-токены для self_score
    existing_additional = tokenizer.additional_special_tokens or []
    to_add = [t for t in GRADE_TOKENS if t not in existing_additional]
    if to_add:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing_additional + to_add}
        )
        print(f"[INFO] Added grade tokens: {to_add}")
    else:
        print("[INFO] Grade tokens already present in tokenizer.")

    print("[INFO] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"[INFO] Loading existing LoRA adapter from {LORA_INIT}...")
    # ВАЖНО: is_trainable=True, иначе градиентов не будет
    model = PeftModel.from_pretrained(
        base_model,
        LORA_INIT,
        is_trainable=True,
    )

    # Ресайз эмбеддингов под новые токены
    model.resize_token_embeddings(len(tokenizer))

    model.print_trainable_parameters()

    print("[INFO] Loading datasets...")
    train_ds = CipherDataset(train_path, tokenizer, max_len=max_len)
    eval_set = CipherEvalSet(val_path)

    args_tr = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        bf16=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )

    online_cb = OnlineEvalCallback(
        tokenizer=tokenizer,
        eval_set=eval_set,
        every_n_steps=eval_every_n_steps,
        n_samples=eval_samples,
        max_new_tokens=eval_max_new_tokens,
    )
    trainer.add_callback(online_cb)

    print("[INFO] Start training with online eval + self_score...")
    trainer.train()

    print("[INFO] Saving model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

