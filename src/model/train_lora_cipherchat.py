import argparse
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

GRADE_TOKENS = ["<grade_1>", "<grade_2>", "<grade_3>", "<grade_4>", "<grade_5>"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train_path", default="data/processed/train_compute.jsonl")
    ap.add_argument("--val_path", default="data/processed/val_cipher.jsonl")
    ap.add_argument("--out_dir", default="checkpoints/qwen-cipherchat-compute")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--log_steps", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Добавляем grade-токены
    existing = tok.additional_special_tokens or []
    to_add = [t for t in GRADE_TOKENS if t not in existing]
    if to_add:
        tok.add_special_tokens({"additional_special_tokens": existing + to_add})
        print(f"[TRAIN] Added {len(to_add)} grade tokens")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.resize_token_embeddings(len(tok))

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": args.train_path, "validation": args.val_path})
    ds = ds.map(lambda ex: {"text": ex["instruction"] + "\n" + json.dumps(ex["json"], ensure_ascii=False)}, 
                remove_columns=ds["train"].column_names)

    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len, padding=False), 
                batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    train_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)

    print(f"Training finished. Model saved to {args.out_dir}")

if __name__ == "__main__":
    main()
