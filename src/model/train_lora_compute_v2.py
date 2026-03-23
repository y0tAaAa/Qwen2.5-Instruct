import argparse
import json
import os
import logging
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("train_compute_v2")

GRADE_TOKENS = ["<grade_1>", "<grade_2>", "<grade_3>", "<grade_4>", "<grade_5>"]


class CompletionOnlyCollator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tok.pad_token_id

        input_ids, attn, labels = [], [], []
        for x in features:
            ids = x["input_ids"]
            am = x["attention_mask"]
            lab = x["labels"]

            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append(am + [0] * pad)
            labels.append(lab + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_example(tok, instruction: str, answer_obj: dict, max_len: int):
    prompt = instruction
    completion = json.dumps(answer_obj, ensure_ascii=False) + tok.eos_token

    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    comp_ids = tok(completion, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + comp_ids
    if len(input_ids) > max_len:
        overflow = len(input_ids) - max_len
        prompt_ids = prompt_ids[overflow:]
        input_ids = (prompt_ids + comp_ids)[-max_len:]

    labels = [-100] * len(prompt_ids) + comp_ids
    labels = labels[-len(input_ids):]

    attn = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def _load_model_compat(base: str):
    # У вас torch_dtype deprecated -> пробуем dtype, если не поддерживается — fallback
    try:
        return AutoModelForCausalLM.from_pretrained(
            base,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )


def _enable_lora_checkpointing_grads(model: torch.nn.Module):
    # Важно для LoRA+gradient checkpointing:
    # нужно чтобы входные эмбеддинги требовали grad, иначе loss без grad_fn.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def _make_emb_require_grad(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)
        emb = model.get_input_embeddings()
        emb.register_forward_hook(_make_emb_require_grad)

    model.config.use_cache = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train_path", default="data/processed/train_compute.jsonl")
    ap.add_argument("--val_path", default=None)
    ap.add_argument("--out_dir", default="checkpoints/qwen-cipherchat-compute-v2")
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

    existing = tok.additional_special_tokens or []
    to_add = [t for t in GRADE_TOKENS if t not in existing]
    if to_add:
        tok.add_special_tokens({"additional_special_tokens": existing + to_add})
        log.info(f"Added grade tokens: {to_add}")

    model = _load_model_compat(args.base)
    model.resize_token_embeddings(len(tok))
    model.config.use_cache = False

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # <<< FIX: LoRA + checkpointing gradients >>>
    _enable_lora_checkpointing_grads(model)

    files = {"train": args.train_path}
    if args.val_path:
        files["validation"] = args.val_path

    ds = load_dataset("json", data_files=files)
    ds = ds.map(lambda ex: build_example(tok, ex["instruction"], ex["json"], args.max_len),
                remove_columns=ds["train"].column_names)

    collator = CompletionOnlyCollator(tok)

    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,

        # <<< FIX: у вас именно eval_strategy >>>
        eval_strategy=("steps" if args.val_path else "no"),
        eval_steps=args.save_steps,

        save_total_limit=3,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=(ds["validation"] if args.val_path else None),
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    log.info(f"✅ saved → {args.out_dir}")


if __name__ == "__main__":
    main()
