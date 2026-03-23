# scripts/test_cypher_project.py
# -*- coding: utf-8 -*-

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SYSTEM = "Respond ONLY with the final string, no explanations."

def build_prompt(tokenizer, user_text: str) -> str:
    # как в model card: system+user, через chat template
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # fallback если вдруг нет шаблона
    return f"{SYSTEM}\n\nUser: {user_text}\nAssistant:"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/llama-3.1-8b-bnb-4bit")
    ap.add_argument("--lora", default="y0ta/Cypher_project")
    ap.add_argument("--text", required=True, help="formalized prompt line")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,   # base именно bnb-4bit
    )
    model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    prompt = build_prompt(tok, args.text)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            pad_token_id=tok.eos_token_id,
        )

    decoded = tok.decode(out[0], skip_special_tokens=True)
    # модель должна вернуть только строку; берём последнюю строку
    answer = decoded.split("\n")[-1].strip()
    print(answer)

if __name__ == "__main__":
    main()

