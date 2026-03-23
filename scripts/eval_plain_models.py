# scripts/eval_plain_models.py
# -*- coding: utf-8 -*-

import argparse
import json
import re
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


_ws_re = re.compile(r"\s+")

def norm(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\u200b", "")
    s = _ws_re.sub(" ", s)
    return s

def cer(ref: str, hyp: str) -> float:
    ref = ref or ""
    hyp = hyp or ""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m] / max(1, n)

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def extract_fields(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # твой основной формат: {"instruction":..., "json": {...}}
    if "json" in obj and isinstance(obj["json"], dict):
        j = obj["json"]
        mode = (j.get("mode") or obj.get("mode") or "").lower().strip()
        out = {
            "mode": mode,  # encrypt/decrypt
            "cipher_type": j.get("cipher_type") or j.get("cipher") or j.get("type"),
            "key": j.get("key"),
            "cipher_text": j.get("cipher_text"),
            "plaintext": j.get("plaintext"),
            "lang": obj.get("lang") or j.get("lang") or "en",
        }
        if out["cipher_type"] and out["key"] is not None and out["cipher_text"] is not None and out["plaintext"] is not None:
            return out

    # fallback flat
    if "cipher_text" in obj and "plaintext" in obj:
        mode = (obj.get("mode") or "").lower().strip()
        out = {
            "mode": mode,
            "cipher_type": obj.get("cipher_type") or obj.get("cipher") or obj.get("type"),
            "key": obj.get("key"),
            "cipher_text": obj.get("cipher_text"),
            "plaintext": obj.get("plaintext"),
            "lang": obj.get("lang") or "en",
        }
        if out["cipher_type"] and out["key"] is not None:
            return out

    return None

def build_prompt(fields: Dict[str, Any], style: str) -> str:
    """
    Генерим "плоский" prompt, чтобы text-модели могли хоть как-то реагировать.
    ВАЖНО: под encrypt даём PLAINTEXT, под decrypt даём CIPHERTEXT.
    """
    mode = fields["mode"]
    ctype = str(fields["cipher_type"])
    key = str(fields["key"])
    lang = fields.get("lang", "en")

    if style == "formal":
        if mode == "decrypt":
            return (
                f"DECRYPT: CIPHERTEXT={fields['cipher_text']}; METHOD={ctype.upper()}; KEY={key}; LANG={lang}; PLAINTEXT="
            )
        else:
            return (
                f"ENCRYPT: PLAINTEXT={fields['plaintext']}; METHOD={ctype.upper()}; KEY={key}; LANG={lang}; CIPHERTEXT="
            )
    else:
        # plain
        if mode == "decrypt":
            return (
                f"LANG={lang}\nCIPHER={ctype}\nKEY={key}\nCIPHERTEXT={fields['cipher_text']}\nPLAINTEXT="
            )
        else:
            return (
                f"LANG={lang}\nCIPHER={ctype}\nKEY={key}\nPLAINTEXT={fields['plaintext']}\nCIPHERTEXT="
            )

def maybe_chat_wrap(tok, user_text: str) -> str:
    # для llama-инструкт шаблонов иногда помогает
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        msgs = [
            {"role": "system", "content": "Output ONLY the final answer string. No explanations."},
            {"role": "user", "content": user_text},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return user_text

@torch.no_grad()
def generate_continuation(model, tok, prompt: str, max_new_tokens: int) -> str:
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    return text.strip()

def load_gpt2(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    model.eval()
    return tok, model

def load_lora(base_id: str, lora_id: str):
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, lora_id)
    model.eval()
    return tok, model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_path", required=True)
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=64)

    ap.add_argument("--model_kind", required=True, choices=["gpt2", "lora"])
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--base_id", default="unsloth/llama-3.1-8b-bnb-4bit")

    ap.add_argument("--prompt_style", choices=["plain", "formal"], default="formal")
    ap.add_argument("--chat_wrap", action="store_true")
    args = ap.parse_args()

    if args.model_kind == "gpt2":
        tok, model = load_gpt2(args.model_id)
    else:
        tok, model = load_lora(args.base_id, args.model_id)

    # counters
    used = 0
    skipped_no_fields = 0
    skipped_bad_mode = 0

    stats = {
        "encrypt": {"n": 0, "em": 0, "cer_sum": 0.0, "empty": 0},
        "decrypt": {"n": 0, "em": 0, "cer_sum": 0.0, "empty": 0},
    }

    for obj in iter_jsonl(args.eval_path):
        fields = extract_fields(obj)
        if fields is None:
            skipped_no_fields += 1
            continue

        mode = fields["mode"]
        if mode not in ("encrypt", "decrypt"):
            skipped_bad_mode += 1
            continue

        prompt = build_prompt(fields, args.prompt_style)
        if args.chat_wrap:
            prompt = maybe_chat_wrap(tok, prompt)

        pred = norm(generate_continuation(model, tok, prompt, args.max_new_tokens))

        # ref зависит от режима
        if mode == "decrypt":
            ref = norm(fields["plaintext"])
        else:
            ref = norm(fields["cipher_text"])

        if pred == "":
            stats[mode]["empty"] += 1

        if pred == ref:
            stats[mode]["em"] += 1

        stats[mode]["cer_sum"] += cer(ref, pred)
        stats[mode]["n"] += 1
        used += 1

        if used % 50 == 0:
            d = stats["decrypt"]
            e = stats["encrypt"]
            def fmt(x): return f"{x:.3f}"
            print(
                f"[eval] used={used} | "
                f"DEC n={d['n']} EM={fmt(d['em']/max(1,d['n']))} CER={fmt(d['cer_sum']/max(1,d['n']))} | "
                f"ENC n={e['n']} EM={fmt(e['em']/max(1,e['n']))} CER={fmt(e['cer_sum']/max(1,e['n']))}"
            )

        if used >= args.max_samples:
            break

    print("\n==== RESULT ====")
    print(f"model_kind: {args.model_kind}")
    print(f"model_id:   {args.model_id}")
    if args.model_kind == "lora":
        print(f"base_id:    {args.base_id}")
    print(f"eval_path:  {args.eval_path}")
    print(f"samples_used(total): {used}")
    print(f"skipped_no_fields: {skipped_no_fields}")
    print(f"skipped_bad_mode:  {skipped_bad_mode}")
    print(f"prompt_style: {args.prompt_style} chat_wrap={args.chat_wrap}")

    for mode in ("decrypt", "encrypt"):
        s = stats[mode]
        n = s["n"]
        if n == 0:
            print(f"{mode.upper()}: n=0")
            continue
        print(
            f"{mode.upper()}: n={n} "
            f"EM={s['em']}/{n}={s['em']/n:.4f} "
            f"AvgCER={s['cer_sum']/n:.4f} "
            f"Empty={s['empty']}/{n}={s['empty']/n:.4f}"
        )

if __name__ == "__main__":
    main()

