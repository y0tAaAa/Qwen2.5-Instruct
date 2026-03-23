import argparse
import json
import os
import re
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ALPHABETS = {
    "en": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "sk": "AÁÄBCČDĎEÉFGHIÍJKLĹĽMNŇOÓÔPQRŔSŠTŤUÚVWXYÝZŽ",
    "uk": "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ",
}


def extract_text_block(instruction: str) -> str:
    m = re.search(r"\bText:\s*\n(.*)\Z", instruction, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_lang(instruction: str) -> Optional[str]:
    m = re.search(r"\[(en|sk|uk)\]", instruction.lower())
    return m.group(1) if m else None


def caesar(text: str, key: int, lang: str, encrypt: bool = True) -> str:
    alpha = ALPHABETS[lang]
    n = len(alpha)
    shift = key % n
    if not encrypt:
        shift = (-shift) % n

    out = []
    for ch in text:
        up = ch.upper()
        if up in alpha:
            idx = alpha.index(up)
            out.append(alpha[(idx + shift) % n])
        else:
            out.append(ch)
    return "".join(out)


def vigenere(text: str, key: str, lang: str, encrypt: bool = True) -> str:
    alpha = ALPHABETS[lang]
    n = len(alpha)
    shifts = [alpha.index(c) for c in key.upper() if c in alpha]
    if not shifts:
        shifts = [0]

    out = []
    ki = 0
    for ch in text:
        up = ch.upper()
        if up in alpha:
            s = shifts[ki % len(shifts)]
            idx = alpha.index(up)
            out.append(alpha[(idx + (s if encrypt else -s)) % n])
            ki += 1
        else:
            out.append(ch)
    return "".join(out)


def substitution_apply(text: str, mapping: Dict[str, str], lang: str, encrypt: bool = True) -> str:
    alpha = ALPHABETS[lang]
    mp = mapping if encrypt else {v: k for k, v in mapping.items()}
    out = []
    for ch in text:
        up = ch.upper()
        if up in alpha and up in mp:
            out.append(mp[up])
        else:
            out.append(ch)
    return "".join(out)


def apply_cipher(cipher_type: str, mode: str, lang: str, key_str: str, text: str) -> str:
    encrypt = (mode == "encrypt")
    if cipher_type == "Caesar":
        return caesar(text, int(str(key_str)), lang, encrypt=encrypt)
    if cipher_type == "Vigenere":
        return vigenere(text, str(key_str), lang, encrypt=encrypt)
    if cipher_type == "Substitution":
        mapping = json.loads(str(key_str))
        return substitution_apply(text, mapping, lang, encrypt=encrypt)
    raise ValueError(cipher_type)


def find_first_json_object(s: str) -> Optional[Dict[str, Any]]:
    # Find first {...} with brace balancing, respecting quoted strings
    start = s.find("{")
    if start < 0:
        return None

    in_str = False
    esc = False
    depth = 0

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = s[start : i + 1]
                try:
                    return json.loads(chunk)
                except Exception:
                    return None
    return None


def norm_key(x: Any) -> str:
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def load_model(base: str, adapter: str):
    """
    Fix for size mismatch:
    - load tokenizer from adapter dir (it was saved during training)
    - resize base model embeddings to match tokenizer length
    - then load PEFT adapter
    """
    tok_src = adapter if os.path.isdir(adapter) else base
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # dtype compat
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

    # resize vocab to tokenizer from adapter
    vocab_now = base_model.get_input_embeddings().weight.shape[0]
    vocab_need = len(tok)
    if vocab_now != vocab_need:
        base_model.resize_token_embeddings(vocab_need)

    model = PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    full = tok.decode(out[0], skip_special_tokens=False)
    pref = tok.decode(inputs["input_ids"][0], skip_special_tokens=False)
    return full[len(pref) :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", choices=["compute", "detect"], required=True)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--report_out", default="")
    args = ap.parse_args()

    tok, model = load_model(args.base, args.adapter)

    total = 0
    parse_ok = 0

    cipher_type_ok = 0
    key_ok = 0
    plaintext_ok = 0
    cipher_text_ok = 0
    algo_ok = 0

    rows = []

    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)
            instruction = ex["instruction"]
            gt = ex["json"]

            total += 1
            if args.max_samples and total > args.max_samples:
                break

            completion = generate(tok, model, instruction, max_new_tokens=args.max_new_tokens)
            pred = find_first_json_object(completion)

            lang = extract_lang(instruction) or gt.get("lang") or "en"
            input_text = extract_text_block(instruction)

            ok_parse = pred is not None
            if ok_parse:
                parse_ok += 1

                # predicted
                p_mode = pred.get("mode", gt.get("mode"))
                p_ct = pred.get("cipher_type", pred.get("cipher", pred.get("cipherType", "")))
                p_key = pred.get("key", "")
                p_plain = pred.get("plaintext", "")
                p_ciph = pred.get("cipher_text", pred.get("ciphertext", ""))

                # ground truth
                g_mode = gt.get("mode", "")
                g_ct = gt.get("cipher_type", "")
                g_key = gt.get("key", "")
                g_plain = gt.get("plaintext", "")
                g_ciph = gt.get("cipher_text", "")

                if str(p_ct) == str(g_ct):
                    cipher_type_ok += 1
                if norm_key(p_key) == norm_key(g_key):
                    key_ok += 1
                if str(p_plain) == str(g_plain):
                    plaintext_ok += 1
                if str(p_ciph) == str(g_ciph):
                    cipher_text_ok += 1

                # Algorithmic correctness
                try:
                    if args.task == "compute":
                        # cipher_type and key are given (gt), so check the produced output matches true transform of prompt input_text
                        expected = apply_cipher(g_ct, g_mode, lang, norm_key(g_key), input_text)
                        if g_mode == "encrypt":
                            ok_algo = (str(p_ciph) == expected)
                        else:
                            ok_algo = (str(p_plain) == expected)
                    else:
                        # detect: cipher_type/key inferred, verify by applying predicted (ct,key) to input text and comparing with GT output
                        expected = apply_cipher(str(p_ct), g_mode, lang, norm_key(p_key), input_text)
                        if g_mode == "encrypt":
                            ok_algo = (expected == g_ciph)
                        else:
                            ok_algo = (expected == g_plain)
                except Exception:
                    ok_algo = False

                if ok_algo:
                    algo_ok += 1

                rows.append(
                    {
                        "ok_parse": True,
                        "ok_algo": ok_algo,
                        "gt_cipher_type": g_ct,
                        "pred_cipher_type": str(p_ct),
                        "gt_key": norm_key(g_key),
                        "pred_key": norm_key(p_key),
                        "gt_mode": g_mode,
                    }
                )
            else:
                rows.append({"ok_parse": False, "ok_algo": False, "raw_completion_prefix": completion[:400]})

    def pct(x):
        return 0.0 if total == 0 else 100.0 * x / total

    summary = {
        "total": total,
        "json_parse_rate": pct(parse_ok),
        "cipher_type_acc": pct(cipher_type_ok),
        "key_acc": pct(key_ok),
        "plaintext_acc": pct(plaintext_ok),
        "cipher_text_acc": pct(cipher_text_ok),
        "algorithmic_acc": pct(algo_ok),
    }

    print("=== EVAL SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as out:
            json.dump({"summary": summary, "rows": rows[:300]}, out, ensure_ascii=False, indent=2)
        print(f"✅ report -> {args.report_out}")


if __name__ == "__main__":
    main()
