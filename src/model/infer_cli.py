# src/model/infer_cli.py
import argparse, json, os, re
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ALLOWED_GRADES = {"<grade_1>", "<grade_2>", "<grade_3>", "<grade_4>", "<grade_5>"}

def extract_last_json(text: str) -> Optional[Dict[str, Any]]:
    """Берём последний JSON-объект из вывода модели."""
    # ищем последний блок { ... } (жадно)
    starts = [m.start() for m in re.finditer(r"\{", text)]
    if not starts:
        return None
    for s in reversed(starts):
        tail = text[s:]
        # обрежем по последней закрывающей скобке
        e = tail.rfind("}")
        if e == -1:
            continue
        chunk = tail[: e + 1]
        try:
            return json.loads(chunk)
        except Exception:
            continue
    return None

def normalize_grade(v: Any) -> str:
    v = (v or "").strip()
    return v if v in ALLOWED_GRADES else "<grade_3>"

def load_model(base_id: str, lora_path: str, dtype=torch.bfloat16) -> Tuple[Any, Any]:
    """
    ВАЖНО:
    - токенизатор берём из LoRA-папки (там мог быть изменён vocab/спецтокены)
    - base модель берём из base_id
    - потом resize под len(tokenizer) и накатываем LoRA
    """
    tok = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # подгоняем embedding под токенизатор LoRA
    base.resize_token_embeddings(len(tok))

    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()

    # фикс pad_token_id везде
    model.config.pad_token_id = tok.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tok.pad_token_id

    return tok, model

def build_prompt(mode: str, cipher_type: str, key: str, text: str, lang: str, task_variant: str) -> str:
    """
    Стиль максимально близок твоим jsonl:
    ### Instruction: ... ### Response (JSON only):
    """
    head = (
        "### Instruction:\n"
        "You are CipherChat — a specialized assistant for encryption and decryption.\n"
        "Follow the instruction exactly.\n"
        "Use the pipeline: DETECT→SOLVE→VERIFY.\n"
        "Return ONLY valid JSON.\n"
        f"[{lang}] {mode.upper()}.\n"
    )

    if task_variant == "detect_solve":
        head += "Cipher: UNKNOWN\nKey: UNKNOWN\n"
    elif task_variant == "noisy_params":
        head += "Cipher: WRONG\nKey: WRONG\n"
    else:
        head += f"Cipher: {cipher_type}\nKey: {key}\n"

    head += (
        "Return ONLY valid JSON with fields: "
        "`mode`, `task_variant`, `cipher_type`, `key`, `cipher_text`, `plaintext`, "
        "`detected_cipher_type`, `detected_key`, `verify`, `reasoning`, `self_score` "
        "(self_score one of <grade_1>,<grade_2>,<grade_3>,<grade_4>,<grade_5>).\n"
        "Text:\n"
        f"{text}\n\n"
        "### Response (JSON only):\n"
    )
    return head

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora", required=True, help="path to checkpoints/..")
    ap.add_argument("--mode", required=True, choices=["encrypt", "decrypt"])
    ap.add_argument("--cipher_type", required=True, choices=["Caesar", "Vigenere", "Substitution"])
    ap.add_argument("--key", required=True, help="For Substitution: JSON-string key is allowed")
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="en", choices=["en", "sk", "uk"])
    ap.add_argument("--task_variant", default="known_params", choices=["known_params", "detect_solve", "noisy_params"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    tok, model = load_model(args.base, args.lora)

    prompt = build_prompt(args.mode, args.cipher_type, args.key, args.text, args.lang, args.task_variant)
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    decoded = tok.decode(out[0], skip_special_tokens=False)
    parsed = extract_last_json(decoded)

    if parsed is None:
        print(json.dumps({"error": "no_json", "raw_tail": decoded[-800:]}, ensure_ascii=False))
        return

    # лёгкая нормализация
    parsed.setdefault("mode", args.mode)
    parsed.setdefault("task_variant", args.task_variant)
    parsed.setdefault("cipher_type", args.cipher_type)
    parsed.setdefault("key", args.key)
    parsed.setdefault("cipher_text", args.text)
    parsed["self_score"] = normalize_grade(parsed.get("self_score"))
    parsed.setdefault("verify", "round_trip_fail")

    print(json.dumps(parsed, ensure_ascii=False))

if __name__ == "__main__":
    main()

