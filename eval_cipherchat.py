"""
eval_cipherchat.py
------------------
Оценка обученной CipherChat модели.
Структура по мотивам qwen_eval-deinerovych.py:
  - Загружает base model один раз
  - Прогоняет каждый пример ДВАЖДЫ: base model и fine-tuned (через disable_adapter)
  - Сравнивает base vs fine-tuned
  - Разбивка: seen (train) vs unseen (val) + cipher_type + lang + task_type

Запуск:
  python eval_cipherchat.py \
      --base    Qwen/Qwen2.5-7B-Instruct \
      --adapter checkpoints/cipherchat-7b-20260306_153334/final_adapter \
      --data    data/eval300.json \
      --out_dir eval_results
"""

import json
import os
import torch
import argparse
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ─────────────────────────────────────────────
# АЛФАВИТЫ И ШИФРЫ
# ─────────────────────────────────────────────

ALPHABETS = {
    "en": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "sk": "AÁÄBCČDĎEÉFGHIÍJKLĹĽMNŇOÓÔPQRŔSŠTŤUÚVWXYÝZŽ",
    "uk": "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ",
}


def alpha_map(lang: str):
    a = ALPHABETS[lang]
    return a, {c: i for i, c in enumerate(a)}


def caesar_transform(text: str, key: int, lang: str, encrypt: bool) -> str:
    alpha, idx = alpha_map(lang)
    n = len(alpha)
    shift = (key % n) if encrypt else ((-key) % n)
    return "".join(alpha[(idx[ch] + shift) % n] if ch in idx else ch for ch in text)


def vigenere_transform(text: str, key: str, lang: str, encrypt: bool) -> str:
    alpha, idx = alpha_map(lang)
    n = len(alpha)
    shifts = [idx[c] for c in key.upper() if c in idx] or [1]
    out, ki = [], 0
    for ch in text:
        if ch in idx:
            s = shifts[ki % len(shifts)]
            out.append(alpha[(idx[ch] + (s if encrypt else -s)) % n])
            ki += 1
        else:
            out.append(ch)
    return "".join(out)


def transposition_encrypt(text: str, key: str) -> str:
    clean = "".join(c for c in text if c.isalpha())
    n_cols = len(key)
    if n_cols < 2:
        return clean
    pad_len = (n_cols - len(clean) % n_cols) % n_cols
    clean += "X" * pad_len
    n_rows = len(clean) // n_cols
    order = sorted(range(n_cols), key=lambda i: key[i])
    result = []
    for col in order:
        for row in range(n_rows):
            result.append(clean[row * n_cols + col])
    return "".join(result)


def transposition_decrypt(ciphertext: str, key: str) -> str:
    n_cols = len(key)
    if n_cols < 2 or len(ciphertext) % n_cols != 0:
        return ciphertext
    n_rows = len(ciphertext) // n_cols
    order = sorted(range(n_cols), key=lambda i: key[i])
    cols = [""] * n_cols
    pos = 0
    for col_idx in order:
        cols[col_idx] = ciphertext[pos: pos + n_rows]
        pos += n_rows
    return "".join(cols[col][row] for row in range(n_rows) for col in range(n_cols))


def apply_cipher(cipher_type: str, text: str, key: str, lang: str, encrypt: bool) -> Optional[str]:
    try:
        if cipher_type == "Caesar":
            return caesar_transform(text, int(key), lang, encrypt)
        elif cipher_type == "Vigenere":
            return vigenere_transform(text, key, lang, encrypt)
        elif cipher_type == "Transposition":
            return transposition_encrypt(text, key) if encrypt else transposition_decrypt(text, key)
    except Exception:
        return None
    return None


# ─────────────────────────────────────────────
# ЗАГРУЗКА МОДЕЛИ (с QLoRA — как при обучении)
# ─────────────────────────────────────────────

def load_model(base: str, adapter: str):
    print(f"  Loading tokenizer from: {adapter}")
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading base model: {base}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    vocab_now  = base_model.get_input_embeddings().weight.shape[0]
    vocab_need = len(tokenizer)
    if vocab_now != vocab_need:
        base_model.resize_token_embeddings(vocab_need)
        print(f"  Resized embeddings: {vocab_now} → {vocab_need}")

    print(f"  Loading LoRA adapter: {adapter}")
    model = PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
    model.eval()
    print(f"  Model ready!\n")
    return tokenizer, model


# ─────────────────────────────────────────────
# ГЕНЕРАЦИЯ
# ─────────────────────────────────────────────

def generate(model, tokenizer, messages: list, max_new_tokens: int = 512) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ─────────────────────────────────────────────
# ПАРСИНГ И СРАВНЕНИЕ
# ─────────────────────────────────────────────

def parse_json_output(raw: str):
    """Парсит JSON из ответа модели. Возвращает (dict|None, bool)."""
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            parts = clean.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part), True
                except Exception:
                    continue
        start = clean.find("{")
        if start < 0:
            return None, False
        in_str, esc, depth = False, False, 0
        for i in range(start, len(clean)):
            ch = clean[i]
            if in_str:
                esc = (not esc and ch == "\\")
                if not esc and ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(clean[start: i + 1]), True
        return None, False
    except Exception:
        return None, False


def verify_algo(pred: Dict, gt: Dict, lang: str) -> bool:
    """Криптографическая проверка: применяем предсказанный key и сравниваем."""
    try:
        p_ct   = str(pred.get("cipher_type", ""))
        p_key  = str(pred.get("key", ""))
        p_mode = str(pred.get("mode", gt.get("mode", "encrypt")))
        p_inp  = str(pred.get("input_text",  gt.get("input_text", "")))
        p_out  = str(pred.get("output_text", ""))
        encrypt = (p_mode == "encrypt")
        computed = apply_cipher(p_ct, p_inp, p_key, lang, encrypt)
        return computed is not None and computed == p_out
    except Exception:
        return False


def compare_outputs(pred_parsed, gt: Dict, lang: str) -> Dict[str, Any]:
    if pred_parsed is None:
        return {
            "output_exact":    False,
            "cipher_type_acc": False,
            "key_acc":         False,
            "algo_correct":    False,
        }
    return {
        "output_exact":    str(pred_parsed.get("output_text","")) == str(gt.get("output_text","")),
        "cipher_type_acc": str(pred_parsed.get("cipher_type","")) == str(gt.get("cipher_type","")),
        "key_acc":         str(pred_parsed.get("key",""))         == str(gt.get("key","")),
        "algo_correct":    verify_algo(pred_parsed, gt, lang),
    }


# ─────────────────────────────────────────────
# СЧЁТЧИКИ МЕТРИК
# ─────────────────────────────────────────────

class Metrics:
    def __init__(self):
        self.total        = 0
        self.valid_json   = 0
        self.output_exact = 0
        self.cipher_acc   = 0
        self.key_acc      = 0
        self.algo_correct = 0

    def add(self, valid_json: bool, cmp: Dict):
        self.total        += 1
        self.valid_json   += int(valid_json)
        self.output_exact += int(cmp.get("output_exact",    False))
        self.cipher_acc   += int(cmp.get("cipher_type_acc", False))
        self.key_acc      += int(cmp.get("key_acc",         False))
        self.algo_correct += int(cmp.get("algo_correct",    False))

    def pct(self, v: int) -> str:
        n = max(self.total, 1)
        return f"{v}/{self.total} ({100*v/n:.1f}%)"

    def as_dict(self) -> Dict:
        return {
            "total":        self.total,
            "valid_json":   self.pct(self.valid_json),
            "algo_correct": self.pct(self.algo_correct),
            "output_exact": self.pct(self.output_exact),
            "cipher_acc":   self.pct(self.cipher_acc),
            "key_acc":      self.pct(self.key_acc),
        }


# ─────────────────────────────────────────────
# ПЕЧАТЬ ТАБЛИЦЫ
# ─────────────────────────────────────────────

METRICS_ORDER = [
    ("valid_json",   "Valid JSON"),
    ("algo_correct", "Algo correct"),
    ("output_exact", "Output exact"),
    ("cipher_acc",   "Cipher type"),
    ("key_acc",      "Key acc"),
]


def print_section(title: str,
                  base_map:  Dict[str, Metrics],
                  ft_map:    Dict[str, Metrics],
                  keys:      list):
    print(f"\n{'─'*80}")
    print(f"  {title}")
    print(f"  {'─'*78}")
    print(f"  {'Group':<14} {'Metric':<16} {'Base':>22} {'Fine-tuned':>22}")
    print(f"  {'─'*78}")
    for key in keys:
        b = base_map.get(key, Metrics())
        f = ft_map.get(key,   Metrics())
        if b.total == 0 and f.total == 0:
            continue
        first = True
        for attr, label in METRICS_ORDER:
            bv = getattr(b, attr)
            fv = getattr(f, attr)
            bn, fn = max(b.total,1), max(f.total,1)
            b_str = f"{bv}/{b.total} ({100*bv/bn:.0f}%)"
            f_str = f"{fv}/{f.total} ({100*fv/fn:.0f}%)"
            k_str = key if first else ""
            print(f"  {k_str:<14} {label:<16} {b_str:>22} {f_str:>22}")
            first = False
        print(f"  {'·'*78}")


# ─────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",           default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter",        required=True)
    ap.add_argument("--data",           default="data/eval300.json")
    ap.add_argument("--out_dir",        default="eval_results")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("CIPHERCHAT EVALUATION — base vs fine-tuned | seen vs unseen")
    print("=" * 80)
    print(f"Adapter : {args.adapter}")
    print(f"Data    : {args.data}")
    print(f"Start   : {ts}")
    print("=" * 80 + "\n")

    # ── Данные ───────────────────────────────────────────────
    with open(args.data, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    n = len(eval_data)
    n_seen   = sum(1 for x in eval_data if x.get("_eval_split") == "seen")
    n_unseen = sum(1 for x in eval_data if x.get("_eval_split") == "unseen")
    print(f"Total: {n}  |  seen(train)={n_seen}  |  unseen(val)={n_unseen}\n")

    # ── Модель ───────────────────────────────────────────────
    tokenizer, finetuned_model = load_model(args.base, args.adapter)

    # ── Счётчики ─────────────────────────────────────────────
    base_global = Metrics()
    ft_global   = Metrics()

    base_by: Dict[str, Dict[str, Metrics]] = {
        "split":        defaultdict(Metrics),
        "cipher":       defaultdict(Metrics),
        "lang":         defaultdict(Metrics),
        "task":         defaultdict(Metrics),
        "cipher_lang":  defaultdict(Metrics),  # Caesar×en, Caesar×uk ...
        "cipher_task":  defaultdict(Metrics),  # Caesar×compute ...
        "lang_task":    defaultdict(Metrics),  # en×detect ...
    }
    ft_by: Dict[str, Dict[str, Metrics]] = {
        "split":        defaultdict(Metrics),
        "cipher":       defaultdict(Metrics),
        "lang":         defaultdict(Metrics),
        "task":         defaultdict(Metrics),
        "cipher_lang":  defaultdict(Metrics),
        "cipher_task":  defaultdict(Metrics),
        "lang_task":    defaultdict(Metrics),
    }

    results = []

    # ── Loop ─────────────────────────────────────────────────
    for i, item in enumerate(eval_data):
        messages   = item["messages"]
        system_msg = messages[0]["content"]
        user_msg   = messages[1]["content"]
        gt         = json.loads(messages[2]["content"])
        ev_split   = item.get("_eval_split", "unseen")

        lang        = gt.get("lang", "en")
        cipher_type = gt.get("cipher_type", "")
        task_type   = "detect" if "TASK: DETECT" in user_msg else "compute"

        prompt_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{n}] split={ev_split} | {cipher_type} | {lang} | {task_type}")

        # ── BASE (адаптер выключен) ───────────────────────────
        with finetuned_model.disable_adapter():
            base_raw = generate(finetuned_model, tokenizer,
                                prompt_messages, args.max_new_tokens)
        base_parsed, base_valid = parse_json_output(base_raw)
        base_cmp = compare_outputs(base_parsed, gt, lang)

        # ── FINE-TUNED ────────────────────────────────────────
        ft_raw = generate(finetuned_model, tokenizer,
                          prompt_messages, args.max_new_tokens)
        ft_parsed, ft_valid = parse_json_output(ft_raw)
        ft_cmp = compare_outputs(ft_parsed, gt, lang)

        # ── Накопление ────────────────────────────────────────
        base_global.add(base_valid, base_cmp)
        ft_global.add(ft_valid, ft_cmp)

        for dim, key in [("split", ev_split), ("cipher", cipher_type),
                         ("lang", lang),       ("task",   task_type),
                         ("cipher_lang", f"{cipher_type}×{lang}"),
                         ("cipher_task", f"{cipher_type}×{task_type}"),
                         ("lang_task",   f"{lang}×{task_type}")]:
            base_by[dim][key].add(base_valid, base_cmp)
            ft_by[dim][key].add(ft_valid,   ft_cmp)

        results.append({
            "idx":         i + 1,
            "eval_split":  ev_split,
            "lang":        lang,
            "cipher_type": cipher_type,
            "task_type":   task_type,
            "gt_key":      str(gt.get("key", ""))[:60],
            "gt_output":   str(gt.get("output_text", ""))[:80],
            "base_model": {
                "raw_output":  base_raw[:300],
                "parsed_json": base_parsed,
                "valid_json":  base_valid,
                **base_cmp,
            },
            "finetuned_model": {
                "raw_output":  ft_raw[:300],
                "parsed_json": ft_parsed,
                "valid_json":  ft_valid,
                **ft_cmp,
            },
        })

        # ── Промежуточное сохранение каждые 50 примеров ──────
        if (i + 1) % 50 == 0:
            ckpt_path = os.path.join(args.out_dir, f"eval300_{ts}_ckpt{i+1}.json")
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"meta": {"checkpoint": i + 1, "total": n},
                           "results": results}, f, ensure_ascii=False)
            print(f"  💾 Checkpoint saved → {ckpt_path}")
            # Очищаем VRAM кэш чтобы не накапливался
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

    # ── ФИНАЛЬНЫЕ ТАБЛИЦЫ ────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Общая таблица
    print(f"\n  {'Metric':<20} {'Base':>22} {'Fine-tuned':>22}")
    print(f"  {'─'*66}")
    for attr, label in METRICS_ORDER:
        bv = getattr(base_global, attr)
        fv = getattr(ft_global,   attr)
        bn = max(base_global.total, 1)
        fn = max(ft_global.total,   1)
        print(f"  {label:<20} {bv}/{bn} ({100*bv/bn:.1f}%):>22 "
              f"{fv}/{fn} ({100*fv/fn:.1f}%):>22")

    # Чистая печать общей таблицы
    print(f"\n  {'Metric':<20} {'Base':>24} {'Fine-tuned':>24}")
    print(f"  {'─'*70}")
    for attr, label in METRICS_ORDER:
        bv = getattr(base_global, attr)
        fv = getattr(ft_global,   attr)
        bn = max(base_global.total, 1)
        fn = max(ft_global.total,   1)
        b_str = f"{bv}/{bn} ({100*bv/bn:.1f}%)"
        f_str = f"{fv}/{fn} ({100*fv/fn:.1f}%)"
        print(f"  {label:<20} {b_str:>24} {f_str:>24}")

    print_section("BY SPLIT  [seen = from train | unseen = from val]",
                  base_by["split"], ft_by["split"],
                  ["seen", "unseen"])

    print_section("BY CIPHER TYPE",
                  base_by["cipher"], ft_by["cipher"],
                  ["Caesar", "Vigenere", "Transposition"])

    print_section("BY LANGUAGE",
                  base_by["lang"], ft_by["lang"],
                  ["en", "sk", "uk"])

    print_section("BY TASK TYPE",
                  base_by["task"], ft_by["task"],
                  ["compute", "detect"])

    print_section("BY CIPHER × LANGUAGE",
                  base_by["cipher_lang"], ft_by["cipher_lang"],
                  [f"{c}×{l}" for c in ["Caesar","Vigenere","Transposition"]
                               for l in ["en","sk","uk"]])

    print_section("BY CIPHER × TASK",
                  base_by["cipher_task"], ft_by["cipher_task"],
                  [f"{c}×{t}" for c in ["Caesar","Vigenere","Transposition"]
                               for t in ["compute","detect"]])

    print_section("BY LANGUAGE × TASK",
                  base_by["lang_task"], ft_by["lang_task"],
                  [f"{l}×{t}" for l in ["en","sk","uk"]
                               for t in ["compute","detect"]])

    print("\n" + "=" * 80)

    # ── Сохранение ───────────────────────────────────────────
    def _serialize_metrics_map(m):
        return {k: v.as_dict() for k, v in m.items()}

    output = {
        "meta": {
            "base":       args.base,
            "adapter":    args.adapter,
            "data":       args.data,
            "timestamp":  ts,
            "n_total":    n,
            "n_seen":     n_seen,
            "n_unseen":   n_unseen,
        },
        "summary": {
            "overall": {
                "base":      base_global.as_dict(),
                "finetuned": ft_global.as_dict(),
            },
            "by_split":       {"base": _serialize_metrics_map(base_by["split"]),
                               "ft":   _serialize_metrics_map(ft_by["split"])},
            "by_cipher":      {"base": _serialize_metrics_map(base_by["cipher"]),
                               "ft":   _serialize_metrics_map(ft_by["cipher"])},
            "by_lang":        {"base": _serialize_metrics_map(base_by["lang"]),
                               "ft":   _serialize_metrics_map(ft_by["lang"])},
            "by_task":        {"base": _serialize_metrics_map(base_by["task"]),
                               "ft":   _serialize_metrics_map(ft_by["task"])},
            "by_cipher_lang": {"base": _serialize_metrics_map(base_by["cipher_lang"]),
                               "ft":   _serialize_metrics_map(ft_by["cipher_lang"])},
            "by_cipher_task": {"base": _serialize_metrics_map(base_by["cipher_task"]),
                               "ft":   _serialize_metrics_map(ft_by["cipher_task"])},
            "by_lang_task":   {"base": _serialize_metrics_map(base_by["lang_task"]),
                               "ft":   _serialize_metrics_map(ft_by["lang_task"])},
        },
        "results": results,
    }

    out_path = os.path.join(args.out_dir, f"eval300_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Report saved → {out_path}")
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
