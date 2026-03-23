"""
generate_dataset_v3.py
-----------------------
V3 датасет — исправление деградации V2.

Проблемы V2 которые решаем:
  1. Valid JSON упал с 96% → 63.7% (catastrophic forgetting формата)
  2. Vigenere деградировал на -25% (был недопредставлен в v2)
  3. UK всё ещё слаб (35% algo)
  4. Transposition algo всё ещё слаб (25%)

Изменения относительно v2:
  - Vigenere weight: 0.30 → 0.35 (восстанавливаем)
  - Transposition weight: 0.40 → 0.35 (чуть снижаем — был перегружен)
  - Caesar weight: 0.30 (без изменений)
  - UK weight: 0.40 → 0.35 (был слишком перекошен)
  - SK/EN: 0.30 → 0.325 каждый
  - JSON-reinforcement примеры: 15% датасета (короткие тексты, акцент на формат)
  - Detect: 50% → 45% (compute чуть больше — там JSON проще верифицировать)

Запуск:
  python generate_dataset_v3.py --num_train 12000 --num_val 1500 \
      --out_train data/train_v3.json --out_val data/val_v3.json --seed 7
"""

import json
import random
import argparse
import os
from collections import Counter
from typing import Optional

# ─────────────────────────────────────────────
# АЛФАВИТЫ
# ─────────────────────────────────────────────
ALPHABETS = {
    "en": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "sk": "AÁÄBCČDĎEÉFGHIÍJKLĹĽMNŇOÓÔPQRŔSŠTŤUÚVWXYÝZŽ",
    "uk": "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ",
}

# ─────────────────────────────────────────────
# КОРПУС (из v2 — без изменений)
# ─────────────────────────────────────────────
CORPUS = {
    "en": [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning can master deterministic cipher algorithms.",
        "Classic ciphers like Caesar and Vigenere are elegant yet breakable.",
        "Security through obscurity is not true security at all.",
        "Cryptography transforms plaintext into unreadable ciphertext.",
        "Frequency analysis reveals patterns hidden in substitution ciphers.",
        "The transposition cipher rearranges letters without changing them.",
        "Neural networks can learn algorithmic rules with sufficient examples.",
        "Every encryption scheme relies on a secret key shared between parties.",
        "Columnar transposition reads columns in a key-defined order.",
        "Vigenere uses a repeating keyword to shift each letter differently.",
        "The alphabet wraps around so Z plus one becomes A again.",
        "Strong encryption protects sensitive data from unauthorized access.",
        "Decryption reverses the transformation to recover the original message.",
        "The key length in Vigenere determines how many distinct shifts are used.",
        "Caesar cipher shifts every letter by the same fixed amount.",
        "Breaking a transposition cipher requires knowing the column order.",
        "Modern cryptography relies on mathematical hardness assumptions.",
    ],
    "sk": [
        "Rýchla hnedá líška skáče cez lenivého psa.",
        "Strojové učenie dokáže zvládnuť deterministické šifrovacie algoritmy.",
        "Klasické šifry ako Caesar a Vigenere sú elegantné, no prelomiteľné.",
        "Bezpečnosť cez utajenie nie je skutočná bezpečnosť.",
        "Kryptografia premieňa čistý text na nečitateľný šifrovaný text.",
        "Frekvenčná analýza odhaľuje vzory skryté v substitučných šifrách.",
        "Transpozičná šifra preusporadúva písmená bez ich zmeny.",
        "Neurónové siete sa dokážu naučiť algoritmické pravidlá.",
        "Každá šifra závisí od tajného kľúča zdieľaného medzi stranami.",
        "Vigenereho šifra používa opakujúce sa kľúčové slovo.",
        "Abeceda sa opakuje, takže za posledným písmenom nasleduje prvé.",
        "Dešifrovanie obracia transformáciu na obnovenie pôvodnej správy.",
        "Caesarova šifra posúva každé písmeno o rovnaký pevný počet miest.",
        "Prelomenie transpozičnej šifry si vyžaduje poznanie poradia stĺpcov.",
    ],
    "uk": [
        "Швидкий бурий лис стрибає через ледачого собаку.",
        "Машинне навчання може опанувати детерміновані алгоритми шифрування.",
        "Класичні шифри Цезаря та Віженера є елегантними, але зламними.",
        "Безпека через приховування не є справжньою безпекою.",
        "Криптографія перетворює відкритий текст на нечитаємий шифртекст.",
        "Частотний аналіз виявляє закономірності в шифрах підстановки.",
        "Шифр перестановки переставляє літери, не змінюючи їх.",
        "Нейронні мережі можуть навчитися алгоритмічним правилам.",
        "Кожна схема шифрування залежить від секретного ключа.",
        "Шифр Віженера використовує ключове слово, що повторюється.",
        "Алфавіт обертається так, що після останньої літери йде перша.",
        "Дешифрування скасовує перетворення для відновлення вихідного тексту.",
        "Шифр Цезаря зсуває кожну літеру на однакову фіксовану кількість позицій.",
        "Для злому транспозиційного шифру потрібно знати порядок стовпців.",
        "Довжина ключа у Віженері визначає кількість різних зсувів.",
        "Стовпчаста перестановка зчитує стовпці у порядку, визначеному ключем.",
        "Шифрування захищає конфіденційні дані від несанкціонованого доступу.",
        "Криптоаналіз — це наука про злом шифрів і кодів.",
        "Алфавіт української мови містить тридцять три літери.",
        "Кожна літера алфавіту має свій унікальний порядковий номер.",
        "Зворотна перестановка відновлює вихідний порядок літер.",
        "Ключ шифрування визначає спосіб перетворення відкритого тексту.",
    ],
}

# Короткие тексты для JSON-reinforcement (4-8 слов, простые)
SHORT_CORPUS = {
    "en": ["HELLO WORLD", "ATTACK AT DAWN", "THE KEY IS ALPHA",
           "MEET ME AT NOON", "SEND REINFORCEMENTS", "THE EAGLE HAS LANDED",
           "CODE RED ALERT", "MISSION COMPLETE", "ALL CLEAR"],
    "sk": ["AHOJ SVET", "UTOK ZA SVITU", "KLUC JE SIFRA",
           "STRETNI SA POLUDNIE", "POSLI POSILY", "OROL PRISTAL",
           "KOD CERVENY", "MISIA SPLNENA"],
    "uk": ["ПРИВІТ СВІТ", "АТАКА НА СВІТАНКУ", "КЛЮЧ ЦЕ ШИФР",
           "ЗУСТРІЧ О ПОЛУДНІ", "НАДІШЛИ ПІДКРІПЛЕННЯ", "ОРЕЛ ПРИЗЕМЛИВСЯ",
           "КОД ЧЕРВОНИЙ", "МІСІЯ ВИКОНАНА"],
}

KEY_VOCAB = {
    "en": ["KEY", "CIPHER", "SECRET", "VECTOR", "MODEL", "TRAIN",
           "SHIFT", "ALPHA", "WORD", "LOCK", "CRYPTO", "SECURE"],
    "sk": ["TAJNE", "HESLO", "SIFRA", "MODEL", "DATA",
           "POSUN", "ZAMOK", "BEZPEC", "KLUC", "KRYPT"],
    "uk": ["КЛЮЧ", "СЕКРЕТ", "ШИФР", "МОДЕЛЬ", "ДАНІ",
           "ПОСУВ", "АЛФА", "ЗАМОК", "КРИПТ", "ЗАХИСТ"],
}

SYSTEM_PROMPT = (
    "You are CipherChat — a cipher computation assistant.\n"
    "Given a task, output ONLY a valid JSON object, no extra text.\n\n"
    "Supported ciphers:\n"
    "  - Caesar:        key is an integer (shift amount, 1..N-1)\n"
    "  - Vigenere:      key is a string keyword\n"
    "  - Transposition: columnar transposition, key is a string keyword\n\n"
    "Output JSON schema:\n"
    "{\n"
    '  "cipher_type": "Caesar" | "Vigenere" | "Transposition",\n'
    '  "mode":        "encrypt" | "decrypt",\n'
    '  "lang":        "en" | "sk" | "uk",\n'
    '  "key":         string (integer as string for Caesar),\n'
    '  "input_text":  the text you received,\n'
    '  "output_text": the transformed result,\n'
    '  "reasoning":   short step-by-step trace (string)\n'
    "}\n\n"
    "Rules:\n"
    "- For COMPUTE tasks: cipher_type and key are given — apply them exactly.\n"
    "- For DETECT tasks: infer cipher_type and key from the ciphertext, then decrypt.\n"
    "- output_text must be the correct cryptographic transform of input_text.\n"
    "- Output ONLY the JSON object."
)

# ─────────────────────────────────────────────
# ВЕСА V3
# Ключевое изменение: Vigenere↑, Transposition↓, UK↓, JSON-reinforce=15%
# ─────────────────────────────────────────────
CIPHER_WEIGHTS = {"Caesar": 0.30, "Vigenere": 0.35, "Transposition": 0.35}
LANG_WEIGHTS   = {"en": 0.325, "sk": 0.325, "uk": 0.35}
TASK_WEIGHTS   = {"compute": 0.55, "detect": 0.45}
JSON_REINFORCE_FRAC = 0.15   # 15% примеров — короткие тексты для закрепления формата


def weighted_choice(weights: dict) -> str:
    return random.choices(list(weights.keys()),
                          weights=list(weights.values()), k=1)[0]


# ─────────────────────────────────────────────
# УТИЛИТЫ
# ─────────────────────────────────────────────

def alpha_map(lang: str):
    a = ALPHABETS[lang]
    return a, {c: i for i, c in enumerate(a)}


def sample_plain(lang: str, short: bool = False) -> str:
    if short:
        return random.choice(SHORT_CORPUS[lang])
    k = random.choice([1, 2, 3])
    return " ".join(random.sample(CORPUS[lang], k=k)).upper()


# ─────────────────────────────────────────────
# ШИФРЫ (идентично v2)
# ─────────────────────────────────────────────

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
    pad = (n_cols - len(clean) % n_cols) % n_cols
    clean += "X" * pad
    n_rows = len(clean) // n_cols
    order = sorted(range(n_cols), key=lambda i: key[i])
    return "".join(clean[r * n_cols + col] for col in order for r in range(n_rows))


def transposition_decrypt(ct: str, key: str) -> str:
    n_cols = len(key)
    if n_cols < 2 or len(ct) % n_cols != 0:
        return ct
    n_rows = len(ct) // n_cols
    order = sorted(range(n_cols), key=lambda i: key[i])
    cols = [""] * n_cols
    pos = 0
    for col in order:
        cols[col] = ct[pos: pos + n_rows]
        pos += n_rows
    return "".join(cols[col][row] for row in range(n_rows) for col in range(n_cols))


def apply_cipher(cipher_type: str, text: str, key, lang: str, encrypt: bool) -> str:
    if cipher_type == "Caesar":
        return caesar_transform(text, int(key), lang, encrypt)
    elif cipher_type == "Vigenere":
        return vigenere_transform(text, str(key), lang, encrypt)
    elif cipher_type == "Transposition":
        return transposition_encrypt(text, str(key)) if encrypt \
               else transposition_decrypt(text, str(key))
    raise ValueError(f"Unknown cipher: {cipher_type}")


def round_trip_ok(cipher_type: str, plain: str, key, lang: str) -> bool:
    try:
        if cipher_type == "Transposition":
            clean = "".join(c for c in plain if c.isalpha())
            n_cols = len(str(key))
            pad = (n_cols - len(clean) % n_cols) % n_cols
            expected = clean + "X" * pad
        else:
            expected = plain
        enc = apply_cipher(cipher_type, plain, key, lang, True)
        dec = apply_cipher(cipher_type, enc, key, lang, False)
        return dec == expected
    except Exception:
        return False


# ─────────────────────────────────────────────
# REASONING (из v2, без изменений)
# ─────────────────────────────────────────────

def make_reasoning_transposition(mode: str, key: str, inp: str, out: str) -> str:
    n_cols = len(key)
    order = sorted(range(n_cols), key=lambda i: key[i])
    lines = [
        f"TASK: {mode.upper()} | cipher=Transposition | key={key}",
        f"n_cols={n_cols} | key_chars=[{', '.join(key)}] | sorted_order={order}",
    ]
    if mode == "encrypt":
        clean = "".join(c for c in inp if c.isalpha())
        pad = (n_cols - len(clean) % n_cols) % n_cols
        padded = clean + "X" * pad
        n_rows = len(padded) // n_cols
        lines.append(f"padded_input={padded[:60]}{'...' if len(padded)>60 else ''}")
        lines.append(f"n_rows={n_rows} | padding={pad}x'X'")
        for r in range(min(n_rows, 4)):
            row_str = " ".join(padded[r * n_cols + c] for c in range(n_cols))
            lines.append(f"  row{r}: [{row_str}]")
        if n_rows > 4:
            lines.append(f"  ... ({n_rows} rows total)")
        for rank, col in enumerate(order):
            col_chars = "".join(padded[r * n_cols + col] for r in range(n_rows))
            lines.append(f"  rank{rank} -> col{col} (key='{key[col]}'): {col_chars[:20]}")
        lines.append(f"OUTPUT={out[:60]}{'...' if len(out)>60 else ''}")
    else:
        n_rows = len(inp) // n_cols if n_cols else 0
        lines.append(f"ciphertext_len={len(inp)} | n_rows={n_rows}")
        for rank, col in enumerate(order):
            chunk = inp[rank * n_rows: (rank + 1) * n_rows]
            lines.append(f"  rank{rank} -> col{col} (key='{key[col]}'): {chunk[:20]}")
        lines.append(f"read row-by-row -> OUTPUT={out[:60]}{'...' if len(out)>60 else ''}")
    lines.append("VERIFY: round-trip OK")
    return " | ".join(lines)


def make_reasoning_caesar(mode: str, lang: str, key: int,
                           inp: str, out: str, max_steps: int = 8) -> str:
    alpha, idx = alpha_map(lang)
    n = len(alpha)
    shift = (key % n) if mode == "encrypt" else ((-key) % n)
    lines = [
        f"TASK: {mode.upper()} | cipher=Caesar | lang={lang} | key={key}",
        f"alphabet_size={n} | shift={shift}",
    ]
    steps = 0
    for i, ch in enumerate(inp):
        if ch in idx and steps < max_steps:
            a = idx[ch]
            b = (a + shift) % n
            lines.append(
                f"  [{i}] '{ch}'(pos={a}) + {shift} % {n} = {b} -> '{out[i] if i<len(out) else '?'}'"
            )
            steps += 1
    if len([c for c in inp if c in idx]) > max_steps:
        lines.append(f"  ... (first {max_steps} alpha chars shown)")
    lines.append("VERIFY: round-trip OK")
    return " | ".join(lines)


def make_reasoning_vigenere(mode: str, lang: str, key: str,
                             inp: str, out: str, max_steps: int = 8) -> str:
    alpha, idx = alpha_map(lang)
    n = len(alpha)
    shifts = [idx[c] for c in key.upper() if c in idx] or [1]
    lines = [
        f"TASK: {mode.upper()} | cipher=Vigenere | lang={lang} | key={key}",
        f"alphabet_size={n} | key_shifts={shifts}",
    ]
    steps, ki = 0, 0
    for i, ch in enumerate(inp):
        if ch in idx and steps < max_steps:
            s = shifts[ki % len(shifts)]
            a = idx[ch]
            b = (a + (s if mode == "encrypt" else -s)) % n
            lines.append(
                f"  [{i}] '{ch}'(pos={a}) {'+'if mode=='encrypt'else'-'}"
                f"{s}(key[{ki%len(shifts)}]='{key[ki%len(key)] if ki<len(key) else '?'}') "
                f"% {n} = {b} -> '{out[i] if i<len(out) else '?'}'"
            )
            ki += 1
            steps += 1
    if len([c for c in inp if c in idx]) > max_steps:
        lines.append(f"  ... (first {max_steps} alpha chars shown)")
    lines.append("VERIFY: round-trip OK")
    return " | ".join(lines)


def make_reasoning(cipher_type: str, mode: str, lang: str,
                   key, inp: str, out: str) -> str:
    if cipher_type == "Caesar":
        return make_reasoning_caesar(mode, lang, int(key), inp, out)
    elif cipher_type == "Vigenere":
        return make_reasoning_vigenere(mode, lang, str(key), inp, out)
    else:
        return make_reasoning_transposition(mode, str(key), inp, out)


# ─────────────────────────────────────────────
# COMPUTE SAMPLE
# ─────────────────────────────────────────────

def generate_compute_sample(lang: str, cipher_type: str,
                             short: bool = False) -> Optional[dict]:
    alpha, _ = alpha_map(lang)
    n = len(alpha)
    plain = sample_plain(lang, short=short)
    mode = random.choice(["encrypt", "decrypt"])

    if cipher_type == "Caesar":
        key_str = str(random.randint(1, n - 1))
    elif cipher_type == "Vigenere":
        key_str = random.choice(KEY_VOCAB[lang])
    else:
        key_str = random.choice(KEY_VOCAB["en"])

    if not round_trip_ok(cipher_type, plain, key_str, lang):
        return None

    cipher_text = apply_cipher(cipher_type, plain, key_str, lang, encrypt=True)

    if cipher_type == "Transposition":
        clean = "".join(c for c in plain if c.isalpha())
        pad = (len(key_str) - len(clean) % len(key_str)) % len(key_str)
        plain_padded = clean + "X" * pad
    else:
        plain_padded = plain

    inp, out = (plain_padded, cipher_text) if mode == "encrypt" else (cipher_text, plain_padded)
    reasoning = make_reasoning(cipher_type, mode, lang, key_str, inp, out)

    user_msg = (
        f"TASK: COMPUTE\nMode: {mode.upper()}\nLanguage: {lang}\n"
        f"Cipher: {cipher_type}\nKey: {key_str}\n\nINPUT_TEXT:\n{inp}"
    )
    answer = {
        "cipher_type": cipher_type, "mode": mode, "lang": lang,
        "key": key_str, "input_text": inp, "output_text": out,
        "reasoning": reasoning,
    }
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
    }


# ─────────────────────────────────────────────
# DETECT SAMPLE
# ─────────────────────────────────────────────

def _best_caesar_guess(ciphertext: str, lang: str):
    alpha, idx = alpha_map(lang)
    best_key, best_pt, best_sc = "1", "", -1e9
    for k in range(1, len(alpha)):
        pt = caesar_transform(ciphertext, k, lang, encrypt=False)
        sc = sum(pt.count(c) for c in alpha[:5])
        if sc > best_sc:
            best_sc, best_key, best_pt = sc, str(k), pt
    return best_key, best_pt


def _best_vigenere_guess(ciphertext: str, lang: str):
    best_key, best_pt, best_sc = KEY_VOCAB[lang][0], "", -1e9
    for key in KEY_VOCAB[lang]:
        pt = vigenere_transform(ciphertext, key, lang, encrypt=False)
        sc = sum(pt.count(c) for c in ALPHABETS[lang][:5])
        if sc > best_sc:
            best_sc, best_key, best_pt = sc, key, pt
    return best_key, best_pt


def generate_detect_sample(lang: str, cipher_type: str,
                            short: bool = False) -> Optional[dict]:
    alpha, _ = alpha_map(lang)
    plain = sample_plain(lang, short=short)

    if cipher_type == "Caesar":
        true_key = str(random.randint(1, len(alpha) - 1))
    elif cipher_type == "Vigenere":
        true_key = random.choice(KEY_VOCAB[lang])
    else:
        true_key = random.choice(KEY_VOCAB["en"])

    if not round_trip_ok(cipher_type, plain, true_key, lang):
        return None

    cipher_text = apply_cipher(cipher_type, plain, true_key, lang, encrypt=True)

    if cipher_type == "Transposition":
        clean = "".join(c for c in plain if c.isalpha())
        pad = (len(true_key) - len(clean) % len(true_key)) % len(true_key)
        true_plain = clean + "X" * pad
    else:
        true_plain = plain

    if cipher_type == "Caesar":
        pred_key, pred_plain = _best_caesar_guess(cipher_text, lang)
        if pred_key != true_key:
            pred_key, pred_plain = true_key, true_plain
    elif cipher_type == "Vigenere":
        pred_key, pred_plain = _best_vigenere_guess(cipher_text, lang)
        if pred_key != true_key:
            pred_key, pred_plain = true_key, true_plain
    else:
        pred_key, pred_plain = true_key, true_plain

    reasoning = make_reasoning(cipher_type, "decrypt", lang,
                                pred_key, cipher_text, pred_plain)

    user_msg = f"TASK: DETECT\nLanguage: {lang}\n\nCIPHERTEXT:\n{cipher_text}"
    if cipher_type == "Transposition":
        crib = true_plain[:len(true_key) * 2]
        user_msg += f"\n\nKNOWN_PLAINTEXT_CRIB (first {len(crib)} chars): {crib}"

    answer = {
        "cipher_type": cipher_type, "mode": "decrypt", "lang": lang,
        "key": pred_key, "input_text": cipher_text, "output_text": pred_plain,
        "reasoning": reasoning,
    }
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
    }


# ─────────────────────────────────────────────
# ГЕНЕРАЦИЯ СПЛИТА
# ─────────────────────────────────────────────

def generate_split(n_samples: int) -> list:
    results = []
    attempts = 0
    max_attempts = n_samples * 8
    n_reinforce = int(n_samples * JSON_REINFORCE_FRAC)

    while len(results) < n_samples and attempts < max_attempts:
        attempts += 1

        # 15% — короткие тексты для закрепления JSON формата
        is_reinforce = len(results) < n_reinforce and random.random() < 0.3

        lang        = weighted_choice(LANG_WEIGHTS)
        cipher_type = weighted_choice(CIPHER_WEIGHTS)
        task_type   = weighted_choice(TASK_WEIGHTS)

        if task_type == "compute":
            sample = generate_compute_sample(lang, cipher_type, short=is_reinforce)
        else:
            sample = generate_detect_sample(lang, cipher_type, short=is_reinforce)

        if sample is not None:
            results.append(sample)
            if len(results) % 1000 == 0:
                print(f"  {len(results)}/{n_samples} (attempts={attempts})")

    if len(results) < n_samples:
        print(f"WARNING: only {len(results)}/{n_samples} generated")
    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_train", type=int, default=12000)
    ap.add_argument("--num_val",   type=int, default=1500)
    ap.add_argument("--out_train", default="data/train_v3.json")
    ap.add_argument("--out_val",   default="data/val_v3.json")
    ap.add_argument("--seed",      type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)

    print("CipherChat V3 dataset")
    print(f"  Ciphers : Vigenere=35% | Transposition=35% | Caesar=30%  (↑Vigenere vs v2)")
    print(f"  Langs   : UK=35% | SK=32.5% | EN=32.5%               (↓UK vs v2)")
    print(f"  Tasks   : compute=55% | detect=45%                    (↑compute vs v2)")
    print(f"  JSON-reinforce: {JSON_REINFORCE_FRAC*100:.0f}% short examples  (новое в v3)")
    print()

    print(f"Generating TRAIN ({args.num_train})...")
    train_data = generate_split(args.num_train)

    random.seed(args.seed + 1)
    print(f"\nGenerating VAL ({args.num_val})...")
    val_data = generate_split(args.num_val)

    for name, data in [("TRAIN", train_data), ("VAL", val_data)]:
        ciphers = Counter(
            json.loads(d["messages"][2]["content"])["cipher_type"] for d in data)
        langs = Counter(
            json.loads(d["messages"][2]["content"])["lang"] for d in data)
        tasks = Counter(
            "detect" if "TASK: DETECT" in d["messages"][1]["content"] else "compute"
            for d in data)
        # Считаем короткие примеры (по длине user msg < 200 символов)
        short_cnt = sum(1 for d in data if len(d["messages"][1]["content"]) < 200)
        print(f"\n{name}: {len(data)}")
        print(f"  ciphers    : {dict(ciphers)}")
        print(f"  langs      : {dict(langs)}")
        print(f"  tasks      : {dict(tasks)}")
        print(f"  short(reinforce): {short_cnt} ({100*short_cnt/max(len(data),1):.1f}%)")

    os.makedirs(os.path.dirname(args.out_train) or ".", exist_ok=True)
    with open(args.out_train, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Train -> {args.out_train}")

    os.makedirs(os.path.dirname(args.out_val) or ".", exist_ok=True)
    with open(args.out_val, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Val   -> {args.out_val}")


if __name__ == "__main__":
    main()
