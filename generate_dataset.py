"""
generate_dataset.py
-------------------
Генерирует тренировочный датасет для трёх шифров:
  - Caesar      (ключ: целое число)
  - Vigenere    (ключ: строка-слово)
  - Transposition  (колончатая перестановка, ключ: строка-слово)

Три языка: en / sk / uk
Два режима задачи: compute (дан шифр+ключ) и detect (дан только шифртекст)

ПРИНЦИПЫ (исправляющие ошибки CipherChat):
  1. В датасет попадают ТОЛЬКО примеры с ok_roundtrip=True
  2. Reasoning — всегда строка (нет конфликтов типов)
  3. Нет self_score / grade токенов — это балласт для loss
  4. Один унифицированный формат messages (system/user/assistant)
  5. Detect для Transposition — с partial crib (иначе задача неразрешима)

Запуск:
  python generate_dataset.py --num_train 14000 --num_val 1500 \
      --out_train train.json --out_val val.json --seed 42
"""

import json
import random
import argparse
import os
from collections import Counter
import math
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
# КОРПУС (10+ предложений на язык)
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
    ],
}

# Контролируемые ключи Vigenere (без пустых сдвигов)
KEY_VOCAB = {
    "en": ["KEY", "CIPHER", "SECRET", "VECTOR", "MODEL", "TRAIN", "SHIFT", "ALPHA", "WORD", "LOCK"],
    "sk": ["TAJNE", "HESLO", "SIFRA", "MODEL", "DATA", "POSUN", "ZAMOK", "KLUČ", "BEZPEC"],
    "uk": ["КЛЮЧ", "СЕКРЕТ", "ШИФР", "МОДЕЛЬ", "ДАНІ", "ПОСУВ", "АЛФА", "ЗАМОК"],
}

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
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
# УТИЛИТЫ
# ─────────────────────────────────────────────

def alpha_map(lang: str):
    """Возвращает (alphabet_str, char→index dict)."""
    a = ALPHABETS[lang]
    return a, {c: i for i, c in enumerate(a)}


def sample_plain(lang: str) -> str:
    k = random.choice([1, 2, 3])
    sents = random.sample(CORPUS[lang], k=k)
    return " ".join(sents).upper()


# ─────────────────────────────────────────────
# РЕАЛИЗАЦИИ ШИФРОВ
# ─────────────────────────────────────────────

def caesar_transform(text: str, key: int, lang: str, encrypt: bool) -> str:
    alpha, idx = alpha_map(lang)
    n = len(alpha)
    shift = (key % n) if encrypt else ((-key) % n)
    out = []
    for ch in text:
        if ch in idx:
            out.append(alpha[(idx[ch] + shift) % n])
        else:
            out.append(ch)
    return "".join(out)


def vigenere_transform(text: str, key: str, lang: str, encrypt: bool) -> str:
    alpha, idx = alpha_map(lang)
    n = len(alpha)
    key_upper = key.upper()
    # Только символы из алфавита языка
    shifts = [idx[c] for c in key_upper if c in idx]
    if not shifts:
        shifts = [1]
    out = []
    ki = 0
    for ch in text:
        if ch in idx:
            s = shifts[ki % len(shifts)]
            delta = s if encrypt else -s
            out.append(alpha[(idx[ch] + delta) % n])
            ki += 1
        else:
            out.append(ch)
    return "".join(out)


def transposition_encrypt(text: str, key: str) -> str:
    """Колончатая перестановка: ключ определяет порядок чтения столбцов."""
    # Работаем только с буквами (убираем пробелы и пунктуацию)
    clean = "".join(c for c in text if c.isalpha())
    n_cols = len(key)
    if n_cols < 2:
        return clean

    # Дополняем до кратного n_cols символом X
    pad_len = (n_cols - len(clean) % n_cols) % n_cols
    clean += "X" * pad_len
    n_rows = len(clean) // n_cols

    # Порядок столбцов: сортируем по символам ключа
    order = sorted(range(n_cols), key=lambda i: key[i])

    # Читаем столбцы в sorted порядке
    result = []
    for col in order:
        for row in range(n_rows):
            result.append(clean[row * n_cols + col])
    return "".join(result)


def transposition_decrypt(ciphertext: str, key: str) -> str:
    """Обратная колончатая перестановка."""
    n_cols = len(key)
    if n_cols < 2 or len(ciphertext) % n_cols != 0:
        return ciphertext
    n_rows = len(ciphertext) // n_cols

    order = sorted(range(n_cols), key=lambda i: key[i])

    # Разбиваем шифртекст на столбцы
    cols = [""] * n_cols
    pos = 0
    for col_idx in order:
        cols[col_idx] = ciphertext[pos: pos + n_rows]
        pos += n_rows

    # Читаем по строкам
    result = []
    for row in range(n_rows):
        for col in range(n_cols):
            result.append(cols[col][row])
    return "".join(result)


def apply_cipher(cipher_type: str, text: str, key, lang: str, encrypt: bool) -> str:
    if cipher_type == "Caesar":
        return caesar_transform(text, int(key), lang, encrypt)
    elif cipher_type == "Vigenere":
        return vigenere_transform(text, str(key), lang, encrypt)
    elif cipher_type == "Transposition":
        if encrypt:
            return transposition_encrypt(text, str(key))
        else:
            return transposition_decrypt(text, str(key))
    raise ValueError(f"Unknown cipher: {cipher_type}")


def round_trip_ok(cipher_type: str, plaintext: str, key, lang: str) -> bool:
    """Проверка: encrypt → decrypt == исходный текст."""
    try:
        # Для транспозиции работаем с очищенным текстом
        if cipher_type == "Transposition":
            clean = "".join(c for c in plaintext if c.isalpha())
            n_cols = len(str(key))
            pad_len = (n_cols - len(clean) % n_cols) % n_cols
            expected = clean + "X" * pad_len
        else:
            expected = plaintext

        enc = apply_cipher(cipher_type, plaintext, key, lang, encrypt=True)
        dec = apply_cipher(cipher_type, enc, key, lang, encrypt=False)
        return dec == expected
    except Exception:
        return False


# ─────────────────────────────────────────────
# ГЕНЕРАЦИЯ REASONING TRACE
# ─────────────────────────────────────────────

def make_reasoning(cipher_type: str, mode: str, lang: str, key,
                   inp: str, out: str, max_steps: int = 6) -> str:
    alpha, idx = alpha_map(lang)
    n = len(alpha)
    lines = [
        f"TASK: {mode.upper()} | cipher={cipher_type} | lang={lang} | key={key}",
        f"alphabet_size={n}",
    ]

    if cipher_type == "Caesar":
        shift = int(key) % n
        actual_shift = shift if mode == "encrypt" else (-shift) % n
        lines.append(f"shift={actual_shift} ({'forward' if mode == 'encrypt' else 'backward'})")
        steps = 0
        for i, ch in enumerate(inp):
            if ch in idx and steps < max_steps:
                a = idx[ch]
                b = (a + actual_shift) % n
                lines.append(f"  [{i}] '{ch}'(pos={a}) + {actual_shift} % {n} = {b} -> '{out[i] if i < len(out) else '?'}'")
                steps += 1
        if len(inp) > max_steps:
            lines.append(f"  ... ({len(inp)} chars total, showing first {max_steps})")

    elif cipher_type == "Vigenere":
        shifts = [idx[c] for c in key.upper() if c in idx] or [1]
        lines.append(f"key_shifts={shifts[:8]}{'...' if len(shifts) > 8 else ''}")
        steps, ki = 0, 0
        for i, ch in enumerate(inp):
            if ch in idx and steps < max_steps:
                s = shifts[ki % len(shifts)]
                a = idx[ch]
                delta = s if mode == "encrypt" else -s
                b = (a + delta) % n
                lines.append(
                    f"  [{i}] '{ch}'(pos={a}) {'+'if mode=='encrypt' else '-'}"
                    f"{s}(key[{ki % len(shifts)}]) % {n} = {b} -> '{out[i] if i < len(out) else '?'}'"
                )
                ki += 1
                steps += 1
        if len(inp) > max_steps:
            lines.append(f"  ... ({len(inp)} chars total, showing first {max_steps})")

    elif cipher_type == "Transposition":
        n_cols = len(str(key))
        order = sorted(range(n_cols), key=lambda i: str(key)[i])
        lines.append(f"n_cols={n_cols} | column_order={order}")
        lines.append(f"input_len={len(inp)} -> output_len={len(out)}")
        lines.append(f"columns read in order: {[str(key)[i] for i in order]}")

    lines.append("VERIFY: round-trip OK")
    return " | ".join(lines)


# ─────────────────────────────────────────────
# IoC — для detect reasoning
# ─────────────────────────────────────────────

def index_of_coincidence(text: str, lang: str) -> float:
    alpha, idx = alpha_map(lang)
    counts = Counter(ch for ch in text if ch in idx)
    N = sum(counts.values())
    if N <= 1:
        return 0.0
    return sum(v * (v - 1) for v in counts.values()) / (N * (N - 1))


# ─────────────────────────────────────────────
# ГЕНЕРАЦИЯ ПРИМЕРОВ
# ─────────────────────────────────────────────

def generate_compute_sample(lang: str, cipher_type: str) -> Optional[dict]:
    """
    COMPUTE задача: пользователю даны cipher_type + key + mode + input_text.
    Модель должна вернуть output_text.
    """
    alpha, _ = alpha_map(lang)
    n = len(alpha)
    plain = sample_plain(lang)
    mode = random.choice(["encrypt", "decrypt"])

    # Выбор ключа
    if cipher_type == "Caesar":
        key = random.randint(1, n - 1)
        key_str = str(key)
    elif cipher_type == "Vigenere":
        key = random.choice(KEY_VOCAB[lang])
        key_str = key
    else:  # Transposition
        key = random.choice(KEY_VOCAB["en"])  # Ключ-слово, всегда латиница
        key_str = key

    # Верификация round-trip ДО генерации примера
    if not round_trip_ok(cipher_type, plain, key_str, lang):
        return None  # Пропускаем — только корректные примеры

    # Вычисляем ciphertext
    cipher_text = apply_cipher(cipher_type, plain, key_str, lang, encrypt=True)

    # Для транспозиции: "правильный" plaintext — очищенный + padding
    if cipher_type == "Transposition":
        n_cols = len(key_str)
        clean = "".join(c for c in plain if c.isalpha())
        pad_len = (n_cols - len(clean) % n_cols) % n_cols
        plain_padded = clean + "X" * pad_len
    else:
        plain_padded = plain

    if mode == "encrypt":
        inp, out = plain_padded, cipher_text
    else:
        inp, out = cipher_text, plain_padded

    reasoning = make_reasoning(cipher_type, mode, lang, key_str, inp, out)

    user_msg = (
        f"TASK: COMPUTE\n"
        f"Mode: {mode.upper()}\n"
        f"Language: {lang}\n"
        f"Cipher: {cipher_type}\n"
        f"Key: {key_str}\n\n"
        f"INPUT_TEXT:\n{inp}"
    )

    answer = {
        "cipher_type": cipher_type,
        "mode": mode,
        "lang": lang,
        "key": key_str,
        "input_text": inp,
        "output_text": out,
        "reasoning": reasoning,
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
    }


def _best_caesar_guess(ciphertext: str, lang: str):
    """Простейший брутфорс Caesar по частотному score."""
    alpha, idx = alpha_map(lang)
    # Биграмм нет — используем только IoC-подобный score по top символу
    best_key, best_pt, best_sc = 1, "", -1e9
    for k in range(1, len(alpha)):
        pt = caesar_transform(ciphertext, k, lang, encrypt=False)
        # Score: частота самых распространённых букв алфавита EN/SK/UK
        sc = sum(pt.count(c) for c in alpha[:5])
        if sc > best_sc:
            best_sc, best_key, best_pt = sc, k, pt
    return best_key, best_pt


def _best_vigenere_guess(ciphertext: str, lang: str):
    best_key, best_pt, best_sc = KEY_VOCAB[lang][0], "", -1e9
    for key in KEY_VOCAB[lang]:
        pt = vigenere_transform(ciphertext, key, lang, encrypt=False)
        sc = sum(pt.count(c) for c in ALPHABETS[lang][:5])
        if sc > best_sc:
            best_sc, best_key, best_pt = sc, key, pt
    return best_key, best_pt


def generate_detect_sample(lang: str, cipher_type: str) -> Optional[dict]:
    """
    DETECT задача: пользователю дан только шифртекст (и crib для Transposition).
    Модель должна определить cipher_type, ключ и расшифровать.
    """
    alpha, _ = alpha_map(lang)
    plain = sample_plain(lang)

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
        n_cols = len(true_key)
        clean = "".join(c for c in plain if c.isalpha())
        pad_len = (n_cols - len(clean) % n_cols) % n_cols
        true_plain = clean + "X" * pad_len
    else:
        true_plain = plain

    # Определяем что модель должна предсказать
    if cipher_type == "Caesar":
        pred_key, pred_plain = _best_caesar_guess(cipher_text, lang)
        pred_key = str(pred_key)
        # Проверяем: правильно ли угадали
        ok = (pred_key == true_key)
        if not ok:
            pred_key = true_key  # Подсказываем правильный — модель должна учиться
            pred_plain = true_plain
    elif cipher_type == "Vigenere":
        pred_key, pred_plain = _best_vigenere_guess(cipher_text, lang)
        ok = (pred_key == true_key)
        if not ok:
            pred_key = true_key
            pred_plain = true_plain
    else:  # Transposition — без crib нельзя решить, даём подсказку
        pred_key = true_key
        pred_plain = true_plain
        ok = True

    ioc_val = index_of_coincidence(cipher_text, lang)
    reasoning = (
        f"TASK: DETECT | lang={lang} | "
        f"IoC={ioc_val:.4f} (Caesar~0.065, Vigenere~0.045, Transposition preserves freqs) | "
        f"top_chars={Counter(ch for ch in cipher_text if ch in alpha).most_common(3)} | "
        f"hypothesis: cipher={cipher_type}, key={pred_key} | "
        f"VERIFY: re-encrypt(predicted_plain)==ciphertext OK"
    )

    user_msg = f"TASK: DETECT\nLanguage: {lang}\n\nCIPHERTEXT:\n{cipher_text}"
    if cipher_type == "Transposition":
        # Даём crib: первые n_cols символов plaintext известны
        crib = true_plain[:len(true_key) * 2]
        user_msg += f"\n\nKNOWN_PLAINTEXT_CRIB (first {len(crib)} chars): {crib}"

    answer = {
        "cipher_type": cipher_type,
        "mode": "decrypt",
        "lang": lang,
        "key": pred_key,
        "input_text": cipher_text,
        "output_text": pred_plain,
        "reasoning": reasoning,
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
    }


# ─────────────────────────────────────────────
# ГЛАВНАЯ ЛОГИКА
# ─────────────────────────────────────────────

def generate_split(n_samples: int, seed_offset: int = 0) -> list:
    """Генерирует n_samples примеров, 70% compute / 30% detect."""
    results = []
    attempts = 0
    max_attempts = n_samples * 5  # Страховка от бесконечного цикла

    langs = ["en", "sk", "uk"]
    ciphers = ["Caesar", "Vigenere", "Transposition"]

    while len(results) < n_samples and attempts < max_attempts:
        attempts += 1
        lang = random.choice(langs)
        cipher_type = random.choice(ciphers)
        task_type = random.choices(["compute", "detect"], weights=[0.70, 0.30])[0]

        if task_type == "compute":
            sample = generate_compute_sample(lang, cipher_type)
        else:
            sample = generate_detect_sample(lang, cipher_type)

        if sample is not None:  # round-trip OK
            results.append(sample)
            if len(results) % 1000 == 0:
                print(f"  generated {len(results)}/{n_samples} (attempts={attempts})")

    if len(results) < n_samples:
        print(f"WARNING: only {len(results)}/{n_samples} valid examples generated "
              f"(check round-trip logic)")
    return results


def main():
    ap = argparse.ArgumentParser(description="Generate cipher training dataset")
    ap.add_argument("--num_train", type=int, default=14000)
    ap.add_argument("--num_val",   type=int, default=1500)
    ap.add_argument("--out_train", default="train.json")
    ap.add_argument("--out_val",   default="val.json")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print(f"Generating TRAIN set ({args.num_train} samples)...")
    train_data = generate_split(args.num_train)

    random.seed(args.seed + 1)
    print(f"\nGenerating VAL set ({args.num_val} samples)...")
    val_data = generate_split(args.num_val)

    # Статистика
    for name, data in [("TRAIN", train_data), ("VAL", val_data)]:
        ciphers = Counter(
            json.loads(d["messages"][2]["content"])["cipher_type"] for d in data
        )
        langs = Counter(
            json.loads(d["messages"][2]["content"])["lang"] for d in data
        )
        tasks = Counter(
            "detect" if "DETECT" in d["messages"][1]["content"] else "compute"
            for d in data
        )
        print(f"\n{name}: {len(data)} samples")
        print(f"  ciphers: {dict(ciphers)}")
        print(f"  langs:   {dict(langs)}")
        print(f"  tasks:   {dict(tasks)}")

    with open(args.out_train, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Train saved → {args.out_train} ({len(train_data)} samples)")

    with open(args.out_val, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Val saved   → {args.out_val} ({len(val_data)} samples)")


if __name__ == "__main__":
    main()
