import json
import random
import argparse
import os
from typing import Dict, Any, List, Tuple

def set_seed(seed: int):
    random.seed(seed)

CORPUS = {
    "en": [
        "Classic ciphers like Caesar and Vigenere are fun to break.",
        "Machine learning can learn algorithmic rules very effectively.",
        "The quick brown fox jumps over the lazy dog.",
        "Encryption and decryption are deterministic transforms.",
        "We are training a reasoning model for cipher tasks.",
        "Cryptography is the art of writing and solving codes.",
        "Advanced models can understand and solve classical ciphers.",
        "Security through obscurity is not real security.",
        "Vigenere cipher was considered unbreakable for centuries.",
        "Substitution ciphers are vulnerable to frequency analysis."
    ],
    "sk": [
        "Klasické šifry ako Caesar a Vigenere sú zaujímavé.",
        "Strojové učenie môže naučiť algoritmické pravidlá.",
        "Rýchla hnedá líška skáče cez lenivého psa.",
        "Dešifrovanie je dôležitá súčasť kybernetickej bezpečnosti.",
        "Trénujeme model na šifrovanie a dešifrovanie s vysvetlením.",
        "Vigenereho šifra bola dlho považovaná za nezlomiteľnú."
    ],
    "uk": [
        "Класичні шифри як Цезар і Віженер цікаві для вивчення.",
        "Машинне навчання може вивчити алгоритмічні правила.",
        "Швидкий бурий лис стрибає через ледачого собаку.",
        "Шифрування та дешифрування — детерміновані перетворення.",
        "Ми навчаємо модель шифруванню та дешифруванню з поясненням.",
        "Шифр Віженера довго вважався незламним."
    ]
}

ALPHABETS = {
    "en": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "sk": "AÁÄBCČDĎEÉFGHIÍJKLĹĽMNŇOÓÔPQRŔSŠTŤUÚVWXYÝZŽ",
    "uk": "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
}

# Контролируемые ключи Vigenere, чтобы не было “пустых сдвигов”
KEY_VOCAB = {
    "en": ["KEY", "CIPHER", "SECRET", "VECTOR", "MODEL", "TRAIN", "SHIFT", "ALPHA"],
    "sk": ["TAJNE", "HESLO", "KLUČ", "SIFRA", "MODEL", "DATA", "POSUN", "ZAMEK"],
    "uk": ["КЛЮЧ", "СЕКРЕТ", "ШИФР", "МОДЕЛЬ", "ДАНІ", "ПОСУВ", "АЛФА", "ЗЛАМ"]
}

def normalize(text: str) -> str:
    return text.upper()

def alpha_maps(lang: str):
    alpha = ALPHABETS[lang]
    idx = {ch: i for i, ch in enumerate(alpha)}
    return alpha, idx

def caesar(text: str, key: int, lang: str, encrypt: bool = True) -> str:
    alpha, idx = alpha_maps(lang)
    n = len(alpha)
    shift = key % n
    if not encrypt:
        shift = -shift
    out = []
    for ch in normalize(text):
        if ch in idx:
            out.append(alpha[(idx[ch] + shift) % n])
        else:
            out.append(ch)
    return "".join(out)

def vigenere(text: str, key: str, lang: str, encrypt: bool = True) -> str:
    alpha, idx = alpha_maps(lang)
    n = len(alpha)
    key_u = normalize(key)
    shifts = [idx[c] for c in key_u if c in idx]
    if not shifts:
        shifts = [0]
    out = []
    k = 0
    for ch in normalize(text):
        if ch in idx:
            s = shifts[k % len(shifts)]
            out.append(alpha[(idx[ch] + (s if encrypt else -s)) % n])
            k += 1
        else:
            out.append(ch)
    return "".join(out)

def subst_key_perm(lang: str) -> str:
    alpha = ALPHABETS[lang]
    perm = list(alpha)
    random.shuffle(perm)
    return "".join(perm)

def substitution(text: str, key_perm: str, lang: str, encrypt: bool = True) -> str:
    alpha, idx = alpha_maps(lang)
    fwd = {a: b for a, b in zip(alpha, key_perm)}
    rev = {b: a for a, b in zip(alpha, key_perm)}
    mp = fwd if encrypt else rev
    out = []
    for ch in normalize(text):
        out.append(mp.get(ch, ch))
    return "".join(out)

def sample_plain(lang: str) -> str:
    # делаем тексты подлиннее, чтобы было меньше “угадывания по теме”
    k = random.choice([1, 2, 3])
    sents = random.sample(CORPUS[lang], k=k)
    glue = " " if lang == "en" else " "
    return normalize(glue.join(sents))

def grade(ok: bool) -> str:
    return "<grade_5>" if ok else "<grade_2>"

def compute_trace_caesar(inp: str, out: str, lang: str, key: int, encrypt: bool, max_steps: int = 8) -> List[str]:
    alpha, idx = alpha_maps(lang)
    n = len(alpha)
    lines = []
    steps = 0
    for i, ch in enumerate(inp):
        if ch in idx:
            a = idx[ch]
            b = (a + (key if encrypt else -key)) % n
            lines.append(f"{i}: '{ch}' pos={a} -> {a} {'+' if encrypt else '-'} {key} = {b} -> '{out[i]}'")
            steps += 1
            if steps >= max_steps:
                break
    return lines

def compute_trace_vigenere(inp: str, out: str, lang: str, key: str, encrypt: bool, max_steps: int = 8) -> List[str]:
    alpha, idx = alpha_maps(lang)
    n = len(alpha)
    key_u = normalize(key)
    key_shifts = [idx[c] for c in key_u if c in idx] or [0]
    lines = []
    steps = 0
    kpos = 0
    for i, ch in enumerate(inp):
        if ch in idx:
            s = key_shifts[kpos % len(key_shifts)]
            a = idx[ch]
            b = (a + (s if encrypt else -s)) % n
            kch = alpha[s]
            lines.append(f"{i}: '{ch}' pos={a}, key='{kch}' shift={s} -> {b} -> '{out[i]}'")
            kpos += 1
            steps += 1
            if steps >= max_steps:
                break
    return lines

def compute_trace_subst(inp: str, out: str, lang: str, key_perm: str, encrypt: bool, max_steps: int = 8) -> List[str]:
    alpha, idx = alpha_maps(lang)
    fwd = {a: b for a, b in zip(alpha, key_perm)}
    rev = {b: a for a, b in zip(alpha, key_perm)}
    mp = fwd if encrypt else rev
    lines = []
    steps = 0
    for i, ch in enumerate(inp):
        if ch in idx or ch in mp:
            mapped = mp.get(ch, ch)
            lines.append(f"{i}: '{ch}' -> '{mapped}'")
            steps += 1
            if steps >= max_steps:
                break
    # добавим небольшой “срез” ключа
    lines.append(f"key_prefix: {key_perm[:min(40, len(key_perm))]}")
    return lines

def generate_sample() -> Dict[str, Any]:
    lang = random.choice(["en", "sk", "uk"])
    cipher_type = random.choice(["Caesar", "Vigenere", "Substitution"])
    mode = random.choice(["encrypt", "decrypt"])

    plain = sample_plain(lang)

    if cipher_type == "Caesar":
        key = random.randint(1, len(ALPHABETS[lang]) - 1)
        cipher = caesar(plain, key, lang, encrypt=True)
        if mode == "encrypt":
            inp, out = plain, cipher
        else:
            inp, out = cipher, plain
        trace = compute_trace_caesar(inp, out, lang, key, encrypt=(mode == "encrypt"))

    elif cipher_type == "Vigenere":
        key = random.choice(KEY_VOCAB[lang])
        cipher = vigenere(plain, key, lang, encrypt=True)
        if mode == "encrypt":
            inp, out = plain, cipher
        else:
            inp, out = cipher, plain
        trace = compute_trace_vigenere(inp, out, lang, key, encrypt=(mode == "encrypt"))

    else:
        key = subst_key_perm(lang)
        cipher = substitution(plain, key, lang, encrypt=True)
        if mode == "encrypt":
            inp, out = plain, cipher
        else:
            inp, out = cipher, plain
        trace = compute_trace_subst(inp, out, lang, key, encrypt=(mode == "encrypt"))

    # VERIFY (round-trip)
    if mode == "encrypt":
        # out == encrypt(inp)
        if cipher_type == "Caesar":
            ok = (caesar(inp, key, lang, True) == out)
        elif cipher_type == "Vigenere":
            ok = (vigenere(inp, key, lang, True) == out)
        else:
            ok = (substitution(inp, key, lang, True) == out)
    else:
        # out == decrypt(inp)
        if cipher_type == "Caesar":
            ok = (caesar(inp, key, lang, False) == out)
        elif cipher_type == "Vigenere":
            ok = (vigenere(inp, key, lang, False) == out)
        else:
            ok = (substitution(inp, key, lang, False) == out)

    reasoning = {
        "style": "compute",
        "cipher": cipher_type,
        "lang": lang,
        "mode": mode,
        "key": key,
        "trace": trace,
        "verify": {"round_trip_ok": ok}
    }

    # PROMPT: для compute мы даём cipher+key+mode и ТЕКСТ ВХОДА (не “всегда cipher”)
    instruction = f"""### Instruction:
You are CipherChat.
Return ONLY valid JSON (no extra text).
Task: {mode.upper()} using the given cipher and key.
Language: {lang}
Cipher: {cipher_type}
Key: {key}

INPUT_TEXT:
{inp}

### Answer JSON:
"""

    # label JSON: хранить и input, и обе стороны полезно для eval
    answer = {
        "mode": mode,
        "lang": lang,
        "cipher_type": cipher_type,
        "key": key,
        "input_text": inp,
        "cipher_text": cipher,
        "plaintext": plain,
        "output_text": out,
        "reasoning": reasoning,
        "self_score": grade(ok)
    }

    return {"instruction": instruction, "json": answer}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=15000)
    ap.add_argument("--out", default="data/processed/train_compute.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.num):
            ex = generate_sample()
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            if i % 1000 == 0:
                print(f"[COMPUTE_V2] {i}/{args.num}")

    print(f"✅ saved → {args.out} ({args.num})")

if __name__ == "__main__":
    main()
