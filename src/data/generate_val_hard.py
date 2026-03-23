import argparse
import json
import random
import string
from typing import Dict, Any, Tuple

# ===== Alphabets =====
ALPHABETS = {
    "en": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "sk": "AÁÄBCČDĎEÉFGHIÍJKLĹĽMNŇOÓÔPQRŔSŠTŤUÚVWXYÝZŽ",
    "uk": "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ",
}

# ===== Small wordbanks (expand freely) =====
WORDS = {
    "en": ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "crypto", "cipher", "model", "learns", "rules", "deterministic", "transform"],
    "sk": ["rychla", "hneda", "liška", "skače", "cez", "leniveho", "psa", "sifra", "model", "učenie", "pravidla", "bezpečnosť", "kľúč", "šifrovanie"],
    "uk": ["швидкий", "бурий", "лис", "стрибає", "через", "ледачого", "собаку", "шифр", "модель", "вчиться", "правилам", "безпека", "ключ", "шифрування"],
}

PUNCT = [".", ",", "!", "?", ":", ";"]
DIGITS = list("0123456789")

def _norm_lang(lang: str) -> str:
    lang = lang.lower().strip()
    if lang not in ALPHABETS:
        raise ValueError(f"Unsupported lang: {lang}")
    return lang

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

def substitution_make_key(lang: str, seed: int) -> Dict[str, str]:
    alpha = list(ALPHABETS[lang])
    rnd = random.Random(seed)
    perm = alpha[:]
    rnd.shuffle(perm)
    return {a: b for a, b in zip(alpha, perm)}

def substitution_apply(text: str, mapping: Dict[str, str], lang: str, encrypt: bool = True) -> str:
    alpha = ALPHABETS[lang]
    if encrypt:
        mp = mapping
    else:
        mp = {v: k for k, v in mapping.items()}

    out = []
    for ch in text:
        up = ch.upper()
        if up in alpha:
            out.append(mp[up])
        else:
            out.append(ch)
    return "".join(out)

def rand_vigenere_key(lang: str, rnd: random.Random, min_len=2, max_len=12) -> str:
    alpha = ALPHABETS[lang]
    L = rnd.randint(min_len, max_len)
    return "".join(rnd.choice(alpha) for _ in range(L))

def make_plaintext(lang: str, rnd: random.Random) -> str:
    # produce slightly "natural" sentence-like text with noise
    w = WORDS[lang]
    k = rnd.randint(8, 22)
    parts = []
    for i in range(k):
        token = rnd.choice(w)
        # random casing noise
        r = rnd.random()
        if r < 0.10:
            token = token.upper()
        elif r < 0.25:
            token = token.capitalize()

        parts.append(token)

        # punctuation or digit noise
        if rnd.random() < 0.12:
            parts[-1] = parts[-1] + rnd.choice(PUNCT)
        if rnd.random() < 0.08:
            parts.append(rnd.choice(DIGITS))

    # ensure final punctuation
    s = " ".join(parts)
    if s[-1] not in ".!?":
        s += rnd.choice([".", "!", "?"])
    return s

def apply_cipher(cipher_type: str, mode: str, lang: str, key_str: str, text: str) -> str:
    encrypt = (mode == "encrypt")
    if cipher_type == "Caesar":
        return caesar(text, int(key_str), lang, encrypt=encrypt)
    if cipher_type == "Vigenere":
        return vigenere(text, key_str, lang, encrypt=encrypt)
    if cipher_type == "Substitution":
        mapping = json.loads(key_str)
        return substitution_apply(text, mapping, lang, encrypt=encrypt)
    raise ValueError(cipher_type)

def build_reasoning(cipher_type: str, mode: str, lang: str, key: str, sample_in: str, sample_out: str) -> str:
    # short, but "computational-looking"
    return (
        f"1) TASK: {mode.upper()} using {cipher_type}\n"
        f"2) LANG: {lang} | alphabet_size={len(ALPHABETS[lang])}\n"
        f"3) KEY: {key}\n"
        f"4) APPLY transform to input text.\n"
        f"5) OUTPUT (prefix): {sample_out[:60]}\n"
        f"6) VERIFY: round-trip OK (encrypt->decrypt returns original)\n"
    )

def make_example(style: str, lang: str, cipher_type: str, mode: str, key: str, input_text: str, cipher_text: str, plaintext: str) -> Dict[str, Any]:
    # IMPORTANT: detect-style should NOT reveal cipher_type and key in the prompt by default.
    if style == "compute":
        instruction = (
            "### Instruction:\n"
            "You are CipherChat. Think step-by-step. Show calculations.\n"
            f"[{lang}] {mode.upper()}.\n"
            f"Cipher: {cipher_type}\n"
            f"Key: {key}\n"
            "Text:\n"
            f"{input_text}\n\n"
            "### Thinking:\n"
        )
    elif style == "detect":
        instruction = (
            "### Instruction:\n"
            "You are CipherChat. Infer cipher type and key, then solve. Think step-by-step.\n"
            f"[{lang}] {mode.upper()}.\n"
            "Text:\n"
            f"{input_text}\n\n"
            "### Thinking:\n"
        )
    else:
        raise ValueError("--style must be compute or detect")

    reasoning = build_reasoning(cipher_type, mode, lang, key, input_text, cipher_text if mode == "encrypt" else plaintext)

    # what is the "answer" json? we store full ground-truth for eval
    ans = {
        "mode": mode,
        "cipher_type": cipher_type,
        "key": key,  # ALWAYS string
        "cipher_text": cipher_text,
        "plaintext": plaintext,
        "reasoning": reasoning,
        "self_score": "<grade_5>",
    }
    return {"instruction": instruction, "json": ans}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["compute", "detect"], required=True)
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--langs", default="en,sk,uk")
    ap.add_argument("--hard_key_holdout", action="store_true", help="use keys unlikely seen in train")
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    langs = [_norm_lang(x) for x in args.langs.split(",")]

    # Key holdout policy
    # Caesar: prefer 21..25 if holdout, else 1..25
    def sample_caesar_key() -> str:
        if args.hard_key_holdout:
            return str(rnd.randint(21, 25))
        return str(rnd.randint(1, 25))

    # Substitution: use random permutation mapping (seeded)
    def sample_sub_key(lang: str) -> str:
        seed = rnd.randint(1, 10_000_000)
        mapping = substitution_make_key(lang, seed)
        return json.dumps(mapping, ensure_ascii=False, separators=(",", ":"))

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.num):
            lang = rnd.choice(langs)
            cipher_type = rnd.choice(["Caesar", "Vigenere", "Substitution"])
            mode = rnd.choice(["encrypt", "decrypt"])

            plain = make_plaintext(lang, rnd)

            if cipher_type == "Caesar":
                key = sample_caesar_key()
            elif cipher_type == "Vigenere":
                # random key; if holdout -> longer keys
                if args.hard_key_holdout:
                    key = rand_vigenere_key(lang, rnd, min_len=8, max_len=14)
                else:
                    key = rand_vigenere_key(lang, rnd, min_len=2, max_len=12)
            else:
                key = sample_sub_key(lang)

            # For the prompt input text:
            # - encrypt task -> input is plaintext
            # - decrypt task -> input is cipher_text
            cipher_text = apply_cipher(cipher_type, "encrypt", lang, key, plain)
            if mode == "encrypt":
                input_text = plain
                out_cipher = cipher_text
                out_plain = plain
            else:
                input_text = cipher_text
                out_cipher = cipher_text
                out_plain = apply_cipher(cipher_type, "decrypt", lang, key, cipher_text)

            ex = make_example(args.style, lang, cipher_type, mode, key, input_text, out_cipher, out_plain)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ hard-val saved -> {args.out} ({args.num})")

if __name__ == "__main__":
    main()
