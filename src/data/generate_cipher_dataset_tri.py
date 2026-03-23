import os
import json
import random
import argparse
from typing import Dict, Tuple, List

# ===== Alphabets (UPPERCASE) =====
ALPHABETS = {
    "en": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    # Slovak uppercase letters (incl. diacritics) — practical set for cipher tasks
    "sk": "AÁÄBCČDĎEÉFGHIÍJKLĹĽMNŇOÓÔPQRŔSŠTŤUÚVWXYÝZŽ",
    # Ukrainian uppercase letters (incl. Ґ, Є, І, Ї)
    "uk": "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ",
}

GRADES = ["<grade_1>", "<grade_2>", "<grade_3>", "<grade_4>", "<grade_5>"]

# ===== Some multilingual plaintext pools (you can expand) =====
TEXTS = {
    "en": [
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
        "CLASSIC CIPHERS ARE FUN TO BREAK",
        "MACHINE LEARNING CAN LEARN ALGORITHMIC RULES",
        "ENCRYPTION AND DECRYPTION ARE DETERMINISTIC TRANSFORMS",
    ],
    "sk": [
        "KLASICKÉ ŠIFRY AKO CAESAR A VIGENERE SÚ ZAUJÍMAVÉ",
        "RÝCHLA HNEDÁ LÍŠKA SKÁČE CEZ LENIVÉHO PSA",
        "STROJOVÉ UČENIE SA MÔŽE UČIŤ DETERMINISTICKÉ PRAVIDLÁ",
        "DEŠIFROVANIE JE ALGORITMICKÁ TRANSFORMÁCIA",
    ],
    "uk": [
        "ШВИДКИЙ БУРИЙ ЛИС СТРИБАЄ ЧЕРЕЗ ЛЕДАЧОГО СОБАКУ",
        "КЛАСИЧНІ ШИФРИ ЯК ЦЕЗАР І ВІЖЕНЕР ЦІКАВІ",
        "МАШИННЕ НАВЧАННЯ МОЖЕ ВИВЧИТИ АЛГОРИТМІЧНІ ПРАВИЛА",
        "ДЕШИФРУВАННЯ Є ДЕТЕРМІНІСТИЧНОЮ ТРАНСФОРМАЦІЄЮ",
    ],
}

def _upper_preserve(text: str) -> str:
    # We will operate mainly on uppercase; keep other chars as-is
    return text.upper()

def caesar_shift_char(ch: str, alphabet: str, shift: int) -> str:
    idx = alphabet.find(ch)
    if idx == -1:
        return ch
    return alphabet[(idx + shift) % len(alphabet)]

def caesar_encrypt(plaintext: str, alphabet: str, shift: int) -> str:
    pt = _upper_preserve(plaintext)
    return "".join(caesar_shift_char(c, alphabet, shift) for c in pt)

def caesar_decrypt(cipher_text: str, alphabet: str, shift: int) -> str:
    ct = _upper_preserve(cipher_text)
    return "".join(caesar_shift_char(c, alphabet, -shift) for c in ct)

def vigenere_key_indices(key: str, alphabet: str) -> List[int]:
    key = _upper_preserve(key)
    idxs = [alphabet.find(c) for c in key if alphabet.find(c) != -1]
    return idxs if idxs else [0]  # fallback "A"

def vigenere_encrypt(plaintext: str, alphabet: str, key: str) -> str:
    pt = _upper_preserve(plaintext)
    key_idxs = vigenere_key_indices(key, alphabet)
    out = []
    ki = 0
    for c in pt:
        pos = alphabet.find(c)
        if pos == -1:
            out.append(c)
        else:
            shift = key_idxs[ki % len(key_idxs)]
            out.append(alphabet[(pos + shift) % len(alphabet)])
            ki += 1
    return "".join(out)

def vigenere_decrypt(cipher_text: str, alphabet: str, key: str) -> str:
    ct = _upper_preserve(cipher_text)
    key_idxs = vigenere_key_indices(key, alphabet)
    out = []
    ki = 0
    for c in ct:
        pos = alphabet.find(c)
        if pos == -1:
            out.append(c)
        else:
            shift = key_idxs[ki % len(key_idxs)]
            out.append(alphabet[(pos - shift) % len(alphabet)])
            ki += 1
    return "".join(out)

def make_substitution_mapping(alphabet: str, rng: random.Random) -> Dict[str, str]:
    src = list(alphabet)
    dst = list(alphabet)
    rng.shuffle(dst)
    # Ensure it's not identical mapping (rare, but possible)
    if dst == src:
        rng.shuffle(dst)
    return {s: d for s, d in zip(src, dst)}

def substitution_apply(text: str, mapping: Dict[str, str], alphabet: str) -> str:
    t = _upper_preserve(text)
    out = []
    for c in t:
        if c in mapping:
            out.append(mapping[c])
        else:
            out.append(c)
    return "".join(out)

def substitution_invert(mapping: Dict[str, str]) -> Dict[str, str]:
    return {v: k for k, v in mapping.items()}

def random_caesar_key(rng: random.Random, alphabet: str) -> str:
    shift = rng.randint(1, max(1, len(alphabet) - 1))
    return str(shift)

def random_vigenere_key(rng: random.Random, alphabet: str, min_len=3, max_len=10) -> str:
    L = rng.randint(min_len, max_len)
    return "".join(rng.choice(alphabet) for _ in range(L))

def random_text(rng: random.Random, lang: str) -> str:
    return rng.choice(TEXTS[lang])

def build_instruction(lang: str, mode: str, cipher_type: str, key_str: str, text: str) -> str:
    # This matches your training style: instruction -> JSON only
    return (
        "### Instruction:\n"
        "You are CipherChat — a specialized assistant for encryption and decryption.\n"
        "Follow the instruction exactly.\n"
        "Return ONLY valid JSON.\n"
        f"[{lang}] {mode.upper()}.\n"
        f"Cipher: {cipher_type}\n"
        f"Key: {key_str}\n"
        "Return ONLY valid JSON with fields: `mode`, `task_variant`, `cipher_type`, `key`, "
        "`cipher_text`, `plaintext`, `detected_cipher_type`, `detected_key`, `verify`, `reasoning`, `self_score` "
        "(one of <grade_1>, <grade_2>, <grade_3>, <grade_4>, <grade_5>).\n"
        "Text:\n"
        f"{text}\n\n"
        "### Response (JSON only):\n"
    )

def generate_sample(rng: random.Random, lang: str, cipher_type: str, mode: str) -> Dict:
    alphabet = ALPHABETS[lang]
    pt = random_text(rng, lang)

    task_variant = "known_params"  # keep it simple now (3 ciphers only)

    if cipher_type == "Caesar":
        key_str = random_caesar_key(rng, alphabet)
        shift = int(key_str)
        ct = caesar_encrypt(pt, alphabet, shift)
        out_plain = caesar_decrypt(ct, alphabet, shift)

    elif cipher_type == "Vigenere":
        key_str = random_vigenere_key(rng, alphabet)
        ct = vigenere_encrypt(pt, alphabet, key_str)
        out_plain = vigenere_decrypt(ct, alphabet, key_str)

    elif cipher_type == "Substitution":
        mapping = make_substitution_mapping(alphabet, rng)
        # key must be STRING always → store JSON string
        key_obj = {"alphabet": alphabet, "mapping": mapping}
        key_str = json.dumps(key_obj, ensure_ascii=False)
        ct = substitution_apply(pt, mapping, alphabet)
        inv = substitution_invert(mapping)
        out_plain = substitution_apply(ct, inv, alphabet)

    else:
        raise ValueError(f"Unknown cipher_type: {cipher_type}")

    if mode == "encrypt":
        cipher_text = ct
        plaintext = pt
        # "verify" is whether decrypt(encrypt(pt)) == pt
        verify = "round_trip_ok" if out_plain == _upper_preserve(pt) else "round_trip_fail"
    else:
        cipher_text = ct
        plaintext = out_plain
        verify = "round_trip_ok" if out_plain == _upper_preserve(pt) else "round_trip_fail"

    # Detected fields: for now we teach "use known params" (later you can add detect_solve)
    detected_cipher_type = cipher_type
    detected_key = key_str

    reasoning = "DETECT→SOLVE→VERIFY: alphabet_check; params_known; apply_cipher; round_trip_check"
    self_score = rng.choice(GRADES)

    instruction = build_instruction(lang, mode, cipher_type, key_str, cipher_text)

    j = {
        "mode": mode,
        "task_variant": task_variant,
        "cipher_type": cipher_type,
        "key": str(key_str),  # IMPORTANT: ALWAYS STRING
        "cipher_text": cipher_text,
        "plaintext": plaintext,
        "detected_cipher_type": detected_cipher_type,
        "detected_key": str(detected_key),
        "verify": verify,
        "reasoning": reasoning,
        "self_score": self_score,
    }
    return {"instruction": instruction, "json": j, "lang": lang}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--train_file", default="train_cipher_train.jsonl")
    ap.add_argument("--val_file", default="val_cipher.jsonl")
    ap.add_argument("--n_train", type=int, default=58800)
    ap.add_argument("--n_val", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    langs = ["en", "sk", "uk"]
    ciphers = ["Caesar", "Vigenere", "Substitution"]
    modes = ["encrypt", "decrypt"]

    def write_file(path: str, n: int):
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                lang = rng.choice(langs)
                cipher = rng.choice(ciphers)
                mode = rng.choice(modes)
                sample = generate_sample(rng, lang, cipher, mode)
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    train_path = os.path.join(args.out_dir, args.train_file)
    val_path = os.path.join(args.out_dir, args.val_file)

    write_file(train_path, args.n_train)
    write_file(val_path, args.n_val)

    print(f"[OK] Wrote train: {train_path}  n={args.n_train}")
    print(f"[OK] Wrote val:   {val_path}  n={args.n_val}")

if __name__ == "__main__":
    main()

