# src/data/generate_cipher_dataset.py

import json
import random
import argparse
from pathlib import Path
from ciphers.ciphers_multi import (
    caesar_encrypt,
    vigenere_encrypt,
    substitution_make_key,
    substitution_encrypt,
)

SENTENCES = {
    "en": [
        "This is a test of cipher learning.",
        "Classic ciphers are interesting.",
    ],
    "sk": [
        "Klasické šifry sú zaujímavé.",
        "Šifrovanie a dešifrovanie textu.",
    ],
    "uk": [
        "Класичні шифри цікаві.",
        "Навчання моделі шифрування.",
    ],
}

CIPHERS = ["Caesar", "Vigenere", "Substitution"]

def generate_sample():
    lang = random.choice(["en", "sk", "uk"])
    text = random.choice(SENTENCES[lang])
    cipher_type = random.choice(CIPHERS)
    mode = random.choice(["encrypt", "decrypt"])

    if cipher_type == "Caesar":
        key = random.randint(1, 5)
        cipher_text = caesar_encrypt(text, key, lang)
    elif cipher_type == "Vigenere":
        key = "KEY"
        cipher_text = vigenere_encrypt(text, key, lang)
    else:
        key_dict = substitution_make_key(lang)
        key = key_dict
        cipher_text = substitution_encrypt(text, key_dict, lang)

    if mode == "encrypt":
        inp = text
        out = cipher_text
    else:
        inp = cipher_text
        out = text

    return {
        "instruction": f"[{lang}] {mode} using {cipher_type}",
        "json": {
            "mode": mode,
            "cipher_type": cipher_type,
            "key": str(key),
            "cipher_text": inp if mode == "decrypt" else out,
            "plaintext": out if mode == "decrypt" else inp,
        },
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output", default="data/train.jsonl")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for _ in range(args.num_samples):
            sample = generate_sample()
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

