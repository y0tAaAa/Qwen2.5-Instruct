#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генерация синтетического датасета для Caesar, Vigenere, Substitution
на английском, словацком и украинском.

Выход: data/processed/train_cipher_multi.jsonl
Каждая строка:
{
  "instruction": "### Instruction: ... ### Response (JSON only):\n",
  "json": { ... }
}
"""

import json
import os
import random
from typing import List, Dict

# Спец-токены для самооценки
GRADE_TOKENS = ["<grade_1>", "<grade_2>", "<grade_3>", "<grade_4>", "<grade_5>"]
DEFAULT_SELF_SCORE = "<grade_5>"

# Алфавиты
EN = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SK = "AÁÄBCČDĎEÉFGHIÍJKLĹĽMNŇOÓÔPQRŔSŠTŤUÚVWXYÝZŽ"
UK = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"


def get_alphabet(lang: str) -> str:
    if lang == "en":
        return EN
    if lang == "sk":
        return SK
    if lang == "uk":
        return UK
    raise ValueError(f"Unknown lang: {lang}")


def _normalize_case(ch: str, mapped: str) -> str:
    return mapped.lower() if ch.islower() else mapped


# ===== Caesar =====

def caesar_encrypt(text: str, shift: int, alphabet: str) -> str:
    out = []
    n = len(alphabet)
    for ch in text:
        up = ch.upper()
        if up in alphabet:
            idx = alphabet.index(up)
            out.append(_normalize_case(ch, alphabet[(idx + shift) % n]))
        else:
            out.append(ch)
    return "".join(out)


def caesar_decrypt(text: str, shift: int, alphabet: str) -> str:
    return caesar_encrypt(text, -shift, alphabet)


# ===== Vigenere =====

def vigenere_encrypt(text: str, key: str, alphabet: str) -> str:
    key_idx = [alphabet.index(c) for c in key.upper() if c in alphabet]
    if not key_idx:
        raise ValueError("Empty Vigenere key after filtering")
    out = []
    j = 0
    for ch in text:
        up = ch.upper()
        if up in alphabet:
            t = alphabet.index(up)
            k = key_idx[j % len(key_idx)]
            out.append(_normalize_case(ch, alphabet[(t + k) % len(alphabet)]))
            j += 1
        else:
            out.append(ch)
    return "".join(out)


def vigenere_decrypt(text: str, key: str, alphabet: str) -> str:
    key_idx = [alphabet.index(c) for c in key.upper() if c in alphabet]
    if not key_idx:
        raise ValueError("Empty Vigenere key after filtering")
    out = []
    j = 0
    for ch in text:
        up = ch.upper()
        if up in alphabet:
            c_idx = alphabet.index(up)
            k = key_idx[j % len(key_idx)]
            out.append(_normalize_case(ch, alphabet[(c_idx - k) % len(alphabet)]))
            j += 1
        else:
            out.append(ch)
    return "".join(out)


# ===== Substitution =====

def make_substitution(alphabet: str) -> Dict[str, str]:
    letters = list(alphabet)
    perm = letters[:]
    random.shuffle(perm)
    return {a: b for a, b in zip(letters, perm)}


def subst_encrypt(text: str, mapping: Dict[str, str], alphabet: str) -> str:
    out = []
    for ch in text:
        up = ch.upper()
        if up in alphabet:
            mapped = mapping.get(up, up)
            out.append(_normalize_case(ch, mapped))
        else:
            out.append(ch)
    return "".join(out)


def subst_decrypt(text: str, mapping: Dict[str, str], alphabet: str) -> str:
    inv = {v: k for k, v in mapping.items()}
    return subst_encrypt(text, inv, alphabet)


# ===== Инструкции =====

def build_instruction(lang: str, mode: str, cipher_type: str, key, text: str) -> str:
    """
    Строим промпт с явным указанием на все поля JSON, включая self_score.
    """
    key_str = str(key)
    if lang == "uk":
        if mode == "encrypt":
            msg = (
                f"Зашифруй цей текст шифром {cipher_type} з ключем {key_str}. "
                "Поверни ТІЛЬКИ валідний JSON з полями: "
                "`mode`, `cipher_type`, `key`, `cipher_text`, `plaintext`, "
                "`reasoning`, `self_score` (одне з <grade_1>, <grade_2>, <grade_3>, <grade_4>, <grade_5>)."
                "\nText:\n"
                f"{text}"
            )
        else:
            msg = (
                f"Розшифруй цей текст, зашифрований шифром {cipher_type} з ключем {key_str}. "
                "Поверни ТІЛЬКИ валідний JSON з полями: "
                "`mode`, `cipher_type`, `key`, `cipher_text`, `plaintext`, "
                "`reasoning`, `self_score` (одне з <grade_1>, <grade_2>, <grade_3>, <grade_4>, <grade_5>)."
                "\nText:\n"
                f"{text}"
            )
    elif lang == "sk":
        if mode == "encrypt":
            msg = (
                f"Zašifruj tento text pomocou šifry {cipher_type} s kľúčom {key_str}. "
                "Vráť IBA platný JSON s poľami: "
                "`mode`, `cipher_type`, `key`, `cipher_text`, `plaintext`, "
                "`reasoning`, `self_score` (jedno z <grade_1>, <grade_2>, <grade_3>, <grade_4>, <grade_5>)."
                "\nText:\n"
                f"{text}"
            )
        else:
            msg = (
                f"Dešifruj tento text pomocou šifry {cipher_type} s kľúčом {key_str}. "
                "Vráť IBA platný JSON s poľami: "
                "`mode`, `cipher_type`, `key`, `cipher_text`, `plaintext`, "
                "`reasoning`, `self_score` (jedno z <grade_1>, <grade_2>, <grade_3>, <grade_4>, <grade_5>)."
                "\nText:\n"
                f"{text}"
            )
    else:
        if mode == "encrypt":
            msg = (
                f"Encrypt this text using {cipher_type} cipher with key {key_str}. "
                "Return ONLY valid JSON with fields: "
                "`mode`, `cipher_type`, `key`, `cipher_text`, `plaintext`, "
                "`reasoning`, `self_score` (one of <grade_1>, <grade_2>, <grade_3>, <grade_4>, <grade_5>)."
                "\nText:\n"
                f"{text}"
            )
        else:
            msg = (
                f"Decrypt this text encrypted with {cipher_type} cipher using key {key_str}. "
                "Return ONLY valid JSON with fields: "
                "`mode`, `cipher_type`, `key`, `cipher_text`, `plaintext`, "
                "`reasoning`, `self_score` (one of <grade_1>, <grade_2>, <grade_3>, <grade_4>, <grade_5>)."
                "\nText:\n"
                f"{text}"
            )

    return "### Instruction:\n" + msg + "\n\n### Response (JSON only):\n"


def make_pair(text: str, lang: str, cipher_type: str) -> List[dict]:
    """
    Для одного текста и шифра генерим encrypt+decrypt пары с self_score.
    """
    alphabet = get_alphabet(lang)

    if cipher_type == "Caesar":
        shift = random.randint(1, len(alphabet) - 1)
        key = str(shift)
        cipher = caesar_encrypt(text, shift, alphabet)
        plain = text
        reasoning_enc = f"Applied Caesar cipher with shift {shift} to the plaintext."
        reasoning_dec = f"Recovered plaintext by Caesar decryption with shift {shift}."
    elif cipher_type == "Vigenere":
        key = "".join(random.choice(alphabet) for _ in range(5))
        cipher = vigenere_encrypt(text, key, alphabet)
        plain = text
        reasoning_enc = f"Applied Vigenere cipher with key {key} to the plaintext."
        reasoning_dec = f"Recovered plaintext by Vigenere decryption with key {key}."
    elif cipher_type == "Substitution":
        mapping = make_substitution(alphabet)
        key = mapping
        cipher = subst_encrypt(text, mapping, alphabet)
        plain = text
        reasoning_enc = "Applied monoalphabetic substitution to the plaintext."
        reasoning_dec = "Recovered plaintext using the inverse substitution mapping."
    else:
        raise ValueError(f"Unknown cipher_type: {cipher_type}")

    enc = {
        "instruction": build_instruction(lang, "encrypt", cipher_type, key, plain),
        "json": {
            "mode": "encrypt",
            "cipher_type": cipher_type,
            "key": key,
            "cipher_text": cipher,
            "plaintext": plain,
            "reasoning": reasoning_enc,
            "self_score": DEFAULT_SELF_SCORE,
        },
    }
    dec = {
        "instruction": build_instruction(lang, "decrypt", cipher_type, key, cipher),
        "json": {
            "mode": "decrypt",
            "cipher_type": cipher_type,
            "key": key,
            "cipher_text": cipher,
            "plaintext": plain,
            "reasoning": reasoning_dec,
            "self_score": DEFAULT_SELF_SCORE,
        },
    }
    return [enc, dec]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs_per_lang",
        type=int,
        default=10000,
        help="Сколько пар (encrypt+decrypt) генерировать на каждый язык.",
    )
    args = parser.parse_args()

    random.seed(42)
    out_path = os.path.join("data", "processed", "train_cipher_multi.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    base_texts = {
        "en": [
            "This is a test of cipher learning.",
            "Classic ciphers like Caesar and Vigenere are fun to break.",
            "We are training a reasoning model for encryption and decryption."
        ],
        "sk": [
            "Toto je test učenia šifier.",
            "Klasické šifry ako Caesar a Vigenere sú zaujímavé.",
            "Trénujeme model na šifrovanie a dešifrovanie s vysvetľovaním."
        ],
        "uk": [
            "Це тест навчання шифрів.",
            "Класичні шифри, як-от Цезаря та Віженера, цікаві для декодування.",
            "Ми навчаємо модель шифруванню та дешифруванню з поясненням."
        ],
    }

    cipher_types = ["Caesar", "Vigenere", "Substitution"]
    all_samples: List[dict] = []

    for lang, texts in base_texts.items():
        pairs_made = 0
        while pairs_made < args.pairs_per_lang:
            t = random.choice(texts)
            ctype = random.choice(cipher_types)
            all_samples.extend(make_pair(t, lang, ctype))
            pairs_made += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for obj in all_samples:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved {len(all_samples)} samples to {out_path}")


if __name__ == "__main__":
    main()

