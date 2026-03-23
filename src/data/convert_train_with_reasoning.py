#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Конвертация старого train_with_reasoning.jsonl в новый формат
с encrypt/decrypt и полем self_score.

Вход:  data/raw/train_with_reasoning.jsonl
Выход: data/processed/train_cipher_converted.jsonl
"""

import json
import os
import argparse
from typing import Dict, Any

DEFAULT_SELF_SCORE = "<grade_5>"


def detect_lang(instruction: str) -> str:
    first = instruction.strip().splitlines()[0]
    if first.startswith("Розшифруй") or "Поверни" in first:
        return "uk"
    if first.startswith("Dešifruj") or "Vráť" in first:
        return "sk"
    return "en"


def build_instruction(lang: str, mode: str, cipher_type: str, key, text: str) -> str:
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


def convert_one(obj: Dict[str, Any]) -> (Dict[str, Any], Dict[str, Any]):
    instr = obj.get("instruction", "")
    j = obj.get("json", {})

    cipher_type = str(j.get("cipher_type", "Unknown"))
    key = j.get("key", "")
    cipher_text = j.get("cipher_text", "")
    plaintext = j.get("plaintext", "")
    reasoning = j.get("reasoning", "")

    lang = detect_lang(instr)

    enc_ins = build_instruction(lang, "encrypt", cipher_type, key, plaintext)
    dec_ins = build_instruction(lang, "decrypt", cipher_type, key, cipher_text)

    enc_json = {
        "mode": "encrypt",
        "cipher_type": cipher_type,
        "key": key,
        "cipher_text": cipher_text,
        "plaintext": plaintext,
        "reasoning": reasoning or f"Applied {cipher_type} cipher with key {key}.",
        "self_score": DEFAULT_SELF_SCORE,
    }
    dec_json = {
        "mode": "decrypt",
        "cipher_type": cipher_type,
        "key": key,
        "cipher_text": cipher_text,
        "plaintext": plaintext,
        "reasoning": reasoning or f"Recovered plaintext by {cipher_type} decryption with key {key}.",
        "self_score": DEFAULT_SELF_SCORE,
    }

    return (
        {"instruction": enc_ins, "json": enc_json},
        {"instruction": dec_ins, "json": dec_json},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("data", "raw", "train_with_reasoning.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "processed", "train_cipher_converted.jsonl"),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    total_in = 0
    total_out = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_in += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                enc_obj, dec_obj = convert_one(obj)
            except Exception:
                continue
            for x in (enc_obj, dec_obj):
                json.dump(x, fout, ensure_ascii=False)
                fout.write("\n")
                total_out += 1

    print(f"Input lines: {total_in}, output lines: {total_out}")


if __name__ == "__main__":
    main()

