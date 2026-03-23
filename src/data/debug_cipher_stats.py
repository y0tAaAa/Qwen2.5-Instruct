#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from collections import Counter


def detect_lang(instruction: str) -> str:
    """
    Язык определяем по первой НЕ-### строке в instruction.
    """
    lines = [l for l in instruction.splitlines() if l.strip()]
    text_line = ""
    for l in lines:
        if not l.startswith("###"):
            text_line = l
            break
    t = text_line
    if t.startswith("Зашифруй") or t.startswith("Розшифруй"):
        return "uk"
    if t.startswith("Zašifruj") or t.startswith("Dešifruj"):
        return "sk"
    if t.startswith("Encrypt") or t.startswith("Decrypt"):
        return "en"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join("data", "processed", "train_cipher_train.jsonl"),
        help="Путь к jsonl (train или train_cipher_multi).",
    )
    args = parser.parse_args()

    c_cipher = Counter()
    c_mode = Counter()
    c_lang = Counter()
    c_combo = Counter()

    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            instr = obj["instruction"]
            j = obj["json"]

            lang = detect_lang(instr)
            cipher_type = j.get("cipher_type", "Unknown")
            mode = j.get("mode", "Unknown")

            c_cipher[cipher_type] += 1
            c_mode[mode] += 1
            c_lang[lang] += 1
            c_combo[(lang, cipher_type, mode)] += 1

    print(f"Stats for: {args.path}\n")

    print("Languages:")
    for k, v in c_lang.most_common():
        print(f"  {k}: {v}")

    print("\nCipher types:")
    for k, v in c_cipher.most_common():
        print(f"  {k}: {v}")

    print("\nModes:")
    for k, v in c_mode.most_common():
        print(f"  {k}: {v}")

    print("\n(lang, cipher_type, mode):")
    for (lang, ct, mode), v in sorted(c_combo.items()):
        print(f"  ({lang}, {ct}, {mode}): {v}")


if __name__ == "__main__":
    main()

