#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Берёт чистый train_cipher_train.jsonl и расширяет его примерами с
намеренно испорченными ответами + разными self_score (1–4).

Вход:  data/processed/train_cipher_train.jsonl
Выход: data/processed/train_cipher_train_selfscore.jsonl
"""

import os
import json
import random
import copy
import argparse

GRADE_TOKENS = ["<grade_1>", "<grade_2>", "<grade_3>", "<grade_4>", "<grade_5>"]


def corrupt_field(field: str, value, obj_json: dict) -> any:
    """Как именно ломаем конкретное поле."""
    if field == "cipher_type":
        all_types = ["Caesar", "Vigenere", "Substitution"]
        cur = str(value)
        others = [t for t in all_types if t != cur]
        return random.choice(others) if others else cur

    if field == "key":
        v = str(value)
        if not v:
            return "BADKEY"
        # простая порча: переворот или добавление мусора
        if len(v) > 3:
            return v[::-1]
        return v + "_x"

    if field in ("cipher_text", "plaintext"):
        s = str(value)
        if len(s) < 4:
            return "XXX" + s
        # перемешать куски
        mid = len(s) // 2
        return s[mid:] + s[:mid]

    # по умолчанию просто вернуть то же
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("data", "processed", "train_cipher_train.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "processed", "train_cipher_train_selfscore.jsonl"),
    )
    parser.add_argument(
        "--error_rate",
        type=float,
        default=0.2,
        help="Доля примеров, для которых будет сгенерирована доп. ошибочная версия.",
    )
    args = parser.parse_args()

    random.seed(42)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    total_in = 0
    total_out = 0
    total_noisy = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_in += 1
            obj = json.loads(line)

            base = copy.deepcopy(obj)
            # гарантируем, что чистый пример имеет self_score = <grade_5>
            base["json"]["self_score"] = "<grade_5>"
            json.dump(base, fout, ensure_ascii=False)
            fout.write("\n")
            total_out += 1

            # с вероятностью error_rate создаём шумную версию
            if random.random() > args.error_rate:
                continue

            noisy = copy.deepcopy(base)
            j = noisy["json"]

            # выбираем grade 1–4
            grade = random.randint(1, 4)
            j["self_score"] = f"<grade_{grade}>"

            # сколько полей ломать: чем меньше grade, тем больше полей
            fields = ["cipher_type", "key", "cipher_text", "plaintext"]
            num_errors = min(5 - grade, len(fields))  # 4→1 поле, 1→4 поля
            random.shuffle(fields)
            to_corrupt = fields[:num_errors]

            for f_name in to_corrupt:
                old_val = j.get(f_name, "")
                j[f_name] = corrupt_field(f_name, old_val, j)

            # немного пометим reasoning
            j["reasoning"] = (
                "This answer is partially or fully incorrect; self_score reflects its quality."
            )

            json.dump(noisy, fout, ensure_ascii=False)
            fout.write("\n")
            total_out += 1
            total_noisy += 1

    print(
        f"Input: {total_in} clean samples, "
        f"output: {total_out} (including {total_noisy} noisy variants) "
        f"→ {args.output}"
    )


if __name__ == "__main__":
    main()

