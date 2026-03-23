"""
compare_all.py
--------------
Сравнение всех версий CipherChat: V1 / V2 / V3 / V4 (и любого числа других).

Каждый файл — результат eval_cipherchat.py.
Поддерживает от 2 до N файлов.

Запуск:
  python compare_all.py \
      --evals eval_results/eval300_v1.json \
              eval_results/eval300_v2.json \
              eval_results/eval300_v3.json \
              eval_results/eval300_v4.json \
      --names V1 V2 V3 V4 \
      --out   eval_results/comparison_all.txt
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

# ─────────────────────────────────────────────
# КОНСТАНТЫ
# ─────────────────────────────────────────────

CIPHERS = ["Caesar", "Vigenere", "Transposition"]
LANGS   = ["en", "sk", "uk"]
TASKS   = ["compute", "detect"]
SPLITS  = ["seen", "unseen"]

METRICS = [
    ("valid_json",   "Valid JSON"),
    ("algo_correct", "Algo correct"),
    ("output_exact", "Output exact"),
    ("cipher_acc",   "Cipher type"),
    ("key_acc",      "Key acc"),
]

# Поля в result dict (из eval_cipherchat.py compare_outputs)
RESULT_FIELD = {
    "valid_json":   "valid_json",
    "algo_correct": "algo_correct",
    "output_exact": "output_exact",
    "cipher_acc":   "cipher_type_acc",
    "key_acc":      "key_acc",
}

GREEN = "\033[92m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"


# ─────────────────────────────────────────────
# ЗАГРУЗКА
# ─────────────────────────────────────────────

def load_eval(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Поддержка как финального файла так и checkpoint файла
    if isinstance(data, list):
        return data
    if "results" in data:
        return data["results"]
    return []


def get_ft(r: dict) -> dict:
    return r.get("finetuned_model", {})


# ─────────────────────────────────────────────
# АГРЕГАЦИЯ
# ─────────────────────────────────────────────

class Agg:
    def __init__(self):
        self.n    = 0
        self.vals = defaultdict(int)

    def add(self, r: dict):
        self.n += 1
        ft = get_ft(r)
        for mk, rk in RESULT_FIELD.items():
            if ft.get(rk):
                self.vals[mk] += 1

    def pct(self, mk: str) -> float:
        return 100 * self.vals[mk] / max(self.n, 1)


def aggregate(results: List[dict], filter_fn=None) -> Agg:
    agg = Agg()
    for r in results:
        if filter_fn is None or filter_fn(r):
            agg.add(r)
    return agg


def task_of(r: dict) -> str:
    msgs = r.get("messages", [])
    for m in msgs:
        if m.get("role") == "user" and "TASK: DETECT" in m.get("content", ""):
            return "detect"
    # Fallback: check user_msg field if stored
    return "detect" if r.get("task_type") == "detect" else "compute"


# ─────────────────────────────────────────────
# ФОРМАТИРОВАНИЕ
# ─────────────────────────────────────────────

def fmt_cell(val: float, best: float, worst: float, plain: bool) -> str:
    s = f"{val:5.1f}%"
    if plain:
        return s
    if val == best and best != worst:
        return f"{GREEN}{s}{RESET}"
    if val == worst and best != worst:
        return f"{RED}{s}{RESET}"
    return s


def delta_str(val: float, ref: float, plain: bool) -> str:
    d = val - ref
    s = f"{'+'if d>=0 else ''}{d:.1f}%"
    if plain:
        return s
    color = GREEN if d >= 0.5 else (RED if d <= -0.5 else "")
    return f"{color}{s}{RESET}" if color else s


def print_table(title: str, rows: List[tuple], names: List[str],
                out_lines: List[str], plain: bool = False):
    """
    rows: list of (label, {name: float})
    """
    col_w = max(10, max(len(n) for n in names) + 2)
    label_w = max(20, max(len(r[0]) for r in rows) + 2)
    sep = "─" * (label_w + col_w * len(names) + 4)

    def emit(s=""):
        print(s)
        out_lines.append(s)

    emit(sep)
    bold_title = f"{BOLD}{title}{RESET}" if not plain else title
    emit(f"  {bold_title}")
    emit("  " + "─" * (label_w + col_w * len(names)))
    header = f"  {'Group':<{label_w}}" + "".join(f"{n:>{col_w}}" for n in names)
    emit(header)
    emit("  " + "─" * (label_w + col_w * len(names)))

    for label, vals in rows:
        vals_list = [vals.get(n, 0.0) for n in names]
        best  = max(vals_list)
        worst = min(vals_list)
        ref   = vals_list[0]  # сравниваем с первым (обычно V1)
        cells = []
        for i, (n, v) in enumerate(zip(names, vals_list)):
            cell = fmt_cell(v, best, worst, plain)
            if i > 0:
                cell += f" ({delta_str(v, ref, plain)})"
            cells.append(f"{cell:>{col_w}}")
        emit(f"  {label:<{label_w}}" + "".join(cells))

    emit()


# ─────────────────────────────────────────────
# ОСНОВНАЯ ЛОГИКА
# ─────────────────────────────────────────────

def build_report(all_results: Dict[str, List[dict]], names: List[str],
                 plain: bool = False) -> List[str]:
    out = []

    def section(title: str, rows):
        print_table(title, rows, names, out, plain)

    # ── OVERALL ──────────────────────────────
    aggs = {name: aggregate(all_results[name]) for name in names}
    rows = []
    for mk, label in METRICS:
        vals = {n: aggs[n].pct(mk) for n in names}
        rows.append((label, vals))
    section("OVERALL", rows)

    # ── BY SPLIT ─────────────────────────────
    for split in SPLITS:
        aggs_s = {name: aggregate(all_results[name],
                  lambda r, s=split: r.get("eval_split") == s) for name in names}
        rows = []
        for mk, label in METRICS:
            vals = {n: aggs_s[n].pct(mk) for n in names}
            rows.append((label, vals))
        section(f"SPLIT: {split}", rows)

    # ── BY CIPHER ────────────────────────────
    for cipher in CIPHERS:
        aggs_c = {name: aggregate(all_results[name],
                  lambda r, c=cipher: r.get("cipher_type") == c) for name in names}
        rows = []
        for mk, label in METRICS:
            vals = {n: aggs_c[n].pct(mk) for n in names}
            rows.append((label, vals))
        section(f"CIPHER: {cipher}", rows)

    # ── BY LANG ──────────────────────────────
    for lang in LANGS:
        aggs_l = {name: aggregate(all_results[name],
                  lambda r, l=lang: r.get("lang") == l) for name in names}
        rows = []
        for mk, label in METRICS:
            vals = {n: aggs_l[n].pct(mk) for n in names}
            rows.append((label, vals))
        section(f"LANG: {lang}", rows)

    # ── BY TASK ──────────────────────────────
    for task in TASKS:
        aggs_t = {name: aggregate(all_results[name],
                  lambda r, t=task: r.get("task_type") == t) for name in names}
        rows = []
        for mk, label in METRICS:
            vals = {n: aggs_t[n].pct(mk) for n in names}
            rows.append((label, vals))
        section(f"TASK: {task}", rows)

    # ── ALGO MATRIX: CIPHER × LANG ────────────────
    print("  " + "─" * 60)
    print(f"  {BOLD}ALGO CORRECT: cipher × lang matrix{RESET}")
    print("  " + "─" * 60)
    out.append("ALGO CORRECT: cipher × lang matrix")
    header = f"  {'':20}" + "".join(f"{'':>6}{n:>8}" for n in names)
    print(header); out.append(header)
    for cipher in CIPHERS:
        for lang in LANGS:
            aggs_cl = {name: aggregate(all_results[name],
                lambda r, c=cipher, l=lang:
                    r.get("cipher_type") == c and r.get("lang") == l)
                for name in names}
            vals = {n: aggs_cl[n].pct("algo_correct") for n in names}
            vals_list = [vals[n] for n in names]
            best = max(vals_list); worst = min(vals_list); ref = vals_list[0]
            cells = []
            for i, (n, v) in enumerate(zip(names, vals_list)):
                cell = fmt_cell(v, best, worst, plain)
                if i > 0:
                    cell += f"({delta_str(v, ref, plain)})"
                cells.append(f"{cell:>14}")
            label = f"{cipher}×{lang}"
            print(f"  {label:<20}" + "".join(cells))
            out.append(f"  {label:<20}" + "".join([f"{vals[n]:>7.1f}%" for n in names]))
    print(); out.append("")

    # ── SUMMARY ──────────────────────────────
    print("  " + "═" * 60)
    print(f"  {BOLD}SUMMARY — algo_correct{RESET}")
    print("  " + "─" * 60)
    out += ["=" * 60, "SUMMARY — algo_correct", "-" * 60]
    overall = {n: aggregate(all_results[n]).pct("algo_correct") for n in names}
    best_name = max(names, key=lambda n: overall[n])
    for n in names:
        marker = " ← BEST" if n == best_name else ""
        line = f"  {n:>6}: {overall[n]:.1f}%{marker}"
        print(line); out.append(line)
    print(); out.append("")

    return out


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True,
                    help="Пути к JSON файлам eval (в порядке V1, V2, ...)")
    ap.add_argument("--names", nargs="+",
                    help="Имена версий (по умолчанию V1, V2, ...)")
    ap.add_argument("--out",   default=None,
                    help="Путь для сохранения plain-text отчёта")
    args = ap.parse_args()

    if args.names is None:
        args.names = [f"V{i+1}" for i in range(len(args.evals))]

    if len(args.names) != len(args.evals):
        raise ValueError("--names и --evals должны иметь одинаковое количество элементов")

    print(f"\n{'='*80}")
    print(f"  {BOLD}CIPHERCHAT COMPARISON: {' vs '.join(args.names)}{RESET}")
    print(f"{'='*80}")
    for name, path in zip(args.names, args.evals):
        print(f"  {name:>6} : {path}")
    print(f"{'='*80}\n")

    all_results = {}
    for name, path in zip(args.names, args.evals):
        results = load_eval(path)
        print(f"  Loaded {name}: {len(results)} samples from {path}")
        all_results[name] = results

    print()
    out_lines = build_report(all_results, args.names, plain=False)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        plain_lines = build_report(all_results, args.names, plain=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(plain_lines))
        print(f"✅ Plain-text report → {args.out}")


if __name__ == "__main__":
    main()
