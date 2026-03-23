"""
compare_evals.py
----------------
Сравнивает v1 vs v2 по ВСЕМ срезам:
  - Overall
  - By split (seen / unseen)
  - By cipher type
  - By language
  - By task type
  - By cipher × language  (9 комбинаций)
  - By cipher × task      (6 комбинаций)
  - By language × task    (6 комбинаций)

Запуск:
  python compare_evals.py \
      --v1 eval_results/eval300_20260308_201750.json \
      --v2 eval_results/eval300_XXXXXXXX_XXXXXX.json
"""

import argparse
import json
import os
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

# ANSI цвета
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


# ─────────────────────────────────────────────
# УТИЛИТЫ
# ─────────────────────────────────────────────

def parse_pct(s) -> float:
    """Извлекает float из '163/300 (54.3%)' или просто числа."""
    if isinstance(s, (int, float)):
        return float(s)
    try:
        return float(str(s).split("(")[1].rstrip("%)"))
    except Exception:
        return 0.0


def delta_str(v1: float, v2: float) -> str:
    d = v2 - v1
    if d >= 0.5:
        return f"{GREEN}+{d:.1f}%{RESET}"
    elif d <= -0.5:
        return f"{RED}{d:.1f}%{RESET}"
    else:
        return f"{YELLOW}{d:+.1f}%{RESET}"


def get_ft(block: Dict, key: str, metric: str) -> float:
    """Достаёт fine-tuned метрику из блока summary."""
    ft = block.get("ft", block.get("finetuned", {}))
    return parse_pct(ft.get(key, {}).get(metric, "0/0 (0%)"))


def get_base(block: Dict, key: str, metric: str) -> float:
    base = block.get("base", {})
    return parse_pct(base.get(key, {}).get(metric, "0/0 (0%)"))


def get_n(block: Dict, key: str) -> int:
    ft = block.get("ft", block.get("finetuned", {}))
    val = ft.get(key, {}).get("total", 0)
    return int(val) if val else 0


# ─────────────────────────────────────────────
# ПЕЧАТЬ
# ─────────────────────────────────────────────

W = 90

def print_header(title: str):
    print(f"\n{'─'*W}")
    print(f"  {BOLD}{title}{RESET}")
    print(f"  {'─'*(W-2)}")
    print(f"  {'Group':<20} {'Metric':<16} {'V1 base':>10} {'V1 ft':>10} {'V2 ft':>10} {'Delta':>12} {'N':>5}")
    print(f"  {'─'*(W-2)}")


def print_row(group: str, label: str,
              v1_base: float, v1_ft: float, v2_ft: float, n: int,
              first: bool):
    g = group if first else ""
    d = delta_str(v1_ft, v2_ft)
    print(f"  {g:<20} {label:<16} "
          f"{v1_base:>8.1f}%  "
          f"{v1_ft:>8.1f}%  "
          f"{v2_ft:>8.1f}%  "
          f"{d:>18}  "
          f"{n:>5}")


def print_section(title: str, s1: Dict, s2: Dict,
                  section_key: str, keys: List[str]):
    """
    Печатает одну секцию сравнения.
    section_key: 'by_cipher', 'by_lang', etc.
    """
    b1 = s1.get(section_key, {})
    b2 = s2.get(section_key, {})

    print_header(title)
    for key in keys:
        n = get_n(b2, key)
        if n == 0:
            n = get_n(b1, key)

        first = True
        for attr, label in METRICS:
            v1_base = get_base(b1, key, attr)
            v1_ft   = get_ft(b1,   key, attr)
            v2_ft   = get_ft(b2,   key, attr)
            print_row(key, label, v1_base, v1_ft, v2_ft, n if first else 0, first)
            first = False
        print(f"  {'·'*(W-2)}")


# ─────────────────────────────────────────────
# МАТРИЦА algo_correct
# ─────────────────────────────────────────────

def print_matrix(title: str, s1: Dict, s2: Dict,
                 section_key: str, rows: List[str], cols: List[str],
                 sep: str = "×"):
    """
    Печатает матрицу одной метрики (algo_correct) для кросс-срезов.
    """
    b1 = s1.get(section_key, {})
    b2 = s2.get(section_key, {})

    print(f"\n{'─'*W}")
    print(f"  {BOLD}{title}  [algo_correct %]{RESET}")
    print(f"  {'─'*(W-2)}")

    col_w = 22
    hdr = f"  {'':20}" + "".join(f"{c:>{col_w}}" for c in cols)
    print(hdr)
    print(f"  {'─'*(W-2)}")

    for row in rows:
        row_str = f"  {row:<20}"
        for col in cols:
            key = f"{row}{sep}{col}"
            v1 = get_ft(b1, key, "algo_correct")
            v2 = get_ft(b2, key, "algo_correct")
            d  = v2 - v1
            if d >= 0.5:
                cell = f"{GREEN}{v2:.0f}%(+{d:.0f}){RESET}"
            elif d <= -0.5:
                cell = f"{RED}{v2:.0f}%({d:.0f}){RESET}"
            else:
                cell = f"{v2:.0f}%({d:+.0f})"
            row_str += f"{cell:>{col_w}}"
        print(row_str)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True, help="Path to V1 eval JSON")
    ap.add_argument("--v2", required=True, help="Path to V2 eval JSON")
    ap.add_argument("--no_color", action="store_true",
                    help="Disable ANSI colors (for log files)")
    args = ap.parse_args()

    if args.no_color:
        global GREEN, RED, YELLOW, RESET, BOLD
        GREEN = RED = YELLOW = RESET = BOLD = ""

    with open(args.v1, "r", encoding="utf-8") as f:
        r1 = json.load(f)
    with open(args.v2, "r", encoding="utf-8") as f:
        r2 = json.load(f)

    s1 = r1["summary"]
    s2 = r2["summary"]

    # ── Шапка ────────────────────────────────────────────────
    print("=" * W)
    print(f"  {BOLD}CIPHERCHAT EVAL COMPARISON: V1 vs V2{RESET}")
    print("=" * W)
    print(f"  V1 adapter : {r1['meta'].get('adapter', 'n/a')}")
    print(f"  V2 adapter : {r2['meta'].get('adapter', 'n/a')}")
    print(f"  V1 data    : {r1['meta'].get('data', 'n/a')}  (n={r1['meta'].get('n_total','?')})")
    print(f"  V2 data    : {r2['meta'].get('data', 'n/a')}  (n={r2['meta'].get('n_total','?')})")
    print(f"\n  Columns: V1 base = Qwen without adapter")
    print(f"           V1 ft   = after 1st training (2 epochs, lr=2e-4)")
    print(f"           V2 ft   = after focused fine-tune (3 epochs, lr=5e-5)")
    print(f"           Delta   = V2 ft − V1 ft")

    # ── 1. OVERALL ───────────────────────────────────────────
    print_header("OVERALL")
    n_total = r2["meta"].get("n_total", r1["meta"].get("n_total", "?"))
    first = True
    for attr, label in METRICS:
        v1_base = parse_pct(s1["overall"]["base"].get(attr, 0))
        v1_ft   = parse_pct(s1["overall"]["finetuned"].get(attr, 0))
        v2_ft   = parse_pct(s2["overall"]["finetuned"].get(attr, 0))
        print_row("overall", label, v1_base, v1_ft, v2_ft, n_total if first else 0, first)
        first = False

    # ── 2. BY SPLIT ──────────────────────────────────────────
    print_section("BY SPLIT  [seen=train | unseen=val]",
                  s1, s2, "by_split", SPLITS)

    # ── 3. BY CIPHER ─────────────────────────────────────────
    print_section("BY CIPHER TYPE",
                  s1, s2, "by_cipher", CIPHERS)

    # ── 4. BY LANGUAGE ───────────────────────────────────────
    print_section("BY LANGUAGE",
                  s1, s2, "by_lang", LANGS)

    # ── 5. BY TASK ───────────────────────────────────────────
    print_section("BY TASK TYPE",
                  s1, s2, "by_task", TASKS)

    # ── 6. BY CIPHER × LANGUAGE ──────────────────────────────
    cl_keys = [f"{c}×{l}" for c in CIPHERS for l in LANGS]
    print_section("BY CIPHER × LANGUAGE",
                  s1, s2, "by_cipher_lang", cl_keys)

    # ── 7. BY CIPHER × TASK ──────────────────────────────────
    ct_keys = [f"{c}×{t}" for c in CIPHERS for t in TASKS]
    print_section("BY CIPHER × TASK",
                  s1, s2, "by_cipher_task", ct_keys)

    # ── 8. BY LANGUAGE × TASK ────────────────────────────────
    lt_keys = [f"{l}×{t}" for l in LANGS for t in TASKS]
    print_section("BY LANGUAGE × TASK",
                  s1, s2, "by_lang_task", lt_keys)

    # ── 9. МАТРИЦЫ algo_correct ──────────────────────────────
    print_matrix("CIPHER × LANGUAGE matrix",
                 s1, s2, "by_cipher_lang",
                 rows=CIPHERS, cols=LANGS, sep="×")

    print_matrix("CIPHER × TASK matrix",
                 s1, s2, "by_cipher_task",
                 rows=CIPHERS, cols=TASKS, sep="×")

    print_matrix("LANGUAGE × TASK matrix",
                 s1, s2, "by_lang_task",
                 rows=LANGS, cols=TASKS, sep="×")

    print(f"\n{'='*W}")
    print(f"  {GREEN}Green{RESET} = improvement ≥0.5% | {RED}Red{RESET} = degradation ≤-0.5%")
    print(f"  Matrix cells: V2_ft%(delta vs V1_ft)")
    print(f"{'='*W}\n")

    # ── Сохранение текстового отчёта ─────────────────────────
    out_dir = os.path.dirname(args.v2) or "eval_results"
    out_path = os.path.join(out_dir, "comparison_v1_vs_v2.txt")

    # Перезапускаем без цветов для файла
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, __file__,
         "--v1", args.v1, "--v2", args.v2, "--no_color"],
        capture_output=True, text=True
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)
    print(f"✅ Plain-text report saved → {out_path}")


if __name__ == "__main__":
    main()
