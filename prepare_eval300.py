"""
prepare_eval300.py
------------------
Формирует eval-сет из 300 примеров:
  - 150 из train.json  (seen   — проверяем запоминание)
  - 150 из val.json    (unseen — проверяем обобщение)

Стратифицированная выборка: равномерно по cipher_type × lang × task_type.

Запуск:
  python prepare_eval300.py \
      --train_file data/train.json \
      --val_file   data/val.json   \
      --out        data/eval300.json
"""

import argparse
import json
import random
from collections import defaultdict


def get_stratum(item: dict) -> str:
    """Ключ стратификации: cipher_type × lang × task_type."""
    msg_user = item["messages"][1]["content"]
    ans = json.loads(item["messages"][2]["content"])
    task = "detect" if "TASK: DETECT" in msg_user else "compute"
    return f"{ans['cipher_type']}|{ans['lang']}|{task}"


def stratified_sample(data: list, n: int, seed: int) -> list:
    """
    Берёт n примеров из data равномерно по стратам.
    Если страт не хватает — добирает случайно.
    """
    rng = random.Random(seed)

    # Группируем по стратам
    buckets = defaultdict(list)
    for i, item in enumerate(data):
        buckets[get_stratum(item)].append(i)

    strata = sorted(buckets.keys())
    n_strata = len(strata)
    per_stratum = max(1, n // n_strata)

    chosen_indices = set()
    for stratum in strata:
        pool = buckets[stratum]
        k = min(per_stratum, len(pool))
        chosen_indices.update(rng.sample(pool, k))

    # Доберём если не хватает
    if len(chosen_indices) < n:
        remaining = [i for i in range(len(data)) if i not in chosen_indices]
        rng.shuffle(remaining)
        chosen_indices.update(remaining[: n - len(chosen_indices)])

    # Обрежем если лишнее (из-за округлений)
    chosen = sorted(chosen_indices)[:n]
    return [data[i] for i in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="data/train.json")
    ap.add_argument("--val_file",   default="data/val.json")
    ap.add_argument("--out",        default="data/eval300.json")
    ap.add_argument("--n_seen",     type=int, default=150,
                    help="Samples from train (seen during training)")
    ap.add_argument("--n_unseen",   type=int, default=150,
                    help="Samples from val (unseen during training)")
    ap.add_argument("--seed",       type=int, default=99)
    args = ap.parse_args()

    print(f"Loading train: {args.train_file}")
    with open(args.train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print(f"Loading val  : {args.val_file}")
    with open(args.val_file, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    print(f"Sampling {args.n_seen} seen (from train)...")
    seen_samples = stratified_sample(train_data, args.n_seen, seed=args.seed)
    for s in seen_samples:
        s["_eval_split"] = "seen"

    print(f"Sampling {args.n_unseen} unseen (from val)...")
    unseen_samples = stratified_sample(val_data, args.n_unseen, seed=args.seed + 1)
    for s in unseen_samples:
        s["_eval_split"] = "unseen"

    eval_data = seen_samples + unseen_samples
    random.Random(args.seed).shuffle(eval_data)

    # Статистика по стратам
    from collections import Counter
    strata_seen   = Counter(get_stratum(s) for s in seen_samples)
    strata_unseen = Counter(get_stratum(s) for s in unseen_samples)

    print(f"\nEval300 composition:")
    print(f"  seen   (train): {len(seen_samples)}")
    print(f"  unseen (val)  : {len(unseen_samples)}")
    print(f"  total         : {len(eval_data)}")

    print(f"\nStratum distribution (seen):")
    for k in sorted(strata_seen):
        print(f"  {k:<35} : {strata_seen[k]}")

    print(f"\nStratum distribution (unseen):")
    for k in sorted(strata_unseen):
        print(f"  {k:<35} : {strata_unseen[k]}")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved → {args.out} ({len(eval_data)} samples)")


if __name__ == "__main__":
    main()
