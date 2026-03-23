# src/data/split_train_val.py
import argparse, json, os, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train_out", required=True)
    ap.add_argument("--val_out", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.val_out), exist_ok=True)

    rng = random.Random(args.seed)
    lines = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    rng.shuffle(lines)
    n = len(lines)
    n_val = max(1, int(n * args.val_ratio))
    val = lines[:n_val]
    train = lines[n_val:]

    with open(args.train_out, "w", encoding="utf-8") as f:
        f.write("\n".join(train) + "\n")
    with open(args.val_out, "w", encoding="utf-8") as f:
        f.write("\n".join(val) + "\n")

    print(f"Total: {n} | Train: {len(train)} | Val: {len(val)}")
    print(f"Saved train → {args.train_out}")
    print(f"Saved val   → {args.val_out}")

if __name__ == "__main__":
    main()

