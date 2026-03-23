import json
import argparse
from collections import Counter

def validate(path: str, max_show: int = 5):
    key_types = Counter()
    bad_key = 0
    bad_json = 0
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad_json += 1
                if bad_json <= max_show:
                    print(f"[BAD JSON] line {i}: {line[:120]}")
                continue

            j = obj.get("json", {})
            k = j.get("key", None)
            key_types[type(k).__name__] += 1
            if not isinstance(k, str):
                bad_key += 1
                if bad_key <= max_show:
                    print(f"[BAD KEY TYPE] line {i}: type={type(k)} key={k}")

    print(f"\nFile: {path}")
    print(f"Total lines: {total}")
    print(f"Bad JSON:    {bad_json}")
    print(f"Key types:   {dict(key_types)}")
    print(f"Bad key(not str): {bad_key}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    args = ap.parse_args()
    validate(args.path)

if __name__ == "__main__":
    main()

