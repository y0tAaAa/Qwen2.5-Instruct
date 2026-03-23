import json
import argparse
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    args = ap.parse_args()

    c_lang = Counter()
    c_cipher = Counter()
    c_mode = Counter()
    c_combo = Counter()

    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lang = obj.get("lang", "??")
            j = obj.get("json", {})
            cipher = j.get("cipher_type", "??")
            mode = j.get("mode", "??")
            c_lang[lang] += 1
            c_cipher[cipher] += 1
            c_mode[mode] += 1
            c_combo[(lang, cipher, mode)] += 1

    print("\nLanguages:")
    for k, v in c_lang.most_common():
        print(f"  {k}: {v}")

    print("\nCipher types:")
    for k, v in c_cipher.most_common():
        print(f"  {k}: {v}")

    print("\nModes:")
    for k, v in c_mode.most_common():
        print(f"  {k}: {v}")

    print("\n(lang, cipher, mode) top 20:")
    for k, v in c_combo.most_common(20):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()

