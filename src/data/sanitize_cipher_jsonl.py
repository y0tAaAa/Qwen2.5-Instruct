import argparse
import json
import sys
from typing import Any, Dict

def to_str_key(x: Any) -> str:
    # key может быть int/float/str/dict/list
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)
    # dict/list/etc -> сериализуем
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))

def to_reasoning_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))

def sanitize_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    # ожидаем формат: {"instruction": ..., "json": {...}}
    if "json" not in obj or not isinstance(obj["json"], dict):
        raise ValueError("Missing or invalid 'json' field")

    j = obj["json"]

    # key -> always string
    if "key" in j:
        j["key"] = to_str_key(j["key"])

    # reasoning -> always string (чтобы не было struct/string конфликтов)
    if "reasoning" in j:
        j["reasoning"] = to_reasoning_str(j["reasoning"])

    # self_score -> string
    if "self_score" in j and not isinstance(j["self_score"], str):
        j["self_score"] = str(j["self_score"])

    # cipher_type/mode/lang -> string (на всякий)
    for f in ["cipher_type", "mode", "lang"]:
        if f in j and not isinstance(j[f], str):
            j[f] = str(j[f])

    obj["json"] = j
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="input jsonl")
    ap.add_argument("--out", required=True, help="output sanitized jsonl")
    ap.add_argument("--skip_bad", action="store_true", help="skip bad lines instead of failing")
    args = ap.parse_args()

    n_in = 0
    n_out = 0
    n_bad = 0

    with open(args.inp, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                obj = json.loads(line)
                obj = sanitize_obj(obj)
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_out += 1
            except Exception as e:
                n_bad += 1
                msg = f"[BAD LINE {line_no}] {e}\n"
                sys.stderr.write(msg)
                if not args.skip_bad:
                    raise

    print(f"✅ sanitized: in={n_in}, out={n_out}, bad={n_bad}")
    print(f"   wrote -> {args.out}")

if __name__ == "__main__":
    main()
