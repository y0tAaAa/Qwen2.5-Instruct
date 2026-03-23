# src/data/generate_cipher_dataset_variants.py
import argparse, json, os, random
from typing import Dict, Any
from src.ciphers.ciphers_multi import ALPHABETS, encrypt, decrypt, substitution_make_key

CIPHERS = ["Caesar", "Vigenere", "Substitution"]
VARIANTS = ["known_params", "detect_solve", "noisy_params"]

TEXTS = {
    "en": [
        "This is a test of cipher learning.",
        "Classic ciphers like Caesar and Vigenere are fun to break.",
        "Attack at dawn. The quick brown fox jumps over the lazy dog.",
    ],
    "sk": [
        "Klasické šifry ako Caesar a Vigenere sú zaujímavé.",
        "Dnes testujem model na šifrovanie a dešifrovanie.",
        "Rýchla hnedá líška preskočí lenivého psa.",
    ],
    "uk": [
        "Класичні шифри як Цезар і Віженер цікаві для аналізу.",
        "Сьогодні тестую модель для шифрування та дешифрування.",
        "Швидкий бурий лис стрибає через ледачого собаку.",
    ],
}

def sample_key(cipher_type: str, lang: str, rng: random.Random):
    alpha = ALPHABETS[lang]
    if cipher_type == "Caesar":
        return rng.randint(1, len(alpha) - 1)  # int
    if cipher_type == "Vigenere":
        L = rng.randint(3, 8)
        return "".join(rng.choice(alpha) for _ in range(L))  # str
    if cipher_type == "Substitution":
        return substitution_make_key(lang, seed=rng.randint(0, 10_000_000))  # dict
    raise ValueError(cipher_type)

def round_trip_ok(cipher_type: str, lang: str, key_obj, plaintext: str) -> bool:
    c = encrypt(cipher_type, plaintext, key_obj, lang)
    p = decrypt(cipher_type, c, key_obj, lang)
    return p == plaintext

def key_to_string(cipher_type: str, key_obj) -> str:
    """В ДАТАСЕТЕ ключ ВСЕГДА строка."""
    if cipher_type == "Substitution":
        return json.dumps(key_obj, ensure_ascii=False, separators=(",", ":"))
    return str(key_obj)

def build_instruction(lang: str, mode: str, tv: str, cipher_type: str, key_str: str, text_in: str) -> str:
    head = (
        "### Instruction:\n"
        "You are CipherChat — a specialized assistant for encryption and decryption.\n"
        "Follow the instruction exactly.\n"
        "Use the pipeline: DETECT→SOLVE→VERIFY.\n"
        "Return ONLY valid JSON.\n"
        f"[{lang}] {mode.upper()}.\n"
    )

    if tv == "detect_solve":
        head += "Cipher: UNKNOWN\nKey: UNKNOWN\n"
    elif tv == "noisy_params":
        head += "Cipher: WRONG\nKey: WRONG\n"
    else:
        head += f"Cipher: {cipher_type}\nKey: {key_str}\n"

    head += (
        "Return ONLY valid JSON with fields: "
        "`mode`, `task_variant`, `cipher_type`, `key`, `cipher_text`, `plaintext`, "
        "`detected_cipher_type`, `detected_key`, `verify`, `reasoning`, `self_score` "
        "(self_score one of <grade_1>,<grade_2>,<grade_3>,<grade_4>,<grade_5>).\n"
        "Text:\n"
        f"{text_in}\n\n"
        "### Response (JSON only):\n"
    )
    return head

def build_sample(rng: random.Random) -> Dict[str, Any]:
    lang = rng.choice(["en", "sk", "uk"])
    cipher_type = rng.choice(CIPHERS)
    mode = rng.choice(["encrypt", "decrypt"])
    task_variant = rng.choice(VARIANTS)

    plaintext = rng.choice(TEXTS[lang])
    key_obj = sample_key(cipher_type, lang, rng)

    cipher_text = encrypt(cipher_type, plaintext, key_obj, lang)

    if mode == "encrypt":
        model_input_text = plaintext
        target_plain = plaintext
        target_cipher = cipher_text
        out_plain_for_json = cipher_text  # как у тебя раньше: plaintext поле хранит результат операции
    else:
        model_input_text = cipher_text
        target_plain = plaintext
        target_cipher = cipher_text
        out_plain_for_json = plaintext

    key_str = key_to_string(cipher_type, key_obj)

    verify = "round_trip_ok" if round_trip_ok(cipher_type, lang, key_obj, plaintext) else "round_trip_fail"

    out_json = {
        "mode": mode,
        "task_variant": task_variant,
        "cipher_type": cipher_type,
        "key": key_str,                  # <-- ВСЕГДА СТРОКА
        "cipher_text": target_cipher,
        "plaintext": out_plain_for_json,
        "detected_cipher_type": cipher_type,
        "detected_key": key_str,         # <-- ВСЕГДА СТРОКА
        "verify": verify,
        "reasoning": "DETECT→SOLVE→VERIFY: alphabet_check; params_infer; apply_cipher; round_trip_check",
        "self_score": "<grade_5>",
    }

    instr = build_instruction(lang, mode, task_variant, cipher_type, key_str, model_input_text)
    return {"instruction": instr, "json": out_json, "lang": lang}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/processed/train_cipher_variants.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = random.Random(args.seed)

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.num_samples):
            f.write(json.dumps(build_sample(rng), ensure_ascii=False) + "\n")

    print(f"Saved: {args.out} ({args.num_samples} samples)")

if __name__ == "__main__":
    main()

