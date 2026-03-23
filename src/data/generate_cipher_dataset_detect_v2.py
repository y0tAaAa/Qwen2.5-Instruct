import json
import random
import argparse
import os
import math
from collections import Counter
from typing import Dict, Any, Tuple, List

def set_seed(seed: int):
    random.seed(seed)

CORPUS = {
    "en": [
        "Classic ciphers like Caesar and Vigenere are fun to break.",
        "Machine learning can learn algorithmic rules very effectively.",
        "The quick brown fox jumps over the lazy dog.",
        "Encryption and decryption are deterministic transforms.",
        "We are training a reasoning model for cipher tasks.",
        "Cryptography is the art of writing and solving codes.",
        "Advanced models can understand and solve classical ciphers.",
        "Security through obscurity is not real security.",
        "Vigenere cipher was considered unbreakable for centuries.",
        "Substitution ciphers are vulnerable to frequency analysis."
    ],
    "sk": [
        "Klasické šifry ako Caesar a Vigenere sú zaujímavé.",
        "Strojové učenie môže naučiť algoritmické pravidlá.",
        "Rýchla hnedá líška skáče cez lenivého psa.",
        "Dešifrovanie je dôležitá súčasť kybernetickej bezpečnosti.",
        "Trénujeme model na šifrovanie a dešifrovanie s vysvetlením.",
        "Vigenereho šifra bola dlho považovaná za nezlomiteľnú."
    ],
    "uk": [
        "Класичні шифри як Цезар і Віженер цікаві для вивчення.",
        "Машинне навчання може вивчити алгоритмічні правила.",
        "Швидкий бурий лис стрибає через ледачого собаку.",
        "Шифрування та дешифрування — детерміновані перетворення.",
        "Ми навчаємо модель шифруванню та дешифруванню з поясненням.",
        "Шифр Віженера довго вважався незламним."
    ]
}

ALPHABETS = {
    "en": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "sk": "AÁÄBCČDĎEÉFGHIÍJKLĹĽMNŇOÓÔPQRŔSŠTŤUÚVWXYÝZŽ",
    "uk": "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
}

KEY_VOCAB = {
    "en": ["KEY", "CIPHER", "SECRET", "VECTOR", "MODEL", "TRAIN", "SHIFT", "ALPHA"],
    "sk": ["TAJNE", "HESLO", "KLUČ", "SIFRA", "MODEL", "DATA", "POSUN", "ZAMEK"],
    "uk": ["КЛЮЧ", "СЕКРЕТ", "ШИФР", "МОДЕЛЬ", "ДАНІ", "ПОСУВ", "АЛФА", "ЗЛАМ"]
}

def normalize(text: str) -> str:
    return text.upper()

def alpha_maps(lang: str):
    alpha = ALPHABETS[lang]
    idx = {ch: i for i, ch in enumerate(alpha)}
    return alpha, idx

def caesar(text: str, key: int, lang: str, encrypt: bool = True) -> str:
    alpha, idx = alpha_maps(lang)
    n = len(alpha)
    shift = key % n
    if not encrypt:
        shift = -shift
    out = []
    for ch in normalize(text):
        out.append(alpha[(idx[ch] + shift) % n] if ch in idx else ch)
    return "".join(out)

def vigenere(text: str, key: str, lang: str, encrypt: bool = True) -> str:
    alpha, idx = alpha_maps(lang)
    n = len(alpha)
    shifts = [idx[c] for c in normalize(key) if c in idx] or [0]
    out = []
    k = 0
    for ch in normalize(text):
        if ch in idx:
            s = shifts[k % len(shifts)]
            out.append(alpha[(idx[ch] + (s if encrypt else -s)) % n])
            k += 1
        else:
            out.append(ch)
    return "".join(out)

def subst_key_perm(lang: str) -> str:
    alpha = ALPHABETS[lang]
    perm = list(alpha)
    random.shuffle(perm)
    return "".join(perm)

def substitution(text: str, key_perm: str, lang: str, encrypt: bool = True) -> str:
    alpha, _ = alpha_maps(lang)
    fwd = {a: b for a, b in zip(alpha, key_perm)}
    rev = {b: a for a, b in zip(alpha, key_perm)}
    mp = fwd if encrypt else rev
    out = []
    for ch in normalize(text):
        out.append(mp.get(ch, ch))
    return "".join(out)

def sample_plain(lang: str) -> str:
    k = random.choice([1, 2, 3])
    sents = random.sample(CORPUS[lang], k=k)
    return normalize(" ".join(sents))

# ----- language score (простая LM по униграммам из CORPUS) -----
def build_logp(lang: str):
    alpha = ALPHABETS[lang]
    idx = {c: i for i, c in enumerate(alpha)}
    cnt = Counter()
    total = 0
    for s in CORPUS[lang]:
        for ch in normalize(s):
            if ch in idx:
                cnt[ch] += 1
                total += 1
    V = len(alpha)
    logp = {c: math.log((cnt[c] + 1) / (total + V)) for c in alpha}
    unk = math.log(1 / (total + V))
    return logp, unk

LOGP = {lang: build_logp(lang) for lang in CORPUS}

def score_text(pt: str, lang: str) -> float:
    alpha, idx = alpha_maps(lang)
    logp, unk = LOGP[lang]
    s = 0.0
    for ch in pt:
        if ch in idx:
            s += logp.get(ch, unk)
    return s

def ioc(text: str, lang: str) -> float:
    alpha, idx = alpha_maps(lang)
    counts = Counter(ch for ch in text if ch in idx)
    N = sum(counts.values())
    if N <= 1:
        return 0.0
    num = sum(v * (v - 1) for v in counts.values())
    den = N * (N - 1)
    return num / den

# ----- DETECT/SOLVE гипотезы -----
def best_caesar(ciphertext: str, lang: str) -> Tuple[int, str, float]:
    alpha = ALPHABETS[lang]
    best = None
    for k in range(1, len(alpha)):
        pt = caesar(ciphertext, k, lang, encrypt=False)
        sc = score_text(pt, lang)
        if best is None or sc > best[2]:
            best = (k, pt, sc)
    return best  # (key, plaintext, score)

def best_vigenere_from_vocab(ciphertext: str, lang: str) -> Tuple[str, str, float]:
    best = None
    for key in KEY_VOCAB[lang]:
        pt = vigenere(ciphertext, key, lang, encrypt=False)
        sc = score_text(pt, lang)
        if best is None or sc > best[2]:
            best = (key, pt, sc)
    return best  # (key, plaintext, score)

def make_full_crib(lang: str, key_perm: str) -> Tuple[str, str]:
    alpha = ALPHABETS[lang]
    letters = list(alpha)
    random.shuffle(letters)
    base = "".join(letters)  # покрытие всего алфавита
    extra = "".join(random.choice(alpha) for _ in range(max(0, len(alpha) // 2)))
    crib_plain = base + extra
    crib_cipher = substitution(crib_plain, key_perm, lang, encrypt=True)
    return crib_plain, crib_cipher

def derive_perm_from_crib(crib_plain: str, crib_cipher: str, lang: str) -> Tuple[str, bool]:
    alpha = ALPHABETS[lang]
    mapping = {}
    for p, c in zip(crib_plain, crib_cipher):
        if p in alpha and c in alpha:
            mapping[p] = c
    perm = "".join(mapping.get(a, "?") for a in alpha)
    ok = ("?" not in perm) and (len(set(perm)) == len(alpha))
    return perm, ok

def grade(ok: bool, margin: float) -> str:
    if not ok:
        return "<grade_2>"
    if margin > 2.0:
        return "<grade_5>"
    if margin > 1.0:
        return "<grade_4>"
    if margin > 0.3:
        return "<grade_3>"
    return "<grade_2>"

def generate_sample() -> Dict[str, Any]:
    lang = random.choice(["en", "sk", "uk"])
    cipher_type = random.choices(["Caesar", "Vigenere", "Substitution"], weights=[0.4, 0.4, 0.2])[0]
    plain = sample_plain(lang)

    crib_plain = None
    crib_cipher = None

    # ---- generate ciphertext with true key ----
    if cipher_type == "Caesar":
        true_key = random.randint(1, len(ALPHABETS[lang]) - 1)
        ciphertext = caesar(plain, true_key, lang, encrypt=True)

    elif cipher_type == "Vigenere":
        true_key = random.choice(KEY_VOCAB[lang])
        ciphertext = vigenere(plain, true_key, lang, encrypt=True)

    else:
        true_key = subst_key_perm(lang)  # permutation
        ciphertext = substitution(plain, true_key, lang, encrypt=True)
        crib_plain, crib_cipher = make_full_crib(lang, true_key)

    # ---- run hypotheses to make reasoning + margins ----
    cand = []

    # Caesar hypothesis
    k_c, pt_c, sc_c = best_caesar(ciphertext, lang)
    ok_c = (caesar(pt_c, k_c, lang, encrypt=True) == ciphertext)
    cand.append(("Caesar", str(k_c), pt_c, sc_c, ok_c))

    # Vigenere hypothesis (over vocab)
    k_v, pt_v, sc_v = best_vigenere_from_vocab(ciphertext, lang)
    ok_v = (vigenere(pt_v, k_v, lang, encrypt=True) == ciphertext)
    cand.append(("Vigenere", k_v, pt_v, sc_v, ok_v))

    # Substitution hypothesis (with crib only if available)
    if crib_plain is not None:
        perm, ok_map = derive_perm_from_crib(crib_plain, crib_cipher, lang)
        pt_s = substitution(ciphertext, perm, lang, encrypt=False)
        sc_s = score_text(pt_s, lang)
        ok_s = ok_map and (substitution(pt_s, perm, lang, encrypt=True) == ciphertext)
        cand.append(("Substitution", perm, pt_s, sc_s, ok_s))

    # Choose best among ok candidates by score
    ok_cands = [x for x in cand if x[4]]
    ok_cands.sort(key=lambda x: x[3], reverse=True)
    best = ok_cands[0] if ok_cands else max(cand, key=lambda x: x[3])

    # margin for self_score (best - second best among ok)
    if len(ok_cands) >= 2:
        margin = ok_cands[0][3] - ok_cands[1][3]
    else:
        margin = 0.0

    pred_type, pred_key, pred_pt, pred_sc, pred_ok = best

    reasoning = {
        "style": "detect",
        "signals": {
            "ioc": ioc(ciphertext, lang),
            "top_chars": Counter([ch for ch in ciphertext if ch in ALPHABETS[lang]]).most_common(6),
        },
        "hypotheses": [
            {"type": t, "ok": ok, "score": sc, "key_preview": (k[:40] if isinstance(k, str) else str(k))}
            for (t, k, _pt, sc, ok) in cand
        ],
        "chosen": {"cipher_type": pred_type, "key_preview": (pred_key[:40] if isinstance(pred_key, str) else str(pred_key))},
        "verify": {"re_encrypt_matches": pred_ok}
    }

    # PROMPT: НЕТ cipher_type/key
    instruction = f"""### Instruction:
You are CipherChat.
Return ONLY valid JSON (no extra text).
Task: DETECT cipher type + key from ciphertext, then DECRYPT.
Language: {lang}

CIPHERTEXT:
{ciphertext}
"""

    if cipher_type == "Substitution":
        instruction += f"""
KNOWN-PLAINTEXT CRIB:
CRIB_PLAINTEXT:
{crib_plain}
CRIB_CIPHERTEXT:
{crib_cipher}
"""

    instruction += "\n### Answer JSON:\n"

    answer = {
        "mode": "decrypt",
        "lang": lang,
        "cipher_type": pred_type,
        "key": pred_key,
        "cipher_text": ciphertext,
        "plaintext": pred_pt,
        "reasoning": reasoning,
        "self_score": grade(pred_ok, margin)
    }

    return {"instruction": instruction, "json": answer}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=15000)
    ap.add_argument("--out", default="data/processed/train_detect.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.num):
            ex = generate_sample()
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            if i % 1000 == 0:
                print(f"[DETECT_V2] {i}/{args.num}")

    print(f"✅ saved → {args.out} ({args.num})")

if __name__ == "__main__":
    main()
