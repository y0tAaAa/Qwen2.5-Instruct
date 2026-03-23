# src/ciphers/ciphers_multi.py
from __future__ import annotations
from typing import Dict, Tuple, List, Any
import random

# =========================
# ALPHABETS (UPPERCASE)
# =========================
ALPHABETS: Dict[str, str] = {
    "en": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "uk": "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ",
    "sk": "AÁÄBCČDĎEÉFGHIÍJKLĹĽMNŇOÓÔPQRŔSŠTŤUÚVXYÝZŽ",
}

def _norm_lang(lang: str) -> str:
    lang = (lang or "").strip().lower()
    if lang not in ALPHABETS:
        raise ValueError(f"Unsupported lang={lang}. Use one of {list(ALPHABETS.keys())}")
    return lang

def _maps(alpha: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    idx = {ch: i for i, ch in enumerate(alpha)}
    rev = {i: ch for i, ch in enumerate(alpha)}
    return idx, rev

def _case(original: str, out_upper: str) -> str:
    return out_upper.lower() if original.islower() else out_upper

# ---------------- Caesar ----------------
def caesar_encrypt(text: str, shift: int, lang: str) -> str:
    lang = _norm_lang(lang)
    alpha = ALPHABETS[lang]
    idx, rev = _maps(alpha)
    n = len(alpha)
    shift = int(shift) % n

    out = []
    for ch in text:
        up = ch.upper()
        if up in idx:
            out_ch = rev[(idx[up] + shift) % n]
            out.append(_case(ch, out_ch))
        else:
            out.append(ch)
    return "".join(out)

def caesar_decrypt(text: str, shift: int, lang: str) -> str:
    return caesar_encrypt(text, -int(shift), lang)

# ---------------- Vigenere ----------------
def _vig_key_indices(key: str, alpha: str) -> List[int]:
    idx, _ = _maps(alpha)
    key_idx = [idx[c] for c in (key or "").upper() if c in idx]
    if not key_idx:
        raise ValueError("Vigenere key has no characters from the selected alphabet.")
    return key_idx

def vigenere_encrypt(text: str, key: str, lang: str) -> str:
    lang = _norm_lang(lang)
    alpha = ALPHABETS[lang]
    idx, rev = _maps(alpha)
    n = len(alpha)
    kidx = _vig_key_indices(key, alpha)

    out = []
    j = 0
    for ch in text:
        up = ch.upper()
        if up in idx:
            k = kidx[j % len(kidx)]
            out_ch = rev[(idx[up] + k) % n]
            out.append(_case(ch, out_ch))
            j += 1
        else:
            out.append(ch)
    return "".join(out)

def vigenere_decrypt(text: str, key: str, lang: str) -> str:
    lang = _norm_lang(lang)
    alpha = ALPHABETS[lang]
    idx, rev = _maps(alpha)
    n = len(alpha)
    kidx = _vig_key_indices(key, alpha)

    out = []
    j = 0
    for ch in text:
        up = ch.upper()
        if up in idx:
            k = kidx[j % len(kidx)]
            out_ch = rev[(idx[up] - k) % n]
            out.append(_case(ch, out_ch))
            j += 1
        else:
            out.append(ch)
    return "".join(out)

# ---------------- Substitution ----------------
def substitution_make_key(lang: str, seed: int | None = None) -> Dict[str, str]:
    lang = _norm_lang(lang)
    alpha = ALPHABETS[lang]
    rng = random.Random(seed)
    perm = list(alpha)
    rng.shuffle(perm)
    return {alpha[i]: perm[i] for i in range(len(alpha))}

def substitution_encrypt(text: str, key: Dict[str, str], lang: str) -> str:
    lang = _norm_lang(lang)
    alpha = ALPHABETS[lang]
    idx, _ = _maps(alpha)
    key_u = {k.upper(): v.upper() for k, v in key.items()}

    out = []
    for ch in text:
        up = ch.upper()
        if up in idx:
            out_ch = key_u.get(up, up)
            out.append(_case(ch, out_ch))
        else:
            out.append(ch)
    return "".join(out)

def substitution_decrypt(text: str, key: Dict[str, str], lang: str) -> str:
    inv = {v.upper(): k.upper() for k, v in key.items()}
    return substitution_encrypt(text, inv, lang)

# ---------------- Unified API ----------------
def encrypt(cipher_type: str, text: str, key: Any, lang: str) -> str:
    ct = (cipher_type or "").strip().lower()
    if ct == "caesar":
        return caesar_encrypt(text, int(key), lang)
    if ct == "vigenere":
        return vigenere_encrypt(text, str(key), lang)
    if ct == "substitution":
        if not isinstance(key, dict):
            raise ValueError("Substitution key must be a dict.")
        return substitution_encrypt(text, key, lang)
    raise ValueError(f"Unsupported cipher_type={cipher_type}")

def decrypt(cipher_type: str, text: str, key: Any, lang: str) -> str:
    ct = (cipher_type or "").strip().lower()
    if ct == "caesar":
        return caesar_decrypt(text, int(key), lang)
    if ct == "vigenere":
        return vigenere_decrypt(text, str(key), lang)
    if ct == "substitution":
        if not isinstance(key, dict):
            raise ValueError("Substitution key must be a dict.")
        return substitution_decrypt(text, key, lang)
    raise ValueError(f"Unsupported cipher_type={cipher_type}")

