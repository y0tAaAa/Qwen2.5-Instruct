import argparse
import json
import os
import re
import sys
from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
from model.force_selfscore import ForceSelfScoreGrades
GRADE_TOKENS = ["<grade_1>", "<grade_2>", "<grade_3>", "<grade_4>", "<grade_5>"]
_ws_re = re.compile(r"\s+")
def norm(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\u200b", "")
    return _ws_re.sub(" ", s)
def cer(ref: str, hyp: str) -> float:
    ref = ref or ""
    hyp = hyp or ""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m] / max(1, n)
def extract_fields(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    j = obj.get("json", obj)
    return {
        "mode": (j.get("mode") or "").lower().strip(),
        "cipher_type": j.get("cipher_type"),
        "key": j.get("key"),
        "cipher_text": j.get("cipher_text"),
        "plaintext": j.get("plaintext"),
        "detected_cipher_type": j.get("detected_cipher_type"),
        "detected_key": j.get("detected_key"),
        "verify": j.get("verify"),
        "lang": obj.get("lang") or "en",
    } if j.get("cipher_type") and j.get("key") is not None and j.get("cipher_text") and j.get("plaintext") else None
def build_prompt(fields: Dict[str, Any]) -> str:
    tv = fields.get("task_variant", "known_params")
    mode = fields["mode"]
    base = f"### Instruction:\nYou are CipherChat — DETECT→SOLVE→VERIFY. Return ONLY valid JSON.\n[{fields['lang']}] Task: {mode} variant:{tv}\n"
    if tv == "detect_solve":
        base += f"Text:\n{fields['cipher_text']}\n"
    else:
        base += f"Cipher type: {fields['cipher_type']}\nKey: {fields['key']}\nText:\n{fields['cipher_text']}\n"
    base += '### Response (JSON only):\n{"mode":"' + mode + '","task_variant":"' + tv + '","cipher_type":"' + fields['cipher_type'] + '","key":' + json.dumps(str(fields['key']), ensure_ascii=False) + ',"cipher_text":' + json.dumps(fields['cipher_text'], ensure_ascii=False) + ',"detected_cipher_type":"'
    return base
def load_model(base_id: str, lora_path: str):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # CRITICAL: Add grade tokens (they may not have been saved properly)
    existing = tokenizer.additional_special_tokens or []
    to_add = [t for t in GRADE_TOKENS if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": existing + to_add})
        print(f"[EVAL] Added missing grade tokens: {to_add}")
    else:
        print(f"[EVAL] Grade tokens already present.")
    print(f"[EVAL] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[EVAL] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        device = torch.device("cuda:0")
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        ).to(device)
    else:
        device = torch.device("cpu")
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        ).to(device)
    # Resize embeddings AFTER adding tokens
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    model.to(device)
    p = next(model.parameters())
    print(f"[EVAL] model param device = {p.device}, dtype={p.dtype}")
    return tokenizer, model, device
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora_path", required=True)
    ap.add_argument("--eval_path", required=True)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()
    os.makedirs("logs", exist_ok=True)
    tok, model, device = load_model(args.base, args.lora_path)
    processor = ForceSelfScoreGrades(tok)
    stats = {"n": 0, "json_valid": 0, "em": 0, "cer_sum": 0.0, "detect_acc": 0, "round_pass": 0, "empty": 0}
    with open(args.eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            fields = extract_fields(obj)
            if not fields: continue
            prompt = build_prompt(fields)
            enc = tok(prompt, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                    logits_processor=[processor],
                )
            decoded = tok.decode(out[0], skip_special_tokens=False)
            parsed = None
            start = decoded.rfind("{")
            end = decoded.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(decoded[start:end + 1])
                except: pass
            if parsed:
                stats["json_valid"] += 1
                pred_pt = norm(parsed.get("plaintext", ""))
                ref_pt = norm(fields["plaintext"])
                if pred_pt == ref_pt: stats["em"] += 1
                stats["cer_sum"] += cer(ref_pt, pred_pt)
                if parsed.get("detected_cipher_type") == fields["cipher_type"]: stats["detect_acc"] += 1
                if parsed.get("verify") == "round_trip_ok": stats["round_pass"] += 1
                if pred_pt == "": stats["empty"] += 1
            stats["n"] += 1
            if stats["n"] % args.log_every == 0:
                print(f"[EVAL] processed={stats['n']} json_valid={stats['json_valid']/stats['n']:.4f} EM={stats['em']/stats['n']:.4f} CER={stats['cer_sum']/stats['n']:.4f} DETACC={stats['detect_acc']/stats['n']:.4f} RTPASS={stats['round_pass']/stats['n']:.4f}")
            if stats["n"] >= args.max_samples: break
    n = stats["n"] or 1
    print("\n==== EVAL RESULT ====")
    print(f"Samples: {stats['n']}")
    print(f"JSON validity: {stats['json_valid']/n:.4f}")
    print(f"Exact Match:   {stats['em']/n:.4f}")
    print(f"Avg CER:       {stats['cer_sum']/n:.4f}")
    print(f"Detect Acc:    {stats['detect_acc']/n:.4f}")
    print(f"Round-trip OK: {stats['round_pass']/n:.4f}")
    print(f"Empty preds:   {stats['empty']/n:.4f}")
if __name__ == "__main__":
    main()
