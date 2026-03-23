# scripts/test_fine_tuned_model.py
# -*- coding: utf-8 -*-

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="y0ta/fine_tuned_model")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="auto")
    model.eval()

    inputs = tok(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.eos_token_id,
        )

    decoded = tok.decode(out[0], skip_special_tokens=True)
    print(decoded)

if __name__ == "__main__":
    main()

