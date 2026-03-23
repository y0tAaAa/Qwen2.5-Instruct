# CipherChat-7B

A QLoRA fine-tune of [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) for classical cipher computation and detection in three languages.

---

## What it does

Given a cipher task in natural language, the model outputs a structured JSON object containing the cryptographic result and a step-by-step reasoning trace.

**Supported ciphers:** Caesar · Vigenere · Transposition (columnar)  
**Supported languages:** English · Slovak · Ukrainian  
**Supported tasks:**
- `compute` — encrypt or decrypt a given text using a known cipher and key
- `detect` — identify the cipher type and key from ciphertext alone, then decrypt

---

## Example

**Input:**
```
TASK: COMPUTE
Mode: ENCRYPT
Language: en
Cipher: Vigenere
Key: SECRET

INPUT_TEXT:
ATTACK AT DAWN
```

**Output:**
```json
{
  "cipher_type": "Vigenere",
  "mode": "encrypt",
  "lang": "en",
  "key": "SECRET",
  "input_text": "ATTACK AT DAWN",
  "output_text": "SXVRGC SX ZSNP",
  "reasoning": "TASK: ENCRYPT | cipher=Vigenere | key=SECRET | shifts=[18,4,2,17,4,19] | [0]'A'(0)+18=18->'S' ..."
}
```

---

## Performance

Evaluated on 300 stratified examples (150 seen / 150 unseen) covering all cipher × language × task combinations.

| Metric | Base model | CipherChat-7B |
|---|---|---|
| Valid JSON | 81.7% | **96.0%** |
| Algo correct | 0.3% | **54.3%** |
| Output exact | 0.3% | **71.3%** |
| Cipher type | 52.7% | **95.3%** |
| Key accuracy | 41.3% | **75.3%** |

**By cipher (algo correct):**

| Cipher | Base | Fine-tuned |
|---|---|---|
| Caesar | 0.0% | 59.8% |
| Vigenere | 0.0% | 70.7% |
| Transposition | 1.0% | 32.3% |

**By language (algo correct):**

| Language | Base | Fine-tuned |
|---|---|---|
| English | 1.0% | 60.8% |
| Slovak | 0.0% | 53.5% |
| Ukrainian | 0.0% | 48.5% |

Seen / unseen gap: 58.0% vs 50.7% — no significant overfitting.

---

## Training details

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Method | QLoRA (4-bit NF4, double quantization, bfloat16) |
| LoRA r / alpha | 64 / 128 |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o/gate/up/down_proj |
| Learning rate | 2e-4 (cosine + 5% warmup) |
| Epochs | 2 |
| Effective batch size | 16 (2 × 8 grad accum) |
| Train samples | 12 000 |
| Max sequence length | 1 536 |
| Optimizer | paged_adamw_8bit |
| Hardware | 1× NVIDIA A100-SXM4-40GB |
| Training time | ~3.4 hours |
| Final train loss | 0.1637 |

---

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch, json

BASE    = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "YOUR_USERNAME/cipherchat-7b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

SYSTEM = (
    "You are CipherChat — a cipher computation assistant.\n"
    "Given a task, output ONLY a valid JSON object, no extra text.\n\n"
    "Output JSON schema:\n"
    '{"cipher_type": "Caesar"|"Vigenere"|"Transposition", '
    '"mode": "encrypt"|"decrypt", "lang": "en"|"sk"|"uk", '
    '"key": string, "input_text": string, "output_text": string, '
    '"reasoning": string}'
)

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": (
        "TASK: COMPUTE\nMode: ENCRYPT\nLanguage: en\n"
        "Cipher: Caesar\nKey: 13\n\nINPUT_TEXT:\nHELLO WORLD"
    )},
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=512,
        do_sample=False, repetition_penalty=1.1,
    )

response = tokenizer.decode(
    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
result = json.loads(response)
print(result["output_text"])  # URYYB JBEYQ
```

---

## Experiment history

This release is V1 — the best performing checkpoint across four training runs.
Three follow-up experiments (V2–V4) attempted to improve on specific weaknesses
but all degraded overall performance. The results are documented below.

### V2 — focused fine-tune ❌

**Goal:** improve Transposition (32% → 50%+) and Ukrainian (48% → 60%+)  
**Changes:** Transposition 40%, UK 40%, detect 50% in dataset; resumed from V1; lr=5e-5, 3 epochs

**Result:** Valid JSON collapsed from 96% to 63.7%. Catastrophic forgetting — the model
lost structured output at lr=5e-5, which was too small to restore but large enough to overwrite.

### V3 — recovery attempt ❌

**Goal:** restore Valid JSON while keeping improved Transposition  
**Changes:** lr=1e-4, 2 epochs, dataset with Vigenere 35% and 15% JSON-reinforce short examples; resumed from V2

**Result:** Valid JSON only partially recovered (68.7%). Transposition reached its best result
across all runs (48.0%, +15.7% over V1), but the JSON format degradation from V2 proved
irreversible through fine-tuning alone.

### V4 — fresh start on V3 dataset ❌

**Goal:** combine V1 training strategy (from scratch, lr=2e-4) with V3 dataset  
**Changes:** from scratch, lr=2e-4, 3 epochs, V3 dataset

**Result:** Valid JSON stayed at 68.3% despite training from scratch. The V3 dataset's
distribution (JSON-reinforce 15%, skewed weights) was suboptimal for lr=2e-4 — the model
overfitted to Transposition at the expense of Caesar and Vigenere.

### Summary

| Metric | V1 ✅ | V2 | V3 | V4 |
|---|---|---|---|---|
| Valid JSON | **96.0%** | 63.7% | 68.7% | 68.3% |
| Algo correct | **54.3%** | 45.3% | 53.7% | 46.3% |
| Transposition | 32.3% | 25.5% | **48.0%** | 43.0% |
| Vigenere | **70.7%** | 45.5% | 53.0% | 46.0% |

**Key finding:** once Valid JSON degrades, it cannot be recovered through further fine-tuning
without returning to the original V1 conditions (from scratch + V1 dataset).
Transposition accuracy is genuinely improved by extended grid-trace reasoning in the dataset,
but this benefit comes at a cost to other ciphers when the dataset distribution is skewed.

---

## License

Apache 2.0 — same as the base model.
