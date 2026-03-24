# CipherChat — Multi-Scale Classical Cipher Models

QLoRA fine-tunes of Qwen2.5-Instruct (3B, 7B, 14B) for classical cipher computation and detection in three languages.

---

## Overview

CipherChat trains language models to solve classical cryptography tasks — both computing cipher results and detecting cipher parameters — across English, Slovak, and Ukrainian. Three model scales are provided, letting users trade off inference speed against accuracy.

**Supported ciphers:** Caesar · Vigenere · Transposition (columnar)  
**Supported languages:** English (`en`) · Slovak (`sk`) · Ukrainian (`uk`)  
**Supported tasks:**
- `compute` — encrypt or decrypt text using a known cipher and key
- `detect` — identify the cipher type and key from ciphertext, then decrypt

---

## Evaluation Results

All models evaluated on the same 300 stratified examples (`data/eval300.json`):
- 150 **seen** (cipher × language × task combinations present during training)
- 150 **unseen** (held-out combinations)

### Performance Comparison — All Models

| Metric | 3B | 7B | 14B |
|---|---|---|---|
| Valid JSON | 96.0% | 96.0% | **96.3%** |
| Algo correct | 39.0% | 54.3% | **55.3%** |
| Output exact | 66.3% | 71.3% | **72.3%** |
| Cipher type | **95.3%** | **95.3%** | 95.0% |
| Key accuracy | 61.7% | 75.3% | **76.7%** |

**Recommendation:** 7B offers the best efficiency (54.3% accuracy, faster inference than 14B). Use 14B for maximum accuracy (55.3%).

---

## Detailed Results — 3B Model

Model: `cipher-instruct-3B-20260320_235526` · Base: `Qwen/Qwen2.5-3B-Instruct`

### Overall Metrics

| Metric | Base | Fine-tuned |
|---|---|---|
| Valid JSON | 78.0% | **96.0%** |
| Algo correct | 0.0% | **39.0%** |
| Output exact | 0.0% | **66.3%** |
| Cipher type | 48.3% | **95.3%** |
| Key accuracy | 33.0% | **61.7%** |

### By Split

| Split | Algo correct | Output exact |
|---|---|---|
| Seen | 45.3% | 70.0% |
| Unseen | 32.7% | 62.7% |

### By Cipher Type (Algo correct)

| Cipher | Base | Fine-tuned |
|---|---|---|
| Caesar | 0.0% | 44.1% |
| Vigenere | 0.0% | 55.9% |
| Transposition | 0.0% | 17.6% |

### By Language (Algo correct)

| Language | Base | Fine-tuned |
|---|---|---|
| English | 0.0% | 45.1% |
| Slovak | 0.0% | 37.3% |
| Ukrainian | 0.0% | 34.3% |

### By Task (Algo correct)

| Task | Base | Fine-tuned |
|---|---|---|
| Compute | 0.0% | 40.0% |
| Detect | 0.0% | 38.0% |

---

## Detailed Results — 7B Model

Model: `cipherchat-7b-20260306_153334` · Base: `Qwen/Qwen2.5-7B-Instruct`

### Overall Metrics

| Metric | Base | Fine-tuned |
|---|---|---|
| Valid JSON | 81.7% | **96.0%** |
| Algo correct | 0.3% | **54.3%** |
| Output exact | 0.3% | **71.3%** |
| Cipher type | 52.7% | **95.3%** |
| Key accuracy | 41.3% | **75.3%** |

### By Split

| Split | Algo correct | Output exact |
|---|---|---|
| Seen | 58.0% | 74.7% |
| Unseen | 50.7% | 68.0% |

### By Cipher Type (Algo correct)

| Cipher | Base | Fine-tuned |
|---|---|---|
| Caesar | 0.0% | 59.8% |
| Vigenere | 0.0% | **70.7%** |
| Transposition | 1.0% | 32.3% |

### By Language (Algo correct)

| Language | Base | Fine-tuned |
|---|---|---|
| English | 1.0% | **60.8%** |
| Slovak | 0.0% | 53.5% |
| Ukrainian | 0.0% | 48.5% |

### By Task (Algo correct)

| Task | Base | Fine-tuned |
|---|---|---|
| Compute | 0.3% | **55.2%** |
| Detect | 0.3% | 53.4% |

---

## Detailed Results — 14B Model

Model: `cipher-instruct-14B-20260321_143806` · Base: `Qwen/Qwen2.5-14B-Instruct`

### Overall Metrics

| Metric | Base | Fine-tuned |
|---|---|---|
| Valid JSON | 84.3% | **96.3%** |
| Algo correct | 0.3% | **55.3%** |
| Output exact | 0.3% | **72.3%** |
| Cipher type | 55.3% | 95.0% |
| Key accuracy | 44.7% | **76.7%** |

### By Split

| Split | Algo correct | Output exact |
|---|---|---|
| Seen | 59.3% | 75.3% |
| Unseen | 51.3% | 69.3% |

### By Cipher Type (Algo correct)

| Cipher | Base | Fine-tuned |
|---|---|---|
| Caesar | 0.0% | 61.8% |
| Vigenere | 0.0% | **72.5%** |
| Transposition | 1.0% | 31.4% |

### By Language (Algo correct)

| Language | Base | Fine-tuned |
|---|---|---|
| English | 1.0% | **62.7%** |
| Slovak | 0.0% | 54.9% |
| Ukrainian | 0.0% | 48.0% |

### By Task (Algo correct)

| Task | Base | Fine-tuned |
|---|---|---|
| Compute | 0.3% | **56.0%** |
| Detect | 0.3% | 54.7% |

---

## Key Findings

- **Transposition is the hardest cipher** across all model sizes (17–32% vs 56–72% for Vigenere), suggesting that columnar permutation requires longer multi-step reasoning traces than shift/substitution ciphers.
- **Scaling improves key accuracy significantly:** 3B→7B jump (+13.6 pp) is larger than 7B→14B (+1.4 pp).
- **No significant overfitting** to seen examples: seen/unseen gap is ≤8 pp across all models and metrics.
- **Ukrainian is the hardest language** across all models, likely due to a larger alphabet (33 chars vs 26 for English).
- **Valid JSON rate saturates early:** all three fine-tuned models reach ~96% structured output, suggesting the format is learned quickly regardless of model size.
- **V1 7B is the best checkpoint:** follow-up experiments (V2–V4) on the 7B line all degraded performance — details in [README_en.md](README_en.md).

---

## Training Details

| Parameter | 3B | 7B | 14B |
|---|---|---|---|
| Base model | Qwen2.5-3B-Instruct | Qwen2.5-7B-Instruct | Qwen2.5-14B-Instruct |
| Method | QLoRA (4-bit NF4) | QLoRA (4-bit NF4) | QLoRA (4-bit NF4) |
| LoRA r / alpha | 64 / 128 | 64 / 128 | 64 / 128 |
| LoRA dropout | 0.05 | 0.05 | 0.05 |
| Learning rate | 2e-4 | 2e-4 | 2e-4 |
| Epochs | 2 | 2 | 2 |
| Effective batch size | 16 | 16 | 16 |
| Train samples | 12 000 | 12 000 | 12 000 |
| Max sequence length | 1 536 | 1 536 | 1 536 |
| Optimizer | paged_adamw_8bit | paged_adamw_8bit | paged_adamw_8bit |
| Hardware | 1× A100-40GB | 1× A100-40GB | 1× A100-40GB |

---

## Usage

### Inference Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch, json

# Choose one of the three models
BASE    = "Qwen/Qwen2.5-7B-Instruct"   # or 3B / 14B
ADAPTER = "checkpoints/cipherchat-7b-20260306_153334/final_adapter"

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
        "Cipher: Vigenere\nKey: SECRET\n\nINPUT_TEXT:\nATTACK AT DAWN"
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
print(result["output_text"])  # SXVRGC SX ZSNP
```

---

## Evaluation Scripts

Three evaluation scripts are provided, one per model scale:

```bash
# Evaluate 3B model
python eval_3b.py --data data/eval300.json --out_dir eval_results

# Evaluate 7B model
python eval_7b.py --data data/eval300.json --out_dir eval_results

# Evaluate 14B model
python eval_14b.py --data data/eval300.json --out_dir eval_results
```

All scripts:
- Load the model in 4-bit (QLoRA) and run both the base and fine-tuned versions
- Log output to `logs/eval_<tag>_<timestamp>.out`
- Save JSON results to `eval_results/eval_<tag>_<timestamp>.json`
- Report breakdowns by split, cipher type, language, and task
- Clean VRAM cache every 50 examples

### Hardware Requirements

| Model | VRAM (4-bit inference) |
|---|---|
| 3B | ~4 GB |
| 7B | ~6 GB |
| 14B | ~10 GB |

---

## Data Format

`data/eval300.json` — 300 stratified examples:

```json
[
  {
    "id": "eval_001",
    "split": "seen",
    "cipher": "Caesar",
    "lang": "en",
    "task": "compute",
    "mode": "encrypt",
    "key": "13",
    "input_text": "HELLO WORLD",
    "expected_output": "URYYB JBEYQ",
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user",   "content": "..."}
    ]
  }
]
```

**Fields:**
- `split` — `"seen"` or `"unseen"`
- `cipher` — `"Caesar"`, `"Vigenere"`, or `"Transposition"`
- `lang` — `"en"`, `"sk"`, or `"uk"`
- `task` — `"compute"` or `"detect"`
- `mode` — `"encrypt"` or `"decrypt"` (for compute tasks)
- `key` — numeric string for Caesar, word for Vigenere, word for Transposition
- `expected_output` — ground-truth ciphertext or plaintext

---

## Repository Structure

```
.
├── README.md                  # this file
├── README_en.md               # 7B model card (English)
├── README_ru.md               # project notes (Russian)
├── README_sk.md               # Slovak README
│
├── eval_3b.py                 # evaluation script for 3B model
├── eval_7b.py                 # evaluation script for 7B model
├── eval_14b.py                # evaluation script for 14B model
├── eval_cipherchat.py         # generic evaluation script
│
├── train_lora.py              # LoRA training script
├── train_lora_v3.py           # updated training script (v3 dataset)
├── generate_dataset.py        # dataset generation (v1)
├── generate_dataset_v2.py     # dataset generation (v2)
├── generate_dataset_v3.py     # dataset generation (v3)
├── prepare_eval300.py         # creates data/eval300.json
│
├── checkpoints/               # saved model checkpoints
│   ├── cipher-instruct-3B-20260320_235526/final_adapter
│   ├── cipherchat-7b-20260306_153334/final_adapter
│   └── cipher-instruct-14B-20260321_143806/final_adapter
│
├── data/
│   ├── eval300.json           # 300-example evaluation set
│   └── processed/             # JSONL files for training/validation
│
├── config/                    # training configuration files
├── logs/                      # training and evaluation logs
├── eval_results/              # JSON evaluation outputs
└── src/                       # source modules
```

---

## Future Work

- [ ] Improve Transposition accuracy (currently 17–32% vs 56–72% for Vigenere) via extended grid-trace reasoning in training data
- [ ] Add Substitution cipher support
- [ ] Extend to additional languages (Polish, Czech, Hungarian)
- [ ] Experiment with larger context windows for longer ciphertexts
- [ ] Distill 14B knowledge into 3B via self-score training

---

## License

Apache 2.0 — same as the base Qwen2.5 models.

