# CipherChat-7B

QLoRA doladenie modelu [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) pre výpočet a detekciu klasických šifier v troch jazykoch.

---

## Čo model robí

Na základe zadania v prirodzenom jazyku model vráti štruktúrovaný JSON objekt obsahujúci kryptografický výsledok a postup výpočtu krok za krokom.

**Podporované šifry:** Caesar · Vigenere · Transpozícia (stĺpcová)  
**Podporované jazyky:** Angličtina · Slovenčina · Ukrajinčina  
**Podporované úlohy:**
- `compute` — zašifrovanie alebo dešifrovanie textu so známou šifrou a kľúčom
- `detect` — určenie typu šifry a kľúča iba zo šifrovaného textu, následné dešifrovanie

---

## Príklad

**Vstup:**
```
TASK: COMPUTE
Mode: ENCRYPT
Language: sk
Cipher: Caesar
Key: 5

INPUT_TEXT:
AHOJ SVET
```

**Výstup:**
```json
{
  "cipher_type": "Caesar",
  "mode": "encrypt",
  "lang": "sk",
  "key": "5",
  "input_text": "AHOJ SVET",
  "output_text": "FMTO XJY",
  "reasoning": "TASK: ENCRYPT | cipher=Caesar | lang=sk | key=5 | shift=5 | [0]'A'(0)+5=5->'F' ..."
}
```

---

## Výkonnosť

Hodnotené na 300 stratifikovaných príkladoch (150 videných / 150 nevidených) pokrývajúcich všetky kombinácie šifra × jazyk × úloha.

| Metrika | Základný model | CipherChat-7B |
|---|---|---|
| Platný JSON | 81,7 % | **96,0 %** |
| Algoritmicky správne | 0,3 % | **54,3 %** |
| Presný výstup | 0,3 % | **71,3 %** |
| Typ šifry | 52,7 % | **95,3 %** |
| Presnosť kľúča | 41,3 % | **75,3 %** |

**Podľa šifry (algoritmicky správne):**

| Šifra | Základný model | Po doladení |
|---|---|---|
| Caesar | 0,0 % | 59,8 % |
| Vigenere | 0,0 % | 70,7 % |
| Transpozícia | 1,0 % | 32,3 % |

**Podľa jazyka (algoritmicky správne):**

| Jazyk | Základný model | Po doladení |
|---|---|---|
| Angličtina | 1,0 % | 60,8 % |
| Slovenčina | 0,0 % | 53,5 % |
| Ukrajinčina | 0,0 % | 48,5 % |

Rozdiel videné / nevidené: 58,0 % vs 50,7 % — žiadne výrazné preplnenie (overfitting).

---

## Parametre trénovania

| Parameter | Hodnota |
|---|---|
| Základný model | Qwen/Qwen2.5-7B-Instruct |
| Metóda | QLoRA (4-bit NF4, dvojitá kvantizácia, bfloat16) |
| LoRA r / alpha | 64 / 128 |
| LoRA dropout | 0,05 |
| Cieľové moduly | q/k/v/o/gate/up/down_proj |
| Rýchlosť učenia | 2e-4 (kosínusový rozvrh + 5 % zahriatie) |
| Epochy | 2 |
| Efektívna veľkosť dávky | 16 (2 × 8 akumulácia gradientov) |
| Trénovacie vzorky | 12 000 |
| Maximálna dĺžka sekvencie | 1 536 |
| Optimalizátor | paged_adamw_8bit |
| Hardware | 1× NVIDIA A100-SXM4-40GB |
| Čas trénovania | ~3,4 hodiny |
| Finálna trénovacia strata | 0,1637 |

---

## Použitie

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
        "TASK: COMPUTE\nMode: ENCRYPT\nLanguage: sk\n"
        "Cipher: Caesar\nKey: 5\n\nINPUT_TEXT:\nAHOJ SVET"
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
print(result["output_text"])
```

---

## História experimentov

Toto vydanie zodpovedá verzii V1 — najlepšiemu checkpointu spomedzi štyroch trénovacích behov.
Tri nadväzujúce experimenty (V2–V4) sa pokúšali opraviť konkrétne slabiny, no všetky
zhoršili celkovú výkonnosť. Výsledky sú zdokumentované nižšie.

### V2 — cielené doladenie ❌

**Cieľ:** zlepšiť Transpozíciu (32 % → 50 %+) a ukrajinskú presnosť (48 % → 60 %+)  
**Zmeny:** váhy datasetu Transpozícia 40 %, UK 40 %, detect 50 %; pokračovanie od V1; lr=5e-5, 3 epochy

**Výsledok:** Platný JSON sa prepadol z 96 % na 63,7 %. Katastrofické zabudnutie formátu —
model prestal generovať štruktúrovaný výstup. Príčina: lr=5e-5 bol príliš malý na obnovenie
zabudnutého, no dostatočne veľký na jeho prepísanie.

### V3 — pokus o obnovu ❌

**Cieľ:** obnoviť platný JSON a zachovať zlepšenie Transpozície  
**Zmeny:** lr=1e-4, 2 epochy, dataset s Vigenere 35 % a 15 % krátkymi JSON-reinforce príkladmi; pokračovanie od V2

**Výsledok:** Platný JSON sa obnovil len čiastočne (68,7 %). Transpozícia dosiahla
najlepší výsledok spomedzi všetkých behov (48,0 %, +15,7 % oproti V1), no celkové
zhoršenie formátu z V2 sa ukázalo byť nezvratným.

### V4 — tréning od základu na datasete V3 ❌

**Cieľ:** spojiť stratégiu V1 (od základu, lr=2e-4) s datasetom V3  
**Zmeny:** od základu, lr=2e-4, 3 epochy, dataset V3

**Výsledok:** Platný JSON zostal na 68,3 % napriek trénovaniu od základu. Distribúcia
datasetu V3 (JSON-reinforce 15 %, skreslené váhy) sa ukázala byť neoptimálna pre
lr=2e-4 — model sa prispôsobil na Transpozíciu na úkor Caesara a Vigenera.

### Súhrnná tabuľka

| Metrika | V1 ✅ | V2 | V3 | V4 |
|---|---|---|---|---|
| Platný JSON | **96,0 %** | 63,7 % | 68,7 % | 68,3 % |
| Algoritmicky správne | **54,3 %** | 45,3 % | 53,7 % | 46,3 % |
| Transpozícia | 32,3 % | 25,5 % | **48,0 %** | 43,0 % |
| Vigenere | **70,7 %** | 45,5 % | 53,0 % | 46,0 % |

**Hlavný záver:** akonáhle sa platný JSON zhorší (V2), nedá sa obnoviť ďalším doladením
bez návratu k pôvodným podmienkam V1 (tréning od základu + dataset V1).
Presnosť Transpozície sa dá skutočne zlepšiť rozšíreným popisom mriežky v datasete,
no tento zisk prichádza za cenu ostatných šifier pri skreslenej distribúcii datasetu.

---

## Licencia

Apache 2.0 — rovnaká ako základný model.
