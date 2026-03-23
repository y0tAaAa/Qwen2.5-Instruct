# CipherChat-7B

QLoRA дообучение модели [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) для вычисления и детектирования классических шифров на трёх языках.

---

## Что делает модель

По заданию на естественном языке модель возвращает структурированный JSON-объект с криптографическим результатом и пошаговым объяснением.

**Поддерживаемые шифры:** Цезарь · Виженер · Перестановка (столбчатая)  
**Поддерживаемые языки:** Английский · Словацкий · Украинский  
**Поддерживаемые задачи:**
- `compute` — зашифровать или расшифровать текст с известным шифром и ключом
- `detect` — определить тип шифра и ключ только по шифртексту, затем расшифровать

---

## Пример

**Входные данные:**
```
TASK: COMPUTE
Mode: ENCRYPT
Language: en
Cipher: Vigenere
Key: SECRET

INPUT_TEXT:
ATTACK AT DAWN
```

**Выходные данные:**
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

## Результаты

Оценка на 300 стратифицированных примерах (150 из обучения / 150 новых), покрывающих все комбинации шифр × язык × тип задачи.

| Метрика | Базовая модель | CipherChat-7B |
|---|---|---|
| Валидный JSON | 81,7% | **96,0%** |
| Алгоритмически верно | 0,3% | **54,3%** |
| Точный вывод | 0,3% | **71,3%** |
| Тип шифра | 52,7% | **95,3%** |
| Точность ключа | 41,3% | **75,3%** |

**По шифрам (алгоритмически верно):**

| Шифр | Базовая | После дообучения |
|---|---|---|
| Цезарь | 0,0% | 59,8% |
| Виженер | 0,0% | 70,7% |
| Перестановка | 1,0% | 32,3% |

**По языкам (алгоритмически верно):**

| Язык | Базовая | После дообучения |
|---|---|---|
| Английский | 1,0% | 60,8% |
| Словацкий | 0,0% | 53,5% |
| Украинский | 0,0% | 48,5% |

Разрыв seen/unseen: 58,0% против 50,7% — значимого переобучения нет.

---

## Параметры обучения

| Параметр | Значение |
|---|---|
| Базовая модель | Qwen/Qwen2.5-7B-Instruct |
| Метод | QLoRA (4-bit NF4, двойное квантование, bfloat16) |
| LoRA r / alpha | 64 / 128 |
| LoRA dropout | 0,05 |
| Целевые модули | q/k/v/o/gate/up/down_proj |
| Скорость обучения | 2e-4 (косинусный планировщик + 5% прогрев) |
| Эпохи | 2 |
| Эффективный размер батча | 16 (2 × 8 накопление градиентов) |
| Обучающих примеров | 12 000 |
| Максимальная длина последовательности | 1 536 |
| Оптимизатор | paged_adamw_8bit |
| Оборудование | 1× NVIDIA A100-SXM4-40GB |
| Время обучения | ~3,4 часа |
| Финальный loss | 0,1637 |

---

## Использование

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

## История экспериментов

Этот релиз соответствует версии V1 — лучшему чекпоинту из четырёх учебных запусков.
Три последующих эксперимента (V2–V4) пытались исправить конкретные слабые места,
но все привели к ухудшению общей производительности. Результаты задокументированы ниже.

### V2 — целевое дообучение ❌

**Цель:** улучшить Перестановку (32% → 50%+) и украинский язык (48% → 60%+)  
**Изменения:** веса датасета: Перестановка 40%, украинский 40%, detect 50%;
продолжение от V1; lr=5e-5, 3 эпохи

**Результат:** Валидный JSON рухнул с 96% до 63,7%. Катастрофическое забывание формата —
модель перестала генерировать структурированный вывод. Причина: lr=5e-5 оказался
слишком маленьким чтобы восстановить забытое, но достаточно большим чтобы его перезаписать.

### V3 — попытка восстановления ❌

**Цель:** восстановить валидный JSON, сохранив улучшение Перестановки  
**Изменения:** lr=1e-4, 2 эпохи, датасет с Виженером 35% и 15% коротких JSON-reinforce примеров;
продолжение от V2

**Результат:** Валидный JSON восстановился лишь частично (68,7%). Перестановка достигла
лучшего результата за все запуски (48,0%, +15,7% относительно V1), однако деградация
формата из V2 оказалась необратимой при дообучении.

### V4 — обучение с нуля на датасете V3 ❌

**Цель:** совместить стратегию V1 (с нуля, lr=2e-4) с датасетом V3  
**Изменения:** с нуля, lr=2e-4, 3 эпохи, датасет V3

**Результат:** Валидный JSON остался на 68,3% несмотря на обучение с нуля.
Распределение датасета V3 (JSON-reinforce 15%, смещённые веса) оказалось
неоптимальным для lr=2e-4 — модель переобучилась на Перестановке в ущерб Цезарю и Виженеру.

### Сводная таблица

| Метрика | V1 ✅ | V2 | V3 | V4 |
|---|---|---|---|---|
| Валидный JSON | **96,0%** | 63,7% | 68,7% | 68,3% |
| Алгоритмически верно | **54,3%** | 45,3% | 53,7% | 46,3% |
| Перестановка | 32,3% | 25,5% | **48,0%** | 43,0% |
| Виженер | **70,7%** | 45,5% | 53,0% | 46,0% |

**Главный вывод:** как только валидный JSON деградирует (V2), его невозможно восстановить
дальнейшим дообучением без возврата к исходным условиям V1 (обучение с нуля + датасет V1).
Точность Перестановки реально улучшается благодаря расширенным трассам сетки в датасете,
однако этот выигрыш достигается за счёт других шифров при смещённом распределении датасета.

---

## Лицензия

Apache 2.0 — аналогично базовой модели.
