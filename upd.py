import json

def generate_selfscore_dataset(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            target = data["json"]
            
            # Пример на основе совпадений (можно более точно настроить для ваших данных)
            real_grade = "grade_5"  # Можно заменить на вашу логику оценивания
            target["self_score"] = f"<{real_grade}>"
            
            # Сохраняем новый пример с self_score
            data["json"] = target
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

# Пример использования:
generate_selfscore_dataset("data/processed/train_cipher_train.jsonl", "data/processed/train_selfscore.jsonl")

