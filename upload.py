import torch  # Добавляем импорт
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, Repository
from peft import PeftModel

def upload_model_to_hf(model_path: str, repo_name: str):
    """
    Загружаем модель на Hugging Face Hub.
    
    :param model_path: Путь к сохранённой модели.
    :param repo_name: Имя репозитория на Hugging Face.
    """
    # Логин в Hugging Face
    api = HfApi()
    
    # Загрузка базовой модели (оригинальной) из Hugging Face
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Перезагружаем эмбеддинги для соответствия размерам LoRA
    vocab_size_ckpt = len(tokenizer)
    print(f"[INFO] Adjusting model's token embedding size to {vocab_size_ckpt}")
    base_model.resize_token_embeddings(vocab_size_ckpt)

    # Теперь загружаем адаптер LoRA поверх базовой модели
    model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True)

    # Создание репозитория на Hugging Face (новая папка)
    repo_url = api.create_repo(repo_name, exist_ok=True)
    
    # Клонируем репозиторий в новую папку
    new_dir = f"~/cipherchat/checkpoints/{repo_name}_upload"
    repo = Repository(local_dir=new_dir, clone_from=repo_url)
    
    # Сохраняем модель и токенизатор
    model.save_pretrained(new_dir)
    tokenizer.save_pretrained(new_dir)

    # Выгрузка модели и токенизатора на Hugging Face
    repo.push_to_hub(commit_message="Upload model and tokenizer")

    print(f"[INFO] Model successfully uploaded to {repo_url}")

# Параметры
model_path = "checkpoints/qwen-cipher-lora-selfscore"
repo_name = "qwen-cipher-lora-selfscore"  # Имя репозитория на Hugging Face

upload_model_to_hf(model_path, repo_name)

