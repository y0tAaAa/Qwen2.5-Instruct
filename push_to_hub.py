"""
push_to_hub.py
--------------
Загрузка CipherChat V1 адаптера на HuggingFace Hub.

Запуск:
  python push_to_hub.py \
      --adapter checkpoints/cipherchat-7b-20260306_153334/final_adapter \
      --repo    YOUR_HF_USERNAME/cipherchat-7b \
      --token   hf_XXXXXXXXXXXXXXXXXXXXXXXX

Или через переменную окружения:
  export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX
  python push_to_hub.py \
      --adapter checkpoints/cipherchat-7b-20260306_153334/final_adapter \
      --repo    YOUR_HF_USERNAME/cipherchat-7b
"""

import argparse
import os

from huggingface_hub import HfApi, create_repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="Путь к адаптеру: checkpoints/cipherchat-7b-.../final_adapter")
    ap.add_argument("--repo",    default="y0ta/qwen2.5-7b-cipherchat",
                    help="HF repo id (default: y0ta/qwen2.5-7b-cipherchat)")
    ap.add_argument("--token",   default=None,
                    help="HF токен (или задай HF_TOKEN в env)")
    ap.add_argument("--private", action="store_true",
                    help="Сделать репозиторий приватным")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Укажи --token или задай переменную HF_TOKEN")

    if not os.path.isdir(args.adapter):
        raise FileNotFoundError(f"Adapter not found: {args.adapter}")

    api = HfApi(token=token)

    print(f"\nCreating repo: {args.repo}  (private={args.private})")
    create_repo(
        repo_id=args.repo,
        token=token,
        private=args.private,
        exist_ok=True,
    )

    # Загружаем адаптер
    print(f"Uploading adapter from: {args.adapter}")
    api.upload_folder(
        folder_path=args.adapter,
        repo_id=args.repo,
        repo_type="model",
        commit_message="Upload CipherChat-7B V1 LoRA adapter",
    )

    # Загружаем все три README
    base_dir = os.path.dirname(os.path.abspath(__file__))
    readmes = [
        ("README_en.md", "README.md"),     # английский → главная страница HF
        ("README_sk.md", "README_sk.md"),  # словацкий
        ("README_ru.md", "README_ru.md"),  # русский
    ]
    for src_name, dst_name in readmes:
        src_path = os.path.join(base_dir, src_name)
        if os.path.isfile(src_path):
            api.upload_file(
                path_or_fileobj=src_path,
                path_in_repo=dst_name,
                repo_id=args.repo,
                repo_type="model",
                commit_message=f"Add {dst_name}",
            )
            print(f"  README uploaded: {dst_name}")
        else:
            print(f"  SKIP {src_name} — file not found")

    print(f"\n✅ Done → https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
