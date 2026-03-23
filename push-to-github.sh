#!/bin/bash

# Конфигурация
REPO_NAME="cipherchat"
GITHUB_USER="y0tAaAa"
LOCAL_PATH="/home/vd243wi/cipherchat"
GITHUB_REPO="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Начало выгрузки на GitHub ===${NC}"

# Проверка наличия папки
if [ ! -d "$LOCAL_PATH" ]; then
    echo -e "${RED}Ошибка: папка $LOCAL_PATH не найдена!${NC}"
    exit 1
fi

cd "$LOCAL_PATH"

# Инициализация Git репозитория (если еще не инициализирован)
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Инициализация Git репозитория...${NC}"
    git init
    git config user.name "$(git config --global user.name)"
    git config user.email "$(git config --global user.email)"
fi

# Добавление файлов
echo -e "${YELLOW}Добавление файлов...${NC}"
git add -A

# Проверка изменений
if git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}Нет изменений для коммита${NC}"
else
    # Создание коммита с датой
    COMMIT_MSG="Update: $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${YELLOW}Создание коммита: $COMMIT_MSG${NC}"
    git commit -m "$COMMIT_MSG"
fi

# Добавление удаленного репозитория (если не добавлен)
if ! git config --get remote.origin.url > /dev/null; then
    echo -e "${YELLOW}Добавление удаленного репозитория...${NC}"
    git remote add origin "$GITHUB_REPO"
fi

# Отправка на GitHub
echo -e "${YELLOW}Отправка на GitHub...${NC}"
git push -u origin main 2>/dev/null || git push -u origin master

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Успешно выгружено на $GITHUB_REPO${NC}"
else
    echo -e "${RED}❌ Ошибка при выгрузке${NC}"
    exit 1
fi

echo -e "${GREEN}=== Выгрузка завершена ===${NC}"