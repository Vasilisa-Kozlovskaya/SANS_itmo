import re
import unicodedata
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
import os
from warcio.archiveiterator import ArchiveIterator

def convert_warc_to_text(warc_path: str, output_path: str):
    """Извлекает сырой контент из WARC файла и сохраняет в .txt."""
    with open(warc_path, 'rb') as stream:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for record in ArchiveIterator(stream):
                # Нас интересуют только ответы серверов (responses)
                if record.rec_type == 'response':
                    # Извлекаем содержимое
                    payload = record.content_stream().read()
                    try:
                        # Декодируем, игнорируя ошибки (бинарные данные)
                        text = payload.decode('utf-8', errors='ignore')
                        if text.strip():
                            out_f.write(text + "\n" + "="*50 + "\n")
                    except Exception as e:
                        continue
    print(f"Конвертация завершена. Файл сохранен в: {output_path}")

# Чтобы результаты определения языка были воспроизводимы
DetectorFactory.seed = 0

def clean_html_and_headers(raw_text: str) -> str:
    """Удаляет HTTP-заголовки и HTML-разметку."""
    # 1. Удаление HTTP-заголовков (они отделяются от тела двойным переносом строки)
    if "\r\n\r\n" in raw_text:
        raw_text = raw_text.split("\r\n\r\n", 1)[1]
    
    # 2. Удаление HTML тегов
    # Используем BeautifulSoup для извлечения текста
    soup = BeautifulSoup(raw_text, "html.parser")
    
    # Удаляем скрипты и стили
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
        
    return soup.get_text(separator=" ")

def normalize_text(text: str) -> str:
    """Нормализация пробелов, Unicode и удаление пустых строк."""
    # Нормализация Unicode (NFKC объединяет символы, которые выглядят одинаково)
    text = unicodedata.normalize("NFKC", text)
    
    # Замена всех видов пробельных символов (табы, переносы) на обычные пробелы
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def is_target_language(text: str, target_langs=['ru', 'en']) -> bool:
    """Проверка, входит ли язык текста в список разрешенных."""
    if len(text) < 20:  # Слишком короткие тексты сложно определить
        return False
    try:
        lang = detect(text)
        return lang in target_langs
    except:
        return False

def split_into_chunks(text: str, max_chars=512*4) -> list:
    """
    Разбивает длинный текст на части. 
    Пока мы не знаем размер токенов, берем примерный запас по символам.
    """
    # В ЛР1 сказано обучать на 512, обычно это токены. 
    # Приблизительно 1 токен ~ 4 символа.
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

def remove_duplicates(texts: list) -> list:
    """Удаляет дубликаты из списка строк, сохраняя порядок."""
    seen = set()
    unique_texts = []
    for item in texts:
        if item not in seen:
            unique_texts.append(item)
            seen.add(item)
    return unique_texts