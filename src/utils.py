import re
from langdetect import detect, DetectorFactory
from ftfy import fix_text
##  Очистка и фильтрация
def clean_text(self, text):
    # 1. Нормализация Unicode и исправление "битых" символов
    text = fix_text(text)

    # 2. Удаление неизвестных языков (оставляем только 'en' или 'ru')
    try:
        if detect(text) not in ['en', 'ru']:
            return None
    except:
        return None

        # 3. Нормализация пробелов и удаление пустых строк
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return None

    return text
def segment_text(self, text, max_tokens=800):
        # Разбиение длинных текстов на части (~512-1024 токенов)
    words = text.split()
    segments = [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
    return segments