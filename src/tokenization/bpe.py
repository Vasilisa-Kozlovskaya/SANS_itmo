import sys
import os
import random

# Установка зависимости для BPE
!pip install tokenizers

sys.path.append(os.path.abspath('./src'))
from tokenization.char_word import CharTokenizer, WordTokenizer
from tokenization.bpe import BpeTokenizer

# 1. Загрузка данных
with open("data/final_filtered_text.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

random_sample = random.choice(texts)
print(f"Случайный объект для теста:\n{random_sample[:200]}...")

# --- СИМВОЛЬНАЯ ТОКЕНИЗАЦИЯ ---
char_tok = CharTokenizer()
char_tok.train(texts)
char_encoded = char_tok.encode(random_sample)
print(f"\n[Char Tokenizer]")
print(f"Размер словаря: {char_tok.vocab_size}")
print(f"Длина последовательности: {len(char_encoded)}")

# --- СЛОВНАЯ ТОКЕНИЗАЦИЯ ---
# Используем часть данных для обучения, если памяти мало
word_tok = WordTokenizer(max_vocab_size=30000)
word_tok.train(texts[:10000]) # Обучаем на подмножестве для скорости
word_encoded = word_tok.encode(random_sample)
print(f"\n[Word Tokenizer]")
print(f"Размер словаря: {word_tok.vocab_size}")
print(f"Длина последовательности: {len(word_encoded)}")

# --- BPE ТОКЕНИЗАЦИЯ ---
bpe_tok = BpeTokenizer(vocab_size=30000)
# BPE обучается прямо из файла
bpe_tok.train_from_file("data/final_filtered_text.txt")
bpe_encoded = bpe_tok.encode(random_sample)
print(f"\n[BPE Tokenizer]")
print(f"Размер словаря: {bpe_tok.vocab_size}")
print(f"Длина последовательности: {len(bpe_encoded)}")

# Сохраним BPE для следующих лабораторных
bpe_tok.save("checkpoints/tokenizer_bpe")