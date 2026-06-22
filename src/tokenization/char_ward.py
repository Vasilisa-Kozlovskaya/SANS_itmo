import re
from collections import Counter

class CharTokenizer:
    def __init__(self):
        self.vocab = {}
        self.id_to_char = {}

    def train(self, texts):
        chars = set("".join(texts))
        self.vocab = {char: i for i, char in enumerate(sorted(chars))}
        self.id_to_char = {i: char for char, i in self.vocab.items()}

    def encode(self, text):
        return [self.vocab[char] for char in text if char in self.vocab]

    @property
    def vocab_size(self):
        return len(self.vocab)

class WordTokenizer:
    def __init__(self, max_vocab_size=50000):
        self.vocab = {}
        self.max_vocab_size = max_vocab_size

    def train(self, texts):
        word_counts = Counter()
        for text in texts:
            # Простая токенизация по словам через regex
            words = re.findall(r'\w+|[^\w\s]', text.lower(), re.UNICODE)
            word_counts.update(words)
        
        # Ограничиваем словарь, если он слишком большой
        most_common = word_counts.most_common(self.max_vocab_size)
        self.vocab = {word: i for i, (word, _) in enumerate(most_common)}
        # Добавляем токен для неизвестных слов
        self.vocab['[UNK]'] = len(self.vocab)

    def encode(self, text):
        words = re.findall(r'\w+|[^\w\s]', text.lower(), re.UNICODE)
        return [self.vocab.get(word, self.vocab['[UNK]']) for word in words]

    @property
    def vocab_size(self):
        return len(self.vocab)