from tokenizers import ByteLevelBPETokenizer
import os

class BpeTokenizer:
    def __init__(self, vocab_size=30000):
        self.tokenizer = ByteLevelBPETokenizer()
        self.vocab_size_limit = vocab_size

    def train_from_file(self, file_path):
        """Обучает BPE на текстовом файле."""
        self.tokenizer.train(
            files=[file_path], 
            vocab_size=self.vocab_size_limit, 
            min_frequency=2,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        )

    def save(self, folder_path):
        """Сохраняет файлы токенизатора (vocab.json и merges.txt)."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.tokenizer.save_model(folder_path)
        print(f"Токенизатор успешно сохранен в: {folder_path}")

    def load(self, folder_path):
        """Загружает предобученный токенизатор из папки."""
        self.tokenizer = ByteLevelBPETokenizer.from_file(
            f"{folder_path}/vocab.json", 
            f"{folder_path}/merges.txt"
        )
        print(f"Токенизатор загружен. Размер словаря: {self.vocab_size}")

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()