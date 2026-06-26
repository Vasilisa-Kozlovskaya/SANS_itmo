from tokenizers import ByteLevelBPETokenizer
import os

class BpeTokenizer:
    def __init__(self, vocab_size=30000):
        # Используем ByteLevelBPETokenizer для поддержки UTF-8 байтов
        self.tokenizer = ByteLevelBPETokenizer()
        self.vocab_size_limit = vocab_size

    def train_from_file(self, file_path):
        self.tokenizer.train(files=[file_path], vocab_size=self.vocab_size_limit, min_frequency=2,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

    def load(self, folder_path):
        """Загрузка из vocab.json и merges.txt."""
        vocab_path = os.path.join(folder_path, "vocab.json")
        merges_path = os.path.join(folder_path, "merges.txt")
        self.tokenizer = ByteLevelBPETokenizer.from_file(vocab_path, merges_path)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.tokenizer.save_model(folder)