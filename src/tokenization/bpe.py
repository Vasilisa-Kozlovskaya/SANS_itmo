from tokenizers import ByteLevelBPETokenizer
import os

class BpeTokenizer:
    def __init__(self, vocab_size=30000):
        self.tokenizer = ByteLevelBPETokenizer()
        self.vocab_size_limit = vocab_size

    def train_from_file(self, file_path):
        self.tokenizer.train(files=[file_path], vocab_size=self.vocab_size_limit, min_frequency=2,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.tokenizer.save_model(folder)