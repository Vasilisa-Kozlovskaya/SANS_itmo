from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import re
def run_tokenization_tasks(data):
    texts = [d['text'] for d in data]
    sample_text = texts[0] if texts else "Sample text for tokenization."

    print("\n--- Tokenization Tasks ---")

    # 1. Посимвольная токенизация
    char_vocab = sorted(list(set("".join(texts))))
    char_to_id = {char: i for i, char in enumerate(char_vocab)}
    char_encoded = [char_to_id[c] for c in sample_text if c in char_to_id]
    print(f"Char Vocab Size: {len(char_vocab)}")
    print(f"Sample Char Sequence Length: {len(char_encoded)}")

    # 2. Пословная токенизация
    word_vocab = set()
    for t in texts[:100]: # Ограничение для экономии памяти
        word_vocab.update(re.findall(r'\w+', t.lower()))
    word_to_id = {word: i for i, word in enumerate(word_vocab)}
    word_encoded = [word_to_id[w] for w in re.findall(r'\w+', sample_text.lower()) if w in word_to_id]
    print(f"Word Vocab Size (subset): {len(word_vocab)}")
    print(f"Sample Word Sequence Length: {len(word_encoded)}")

    # 3. BPE Токенизация (Byte Pair Encoding)
    # Используем библиотеку tokenizers от Hugging Face
    bpe_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    trainer = trainers.BpeTrainer(vocab_size=5000, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
    bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Обучаем на всем (или части) очищенном датасете
    bpe_tokenizer.train_from_iterator(texts, trainer)

    encoded = bpe_tokenizer.encode(sample_text)
    print(f"BPE Vocab Size: {bpe_tokenizer.get_vocab_size()}")
    print(f"BPE Encoded Sequence Length: {len(encoded.ids)}")
    return bpe_tokenizer

# bpe_tokenizer = run_tokenization_tasks(dm.final_data)