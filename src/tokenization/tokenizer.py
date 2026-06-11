import re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from collections import defaultdict

class CustomTokenizer:
    def __init__(self, mode="bpe", vocab_size=5000):
        """
        Режимы (mode): 'char', 'word', 'bpe'
        """
        self.mode = mode
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        
        # Специфичные структуры для BPE
        self.bpe_merges = []  # Список выученных слияний в виде кортежей (id1, id2)

    def _get_pair_stats(self, ids_list):
        """Вспомогательный метод: подсчет частоты пар соседних токенов."""
        counts = defaultdict(int)
        for ids in ids_list:
            for i in range(len(ids) - 1):
                counts[(ids[i], ids[i+1])] += 1
        return counts

    def _merge_pair(self, ids_list, pair, new_id):
        """Вспомогательный метод: замена пары токенов на новый ID во всем корпусе."""
        new_ids_list = []
        for ids in ids_list:
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            new_ids_list.append(new_ids)
        return new_ids_list

    def train(self, texts):
        """Обучение токенизатора на корпусе текстов."""
        if self.mode == "char":
            char_vocab = sorted(list(set("".join(texts))))
            self.vocab = {char: i for i, char in enumerate(char_vocab)}
            self.inverse_vocab = {i: char for char, i in self.vocab.items()}
            print(f"Char Vocab Size: {len(self.vocab)}")

        elif self.mode == "word":
            word_vocab = set()
            for t in texts[:100]:  
                word_vocab.update(re.findall(r'\w+', t.lower()))
            
            word_vocab = sorted(list(word_vocab))
            self.vocab = {"[UNK]": 0}
            for i, word in enumerate(word_vocab, start=1):
                self.vocab[word] = i
            self.inverse_vocab = {i: word for word, i in self.vocab.items()}
            print(f"Word Vocab Size (subset): {len(self.vocab)}")

        elif self.mode == "bpe":
            print(f"Training custom BPE tokenizer to vocab_size={self.vocab_size}...")
            
            # Инициализируем базовый посимвольный словарь + специальные токены
            special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
            self.vocab = {tok: i for i, tok in enumerate(special_tokens)}
            
            # Собираем все уникальные символы из текста
            unique_chars = sorted(list(set("".join(texts))))
            for char in unique_chars:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
            
            # Переводим тексты в списки базовых ID символов (используем подвыборку для скорости обучения)
            # Из-за чистого Python обучение на большом датасете может быть долгим
            train_texts = texts[:500] 
            ids_list = [[self.vocab[c] for c in t] for t in train_texts]
            
            # Цикл итеративного слияния пар (BPE алгоритм)
            num_merges = self.vocab_size - len(self.vocab)
            current_new_id = len(self.vocab)
            
            for stage in range(num_merges):
                stats = self._get_pair_stats(ids_list)
                if not stats:
                    break
                    
                # Находим самую частую пару
                best_pair = max(stats, key=stats.get)
                if stats[best_pair] < 2: 
                    break # Если пары встречаются слишком редко, останавливаемся
                
                # Запоминаем слияние
                self.bpe_merges.append(best_pair)
                
                # Создаем текстовое представление нового токена для визуализации/словаря
                # Находим текстовые элементы по их ID в inverse_vocab или vocab
                self.inverse_vocab = {v: k for k, v in self.vocab.items()}
                part1 = self.inverse_vocab.get(best_pair[0], f"id_{best_pair[0]}")
                part2 = self.inverse_vocab.get(best_pair[1], f"id_{best_pair[1]}")
                new_token_str = part1 + part2
                
                self.vocab[new_token_str] = current_new_id
                
                # Проводим замену во всем обучающем корпусе
                ids_list = self._merge_pair(ids_list, best_pair, current_new_id)
                current_new_id += 1
                
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            print(f"Custom BPE trained! Final Vocab Size: {len(self.vocab)}")

    def encode(self, text):
        """Кодирование текста в идентификаторы (IDs)."""
        if self.mode == "char":
            return [self.vocab[c] for c in text if c in self.vocab]
            
        elif self.mode == "word":
            words = re.findall(r'\w+', text.lower())
            return [self.vocab.get(w, self.vocab["[UNK]"]) for w in words]
            
        elif self.mode == "bpe":
             # 1. Предварительная подготовка: переводим текст в список ID базовых символов
             # Кэшируем unk_id, чтобы не лезть в словарь на каждом шаге
            unk_id = self.vocab.get("[UNK]", 0)
            ids = [self.vocab.get(c, unk_id) for c in text]
    
            if len(ids) < 2:
                return ids

             # 2. Создаем словарь правил для быстрого поиска
             # Вместо цикла по self.bpe_merges, используем lookup-таблицу
             # { (id_left, id_right): new_id }
            merge_map = {}
            for pair in self.bpe_merges:
                pair_str = self.inverse_vocab[pair[0]] + self.inverse_vocab[pair[1]]
                if pair_str in self.vocab:
                    merge_map[tuple(pair)] = self.vocab[pair_str]

             # 3. Итеративное применение слияний
             # Мы продолжаем процесс, пока можно слить хотя бы одну пару
            while True:
                 # Ищем все возможные пары в текущем списке ids
                pairs = {}
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i+1])
                    if pair in merge_map:
                        # Храним индекс первого вхождения пары
                        pairs[pair] = i
                        break # Находим только первое слияние согласно приоритету
        
                if not pairs:
                    break # Слияния больше невозможны
            
                 # Применяем первое найденное слияние (самое приоритетное)
                pair, idx = list(pairs.items())[0]
                new_id = merge_map[pair]
        
                 # Обновляем список ids
                ids = ids[:idx] + [new_id] + ids[idx+2:]
        
        return ids

    def token_to_id(self, token):
        """Получение ID токена (для Packed Batching)."""
        return self.vocab.get(token, None)