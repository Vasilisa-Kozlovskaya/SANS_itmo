import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import os
from src.data.processing import clean_html_and_headers, normalize_text, is_target_language, split_into_chunks

class TextDataset(Dataset):
    def __init__(self, file_path):
        # Чтобы не загружать всё в RAM, читаем строки из итогового файла
        with open(file_path, 'r', encoding='utf-8') as f:
            self.texts = f.readlines()
            
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx].strip()

class CommonCrawlDataModule(pl.LightningDataModule):
    def __init__(self, raw_path: str, processed_path: str, batch_size: int = 32, max_objects: int = 50000):
        super().__init__()
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.batch_size = batch_size
        self.max_objects = max_objects # Ограничение, чтобы не переполнить диск/память

    def stream_raw_data(self):
        """Генератор, который читает файл по частям, разделенным '==='."""
        buffer = []
        with open(self.raw_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("="*50):
                    yield "".join(buffer)
                    buffer = []
                else:
                    buffer.append(line)
            if buffer:
                yield "".join(buffer)

    def prepare_data(self):
        """Потоковая обработка данных с записью на диск."""
        if os.path.exists(self.processed_path):
            print(f"Обработанный файл уже существует: {self.processed_path}")
            return

        count = 0
        with open(self.processed_path, 'w', encoding='utf-8') as out_f:
            for raw_item in self.stream_raw_data():
                if count >= self.max_objects:
                    break
                
                # Применяем фильтры
                text = clean_html_and_headers(raw_item)
                text = normalize_text(text)
                
                if is_target_language(text):
                    chunks = split_into_chunks(text)
                    for chunk in chunks:
                        if chunk.strip():
                            out_f.write(chunk.replace('\n', ' ') + "\n")
                            count += 1
                            
                if count % 1000 == 0 and count > 0:
                    print(f"Обработано объектов: {count}")
        
        print(f"Итоговая очистка завершена. Сохранено объектов: {count}")

    def setup(self, stage=None):
        self.train_dataset = TextDataset(self.processed_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)