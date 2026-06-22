import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from src.data.processing import clean_html_and_headers, normalize_text, is_target_language, split_into_chunks

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

class CommonCrawlDataModule(pl.LightningDataModule):
    def __init__(self, raw_path: str, batch_size: int = 32):
        super().__init__()
        self.raw_path = raw_path
        self.batch_size = batch_size
        self.processed_data = []

    def prepare_data(self):
        # Здесь происходит тяжелая обработка (один раз)
        with open(self.raw_path, 'r', encoding='utf-8') as f:
            # Читаем файл (в реальности лучше читать по строкам/блокам)
            content = f.read().split("="*50) 
            
        cleaned_chunks = []
        for raw_item in content:
            # 1. Чистим HTML
            text = clean_html_and_headers(raw_item)
            # 2. Нормализуем
            text = normalize_text(text)
            # 3. Фильтруем язык
            if is_target_language(text):
                # 4. Режем на куски
                chunks = split_into_chunks(text)
                cleaned_chunks.extend(chunks)
        
        self.processed_data = cleaned_chunks
        print(f"Обработка завершена. Получено объектов: {len(self.processed_data)}")

    def setup(self, stage=None):
        self.train_dataset = TextDataset(self.processed_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)