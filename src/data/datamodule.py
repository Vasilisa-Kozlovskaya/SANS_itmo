import os
import gzip
import torch
import requests
from warcio.archiveiterator import ArchiveIterator
from trafilatura import extract
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pytorch_lightning as pl
from src.utils import clean_text, segment_text
from datasets import load_dataset
from src.tokenization.tokenizer import CustomTokenizer

class CommonCrawlDataModule(pl.LightningDataModule):
    def __init__(self, warc_url, raw_dir='data/raw', processed_dir='data/processed', batch_size=4):
        super().__init__()
        # 1. Определяем устройство: если есть CUDA, используем её
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.warc_url = warc_url
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.batch_size = batch_size

        self.tokenizer_gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")

        # 2. Переносим модель на GPU сразу при загрузке
        self.model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.model_gpt2.eval()

        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

    ## 1 Загрузка и конвертация
    def download_and_extract(self):
        warc_path = os.path.join(self.raw_dir, "sample.warc.gz")

        print(f"Downloading from {self.warc_url}...")
        response = requests.get(self.warc_url, stream=True)

        # ПРОВЕРКА Если статус не 200, значит S3 вернул ошибку (XML)
        if response.status_code != 200:
            print(f"Error: Could not download file. Status code: {response.status_code}")
            print("Server response:", response.text) # Тут вы увидите причину ошибки
            return [] # Возвращаем пустой список, так как загрузка не удалась

        with open(warc_path, "wb") as f:
            # Используем response.content для файлов небольшого размера, или iter_content для больших
            # Если файл большой, то нужно итерироваться
            f.write(response.content)

        texts = []
        # Открываем файл как gzip, так как .gz в расширении
        try:
            with gzip.open(warc_path, 'rb') as stream:
                for record in ArchiveIterator(stream):
                    if record.rec_type == 'response':
                        # Извлекаем только текст, игнорируя HTML разметку и хедеры
                        html_content = record.content_stream().read()
                        content = extract(html_content) # trafilatura отлично чистит HTML
                        if content:
                            texts.append(content)
        except gzip.BadGzipFile:
            print(f"Error: {warc_path} is not a valid gzip file. It might be an HTML error page or an invalid file.")
            try:
                with open(warc_path, 'rb') as f:
                    first_bytes = f.read(200)
                    print(f"First 200 bytes of the file: {first_bytes}")
            except Exception as read_err:
                print(f"Could not read first bytes of file: {read_err}")
            return []
        except Exception as e:
            print(f"An error occurred while processing WARC file: {e}")
            return []

        return texts

    ## Энтропия и плотность
    @torch.no_grad()
    def calculate_metrics(self, text):
        # Токенизируем текст
        inputs = self.tokenizer_gpt2(text, return_tensors="pt", truncation=True, max_length=1024)

        # перенос данных на gpu
        # Перебираем все ключи в словаре (input_ids, attention_mask) и кидаем на то же устройство, что и модель
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Теперь вычисления на видеокарте
        outputs = self.model_gpt2(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()

        density = loss / len(text) if len(text) > 0 else 0
        return loss, density

    def prepare_data(self):
        # tqdm для визуализации прогресса, чтобы не гадать, завис код или нет
        from tqdm import tqdm

        raw_texts = self.download_and_extract()
        processed_data = []

        if not raw_texts:
            print("No raw texts to process. Exiting prepare_data.")
            return

        print(f"Processing {len(raw_texts)} records...")
        for raw_text in tqdm(raw_texts): #прогресс-бар
            cleaned = clean_text(raw_text)
            if cleaned:
                segments = segment_text(cleaned)
                for seg in segments:
                    entropy, density = self.calculate_metrics(seg)
                    if 3.0 < entropy < 7.0:
                        processed_data.append({"text": seg, "entropy": entropy, "density": density})

        # Удаление дубликатов по тексту
        unique_data = {d['text']: d for d in processed_data}.values()
        self.final_data = list(unique_data)

        if not self.final_data:
            print("No data after processing and filtering.")
            return

        # Оценка информационной плотности датасета (взвешенное среднее)
        total_len = sum(len(d['text']) for d in self.final_data)
        avg_density = sum(d['density'] * len(d['text']) for d in self.final_data) / total_len
        print(f"Dataset average information density: {avg_density:.6f}")

    def setup(self, stage=None):
        # Здесь обычно происходит разделение на train/val
        pass

# WARC_URL = "https://data.commoncrawl.org/crawl-data/CC-NEWS/2025/02/CC-NEWS-20250201012811-00559.warc.gz"

class WikiTextProcessing(CommonCrawlDataModule):
    def __init__(self, cc_bpe_tokenizer: CustomTokenizer, **kwargs):
        """Принимает наш кастомный объект токенизатора."""
        super().__init__(warc_url=None, **kwargs)
        self.bpe_tokenizer = cc_bpe_tokenizer

    def fetch_wikitext(self):
        print("Loading WikiText from Hugging Face...")
        # Загружаем wikitext-2 (он компактный)
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Извлекаем тексты, убирая пустые строки, которые часто встречаются в raw wikitext
        return [line['text'] for line in dataset if len(line['text'].strip()) > 0]

    def process_wikitext(self):
        raw_texts = self.fetch_wikitext()
        processed_data = []
        
        print("Cleaning and calculating metrics for WikiText...")
        for raw_text in raw_texts:
            cleaned = clean_text(raw_text)
            if cleaned:
                # В WikiText сегментация может не понадобиться, если строки короткие,
                # но мы прогоним через расчет метрик
                entropy, density = self.calculate_metrics(cleaned)
                processed_data.append(cleaned)
        
        print(f"WikiText processed. Remaining objects: {len(processed_data)}")
        return processed_data

    ##  PACKED BATCHING
    def create_packed_batches(self, texts, block_size=512):
        print(f"Starting Packed Batching (block size: {block_size})...")
        
        all_token_ids = []
        # Вызываем метод token_to_id класса
        eos_token_id = self.bpe_tokenizer.token_to_id("[SEP]") 
        if eos_token_id is None:
            eos_token_id = 0

        for  text in texts:
            # Вызываем метод encode класса (возвращает чистый список id)
            encoded_ids = self.bpe_tokenizer.encode(text)
            all_token_ids.extend(encoded_ids + [eos_token_id])

        # Разбиваем длинный список токенов на блоки фиксированной длины
        total_length = len(all_token_ids)
        # Отрезаем остаток, который не влезает в полный блок
        total_length = (total_length // block_size) * block_size
        
        packed_batches = []
        for i in range(0, total_length, block_size):
            batch = all_token_ids[i : i + block_size]
            packed_batches.append(torch.tensor(batch))

        print(f"Created {len(packed_batches)} packed blocks.")
        return packed_batches
