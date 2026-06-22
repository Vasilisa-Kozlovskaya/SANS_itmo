import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import numpy as np

class EntropyCalculator:
    def __init__(self, model_name='gpt2', device='cuda'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Загрузка модели {model_name} на {self.device} (FP16)...")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Добавляем паддинг токен (у GPT-2 его нет по умолчанию)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Загружаем модель в режиме полуточности (half precision) для скорости
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device).half()
        self.model.eval()

    @torch.no_grad()
    def calculate_entropy_batch(self, texts: list):
        """Вычисляет энтропию для списка текстов (батча)."""
        if not texts:
            return []

        # Токенизация всего батча сразу
        inputs = self.tokenizer(
            texts, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512, 
            padding=True
        ).to(self.device)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Прогон через модель
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
        
        # Получаем логиты
        logits = outputs.logits # [batch, seq_len, vocab]
        
        # Вычисляем потери вручную для каждого элемента в батче
        # Сдвигаем логиты и метки для предсказания следующего токена
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        # Функция потерь без усреднения
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Вычисляем лосс для всех токенов
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())
        
        # Усредняем только по не-паддинг токенам для каждого предложения
        masked_loss = loss * shift_mask
        sum_loss = masked_loss.sum(dim=1)
        valid_tokens = shift_mask.sum(dim=1)
        
        batch_entropies = (sum_loss / valid_tokens).cpu().tolist()
        return batch_entropies

def get_dataset_statistics_fast(texts, calc, batch_size=16):
    """Ускоренный расчет энтропии с батчингом."""
    # Сортировка по длине (уменьшает количество паддингов в батчах, что ускоряет GPU)
    indexed_texts = sorted(enumerate(texts), key=lambda x: len(x[1]), reverse=True)
    sorted_texts = [x[1] for x in indexed_texts]
    original_indices = [x[0] for x in indexed_texts]
    
    results_map = {}
    total_entropy_weight = 0
    total_length = 0

    print(f"Вычисление энтропии (батч: {batch_size})...")
    for i in tqdm(range(0, len(sorted_texts), batch_size)):
        batch = sorted_texts[i : i + batch_size]
        entropies = calc.calculate_entropy_batch(batch)
        
        for j, entropy in enumerate(entropies):
            text = batch[j]
            length = len(text)
            results_map[original_indices[i + j]] = {'text': text, 'entropy': entropy}
            
            total_entropy_weight += entropy * length
            total_length += length
            
    # Возвращаем результаты в исходном порядке
    results = [results_map[i] for i in range(len(texts)) if i in results_map]
    dataset_density = total_entropy_weight / total_length if total_length > 0 else 0
    
    return results, dataset_density