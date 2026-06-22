import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import numpy as np

class EntropyCalculator:
    def __init__(self, model_name='gpt2', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def calculate_entropy(self, text: str):
        """Вычисляет среднюю энтропию (NLL) текста."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        
        if input_ids.shape[1] < 2:
            return None

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss # Это и есть средняя кросс-энтропия (NLL)
            
        return loss.item()

def get_dataset_statistics(texts, calc):
    """Считает энтропию для списка текстов и взвешенное среднее."""
    results = []
    total_entropy_weight = 0
    total_length = 0

    print("Вычисление энтропии объектов...")
    for t in tqdm(texts):
        entropy = calc.calculate_entropy(t)
        if entropy is not None:
            length = len(t)
            results.append({'text': t, 'entropy': entropy})
            
            # Для взвешенного среднего информационной плотности
            total_entropy_weight += entropy * length
            total_length += length
            
    dataset_density = total_entropy_weight / total_length if total_length > 0 else 0
    return results, dataset_density