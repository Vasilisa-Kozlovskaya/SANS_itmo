import torch
import torch.nn.functional as F

@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50, 
                  temperature: float = 1.0, top_k: int = None, device: str = "cuda"):
    """
    Функция авторегрессионной генерации текста (Inference) для GPT-like модели.
    
    Args:
        model: Обученная модель (GPTLikeModel или GPTLightningModule)
        tokenizer: Экземпляр вашего CustomTokenizer из ЛР1
        prompt: Начальный текст (затравка)
        max_new_tokens: Сколько токенов сгенерировать суммарно
        temperature: Масштабирование логитов (чем выше, тем разнообразнее текст)
        top_k: Ограничение выборки только топ-K лучшими токенами
        device: Устройство для вычислений ('cuda' или 'cpu')
    """
    model.eval()  # Обязательно переводим модель в режим инференса (отключаем dropout и т.д.)
    
    # 1. Токенизируем входной текст с помощью вашего CustomTokenizer
    # (Предполагается, что метод возвращает список ID токенов)
    token_ids = tokenizer.encode(prompt) 
    
    # Переводим в тензор и добавляем размерность батча -> (1, seq_len)
    tokens_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        # Ограничиваем контекст, если он превышает max_len модели
        current_context = tokens_tensor[:, -1024:] 
        
        # Так как это инференс одной строки, все токены принадлежат одной сессии (id=1)
        sequence_ids = torch.ones_like(current_context, device=device)
        
        # Получаем логиты из модели
        if hasattr(model, "model"):  # Если передан LightningModule, берем внутреннюю модель
            logits, _ = model.model(current_context, sequence_ids)
        else:
            logits, _ = model(current_context, sequence_ids)
            
        # Нас интересуют логиты только для САМОГО ПОСЛЕДНЕГО сгенерированного токена
        next_token_logits = logits[:, -1, :] / temperature
        
        # Опционально: применяем Top-K фильтрацию для исключения маловероятных токенов
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
            
        # Превращаем логиты в вероятности с помощью Softmax
        probs = F.softmax(next_token_logits, dim=-1)
        
        # Сэмплируем следующий токен на основе распределения вероятностей
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Конкатенируем сгенерированный токен к общему контексту
        tokens_tensor = torch.cat((tokens_tensor, next_token), dim=1)
        
        # (Опционально) Если в вашем токенизаторе есть токен конца текста (EOS), можно прервать цикл
        # if next_token.item() == tokenizer.eos_token_id:
        #     break

    # Извлекаем итоговый список ID (убираем батч-размерность)
    generated_ids = tokens_tensor[0].tolist()
    
    # Декодируем ID обратно в читаемую строку текста
    return tokenizer.decode(generated_ids)