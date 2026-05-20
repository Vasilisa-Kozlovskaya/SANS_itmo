import math
import torch
import torch.nn as nn

def compute_positions_vectorized(sequence_ids: torch.Tensor) -> torch.Tensor:
    """
    Векторизованное вычисление локальных позиций токенов внутри packed batch.
    Пример: [1, 1, 1, 2, 2, 0] -> [0, 1, 2, 0, 1, 0]
    """
    B, T = sequence_ids.shape
    
    # Определяем, совпадает ли текущий id последовательности с предыдущим
    same_as_prev = (sequence_ids[:, 1:] == sequence_ids[:, :-1]) & (sequence_ids[:, 1:] != 0)
    same_as_prev = torch.cat([torch.zeros((B, 1), dtype=torch.bool, device=sequence_ids.device), same_as_prev], dim=1)
    
    # Позиции начала новых подпоследовательностей
    starts = ~same_as_prev
    
    # Генерируем базовую матрицу индексов [[0, 1, 2, ...]]
    indices = torch.arange(T, device=sequence_ids.device).unsqueeze(0).expand(B, T)
    
    # Оставляем индексы только там, где начинаются новые последовательности
    start_indices = torch.where(starts, indices, torch.zeros_like(indices))
    
    # Находим индекс старта текущей подпоследовательности через кумулятивный максимум
    latest_start_indices = torch.cummax(start_indices, dim=1)[0]
    
    # Локальная позиция — это смещение от начала текущей подпоследовательности
    positions = indices - latest_start_indices
    
    # Зануляем позиции для паддингов (id == 0)
    return positions * (sequence_ids != 0).long()


class SinusoidalPositionalEncoding(nn.Module):
    """Модуль синусоидального позиционного кодирования."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # register_buffer сохраняет состояние, но не делает тензор обучаемым параметром
        self.register_buffer('pe', pe)
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (B, T)
        # Выход: (B, T, d_model)
        return self.pe[positions]


class BlockMaskedMultiHeadAttention(nn.Module):
    """Модуль многоглавого маскированного внимания (Block-Masked Attention)."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model должно делиться на n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, sequence_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model), sequence_ids: (B, T)
        B, T, _ = x.shape
        
        # Проекции Q, K, V и разделение на головы: (B, n_heads, T, d_k)
        q = self.q_linear(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # Расчет скалярного произведения (scores)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, n_heads, T, T)
        
        # Создание Block-Mask
        # 1. Проверяем принадлежность к одному объекту (s_i == s_j)
        same_seq = (sequence_ids.unsqueeze(-1) == sequence_ids.unsqueeze(-2)) # (B, T, T)
        # 2. Применяем классическую причинно-следственную казуальную маску (j <= i)
        causal_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=x.device)) # (T, T)
        # 3. Исключаем паддинг (s_i != 0)
        not_pad = (sequence_ids.unsqueeze(-1) != 0) # (B, T, 1)
        
        # Результирующая блок-маска
        block_mask = same_seq & causal_mask & not_pad # (B, T, T)
        block_mask = block_mask.unsqueeze(1) # Добавляем размерность для голов: (B, 1, T, T)
        
        # Заполняем запрещенные переходы -inf
        scores = scores.masked_fill(~block_mask, float('-inf'))
        
        # Softmax + Dropout
        attn_probs = torch.softmax(scores, dim=-1)
        # Защита от NaN, если вся строка оказалась замаскирована (например, сплошной паддинг)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs = self.dropout(attn_probs)
        
        # Взвешенное суммирование и объединение голов обратно
        context = torch.matmul(attn_probs, v) # (B, n_heads, T, d_k)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        return self.out_linear(context)


class FeedForward(nn.Module):
    """FFN-модуль."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU() # Стандарт для GPT архитектур
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.act(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Слой-трансформера с применением Post-Norm нормализации."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = BlockMaskedMultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, sequence_ids: torch.Tensor) -> torch.Tensor:
        # Реализация Post-Norm варианта:
        # z1 = LayerNorm(x + Attention(x))
        # z2 = LayerNorm(z1 + FFN(z1))
        z1 = self.ln1(x + self.attn(x, sequence_ids))
        z2 = self.ln2(z1 + self.ffn(z1))
        return z2


class LMHead(nn.Module):
    """Модуль LM-head для получения логитов."""
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Возвращаем «сырые» логиты без Softmax
        return self.linear(x)


class GPTLanguageModel(nn.Module):
    """Полная GPT-like модель, объединяющая все блоки."""
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.lm_head = LMHead(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, sequence_ids: torch.Tensor) -> torch.Tensor:
        # Рассчитываем локальные позиционные индексы динамически по sequence_ids
        positions = compute_positions_vectorized(sequence_ids)
        
        # Эмбеддинги токенов + позиционные эмбеддинги
        out = self.dropout(self.token_embeddings(x) + self.pos_encoding(positions))
        
        # Прогон через стек слоев трансформера
        for block in self.blocks:
            out = block(out, sequence_ids)
            
        # Логиты
        return self.lm_head(out)


def compute_packed_loss(logits: torch.Tensor, targets: torch.Tensor, sequence_ids: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    """
    Расчет функции потерь с маскированием стыков и паддингов при packed batching.
    """
    # Вычисляем маску для loss: M_i = (s_i == s_{i+1}) & (s_i != 0)
    s_i = sequence_ids[:, :-1]
    s_next = sequence_ids[:, 1:]
    loss_mask = (s_i == s_next) & (s_i != 0)
    
    # Сдвигаем логиты и таргеты для авторегрессионного предсказания (следующий токен)
    active_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    active_targets = targets[:, 1:].reshape(-1)
    flat_mask = loss_mask.reshape(-1)
    
    # Фильтруем только валидные переходы внутри одной подпоследовательности
    filtered_logits = active_logits[flat_mask]
    filtered_targets = active_targets[flat_mask]
    
    if filtered_logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
    return criterion(filtered_logits, filtered_targets)