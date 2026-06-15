import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PackedSinusoidalPositionalEncoding(nn.Module):
    """
    Модуль синусоидального позиционного кодирования, адаптированный под Packed Batching.
    Позиции вычисляются локально для каждой подпоследовательности.
    """
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        
        # Генерация статической матрицы позиционных эмбеддингов
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Регистрируем как буфер, чтобы тензор не обновлялся градиентами, но переносился на GPU
        self.register_buffer('pe', pe)

    def forward(self, sequence_ids: torch.Tensor) -> torch.Tensor:
        """
        sequence_ids: (batch_size, seq_len) — ID исходных последовательностей (например, [1, 1, 2, 2, 2, 0])
        """
        B, T = sequence_ids.shape
        
        # Вычисляем локальные позиции внутри упакованного батча динамически
        s_i = sequence_ids.unsqueeze(2)  # (B, T, 1)
        s_j = sequence_ids.unsqueeze(1)  # (B, 1, T)
        tril = torch.tril(torch.ones(T, T, device=sequence_ids.device)).bool()
        
        # Токен принадлежит той же подпоследовательности и находится левее/равно текущему
        mask = (s_i == s_j) & tril & (s_i != 0)
        
        # Локальная позиция — это количество предшествующих токенов той же сессии минус 1
        positions = mask.sum(dim=-1) - 1
        positions = torch.clamp(positions, min=0)  # Убираем -1 для паддингов (id=0)
        
        # Выбираем эмбеддинги с помощью advanced indexing: (B, T) -> (B, T, d_model)
        return self.pe[positions]


class BlockMaskedMultiHeadAttention(nn.Module):
    """
    Модуль многоглавого маскированного внимания (Block-Masked Attention)
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model должен делиться на num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, sequence_ids: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Проекции Q, K, V и разделение на головы
        q = self.q_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, T, d_k)
        k = self.k_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # Расчет скалярного произведения (scores)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Построение Block-Masked Attention
        s_i = sequence_ids.unsqueeze(2)  # (B, T, 1)
        s_j = sequence_ids.unsqueeze(1)  # (B, 1, T)
        tril = torch.tril(torch.ones(T, T, device=x.device)).bool()
        
        # Формула: (M)_{i,j} = (s_i == s_j) & (j <= i) & (s_i != 0)
        block_mask = (s_i == s_j) & tril & (s_i != 0)
        block_mask = block_mask.unsqueeze(1)  # Добавляем размерность для голов: (B, 1, T, T)
        
        # Заполняем запрещенные связи минус бесконечностью
        scores = scores.masked_fill(~block_mask, float('-inf'))
        
        # Softmax по последней размерности
        attn_weights = F.softmax(scores, dim=-1)
        
        # Защита от NaN, если в батче есть строки, целиком состоящие из PAD (0)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Взвешенное суммирование значений V
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_linear(context)


class PositionWiseFeedForward(nn.Module):
    """
    Стандартный FFN-модуль трансформера
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()  # Стандарт для GPT-моделей
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))


class TransformerLayer(nn.Module):
    """
    Слой трансформера с использованием жесткого Post-Norm варианта нормализации
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attn = BlockMaskedMultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, sequence_ids: torch.Tensor) -> torch.Tensor:
        # z1 = LayerNorm(z + Attention(z))
        z1 = self.ln1(x + self.attn(x, sequence_ids))
        # z2 = LayerNorm(z1 + FFN(z1))
        z2 = self.ln2(z1 + self.ffn(z1))
        return z2


class GPTLikeModel(nn.Module):
    """
    Основной класс GPT-like модели с маскированием функции потерь под Packed Batching
    """
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_len: int = 1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PackedSinusoidalPositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        # LM-head для получения сырых логитов (без повторного применения Softmax!)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, sequence_ids: torch.Tensor, targets: torch.Tensor = None):
        # Эмбеддинги токенов + кастомные позиционные эмбеддинги
        x = self.token_emb(tokens) + self.pos_enc(sequence_ids)
        
        # Прогон сквозь слои трансформера
        for layer in self.layers:
            x = layer(x, sequence_ids)
            
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Сдвиг логитов и таргетов для классической авторегрессионной задачи предсказания
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            
            # Сдвиг ID последовательностей для корректного маскирования стыков
            shift_seq_ids = sequence_ids[:, :-1].contiguous()
            next_seq_ids = sequence_ids[:, 1:].contiguous()
            
            # Формула маски лосса: M_i^{loss} = (s_i == s_{i+1}) & (s_i != 0)
            loss_mask = (shift_seq_ids == next_seq_ids) & (shift_seq_ids != 0)
            
            # Вычисляем cross-entropy поэлементно (reduction='none')
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_targets = shift_targets.view(-1)
            
            raw_loss = loss_fn(flat_logits, flat_targets).view(shift_targets.shape)
            
            # Обнуляем лосс на границах разных текстов и на паддингах
            masked_loss = raw_loss * loss_mask.float()
            
            # Усредняем ошибку исключительно по тем позициям, где маска равна 1
            if loss_mask.sum() > 0:
                loss = masked_loss.sum() / loss_mask.sum()
            else:
                loss = masked_loss.sum()
                
        return logits, loss