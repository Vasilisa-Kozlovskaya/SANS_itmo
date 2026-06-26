import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.backend.flash_attention import FlashAttentionFunction

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.model.n_head
        self.n_kv_head = config.model.n_kv_head
        # Количество Query-голов, приходящихся на одну KV-голову
        self.n_groups = self.n_head // self.n_kv_head
        self.head_dim = config.model.n_embd // self.n_head
        
         # Линейные проекции.k и v имеют меньшую размерность (n_kv_head * head_dim)
        self.q_proj = nn.Linear(config.model.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.model.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.model.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.model.n_embd, config.model.n_embd, bias=False)
        
        self.dropout = config.model.dropout

    def forward(self, x, past_key_value=None, use_cache=False):
        B, T, C = x.size()
        
        # 1. Проецируем вход в Q, K, V и меняем размерность для голов
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, H, T, d)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # 2. Логика KV-кэширования (инференс)
        # Если пришел кэш из прошлого шага, "приклеиваем" новые K и V к старым
        if past_key_value is not None:
            prev_k, prev_v = past_key_value
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        
        # Сохраняем текущее состояние KV для будущих шагов
        full_kv = (k, v) if use_cache else None

        # Repeat KV heads для соответствия числу Query heads (GQA logic)
        # (B, G, T, d) -> (B, H, T, d)
        # Размножаем KV-головы, чтобы их количество совпало с Q-головами (для матричного умножения)
        k_repeated = k.repeat_interleave(self.n_groups, dim=1)
        v_repeated = v.repeat_interleave(self.n_groups, dim=1)

        # Используем Triton Flash Attention из ЛР3
        # Кернел ожидает (B, H, T, d)
        if self.training:
            # Во время обучения используем Flash Attention
            # Он экономит память и работает быстрее за счет тайлинга (tiling)
            out = FlashAttentionFunction.apply(q, k_repeated, v_repeated, 1.0/math.sqrt(self.head_dim))
        else:
            # Во время инференса (особенно с KV-cache) можно использовать стандартный torch или Triton
            # так как T обычно равен 1 для нового токена
            out = F.scaled_dot_product_attention(q, k_repeated, v_repeated, is_causal=True if past_key_value is None else False)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), full_kv

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
    
class TransformerBlockGQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.model.n_embd)
        self.attn = GroupedQueryAttention(config)
        self.ln2 = nn.LayerNorm(config.model.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.model.n_embd, 4 * config.model.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.model.n_embd, config.model.n_embd),
            nn.Dropout(config.model.dropout),
        )

    def forward(self, x, past_key_value=None, use_cache=False):
        # 1. Attention + Residual Connection
        attn_out, kv = self.attn(self.ln1(x), past_key_value=past_key_value, use_cache=use_cache)
        x = x + attn_out
        
        # Применяем MLP
        #  MLP + Residual Connection
        x = x + self.mlp(self.ln2(x))
        
        return x, kv # Всегда возвращаем (результат, кэш)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Post-norm variant: x = LayerNorm(x + Sublayer(x))
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.mlp(x))
        return x