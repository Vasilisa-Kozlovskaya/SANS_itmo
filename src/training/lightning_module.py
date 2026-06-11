import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

# Импортируем архитектуру и функцию потерь из предыдущего задания
from src.models.gpt_model import GPTLanguageModel, compute_packed_loss


class GPTLightningModule(pl.LightningModule):
    """
    LightningModule для обучения GPT-like модели с поддержкой Packed Batching.
    """
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.lm_head = LMHead(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.lm_head.linear.weight = self.token_embeddings.weight
        
        # === ДОБАВЬ ЭТИ СТРОКИ В КОНЕЦ __init__ ===
        # Применяем базовую инициализацию ко всем модулям
        self.apply(self._init_weights)
        self.criterion = nn.CrossEntropyLoss()
        
        # Специальное масштабирование весов для слоев проекции после Attention и FFN
        # Это предотвращает разгон дисперсии в остаточных связях (Residual Connections)
        for name, p in self.named_parameters():
            if name.endswith('out_linear.weight') or name.endswith('linear2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor, sequence_ids: torch.Tensor) -> torch.Tensor:
        return self.model(x, sequence_ids)

    def _shared_step(self, batch):
        """Вспомогательный метод для распаковки батча и расчета loss."""
        # Распаковываем батч (наш PackedDataset возвращает кортеж из двух элементов)
        if isinstance(batch, dict):
            tokens = batch["tokens"]
            sequence_ids = batch["sequence_ids"]
        else:
            tokens, sequence_ids = batch
        
        # Прямой проход модели (передаем ОБА обязательных аргумента)
        logits = self(tokens, sequence_ids=sequence_ids)
    
        # Расчет специализированного лосса, который маскирует паддинги и стыки
        loss = compute_packed_loss(logits, tokens, sequence_ids, self.criterion)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        # Считаем перплексию (PPL)
        train_ppl = torch.exp(loss)
        
        # Логируем loss на каждом шаге и средний за эпоху
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_ppl", train_ppl, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Используем общий метод для расчета лосса — он теперь сам правильно 
        # достанет токены и sequence_ids, а также передаст их в forward()
        loss = self._shared_step(batch)
    
        # Считаем перплексию (PPL) на валидации
        val_ppl = torch.exp(loss)
    
        # Логируем метрики по эпохам (on_step=False)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ppl", val_ppl, on_step=False, on_epoch=True, prog_bar=True)
    
        return loss

    def configure_optimizers(self):
        """
        Настройка оптимизатора AdamW с раздельным весовым затуханием (Weight Decay)
        и планировщика Cosine Annealing c Warmup.
        """
        # Разделяем параметры: весовое затухание НЕ применяется к biases и LayerNorm
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Biases, LayerNorm и позиционные эмбеддинги исключаются из weight decay
            if "bias" in name or "ln" in name or "LayerNorm" in name or "pos_encoding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        grouped_parameters = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        # Используем AdamW (стандарт для GPT архитектур)
        optimizer = torch.optim.AdamW(
            grouped_parameters, 
            lr=self.hparams.lr, 
            betas=(0.9, 0.95)
        )
        
        # Лямбда-функция для создания Cosine Decay с линейным Warmup stage
        def lr_lambda(current_step: int):
            # 1. Линейный прогрев (Warmup)
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            
            # 2. Косинусное затухание (Cosine Annealing)
            progress = float(current_step - self.hparams.warmup_steps) / float(
                max(1, self.hparams.max_steps - self.hparams.warmup_steps)
            )
            # Ограничиваем прогресс единицей, чтобы косинус не ушел в рост
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Планировщик обновляется на каждом шаге (step), а не эпохе
                "frequency": 1
            }
        }