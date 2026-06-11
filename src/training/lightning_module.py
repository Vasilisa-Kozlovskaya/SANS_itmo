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
    def __init__(
        self, 
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        max_steps: int = 2000,
        dropout: float = 0.1
    ):
        super().__init__()
        # Сохраняет все переданные аргументы в self.hparams
        self.save_hyperparameters()
        
        # Инициализируем GPT-модель из задания 2.1
        self.model = GPTLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Базовый критерий для кросс-энтропии
        self.criterion = nn.CrossEntropyLoss()

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