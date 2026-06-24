import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.layers import SinusoidalPositionalEncoding, TransformerBlock
import math
from src.training.scheduler import get_cosine_schedule_with_warmup

class GPT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config) # Сохраняет конфиг в чекпоинт
        self.config = config

        # Эмбеддинги
        self.token_embedding = nn.Embedding(config.model.vocab_size, config.model.n_embd)
        self.pos_encoding = SinusoidalPositionalEncoding(config.model.n_embd, config.model.block_size)
        self.dropout = nn.Dropout(config.model.dropout)

        # Стек слоев трансформера
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                config.model.n_embd, 
                config.model.n_head, 
                config.model.block_size, 
                config.model.dropout
            ) for _ in range(config.model.n_layer)
        ])

        # Финальная нормализация и выходная голова
        self.ln_f = nn.LayerNorm(config.model.n_embd)
        self.lm_head = nn.Linear(config.model.n_embd, config.model.vocab_size, bias=False)

        # Веса lm_head и token_embedding обычно связывают (weight tying)
        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

        self.criterion = nn.CrossEntropyLoss()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # нормальное распределение с std=0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
        # Специфика для Post-Norm (масштабирование весов проекционных слоев)
        # Это помогает стабилизировать глубокие сети
        for name, child in module.named_children():
            if name in ['proj', 'mlp']: # слои, которые стоят в конце остаточных связей
                if isinstance(child, nn.Linear):
                    torch.nn.init.normal_(child.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.model.n_layer))

    def forward(self, idx):
        B, T = idx.size()
        
        # Проход через эмбеддинги и позиционное кодирование
        x = self.token_embedding(idx) # (B, T, n_embd)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Проход через блоки трансформера
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Получение логитов
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits

    def training_step(self, batch, batch_idx):
        # В packed batching таргеты — это входные данные, смещенные на 1
        # input: tokens[0...T-1], target: tokens[1...T]
        idx = batch[:, :-1]
        targets = batch[:, 1:]

        logits = self(idx)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        idx = batch[:, :-1]
        targets = batch[:, 1:]

        logits = self(idx)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        perplexity = torch.exp(loss)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_perplexity', perplexity, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.training.learning_rate, 
            weight_decay=self.config.training.weight_decay
       )
    
    # Рассчитываем общее количество шагов обучения (нужно для планировщика)
    # Эти данные мы подтянем из трейнера позже или зададим примерно
        total_steps = self.trainer.estimated_stepping_batches
    
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.config.training.warmup_steps, 
            num_training_steps=total_steps
       )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Обновлять LR каждый шаг, а не каждую эпоху
                },
        }  
    

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Метод для инференса: генерация текста."""
        for _ in range(max_new_tokens):
            # Если последовательность слишком длинная, обрезаем её
            idx_cond = idx[:, -self.config.model.block_size:]
            logits = self(idx_cond)
            # Берем логиты только последнего токена и применяем температуру
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            # Сэмплируем следующий токен
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
