import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.layers import SinusoidalPositionalEncoding, TransformerBlockGQA
from src.training.scheduler import get_cosine_schedule_with_warmup
import math

class GPT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.token_embedding = nn.Embedding(config.model.vocab_size, config.model.n_embd)
        self.pos_encoding = SinusoidalPositionalEncoding(config.model.n_embd, config.model.block_size)
        self.dropout = nn.Dropout(config.model.dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlockGQA(config) for _ in range(config.model.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.model.n_embd)
        self.lm_head = nn.Linear(config.model.n_embd, config.model.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)
        self.criterion = nn.CrossEntropyLoss()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, past_key_values=None, use_cache=False):
        x = self.token_embedding(idx)
        x = self.pos_encoding(x)
    
        new_key_values = []
        for i, block in enumerate(self.blocks):
        # Передаем кэш в каждый блок. Если past_key_values - None, блок создаст новый
            past_kv = past_key_values[i] if past_key_values is not None else None
        
        # Блок теперь возвращает (x, kv)
            x, kv = block(x, past_key_value=past_kv, use_cache=use_cache)
        
            if use_cache:
                new_key_values.append(kv)
    
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, new_key_values
        return logits

    def training_step(self, batch, batch_idx):
        idx = batch[:, :-1]
        targets = batch[:, 1:]
        logits = self(idx)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_perplexity', torch.exp(loss.detach()), on_step=False, on_epoch=True)
        self.log('logits_max', logits.detach().max(), on_step=True, on_epoch=False)
        self.log('logits_min', logits.detach().min(), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        idx = batch[:, :-1]
        targets = batch[:, 1:]
        logits = self(idx)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_perplexity', torch.exp(loss.detach()), on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.training.learning_rate, 
            weight_decay=self.config.training.weight_decay
            betas=(0.9, 0.95)
        )
        
        if self.trainer.max_steps > 0:
            total_steps = self.trainer.max_steps
        else:
        # Считаем на основе данных: (кол-во батчей / аккумулирование) * эпохи
            dataset_size = len(self.trainer.datamodule.train_dataloader()) if self.trainer.datamodule else len(self.trainer.train_dataloader)
            total_steps = (dataset_size // self.trainer.accumulate_grad_batches) * self.trainer.max_epochs

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.config.training.warmup_steps, 
            num_training_steps=total_steps,
            min_lr_ratio=0.1
        )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.model.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx