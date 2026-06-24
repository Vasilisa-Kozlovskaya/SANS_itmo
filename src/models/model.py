import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.layers import SinusoidalPositionalEncoding, TransformerBlock
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
            TransformerBlock(
                config.model.n_embd, 
                config.model.n_head, 
                config.model.block_size, 
                config.model.dropout
            ) for _ in range(config.model.n_layer)
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

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_embedding(idx)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
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
        )
        
        # Безопасное получение total_steps из конфига
        total_steps = self.config.training.get('total_steps', 10000)
        warmup_steps = self.config.training.get('warmup_steps', 500)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
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