import pytorch_lightning as pl
import torch
import math
from src.models.gpt_model import GPTLikeModel

class GPTLightningModule(pl.LightningModule):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, 
                 num_layers: int, max_len: int = 1024, lr: float = 3e-4, weight_decay: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = GPTLikeModel(
            vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, 
            d_ff=d_ff, num_layers=num_layers, max_len=max_len
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, tokens, sequence_ids, targets=None):
        return self.model(tokens, sequence_ids, targets)

    def training_step(self, batch, batch_idx):
        tokens, sequence_ids = batch
        logits, loss = self.model(tokens, sequence_ids, targets=tokens)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_ppl", torch.exp(loss), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, sequence_ids = batch
        logits, loss = self.model(tokens, sequence_ids, targets=tokens)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_ppl", torch.exp(loss), on_epoch=True, prog_bar=True)
        return loss

    # =====================================================================
    # ЗАДАНИЕ 2.3: ЛОГИРОВАНИЕ ЛОКАЛЬНЫХ ГРАДИЕНТНЫХ НОРМ СЛОЕВ
    # =====================================================================
    def on_before_optimizer_step(self, optimizer):
        """
        Вызывается автоматически после backward-прохода, но ДО шага оптимизатора.
        Здесь мы вычисляем локальную норму для каждого крупного подмодуля.
        """
        layer_grad_norms = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Разбиваем имя параметра (например, 'layers.0.attn.q_linear.weight')
                parts = name.split('.')
                
                # Группируем по логическим блокам модели, чтобы графики были читаемыми
                if parts[0] == 'layers' and len(parts) > 1:
                    block_name = f"transformer_layer_{parts[1]}"  # например, transformer_layer_0
                else:
                    block_name = parts[0]  # token_emb, pos_enc или lm_head
                
                # Квадрат нормы текущего тензора градиентов
                param_norm_sq = torch.sum(param.grad.detach() ** 2)
                layer_grad_norms[block_name] = layer_grad_norms.get(block_name, 0.0) + param_norm_sq

        # Логируем итоговую L2 норму для каждого слоя: ||g_layer|| = sqrt(sum(||g_param||^2))
        for block_name, norm_sq in layer_grad_norms.items():
            local_norm = torch.sqrt(norm_sq)
            self.log(f"grad_norm_local/{block_name}", local_norm, on_step=True, on_epoch=False)

    # =====================================================================
    # ЗАДАНИЕ 2.3: ВЫЧИСЛЕНИЕ ГЛОБАЛЬНОЙ НОРМЫ И ОБРЕЗКА (GRADIENT CLIPPING)
    # =====================================================================
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        """
        Кастомное управление обрезкой градиентов. 
        Позволяет замерить глобальную норму ДО и ПОСЛЕ применения формулы из методички.
        """
        # 1. Собираем все градиенты в один плоский вектор для вычисления глобальной нормы ДО обрезки
        grads = [p.grad.detach().flatten() for p in self.parameters() if p.grad is not None]
        if grads:
            global_norm_before = torch.linalg.vector_norm(torch.cat(grads), ord=2)
            self.log("grad_norm_global/unclipped", global_norm_before, on_step=True)
        
        # 2. Выполняем жесткую обрезку по глобальной норме c = 1.0 (согласно требованиям методички)
        self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        
        # 3. Замеряем норму ПОСЛЕ обрезки, чтобы на графиках показать преподавателю, что лимит сработал
        grads_after = [p.grad.detach().flatten() for p in self.parameters() if p.grad is not None]
        if grads_after:
            global_norm_after = torch.linalg.vector_norm(torch.cat(grads_after), ord=2)
            self.log("grad_norm_global/clipped", global_norm_after, on_step=True)

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "LayerNorm" in name or "ln" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_steps, eta_min=self.lr * 0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }