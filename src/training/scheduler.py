import math
import torch
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Расписание: Линейный прогрев до 1.0, затем косинусное затухание до min_lr_ratio.
    
    Args:
        optimizer: Оптимизатор.
        num_warmup_steps: Количество шагов для разогрева.
        num_training_steps: Общее количество шагов (batch_size * epochs / grad_accum).
        min_lr_ratio: Минимальный LR в конце (например, 0.1 от базового).
    """
    def lr_lambda(current_step):
        # 1. Линейный прогрев
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 2. Косинусное затухание
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(1.0, max(0.0, progress))
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Гарантируем, что LR не упадет ниже чем min_lr_ratio * base_lr
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)