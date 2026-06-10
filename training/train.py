import os
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from clearml import Task
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Импортируем созданный ранее LightningModule
from src.training.lightning_module import GPTLightningModule
from src.data.datamodule import PackedDataModule 


class GradNormLoggerCallback(pl.Callback):
    """
    Кастомный Callback для отслеживания локальных норм градиентов каждого слоя 
    и вычисления глобальной нормы для логирования в ClearML/TensorBoard.
    """
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        global_norm_sq = 0.0
        
        # Перебираем все параметры модели
        for name, param in pl_module.model.named_parameters():
            if param.grad is not None:
                # Считаем локальную норму слоя L2
                local_norm = param.grad.norm(2).item()
                global_norm_sq += local_norm ** 2
                
                # Логируем локальные нормы (требование задания 2.3.3)
                pl_module.log(f"grad_norms_local/{name}", local_norm, on_step=True, on_epoch=False)
        
        # Вычисляем и логируем глобальную норму градиентов
        global_norm = global_norm_sq ** 0.5
        pl_module.log("grad_norms/global", global_norm, on_step=True, on_epoch=False, prog_bar=True)


def main():
    # 1. Загружаем переменные окружения из .env файла (Задание 2.2)
    load_dotenv()
    
    # 2. Загружаем конфигурационный файл с помощью OmegaConf (Задание 2.2)
    config_path = os.path.join("configs", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфигурационный файл не найден по пути: {config_path}")
        
    cfg = OmegaConf.load(config_path)
    
    # 3. Подключаем и инициализируем ClearML (Задание 2.3.1)
    task = Task.init(
        project_name=cfg.project.name, 
        task_name=cfg.project.title,
        auto_connect_frameworks={'pytorch': False} # Отключаем авто-трекинг, чтобы Lightning управлял логами самостоятельно
    )
    # Сохраняем конфигурацию модели в интерфейсе ClearML
    task.connect(OmegaConf.to_container(cfg, resolve=True))
    
    # 4. Подключаем TensorBoard логгер (Задание 2.3.1)
    # TensorBoardLogger в PL автоматически транслирует графики в интерфейс ClearML
    tb_logger = TensorBoardLogger(save_dir="tb_logs", name=cfg.project.title)
    
    # 5. Настраиваем Callbacks
    # Коллбэк для автоматического сохранения чекпоинтов (Задание 2.3)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.project.checkpoint_dir,
        filename="gpt-step={step:04d}-val_loss={val_loss:.2f}-val_ppl={val_perplexity:.2f}",
        monitor="val_perplexity", # Отслеживаем метрику перплексии
        mode="min",               # Чем меньше перплексия, тем лучше модель
        save_top_k=1,             # Сохраняем только лучший чекпоинт
        save_last=True            # Сохраняем последний шаг для возможности восстановления
    )
    
    # Коллбэк для отображения графика изменения Learning Rate (с учетом warm-up)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Наш кастомный коллбэк для мониторинга локальных норм весов (Задание 2.3.3)
    grad_logger = GradNormLoggerCallback()
    
    # 6. Инициализируем модель параметрами ИЗ конфигурационного файла
    model = GPTLightningModule(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
        n_layers=cfg.model.n_layers,
        dropout=cfg.model.dropout,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        max_steps=cfg.training.max_steps
    )
    
    # Инициализфция DataModule, созданный в ЛР1
    datamodule = PackedDataModule(batch_size=cfg.training.batch_size)
    
    # 7. Конфигурируем PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        
        # Обрезка градиентов по ГЛОБАЛЬНОЙ норме (Задание 2.3.3)
        gradient_clip_val=cfg.training.gradient_clip_val,
        gradient_clip_algorithm="norm",
        
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor, grad_logger],
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        
        # Автоматический выбор доступного железа (CPU / GPU T4 / MPS)
        accelerator="auto",
        devices="auto"
    )
    
    # 8. Реализация восстановления из чекпоинта (Задание 2.3)
    # Если в папке с чекпоинтами лежит 'last.ckpt', автоматически продолжаем с него
    last_ckpt_path = os.path.join(cfg.project.checkpoint_dir, "last.ckpt")
    ckpt_path_to_load = last_ckpt_path if os.path.exists(last_ckpt_path) else None
    
    if ckpt_path_to_load:
        print(f" Найдена сохраненная сессия. Восстановление обучения из: {ckpt_path_to_load}")
    else:
        print(" Запуск обучения с нулевой итерации.")

    # 9. Запуск цикла обучения и валидации
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path_to_load)
    print(" Инфраструктура успешно инициализирована и готова к запуску!")


if __name__ == "__main__":
    main()