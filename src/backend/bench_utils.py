# src/backend/bench_utils.py
import torch

def compute_causal_attention_flops(b, h, n, d, mode="forward"):
    """
    Вычисляет теоретическое количество операций (FLOPs) для маскированного каузального внимания.
    Поскольку матрица маскирована, вычисляется ровно половина элементов (нижний треугольник).
    
    - Умножение Q * K^T: 2 * B * H * (N^2 / 2) * d
    - Умножение P * V: 2 * B * H * (N^2 / 2) * d
    Итого для Forward: 2 * B * H * N^2 * d
    Для Backward операций требуется примерно в 2.5 раза больше.
    """
    # 2 матрицы перемножения * B * H * (N^2 / 2) * d * 2 (умножение + сложение)
    fwd_flops = 2 * b * h * n * n * d
    if mode == "forward":
        return fwd_flops
    elif mode == "backward":
        # На обратном проходе градиенты считаются и по Q, и по K, и по V
        return int(2.5 * fwd_flops)
    else:
        raise ValueError("Режим должен быть 'forward' или 'backward'")