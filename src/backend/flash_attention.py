import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMIN: tl.constexpr
):
    # Определяем индексы текущего блока по Query и номеру (Batch * Head)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Разделяем соединенный индекс обратно на batch и head
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Инициализируем смещения (offsets) для вычислений внутри блока
    offs_d = tl.arange(0, BLOCK_DMIN)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Указатель на начальный элемент блока Q
    q_ptr = Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    
    # Маска для валидных строк Query (чтобы не выйти за границы последовательности)
    mask_m = offs_m[:, None] < N_CTX
    q = tl.load(q_ptr, mask=mask_m, other=0.0)
    
    # Инициализация переменных для онлайн-софтмакса (алгоритм FlashAttention)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMIN], dtype=tl.float32)
    
    # Масштабируем Q перед перемножением
    q = (q * sm_scale).to(tl.float16)
    
    # Каузальная маска: полностью замаскированные блоки (где все Key > Query) пропускаются.
    # Поэтому мы ограничиваем итерации цикла BLOCK_N только до текущего блока Query.
    end_n = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    
    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Загружаем блоки К и V с учетом их страйдов
        k_ptr = K + off_z * stride_kz + off_h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        v_ptr = V + off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        
        mask_n = offs_n[None, :] < N_CTX
        k = tl.load(k_ptr, mask=mask_n, other=0.0)
        
        # Вычисляем S = Q * K^T
        s = tl.dot(q, k)
        
        # Применяем маскирование внутри блока
        if start_n + BLOCK_N > start_m * BLOCK_M:
            # Если это диагональный блок, маскируем элементы, где индекс Key > индекс Query
            mask_causal = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(mask_causal & mask_m & mask_n, s, float("-inf"))
        else:
            # Для полностью левых блоков проверяем только выход за общие границы последовательности
            s = tl.where(mask_m & mask_n, s, float("-inf"))
        
        # Стандартный шаг коррекции для численной стабильности Softmax (Online Softmax)
        m_ij = tl.max(s, axis=1)
        m_next = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_next)
        p = tl.exp(s - m_next[:, None])
        
        l_ij = tl.sum(p, axis=1)
        l_next = l_i * alpha + l_ij
        
        # Обновляем аккумулятор взвешенной суммы
        acc = acc * alpha[:, None]
        
        v = tl.load(v_ptr, mask=offs_n[:, None] < N_CTX, other=0.0)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc=acc)
        
        # Переходим к следующей итерации
        m_i = m_next
        l_i = l_next
        
    # Финальная нормировка результатов
    acc = acc / l_i[:, None]
    
    # Сохраняем итоговый результат в HBM (глобальную память)
    out_ptr = Out + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptr, acc, mask=mask_m)


def flash_attention_fwd(q, k, v, sm_scale=None):
    """
    Враппер для вызова Triton-кернела прямого прохода Flash Attention.
    Ожидает тензоры с формой (Batch, Head, Seq_Len, Head_Dim) в полуточном формате (float16 или bfloat16).
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        
    B, H, N, d = q.shape
    
    # Выделяем память под выходной тензор той же формы
    out = torch.empty_like(q)
    
    # Конфигурация размеров блоков (классические значения для Triton)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMIN = d  # Размерность хэда (обычно 64 или 128)
    
    # Определение двумерной сетки потоков (Grid)
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    
    # Запуск кернела с передачей страйдов напрямую (без принудительного .contiguous())
    _flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale,
        out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMIN=BLOCK_DMIN
    )
    return out