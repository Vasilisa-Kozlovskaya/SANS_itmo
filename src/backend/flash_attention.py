import torch
import triton
import triton.language as tl

# === 1. МАКСИМАЛЬНО ОПТИМИЗИРОВАННЫЙ FORWARD КЕРНЕЛ (С ИСПРАВЛЕНИЕМ ПОД BACKWARD) ===
@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale, LSE, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMIN: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    offs_d = tl.arange(0, BLOCK_DMIN)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    q_ptr = Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    mask_m = offs_m[:, None] < N_CTX
    q = tl.load(q_ptr, mask=mask_m, other=0.0)
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMIN], dtype=tl.float32)
    
    q = (q * sm_scale).to(tl.float16)
    end_n = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    
    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptr = K + off_z * stride_kz + off_h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        v_ptr = V + off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        
        mask_n = offs_n[None, :] < N_CTX
        k = tl.load(k_ptr, mask=mask_n, other=0.0)
        
        s = tl.dot(q, k)
        if start_n + BLOCK_N > start_m * BLOCK_M:
            mask_causal = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(mask_causal & mask_m & mask_n, s, float("-inf"))
        else:
            s = tl.where(mask_m & mask_n, s, float("-inf"))
        
        m_ij = tl.max(s, axis=1)
        m_next = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_next)
        p = tl.exp(s - m_next[:, None])
        
        l_ij = tl.sum(p, axis=1)
        l_next = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        v = tl.load(v_ptr, mask=offs_n[:, None] < N_CTX, other=0.0)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc=acc)
        
        m_i = m_next
        l_i = l_next
        
    acc = acc / l_i[:, None]
    
    # Сохраняем итоговый результат O
    out_ptr = Out + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptr, acc, mask=mask_m)
    
    # Сохраняем Лог-Сумму Экспонент (LSE) для обратного прохода
    lse_ptr = LSE + off_hz * N_CTX + offs_m
    tl.store(lse_ptr, m_i + tl.log(l_i), mask=offs_m < N_CTX)


# === 2. КЕРНЕЛ ОБРАТНОГО РАСПРОСТРАНЕНИЯ (BACKWARD) ===
@triton.jit
def _flash_attn_bwd_kernel(
    Q, K, V, sm_scale, Out, dO,
    dQ, dK, dV,
    LSE, D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMIN: tl.constexpr
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    offs_d = tl.arange(0, BLOCK_DMIN)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Сборка указателей с проверкой масок границ по N_CTX
    k_ptr = K + off_z * stride_kz + off_h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    v_ptr = V + off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    
    mask_n_2d = offs_n[None, :] < N_CTX # [1, BLOCK_N]
    # Для К нам нужна маска [BLOCK_DMIN, BLOCK_N], для V нужна [BLOCK_N, BLOCK_DMIN]
    k = tl.load(k_ptr, mask=mask_n_2d, other=0.0) 
    v = tl.load(v_ptr, mask=offs_n[:, None] < N_CTX, other=0.0)
    
    dk = tl.zeros([BLOCK_N, BLOCK_DMIN], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMIN], dtype=tl.float32)
    
    # Итерируемся по блокам М
    for start_m_idx in range(0, N_CTX, BLOCK_M):
        offs_m = start_m_idx + tl.arange(0, BLOCK_M)
        mask_m = offs_m[:, None] < N_CTX
        
        q_ptr = Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptr = dO + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        
        q = tl.load(q_ptr, mask=mask_m, other=0.0)
        do = tl.load(do_ptr, mask=mask_m, other=0.0)
        
        lse = tl.load(LSE + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
        di = tl.load(D + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
        
        # Восстановление матрицы S
        s = tl.dot(q, k) * sm_scale
        
        # Каузальное маскирование
        if start_m_idx < (start_n + 1) * BLOCK_N:
            mask_causal = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(mask_causal & mask_m & mask_n_2d, s, float("-inf"))
        else:
            s = tl.where(mask_m & mask_n_2d, s, float("-inf"))
            
        p = tl.exp(s - lse[:, None])
        
        # Вычисление dV и dP
        dv += tl.dot(tl.trans(p.to(tl.float16)), do.to(tl.float16))
        dp = tl.dot(do.to(tl.float16), tl.trans(v.to(tl.float16)))
        
        # Вычисление dS 
        ds = p * (dp - di[:, None])
        
        # Накопление dK (умножаем на sm_scale здесь)
        dk += tl.dot(tl.trans(ds.to(tl.float16)), q.to(tl.float16)) * sm_scale
        
        # Запись в dQ с атомарным сложением и учетом константы sm_scale
        dq_ptr = dQ + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        dq_row = tl.dot(ds.to(tl.float16), tl.trans(k.to(tl.float16))) * sm_scale
        tl.atomic_add(dq_ptr, dq_row.to(tl.float16), mask=mask_m)
        
    # Сохраняем dK и dV для текущего блока N
    dk_ptr = dK + off_z * stride_kz + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    dv_ptr = dV + off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    tl.store(dk_ptr, dk.to(tl.float16), mask=offs_n[:, None] < N_CTX)
    tl.store(dv_ptr, dv.to(tl.float16), mask=offs_n[:, None] < N_CTX)

# === 3. AUTOGRAD ФУНКЦИЯ ДЛЯ СВЯЗИ FORWARD И BACKWARD ===
class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale=None):
        if sm_scale is None:
            sm_scale = 1.0 / (q.shape[-1] ** 0.5)
            
        B, H, N, d = q.shape
        out = torch.empty_like(q)
        
        # Выделяем тензор под LSE (LogSumExp)
        lse = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
        
        BLOCK_M, BLOCK_N = 64, 64
        grid = (triton.cdiv(N, BLOCK_M), B * H)
        
        _flash_attn_fwd_kernel[grid](
            q, k, v, sm_scale, lse, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, H, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMIN=d
        )
        
        # Сохраняем контекст для обратного шага 
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        
        B, H, N, d = q.shape
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Предварительно вычисляем вспомогательный вектор D = rowsum(dO * O)
        # Он необходим для вычисления градиента софтмакса в Online-режиме
        D = torch.sum(do.to(torch.float32) * out.to(torch.float32), dim=-1).view(B * H, N)
        
        BLOCK_M, BLOCK_N = 64, 64
        grid = (triton.cdiv(N, BLOCK_N), B * H)
        
        _flash_attn_bwd_kernel[grid](
            q, k, v, sm_scale, out, do,
            dq, dk, dv,
            lse, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            B, H, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMIN=d
        )
        return dq, dk, dv, None


# === 4. ИТОГОВЫЙ TORCH.NN.MODULE ДЛЯ СЕТИ ===
class FlashCausalAttention(torch.nn.Module):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale=None):
        if sm_scale is None:
            sm_scale = 1.0 / (q.shape[-1] ** 0.5)
            
        B, H, N, d = q.shape
        out = torch.empty_like(q)
        lse = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
        
        BLOCK_M, BLOCK_N = 64, 64
        grid = (triton.cdiv(N, BLOCK_M), B * H)
        
        _flash_attn_fwd_kernel[grid](
            q, k, v, sm_scale, lse, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, H, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMIN=d
        )
        
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        
        B, H, N, d = q.shape
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Вектор D
        D = torch.sum(do.to(torch.float32) * out.to(torch.float32), dim=-1).view(B * H, N)
        
        BLOCK_M, BLOCK_N = 32, 32
        grid = (triton.cdiv(N, BLOCK_N), B * H)
        
        # ИСПРАВЛЕНО: Теперь передаем ВСЕ параметры строго по сигнатуре, включая B (Z)
        _flash_attn_bwd_kernel[grid](
            q, k, v, sm_scale, out, do,
            dq, dk, dv,
            lse, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            B, H, N,  # <-- Передаем B как параметр Z для цикла деления логов
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMIN=d
        )
        return dq, dk, dv, None
    def __init__(self, sm_scale=None):
        super().__init__()
        self.sm_scale = sm_scale

    def forward(self, q, k, v):
        # На вход подаются тензоры (B, H, N, d)
        return FlashAttentionFunction.apply(q, k, v, self.sm_scale)