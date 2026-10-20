"""GDN 反向融合链的私有尾部实现。"""
from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm_add_kernel(q, qrstd, dq, qout, k, krstd, dk0, dk1, kout,
                        T: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                        BT: tl.constexpr, BLOCKS: tl.constexpr):
    block = tl.program_id(0)
    total = triton.cdiv(T, BT)
    base = total // BLOCKS
    extra = total % BLOCKS
    if block < extra:
        count = base + 1
        start = block * count
    else:
        count = base
        start = block * base + extra

    for offset in range(count):
        first = (start + offset) * BT
        p_q = tl.make_block_ptr(q, (T, D), (D, 1), (first, 0), (BT, BD), (1, 0))
        p_qrstd = tl.make_block_ptr(qrstd, (T,), (1,), (first,), (BT,), (0,))
        p_dq = tl.make_block_ptr(dq, (T, D), (D, 1), (first, 0), (BT, BD), (1, 0))
        p_qout = tl.make_block_ptr(qout, (T, D), (D, 1), (first, 0), (BT, BD), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_qrstd = tl.load(p_qrstd, boundary_check=(0,), padding_option="zero").to(tl.float32)
        b_dq = tl.load(p_dq, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_qout = b_dq * b_qrstd[:, None] - tl.sum(b_dq * b_q, 1)[:, None] * b_q * b_qrstd[:, None]
        tl.store(p_qout, b_qout.to(p_qout.dtype.element_ty), boundary_check=(0, 1))

        p_k = tl.make_block_ptr(k, (T, D), (D, 1), (first, 0), (BT, BD), (1, 0))
        p_krstd = tl.make_block_ptr(krstd, (T,), (1,), (first,), (BT,), (0,))
        p_dk0 = tl.make_block_ptr(dk0, (T, D), (D, 1), (first, 0), (BT, BD), (1, 0))
        p_dk1 = tl.make_block_ptr(dk1, (T, D), (D, 1), (first, 0), (BT, BD), (1, 0))
        p_kout = tl.make_block_ptr(kout, (T, D), (D, 1), (first, 0), (BT, BD), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_krstd = tl.load(p_krstd, boundary_check=(0,), padding_option="zero").to(tl.float32)
        b_dk0 = tl.load(p_dk0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_dk1 = tl.load(p_dk1, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        # Match dk.add_(dk1): round to the gradient storage dtype before norm.
        b_dk = (b_dk0 + b_dk1).to(p_dk0.dtype.element_ty).to(tl.float32)
        b_kout = b_dk * b_krstd[:, None] - tl.sum(b_dk * b_k, 1)[:, None] * b_k * b_krstd[:, None]
        tl.store(p_kout, b_kout.to(p_kout.dtype.element_ty), boundary_check=(0, 1))


def _l2norm_bwd_qk_add(q: torch.Tensor, qrstd: torch.Tensor, dq: torch.Tensor,
                       k: torch.Tensor, krstd: torch.Tensor, dk0: torch.Tensor,
                       dk1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """保留原加法舍入及行公式，联合执行 Q/K 反向；支持非整块尾行。"""
    matrices = (q, dq, k, dk0, dk1)
    if any(x.shape != q.shape for x in matrices):
        raise ValueError("Q/K 尾部要求五个矩阵 shape 一致")
    if q.dtype not in (torch.bfloat16, torch.float16) or any(x.dtype != q.dtype for x in matrices):
        raise TypeError("Q/K 尾部要求矩阵使用同一 BF16/FP16 dtype")
    if q.ndim < 2 or q.shape[-1] > 512 or q.shape[-1] <= 0:
        raise ValueError("Q/K 尾部要求 0<D<=512")
    if any(x.device != q.device for x in (*matrices, qrstd, krstd)) or q.device.type != "npu":
        raise ValueError("Q/K 尾部要求输入位于同一 NPU")
    rows, dim = q.numel() // q.shape[-1], q.shape[-1]
    if qrstd.numel() != rows or krstd.numel() != rows:
        raise ValueError("rstd 必须与矩阵行数一致")
    if qrstd.dtype != torch.float32 or krstd.dtype != torch.float32:
        raise TypeError("rstd 必须为 FP32")
    # Queue logical-view copies before draining producers for the raw launch.
    q = q.contiguous()
    qrstd = qrstd.contiguous()
    dq = dq.contiguous()
    k = k.contiguous()
    krstd = krstd.contiguous()
    dk0 = dk0.contiguous()
    dk1 = dk1.contiguous()
    qout = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    kout = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    if rows == 0:
        return qout, kout
    # Drain queued torch_npu producers before the raw-stream Triton launch.
    import torch_npu
    device = q.device.index if q.device.index is not None else torch.npu.current_device()
    try:
        torch_npu._C._npu_getCurrentRawStream(device)
    except AttributeError:
        torch.npu.synchronize(device)
    _qk_norm_add_kernel[(40,)](
        q.view(-1, dim), qrstd.view(-1), dq.view(-1, dim), qout.view(-1, dim),
        k.view(-1, dim), krstd.view(-1), dk0.view(-1, dim), dk1.view(-1, dim), kout.view(-1, dim),
        T=rows, D=dim, BD=triton.next_power_of_2(dim), BT=32, BLOCKS=40, num_warps=4,
    )
    return qout, kout


@triton.jit
def _merge_reverse_chunk_scan(dg0, dg1, out, cu_seqlens, chunk_indices,
                              T: tl.constexpr, H: tl.constexpr,
                              CHUNKS: tl.constexpr, WORKS: tl.constexpr,
                              S0B: tl.constexpr, S0H: tl.constexpr, S0T: tl.constexpr,
                              S1B: tl.constexpr, S1H: tl.constexpr, S1T: tl.constexpr,
                              BT: tl.constexpr, BLOCKS: tl.constexpr,
                              VARLEN: tl.constexpr):
    program = tl.program_id(0)
    base = WORKS // BLOCKS
    extra = WORKS % BLOCKS
    if program < extra:
        count = base + 1
        start = program * count
    else:
        count = base
        start = program * base + extra
    token = tl.arange(0, BT)[None, :]
    # Each iteration retains the validated (1, BT) scan layout.  There is
    # no multi-row chunk bundle and no state carried between chunks.
    for step in range(count):
        work = start + step
        chunk = work % CHUNKS
        head_linear = work // CHUNKS
        head = head_linear % H
        batch = head_linear // H
        if VARLEN:
            sequence = tl.load(chunk_indices + chunk * 2).to(tl.int64)
            local_chunk = tl.load(chunk_indices + chunk * 2 + 1).to(tl.int64)
            begin = tl.load(cu_seqlens + sequence).to(tl.int64)
            end = tl.load(cu_seqlens + sequence + 1).to(tl.int64)
            time = begin + local_chunk * BT + token
            valid = time < end
        else:
            time = chunk * BT + token
            valid = time < T
        a = tl.load(dg0 + batch * S0B + head * S0H + time * S0T,
                    mask=valid, other=0.0).to(tl.float32)
        b = tl.load(dg1 + batch * S1B + head * S1H + time * S1T,
                    mask=valid, other=0.0).to(tl.float32)
        # Preserve the original add_ storage rounding before the FP32 scan.
        merged = (a + b).to(dg0.dtype.element_ty).to(tl.float32)
        scanned = tl.cumsum(merged, axis=1, reverse=True)
        tl.store(out + (batch * H + head) * T + time, scanned, mask=valid)


@lru_cache(maxsize=32)
def _varlen_scan_metadata(cu: tuple[int, ...], pairs: tuple[int, ...],
                          chunk_size: int, tokens: int, device: str, stream: int):
    """按值、设备和stream缓存副本，避免跨stream读取尚未完成的上传。"""
    if len(cu) < 2 or cu[0] != 0 or cu[-1] != tokens:
        raise ValueError("varlen cu_seqlens 必须从0到总token数")
    if any(end < begin for begin, end in zip(cu, cu[1:])):
        raise ValueError("varlen cu_seqlens 必须非递减")
    expected = {(sequence, local) for sequence, (begin, end) in enumerate(zip(cu, cu[1:]))
                for local in range(triton.cdiv(end - begin, chunk_size))}
    if len(pairs) % 2 or len(pairs) != 2 * len(expected) or set(zip(pairs[::2], pairs[1::2])) != expected:
        raise ValueError("chunk_indices 必须恰好覆盖每个序列的所有chunk，可改变排列顺序")
    return (torch.tensor(cu, dtype=torch.int64, device=device),
            torch.tensor(pairs, dtype=torch.int64, device=device))


def _merge_reverse_scan(dg0: torch.Tensor, dg1: torch.Tensor, chunk_size: int,
                        *, cu_seqlens=None, chunk_indices=None) -> torch.Tensor:
    """合并 head-first dg 并反向 scan；每个dense/varlen chunk独立处理。"""
    if dg0.ndim != 3 or dg1.shape != dg0.shape:
        raise ValueError("dg 合并要求两个同 shape 的 [B,H,T] 张量")
    if dg0.dtype not in (torch.float32, torch.bfloat16, torch.float16) or dg1.dtype != dg0.dtype:
        raise TypeError("dg 合并要求同一 FP32/BF16/FP16 dtype")
    if dg0.device.type != "npu" or dg1.device != dg0.device:
        raise ValueError("dg 合并要求输入位于同一 NPU")
    if chunk_size not in (64, 128):
        raise ValueError("dg 合并支持 chunk64/128")
    batch, heads, tokens = dg0.shape
    varlen = cu_seqlens is not None
    cu_device = indices_device = None
    if varlen and isinstance(cu_seqlens, (list, tuple)) and (chunk_indices is None or isinstance(chunk_indices, (list, tuple))):
        if batch != 1:
            raise ValueError("packed varlen dg 要求 B=1")
        cu = tuple(int(x) for x in cu_seqlens)
        if chunk_indices is None:
            pairs = tuple(x for sequence, (begin, end) in enumerate(zip(cu, cu[1:]))
                          for local in range(triton.cdiv(end - begin, chunk_size))
                          for x in (sequence, local))
        else:
            pairs = tuple(int(x) for x in chunk_indices)
        import torch_npu
        device = dg0.device.index if dg0.device.index is not None else torch.npu.current_device()
        try:
            stream = int(torch_npu._C._npu_getCurrentRawStream(device))
        except AttributeError:
            torch.npu.synchronize(device)
            stream = 0
        cu_device, indices_device = _varlen_scan_metadata(cu, pairs, chunk_size, tokens, str(dg0.device), stream)
        chunks = len(pairs) // 2
    elif cu_seqlens is not None or chunk_indices is not None:
        # Preserve the existing adapter for other metadata representations.
        from fla_npu.ops import ascendc
        return ascendc.chunk_local_cumsum(
            (dg0 + dg1).contiguous(), chunk_size=chunk_size,
            reverse=True, scale=1.0, cu_seqlens=cu_seqlens,
            chunk_indices_out=chunk_indices, head_first=True, output_dtype="float32",
        )
    else:
        chunks = triton.cdiv(tokens, chunk_size)
    out = torch.empty(dg0.shape, device=dg0.device, dtype=torch.float32)
    if dg0.numel() == 0:
        return out
    import torch_npu
    device = dg0.device.index if dg0.device.index is not None else torch.npu.current_device()
    try:
        torch_npu._C._npu_getCurrentRawStream(device)
    except AttributeError:
        torch.npu.synchronize(device)
    _merge_reverse_chunk_scan[(40,)](
        dg0, dg1, out, cu_device, indices_device,
        T=tokens, H=heads, CHUNKS=chunks, WORKS=batch * heads * chunks,
        S0B=dg0.stride(0), S0H=dg0.stride(1), S0T=dg0.stride(2),
        S1B=dg1.stride(0), S1H=dg1.stride(1), S1T=dg1.stride(2),
        BT=chunk_size, BLOCKS=40, VARLEN=varlen, num_warps=4, num_stages=1,
    )
    return out
