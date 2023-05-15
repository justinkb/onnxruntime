# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from types import ModuleType
from typing import Tuple
import os

import torch

from onnxruntime.training import ortmodule

from .._cache import ModuleCache, PyCodeCache
from .._utils import next_power_of_2


_DEBUG_MODE = ortmodule._defined_from_envvar("ORTMODULE_TRITON_DEBUG", 0) != 0


mm_template = """
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
{autotune_configs}
    ],
    key=["M", "N", "K"],
)
@triton.jit
def {kernel_name}(
    A, B, C, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr
):
    M = {M}
    N = {N}
    K = {K}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * K + rk[None, :])
    B = B + (rk[:, None] * N + rbn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype={acc_dtype})
    for k in range(K, 0, -BLOCK_K):
        if {even_k}:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32={allow_tf32})
        A += BLOCK_K
        B += BLOCK_K * N

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    C = C + (idx_m * N + idx_n)
    tl.store(C, acc{cast_to_target}, mask=mask)


def {func_name}(a, b):
    # Allocates output.
    c = torch.empty(({M}, {N}), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv({M}, META["BLOCK_M"]) * triton.cdiv({N}, META["BLOCK_N"]),)
    {kernel_name}[grid](a, b, c, {M}, {N}, {K})
    return c
"""


bmm_template = """
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
{autotune_configs}
    ],
    key=["M", "N", "K"],
)
@triton.jit
def {kernel_name}(
    A, B, C, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr
):
    M = {M}
    N = {N}
    K = {K}

    stride_aq = M * K
    stride_bq = K * N
    stride_cq = M * N

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1)  # batch dimension for BMM
    A = A + (ram[:, None] * K + rk[None, :] + idx_q * stride_aq)
    B = B + (rk[:, None] * N + rbn[None, :] + idx_q * stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype={acc_dtype})
    for k in range(K, 0, -BLOCK_K):
        if {even_k}:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32={allow_tf32})
        A += BLOCK_K
        B += BLOCK_K * N

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    C = C + (idx_m * N + idx_n + idx_q * stride_cq)
    tl.store(C, acc{cast_to_target}, mask=mask)


def {func_name}(a, b):
    # Allocates output.
    shape = list(a.shape)
    shape[-1] = b.shape[-1]
    batch = a.numel() // ({M} * {K})
    c = torch.empty(*shape, device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv({M}, META["BLOCK_M"]) * triton.cdiv({N}, META["BLOCK_N"]), batch, 1)
    {kernel_name}[grid](a, b, c, {M}, {N}, {K})
    return c
"""


def _mm_configs(dtype, m, n, k):
    condidates = [
        # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
        (32, 32, 16, 1, 2),
        (64, 64, 16, 2, 4),
        (64, 64, 32, 2, 4),
        (64, 128, 32, 3, 4),
        (128, 64, 32, 3, 4),
        (64, 128, 32, 4, 8),
        (128, 64, 32, 4, 8),
        (64, 32, 32, 5, 8),
        (32, 64, 32, 5, 8),
        (128, 128, 32, 2, 8),
        (64, 64, 64, 3, 8),
        (32, 32, 128, 2, 4),
    ]
    tm = max(next_power_of_2(m), 16)
    tn = max(next_power_of_2(n), 16)
    tk = max(next_power_of_2(k), 16)
    config_set = set()
    config_strs = []
    max_bk = 1
    for bm, bn, bk, num_stages, num_warps in condidates:
        bm = min(bm, tm)
        bn = min(bn, tn)
        bk = min(bk, tk)
        if (bm, bn, bk) in config_set:
            continue
        config_set.add((bm, bn, bk))
        if bk > max_bk:
            max_bk = bk
        num_warps = min(num_warps, bm * bn // 256)
        config_strs.append(
            f"        triton.Config({{\"BLOCK_M\": {bm}, \"BLOCK_N\": {bn}, \"BLOCK_K\": {bk}, \"GROUP_M\": 8}}, "
            f"num_stages={num_stages}, num_warps={num_warps}),"
        )
    autotune_configs = "\n".join(config_strs)
    return dict(
        autotune_configs=autotune_configs,
        M=m,
        N=n,
        K=k,
        acc_dtype="tl.float32",
        even_k=k % max_bk == 0,
        allow_tf32=dtype == torch.backends.cuda.matmul.allow_tf32,
        cast_to_target=".to(tl.float16)" if dtype == torch.float16 else "",
    )


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]

def _gen_key(mm_type: str, dtype: torch.dtype, m: int, n: int, k: int) -> int:
    return hash(f"{mm_type}|{_dtype_str(dtype)}|{m}|{n}|{k}") % (10**8)


def _gen_module(mm_type: str, dtype: torch.dtype, m: int, n: int, k: int) -> Tuple[str, ModuleType]:
    func_name = f"{mm_type}_{_dtype_str(dtype)}_{m}_{n}_{k}"
    code_template = mm_template if mm_type == "mm" else bmm_template
    kwargs = _mm_configs(dtype, m, n, k)
    kwargs["kernel_name"] = f"kernel_{func_name}"
    kwargs["func_name"] = func_name
    src_code = code_template.format(**kwargs)
    if _DEBUG_MODE:
        os.makedirs(os.path.dirname("triton_debug/"), exist_ok=True)
        with open(f"triton_debug/{func_name}.py", "w") as f:
            f.write(src_code)
    return func_name, PyCodeCache().load(src_code)


def _call_kernel(mm_type, a, b):
    m = a.shape[-2]
    n = b.shape[-1]
    k = a.shape[-1]
    dtype = a.dtype
    func_name, mod = ModuleCache.load(_gen_key, _gen_module, mm_type, dtype, m, n, k)
    func = getattr(mod, func_name)
    return func(a, b)


def mm(a, b):
    assert(len(a.shape) == 2 and len(b.shape) == 2)
    return _call_kernel("mm", a, b)


def bmm(a, b):
    assert(len(a.shape) >= 3 and len(b.shape) >= 3)
    assert(a.shape[:-2] == b.shape[:-2])
    return _call_kernel("bmm", a, b)
