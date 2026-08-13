"""Self-contained Triton seed for the GLM-5.1-FP8 MoE stage-1 GEMM (g1u1).

Target op:  aiter::fmoe_fp8_blockscale_g1u1  (aiter/fused_moe.py: fused_moestage)
            The production kernel is a hand-written AMD ASM kernel
            (_ZN5aiter50fmoe_bf16_blockscaleFp8_g1u1_vs_silutg_psx256E), measured at
            ~7.1% of GPU time and ~26% roofline efficiency at the decode shape in
            the GLM-5.1-FP8 phaseC2 trace (compute-bound, real headroom).

What this kernel computes (MoE expert stage-1, "g1u1" = gate + up, fused SiLU*mul):
    For each routed (token, expert) pair, with per-expert weight W1[e] = [2N, K]
    (first N rows = gate, second N rows = up):
        gate = (x @ Wgate[e]^T)                 # [., N]
        up   = (x @ Wup[e]^T)                   # [., N]
        out  = silu(gate) * up                  # [., N]
    x is FP8 (e4m3) with per-token, per-128-K-group activation scales.
    W1 is FP8 (e4m3) with per-expert [128 x 128] block scales.
    Accumulation is FP32. Stage-1 does NOT apply the router weight (that happens
    in stage-2 / the down projection), matching vLLM/aiter convention.

This file is the kernel-under-optimization. GEAK copies it into baseline/ and
workspace/; test_harness.py (co-located) loads THIS file's `moe_g1u1_fp8` launcher,
checks it against a pure-torch fp32 oracle, and CUDA-event benchmarks it.

The token layout (expert-sorted, block-padded) is produced host-side by
`moe_align_block_size` (below) exactly like vLLM's fused MoE path, so the Triton
kernel is a straight grouped GEMM with an expert id per BLOCK_M row-block.
"""
import torch
import triton
import triton.language as tl

_FP8 = torch.float8_e4m3fn


@triton.jit
def _moe_g1u1_fp8_kernel(
    # pointers
    a_ptr,            # [num_valid_tokens, K]  fp8   (expanded, expert-sorted rows via sorted_token_ids)
    w1_ptr,           # [E, 2N, K]             fp8
    out_ptr,          # [num_valid_tokens, N]  bf16  (silu(gate)*up)
    a_scale_ptr,      # [num_tokens, K//group_k]        fp32 (indexed by real token id)
    w1_scale_ptr,     # [E, (2N)//group_n, K//group_k]  fp32
    sorted_token_ids_ptr,   # [EM] int32   (row id in the expanded a; >= num_valid_tokens => pad)
    expert_ids_ptr,         # [EM//BLOCK_M] int32
    num_tokens_post_padded_ptr,
    # sizes
    N, K, EM,
    num_valid_tokens,
    top_k,
    # strides
    stride_am, stride_ak,
    stride_we, stride_wn, stride_wk,
    stride_om, stride_on,
    stride_asm, stride_ask,
    stride_wse, stride_wsn, stride_wsk,
    # meta
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Grouped FP8 block-scaled GEMM with fused SiLU*mul epilogue (gate|up)."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # a row = expanded token index (offs_token). The real activation scale is
    # indexed by the *real* token id = offs_token // top_k.
    a_row = offs_token
    a_ptrs = a_ptr + (a_row[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # gate cols = offs_bn ; up cols = N + offs_bn  (within the 2N rows of W1)
    wk = offs_k
    w_gate_ptrs = w1_ptr + off_experts * stride_we + (offs_bn[None, :] * stride_wn + wk[:, None] * stride_wk)
    w_up_ptrs = w1_ptr + off_experts * stride_we + ((offs_bn + N)[None, :] * stride_wn + wk[:, None] * stride_wk)

    # scale pointers
    a_scale_row = (offs_token // top_k)
    offs_bsn = offs_bn // group_n
    offs_bsn_up = (offs_bn + N) // group_n

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_groups = tl.cdiv(K, BLOCK_K)
    for kk in range(0, k_groups):
        k_remaining = K - kk * BLOCK_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_remaining), other=0.0)
        b_gate = tl.load(w_gate_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        b_up = tl.load(w_up_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        # block scales for this k-group (BLOCK_K == group_k)
        ks = kk  # group index along K
        a_scale = tl.load(
            a_scale_ptr + a_scale_row * stride_asm + ks * stride_ask,
            mask=token_mask, other=0.0,
        )
        wg_scale = tl.load(w1_scale_ptr + off_experts * stride_wse + offs_bsn * stride_wsn + ks * stride_wsk)
        wu_scale = tl.load(w1_scale_ptr + off_experts * stride_wse + offs_bsn_up * stride_wsn + ks * stride_wsk)

        acc_gate += tl.dot(a, b_gate) * a_scale[:, None] * wg_scale[None, :]
        acc_up += tl.dot(a, b_up) * a_scale[:, None] * wu_scale[None, :]

        a_ptrs += BLOCK_K * stride_ak
        w_gate_ptrs += BLOCK_K * stride_wk
        w_up_ptrs += BLOCK_K * stride_wk

    # fused SiLU(gate) * up
    gate = acc_gate
    silu = gate * (1.0 / (1.0 + tl.exp(-gate)))
    out = silu * acc_up
    out = out.to(out_ptr.dtype.element_ty)

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + offs_token[:, None] * stride_om + offs_cn[None, :] * stride_on
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(out_ptrs, out, mask=c_mask)


_SCRATCH = {}


def _iota(n, device):
    """Shape-derived cached iota (int64). Read-only helper buffer, never an output."""
    key = ("iota", n, str(device))
    t = _SCRATCH.get(key)
    if t is None:
        t = torch.arange(n, device=device, dtype=torch.int64)
        _SCRATCH[key] = t
    return t


def _ones64(n, device):
    key = ("ones", n, str(device))
    t = _SCRATCH.get(key)
    if t is None:
        t = torch.ones(n, device=device, dtype=torch.int64)
        _SCRATCH[key] = t
    return t


def moe_align_block_size(topk_ids: torch.Tensor, block_m: int, num_experts: int):
    """Fully vectorized, sync-free reimplementation of vLLM's moe_align_block_size.

    topk_ids: [num_tokens, top_k] int32 expert assignments.
    Returns sorted_token_ids (expanded row ids, pad-filled), expert_ids (per block),
    num_tokens_post_padded. Expanded row id = token * top_k + slot, so the kernel
    can recover the real token id via // top_k.

    No `.item()`, no `.cpu()`, no python loop over experts.  The output buffers are
    sized from the host-computable static upper bound
        EM_max = ceil_bm(numel + min(E, numel) * (block_m - 1))
    and `num_tokens_post_padded` stays on device; the kernel early-exits on the
    tail blocks.
    """
    num_tokens, top_k = topk_ids.shape
    numel = num_tokens * top_k
    device = topk_ids.device
    flat = topk_ids.reshape(-1)

    # Collapse the ~20 small align dispatches into a single graph replay. The graph
    # is keyed ONLY by (shape, dtype, block_m, num_experts) -- never by tensor
    # identity or content -- and the real expert ids are copied into the static
    # input every call, so every invocation does the full real work.
    g = _graph_for(numel, block_m, num_experts, device, flat.dtype)
    if g is None:
        return _align_eager(flat, numel, block_m, num_experts, device)
    static_in, graph, outs = g
    static_in.copy_(flat)
    graph.replay()
    return outs


_GRAPHS = {}


def _graph_for(numel, block_m, num_experts, device, dtype):
    key = ("g", numel, block_m, num_experts, str(device), str(dtype))
    if key in _GRAPHS:
        return _GRAPHS[key]
    entry = None
    try:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("nested capture")
        static_in = torch.zeros(numel, device=device, dtype=dtype)
        # warm the eager path (allocator + kernel modules) on a side stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _align_eager(static_in, numel, block_m, num_experts, device)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(graph, pool=pool):
            outs = _align_eager(static_in, numel, block_m, num_experts, device)
        entry = (static_in, graph, outs)
    except Exception:
        entry = None
    _GRAPHS[key] = entry
    return entry


def _em_max(numel, block_m, num_experts):
    """Host-computable STATIC upper bound on the padded row count."""
    nnz_max = num_experts if num_experts < numel else numel
    em = numel + nnz_max * (block_m - 1)
    em = ((em + block_m - 1) // block_m) * block_m
    return em if em >= block_m else block_m


def _align_eager(flat, numel, block_m, num_experts, device):
    flat64 = flat if flat.dtype == torch.int64 else flat.to(torch.int64)

    # counts per expert (scatter_add avoids bincount's device->host max() sync)
    counts = torch.zeros(num_experts, device=device, dtype=torch.int64)
    counts.scatter_add_(0, flat64, _ones64(numel, device))

    padded = ((counts + (block_m - 1)) // block_m) * block_m
    # one cumsum for both counts and padded (2 rows -> 1 dispatch)
    cums = torch.cumsum(torch.stack((padded, counts)), 1)
    cum_padded = cums[0]
    # base[e] = (start row of expert e's padded region) - (start of e in sorted order)
    base = (cum_padded - padded) - (cums[1] - counts)

    order = torch.argsort(flat64, stable=True)       # == sorted_rows (expanded_row is iota)
    dest = base[flat64][order] + _iota(numel, device)

    em_max = _em_max(numel, block_m, num_experts)
    sorted_token_ids = torch.full((em_max,), numel, device=device, dtype=torch.int32)
    sorted_token_ids.scatter_(0, dest, order.to(torch.int32))

    # expert id per BLOCK_M row-block: smallest e with cum_padded_blocks[e] > b.
    # Tail blocks (b >= total_blocks) get num_experts; the kernel's
    # `pid_m*BLOCK_M >= num_tokens_post_padded` guard returns before reading them.
    cum_blocks = cum_padded // block_m
    expert_ids = torch.bucketize(_iota(em_max // block_m, device), cum_blocks,
                                 right=True, out_int32=True)

    num_tokens_post_padded = cum_padded[-1:].to(torch.int32)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def moe_g1u1_fp8(
    a_fp8, a_scale, w1_fp8, w1_scale, topk_ids,
    N, group_n=128, group_k=128,
    block_m=16, block_n=64, block_k=128, group_m=1,
    num_warps=4, num_stages=1,
):
    """Launcher for the MoE stage-1 g1u1 fused GEMM.

    a_fp8:    [num_tokens*top_k, K] fp8  (already expanded per top_k slot)
    a_scale:  [num_tokens, K//group_k]   fp32
    w1_fp8:   [E, 2N, K] fp8
    w1_scale: [E, 2N//group_n, K//group_k] fp32
    topk_ids: [num_tokens, top_k] int32
    Returns out: [num_tokens*top_k, N] bf16.
    """
    # The k-loop uses one block scale per iteration (ks = kk), which is only correct
    # when the K tile matches the scale group. Measured sweep also shows BLOCK_K=256
    # is slower on every case, so pin it.
    block_k = group_k
    num_tokens, top_k = topk_ids.shape
    E = w1_fp8.shape[0]
    K = a_fp8.shape[1]
    num_valid = num_tokens * top_k

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_m, E)
    EM = sorted_token_ids.numel()

    # Every valid row is fully written by the kernel (each expanded row id appears
    # exactly once in sorted_token_ids); pad rows are masked off. No memset needed.
    out = torch.empty((num_valid, N), device=a_fp8.device, dtype=torch.bfloat16)

    grid = (triton.cdiv(EM, block_m) * triton.cdiv(N, block_n),)
    _moe_g1u1_fp8_kernel[grid](
        a_fp8, w1_fp8, out, a_scale, w1_scale,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_valid, top_k,
        a_fp8.stride(0), a_fp8.stride(1),
        w1_fp8.stride(0), w1_fp8.stride(1), w1_fp8.stride(2),
        out.stride(0), out.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        w1_scale.stride(0), w1_scale.stride(1), w1_scale.stride(2),
        group_n=group_n, group_k=group_k,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
