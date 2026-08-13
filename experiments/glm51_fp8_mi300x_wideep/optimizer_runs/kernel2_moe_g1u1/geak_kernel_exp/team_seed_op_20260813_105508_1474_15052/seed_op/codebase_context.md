# Codebase Context — GLM-5.1-FP8 MoE stage-1 g1u1 (Triton)

## Files
- `moe_fp8_blockscale_g1u1.py` — THE kernel under optimization (only editable source).
  Contains three pieces, ALL of them fair game:
  1. `_moe_g1u1_fp8_kernel` (@triton.jit) — the grouped fp8 block-scaled GEMM + silu*mul epilogue.
  2. `moe_align_block_size` — pure-torch host-side expert-sort / block-pad. **Currently a Python
     `for e in range(256)` loop with `.item()` syncs — measured as 65-72% of total latency.**
  3. `moe_g1u1_fp8` — the launcher timed by the harness.
- `test_harness.py` — correctness oracle + CUDA-event benchmark (DO NOT edit; read-only contract).
- `config.yaml` — target kernel names, commands, tolerance.

## Hardware (detected on box)
gfx942 (MI300X-class), 304 CUs, 64 KB LDS/CU, 4 MB L2, ~5.3 TB/s HBM. wave64.
Triton 3.6.0, torch 2.11.

## Shapes (all 4 cases, E=256, top_k=8, N=2048, K=6144, fp8 block [128,128])
| case | num_tokens | rows (nt*8) | EM (padded, BLOCK_M=16) | M-blocks |
|---|---|---|---|---|
| 0 | 16 | 128 | 1648 | 103 |
| 1 | 32 | 256 | 2704 | 169 |
| 2 | 64 | 512 | 3504 | 219 |
| 3 | 128 | 1024 | 4000 | 250 |

Note the padding waste: case 0 has 128 real rows but 1648 padded rows -> **12.9x** M-padding waste.
Every one of the ~103 active experts sees 1-2 tokens but is padded to BLOCK_M=16.

## MEASURED breakdown (CUDA events, median of 30, this box)
| case | full launcher | GEMM kernel only | host `moe_align_block_size` | `torch.zeros` out |
|---|---|---|---|---|
| 0 (T=16)  | 10.91 ms | 2.97 ms | **7.92 ms (73%)** | 0.016 ms |
| 1 (T=32)  | 15.05 ms | 4.86 ms | **10.18 ms (68%)** | 0.017 ms |
| 2 (T=64)  | 18.51 ms | 6.37 ms | **12.11 ms (65%)** | 0.017 ms |
| 3 (T=128) | 20.45 ms | 7.22 ms | **13.21 ms (65%)** | 0.017 ms |

`moe_align_block_size` loops over all 256 experts in Python and calls `int(counts[e].item())` /
`int(padded[e].item())` twice per expert -> ~512 device->host syncs + ~256 tiny slice-assign kernel
launches PER CALL. This dwarfs the GEMM. **This is the #1 target and it is pure host_runtime work.**

## Prototype already measured by the TechLead (bit-exact vs original, verified with torch.equal
on sorted_token_ids / expert_ids / num_tokens_post_padded for cases 0 and 1)

```python
def align_vec(topk_ids, block_m, E):
    nt, tk = topk_ids.shape; dev = topk_ids.device
    flat  = topk_ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    sorted_rows = order.to(torch.int32)          # expanded_row[order] == order
    counts = torch.bincount(flat, minlength=E)
    padded = ((counts + block_m - 1)//block_m)*block_m
    starts = torch.cumsum(padded,0) - padded
    csum   = torch.cumsum(counts,0) - counts
    numel  = nt*tk
    total  = int(padded.sum())                   # ONE sync remains
    sti  = torch.full((total,), numel, device=dev, dtype=torch.int32)
    idx_in_e = torch.arange(numel, device=dev) - csum.repeat_interleave(counts)
    dest = starts.repeat_interleave(counts) + idx_in_e
    sti[dest] = sorted_rows
    nblk = padded // block_m
    eids = torch.repeat_interleave(torch.arange(E,device=dev,dtype=torch.int32), nblk)
    ntp  = torch.tensor([total], device=dev, dtype=torch.int32)
    return sti, eids, ntp
```
Measured: **0.50 ms** (case 0) and **0.49 ms** (case 1) vs 7.82 / 9.93 ms for the original —
a ~16-20x cut of the dominant term, on its own worth roughly **3.0x** end-to-end on case 0.
It is still ~0.5 ms (many small torch ops + one `.item()` sync); a **single fused Triton align
kernel** (or a HIP-graph / persistent-buffer scheme + a device-side `num_tokens_post_padded` with a
worst-case-sized grid so no `.item()` sync is needed) should push it toward tens of microseconds.
This is a further ~2x on top. Treat 0.5 ms as an intermediate milestone, not the goal.

## GEMM-side facts
- Weight traffic dominates the GEMM: `M_blocks * 2N * K` fp8 bytes = 2.59 GB (case 0) / 5.51 GB
  (case 2). At 5.3 TB/s that is a **0.49 ms / 1.04 ms HBM floor**, vs 2.97 / 6.37 ms measured ->
  the GEMM runs at ~16% of the bandwidth roofline. Matches the "26% of roofline" note in the task.
- The GEMM is **weight-bandwidth bound, not MFMA bound**, at decode. Each M-block re-streams a full
  `2N x K` expert weight slab. If an expert has 2 tokens split across 1 block that is unavoidable;
  but note **multiple M-blocks can share the same expert** when an expert got >16 tokens, and more
  importantly with BLOCK_N=64 there are `N/BLOCK_N = 32` programs per M-block each re-reading its
  own `2*64*K` slice — those are distinct columns so no duplication there, but the L2 reuse across
  M-blocks of the same expert is what matters.
- Tile sweep already measured (GEMM only, BLOCK_M=16, BLOCK_K=128):
  - case 0: `BLOCK_N=64, num_warps=4, num_stages=1` -> **2.517 ms** (baseline ns=2: 2.974) = 1.18x
  - case 2: `BLOCK_N=64, num_warps=4, num_stages=1` -> **5.579 ms** (baseline ns=2: 6.374) = 1.14x
  So `num_stages=1` is a free ~15% on the GEMM. `num_warps=8` and `BLOCK_N=128/256` were worse.
  This matches the perf_knowledge Triton card (wave64: num_warps=4, num_stages 1-2).
- `GROUP_M=1` currently: no L2 tile grouping at all. The card recommends `GROUP_SIZE_M` a multiple
  of XCD=8. Untested here — a candidate.
- BLOCK_M=16 with ~1-2 real tokens/expert wastes 12.9x of the MFMA lanes on case 0. Levers: a
  **masked / variable-M grouped GEMM** (DeepGEMM style) that skips pad blocks, BLOCK_M=8 or a
  GEMV-style path when `m_e == 1`, or K-split to raise tile count.

## Full current kernel source
```python
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


def moe_align_block_size(topk_ids: torch.Tensor, block_m: int, num_experts: int):
    """Pure-torch reimplementation of vLLM's moe_align_block_size.

    topk_ids: [num_tokens, top_k] int32 expert assignments.
    Returns sorted_token_ids (expanded row ids, pad-filled), expert_ids (per block),
    num_tokens_post_padded. Expanded row id = token * top_k + slot, so the kernel
    can recover the real token id via // top_k.
    """
    num_tokens, top_k = topk_ids.shape
    device = topk_ids.device
    flat_expert = topk_ids.reshape(-1)                       # [num_tokens*top_k]
    expanded_row = torch.arange(num_tokens * top_k, device=device, dtype=torch.int32)

    order = torch.argsort(flat_expert, stable=True)
    sorted_experts = flat_expert[order]
    sorted_rows = expanded_row[order]

    counts = torch.bincount(flat_expert, minlength=num_experts)  # tokens per expert
    padded = ((counts + block_m - 1) // block_m) * block_m
    num_padded = int(padded.sum().item())

    numel = num_tokens * top_k
    sorted_token_ids = torch.full((num_padded,), numel, device=device, dtype=torch.int32)
    expert_ids = torch.zeros((num_padded // block_m,), device=device, dtype=torch.int32)

    cur = 0
    blk = 0
    csum = 0
    for e in range(num_experts):
        c = int(counts[e].item())
        if c == 0:
            continue
        p = int(padded[e].item())
        rows_e = sorted_rows[csum:csum + c]
        sorted_token_ids[cur:cur + c] = rows_e
        nblocks = p // block_m
        expert_ids[blk:blk + nblocks] = e
        cur += p
        blk += nblocks
        csum += c

    num_tokens_post_padded = torch.tensor([num_padded], device=device, dtype=torch.int32)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def moe_g1u1_fp8(
    a_fp8, a_scale, w1_fp8, w1_scale, topk_ids,
    N, group_n=128, group_k=128,
    block_m=16, block_n=64, block_k=128, group_m=1,
    num_warps=4, num_stages=2,
):
    """Launcher for the MoE stage-1 g1u1 fused GEMM.

    a_fp8:    [num_tokens*top_k, K] fp8  (already expanded per top_k slot)
    a_scale:  [num_tokens, K//group_k]   fp32
    w1_fp8:   [E, 2N, K] fp8
    w1_scale: [E, 2N//group_n, K//group_k] fp32
    topk_ids: [num_tokens, top_k] int32
    Returns out: [num_tokens*top_k, N] bf16.
    """
    num_tokens, top_k = topk_ids.shape
    E = w1_fp8.shape[0]
    K = a_fp8.shape[1]
    num_valid = num_tokens * top_k

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_m, E)
    EM = sorted_token_ids.numel()

    out = torch.zeros((num_valid, N), device=a_fp8.device, dtype=torch.bfloat16)

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
```
