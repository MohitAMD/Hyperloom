# Codebase Context — GLM-5.1-FP8 MoE stage-1 g1u1 (Triton)

WORKSPACE: `/shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/geak_kernel_exp/team_seed_op_20260813_102220_386_9297/seed_op/workspace`

Files:
- `moe_fp8_blockscale_g1u1.py` — THE ONLY MODIFIABLE FILE. Contains `_moe_g1u1_fp8_kernel`
  (@triton.jit), `moe_align_block_size` (pure-torch host helper), `moe_g1u1_fp8` (launcher).
- `config.yaml`, `test_harness.py` — READ-ONLY (mode 0444). Oracle + benchmark.

Device: gfx942 (MI300X class, 304 CU, wave64). torch 2.11.0, triton 3.6.0.

## Harness contract
- 4 cases: T = 16 / 32 / 64 / 128, all with top_k=8, E=256, N=2048, K=6144, fp8 block [128,128].
- `--benchmark` picks 3 of 4 (indices 0,2,3 via `_pick`); `--full-benchmark` runs all 4.
- Metric: median-of-50 CUDA-event latency per case, geomean printed as `GEAK_RESULT_LATENCY_MS`.
  NOTE: the timed region is `run_kernel` = the WHOLE `moe_g1u1_fp8` launcher including
  `moe_align_block_size`. Host time inside the event window counts (events are recorded around a
  region containing host syncs, so the host cost is fully captured).
- Correctness: pure-torch fp32 dequant oracle, rtol=atol=5e-2, err_ratio<=0.05, all 4 cases.
- Inputs are regenerated per case by `setup_inputs`; `topk_ids` = top_k distinct random experts
  per token (`torch.randperm`), so routing is near-uniform over 256 experts.

## Measured baseline (this box, --full-benchmark)
| case | T | total ms |
|---|---|---|
| 0 | 16 | 10.4272 |
| 1 | 32 | 14.4246 |
| 2 | 64 | 17.6928 |
| 3 | 128 | 19.5626 |
| geomean(4) | | **15.1051** |

## THE DOMINANT FINDING — host `moe_align_block_size` is 65-75% of wall clock
Direct decomposition:

| case | total | `moe_align_block_size` (host) | Triton kernel |
|---|---|---|---|
| 0 (T=16) | 10.997 ms | **7.853 ms** | 2.961 ms |
| 3 (T=128) | 20.552 ms | **12.672 ms** | 7.200 ms |

`moe_align_block_size` runs a Python `for e in range(num_experts)` loop over **256 experts**, and
inside it does `int(counts[e].item())` and `int(padded[e].item())` — two GPU->host syncs per
expert, i.e. **512 device syncs plus 256 slice-assign kernel launches per call**. Plus
`int(padded.sum().item())`. This is pure launch/sync overhead and is completely independent of
problem size, which is exactly why case 0 (which computes 8x less than case 3) is only 1.9x faster.

**This is the single highest-leverage target in the whole task.** It is fully vectorizable with
torch ops (cumsum of padded counts + `repeat_interleave`/`scatter`) with ZERO device syncs, or
better, written as a small Triton kernel. Expected: 7.9 ms -> <0.2 ms on case 0, i.e. ~3x on the
geomean before touching the GEMM at all.

## Kernel-side facts
- Grid is 1-D: `cdiv(EM, BLOCK_M) * cdiv(N, BLOCK_N)`, with GROUP_M=1 swizzle (so the swizzle is a
  no-op: `pid_m = pid // num_pid_n`, `pid_n = pid % num_pid_n`).
- Decode padding waste: case 0 has EM=1648 rows for only 128 real (token,expert) pairs across
  **103 active experts**, mean **1.24 real tokens per 16-row block** => ~92% of MFMA M-lanes are
  padding. Case 3: 250 active experts, mean 4.10 tokens each.
- Per-K-step the kernel issues `tl.dot(a, b_gate)` and `tl.dot(a, b_up)` — the `a` tile is loaded
  once (good) but with BLOCK_M=16 the two dots are 16x64x128 each; on wave64 MFMA 16x16 that is
  heavily under-utilized.
- W1 traffic dominates: 103 active experts x 2N x K fp8 = ~2.6 GB read per call on case 0 =>
  ~0.49 ms at ~5.3 TB/s. The kernel measures 2.96 ms, i.e. **~6x above its own HBM floor**, so
  even holding the algorithm fixed there is 5-6x of kernel headroom. Note W1 is read once per
  `pid_m` block (each expert's blocks re-stream the same weight rows for each pid_n only once, but
  an expert with 2 padded M-blocks reads its full W1 twice).
- FLOP floor is negligible: case 0 is 2*128*2*2048*6144 = 6.4 GFLOP => ~64 us at fp8 MFMA peak.
  **The GEMM is memory bound, not compute bound**, at these decode shapes.
- LDS ceiling: BLOCK_N=256 with BLOCK_K=128 num_stages=2 already fails to compile
  (68608 > 65536 bytes). BLOCK_N=128 + num_stages=1 fits and measured slightly faster than
  baseline (2.823 vs 2.962 ms).

### Quick tiling probe (case 0, kernel-only, BLOCK_M=16, BLOCK_K=128)
| config | ms |
|---|---|
| BN=64  w=4 st=2 (baseline) | 2.962 |
| BN=128 w=4 st=1 | **2.823** |
| BN=128 w=4 st=2 | 4.214 |
| BN=128 w=8 st=2 | 3.377 |
| BN=64  w=4 st=3 | 4.552 |
| BN=128 w=2 st=2 | 6.180 |
| BN=256 w=4 st=2 | LDS compile fail |

Naive tile tuning alone buys only ~5%. The real kernel-side levers are structural: cut the number
of times W1 is streamed, and stop paying 16-row tiles for 1.24-token experts.

## Full kernel source (baseline)

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
