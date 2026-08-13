# Codebase context — DSA sparse-MLA indexer prefill logits (Triton, gfx942)

## Layout
- WORKSPACE: `.../seed_op/workspace/`
  - `triton_fp8_mqa_logits.py` — **the only modifiable file** (kernel `_fp8_mqa_logits_kernel` + launcher `fp8_mqa_logits_gfx942`)
  - `test_harness.py`, `config.yaml` — IMMUTABLE (correctness oracle + benchmark)
- Baseline (speedup denominator): `.../seed_op/baseline/triton_fp8_mqa_logits.py` (pristine copy).

## What the op computes
`logits[i,j] = sum_h relu( (q[i,h,:] . k[j,:]) * kv_scale[j] ) * weight[i,h]` for
`j in [cu_starts[i], cu_ends[i])`, `-inf` outside. Shapes: M=N=seq_len, H=64, D=128,
q/k are `torch.float8_e4m3fn` (OCP), everything else fp32.

Note the ReLU sits **between** the kv_scale multiply and the weight multiply, and kv_scale>0 in the
harness — but do not assume positivity in general. The head reduction is a plain fp32 sum over H=64.

## Structure
One workgroup per query row (`grid=(seq_len,)`, 4 warps). Q[64,128] is loaded once into registers and
is loop-invariant; the loop streams KV tiles `[128, BLOCK_KV]`, does one `tl.dot` -> `[64, BLOCK_KV]`
fp32 scores, applies scale/relu/weight, reduces over the 64 head rows, and stores `[BLOCK_KV]` fp32.
There is a full-tile loop plus one masked epilogue tile.

## Measured baseline (this box, gfx942/MI300X)
| case | shape | ms |
|---|---|---|
| 0 | M=N=2048 | 1.0577 |
| 2 | M=N=8000 | 14.6895 |
| 3 | M=N=8192 | 15.0611 |
geomean = **6.1624 ms** (harness `--benchmark` uses cases [0,2,3]).

Arithmetic: 2*M*N*H*D flops. At M=8000 that is 1.049 PFLOP-equivalent... concretely 14.69 ms =>
**~71 TFLOPS**. MI300X fp8 peak is ~2600 TFLOPS dense. The baseline is at ~2.7% of fp8 peak. Memory
traffic is trivial by comparison (KV is re-read per row but is fully L2/LDS-friendly; output is
M*N*4 B = 256 MB at M=8000 => ~17 GB/s at baseline, nowhere near the 5.3 TB/s HBM roofline). **This
op is compute/MFMA-issue bound, not memory bound**, until it gets ~10x faster; the fp32 output store
(256 MB / 5.3 TB/s ~= 48 us... plus L2 write-through) only becomes relevant well below ~1 ms.

## THE headline finding (ISA-verified)
`tl.dot` on `torch.float8_e4m3fn` operands does **NOT** emit fp8 MFMA on gfx942. Dumped ISA:
```
v_mfma_f32_16x16x16_f16   x64      # nkd=16
v_mfma_f32_32x32x8_f16             # nkd=32
```
i.e. Triton upconverts the fp8 operands to fp16 and runs the fp16 MFMA pipe. CDNA3 only has native
MFMA for the **fnuz** fp8 encodings. Bitcasting to `tl.float8e4b8` before the dot:
```
v_mfma_f32_16x16x32_fp8_fp8        # nkd=16
v_mfma_f32_32x32x16_fp8_fp8        # nkd=32
```
Measured at M=8000: 7.18 ms -> **1.75 ms** (146 -> 600 TFLOPS).

### Numerics of the fnuz bitcast (this is exact, not an approximation)
- OCP `e4m3` has exponent bias 7; `e4m3fnuz` has bias 8. A raw bitcast therefore reinterprets every
  finite value as exactly `v/2`. Two bitcast operands => the dot result is exactly `(q.k)/4`.
  Fold a `* 4.0` into the already-present per-column `kv_scales` multiply — **zero extra cost**.
- Byte `0x80` is `-0.0` in OCP e4m3 but **NaN** in e4m3fnuz. Skipping the scrub gives err_ratio 0.918.
  The harness data has 101755 such bytes in Q and 1654 in K at M=8000. Map `0x80 -> 0x00`.
- OCP e4m3 has `0x7F/0xFF` = NaN and max finite 448; fnuz has no Inf, `0x80`=NaN, max finite 240.
  With bias-halving the OCP max 448 maps to 224 < 240, so **no overflow is possible** — only the
  `0x80` case needs handling. (The launcher/harness clamps to +-448 anyway.)
- Measured accuracy with the compensation: err_ratio 0.0077, cos_diff 4.0e-3 — same as the f16 path,
  well inside the 5e-2 / 0.02 gate.

### Where to put the scrub (measured)
| scrub placement | ms @ M=8000 | correct |
|---|---|---|
| none | 1.737 | NO (err 0.918) |
| in-kernel on both Q tile and every KV tile | 2.521 | yes |
| in-kernel on Q only + one-shot host scrub of K | **1.760** | yes |
| host scrub of both Q and K | 1.747 | yes |
The KV scrub inside the loop costs ~0.78 ms of VALU because it runs per tile per row. A one-shot host
scrub of K is `torch.where(k.view(uint8)==0x80, 0, ...)` and measured **0.027 ms** for N=8192.
Q is loaded once per workgroup so the in-kernel Q scrub is free.

## Secondary tuning findings (all measured at M=8000, f16-MFMA path unless noted)
| change | ms | note |
|---|---|---|
| baseline tile BLOCK_KV=64, nkd=32 | 14.24 | what the launcher picks today for (H=64,D=128) |
| BLOCK_KV=128, nkd=32 | 7.78 | 1.83x — the LDS heuristic is far too conservative |
| BLOCK_KV=128, nkd=16 | 7.18 | another 1.08x |
| BLOCK_KV=256 | 8.22 | slower AND err_ratio 0.023 fails the gate |
| num_warps=8 | 15.04 | hard regression, do not use |
| waves_per_eu=1 | 11.79 | regression; 2 and 4 tie |
| num_stages=2 (with BKV=128) | 7.68 | ~1% |
| `input_precision="tf32"` vs `"ieee"` | 7.80 | no effect (operands are fp8) |
| BLOCK_Q=2 (128 dot rows, reshape+sum over H) | 7.99 | no win over BLOCK_Q=1/BKV=128 |
| hoisting the kv_scale multiply past the ReLU into the post-sum | 7.18 | no effect |

In the fnuz path: nkd=16 (1.75 ms) beats nkd=32 (2.26 ms) by 1.30x. BLOCK_KV=256 is still incorrect.

## Launcher issues
1. `_gfx942_default_tile_fits_lds(64,128)` returns False, forcing `(BLOCK_KV=64, num_stages=1)`. The
   LDS accounting is wrong-headed: it charges `num_heads*BLOCK_KV*4` for the fp32 scores even though
   they live in VGPRs, and its occupancy gate uses a ~28.8 KiB budget. BLOCK_KV=128 measurably works.
2. `matrix_instr_nonkdim=32` for `seq_len>1024`; 16 is better on every case measured.
3. `torch.full((M,N), -inf)` costs 0.060 ms at M=8000 (vs 0.0036 ms for `torch.empty`). That is 0.4%
   of the baseline but **3.4%** of a 1.75 ms kernel. The `-inf` fill is only needed outside
   `[cu_starts[i], cu_ends[i])`; the harness always uses the full range. Preserve the semantic
   (e.g. have the kernel write `-inf` on the out-of-window columns, or fill only the edges) rather
   than silently dropping it.

## Roofline / target
At 600 TFLOPS the fnuz path is already ~23% of MI300X fp8 dense peak. Remaining structural cost:
each of the M workgroups re-reads all of KV (M*N*D bytes of L2 traffic = 8 GB at M=8000 => ~4.6 TB/s
of L2 read at 1.75 ms — approaching the L2/Infinity-Cache limit) and writes M*N*4 = 256 MB (146 GB/s).
So after the MFMA fix the op moves toward **L2-bandwidth / KV-reuse bound**, and the next lever is
increasing Q-row reuse per KV read (BLOCK_Q, or a 2-D grid over (Q-block, KV-block) so a loaded KV
tile serves several query rows). Note BLOCK_Q=2 did NOT help in the f16 path — it must be re-measured
in the fnuz path where the balance is completely different.

## Full baseline source
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary gfx942 fallback for AITER's fp8_mqa_logits kernel.

This module vendors AITER's Triton fp8_mqa_logits kernel with the gfx942
tile-size workaround from ROCm/aiter#3257. It is used only while vLLM's
pinned AITER version lacks that fix.

TODO: Remove this vendored copy once vLLM pins an AITER version that includes
ROCm/aiter#3257 bugfix for gfx942.
"""

import torch

from vllm.triton_utils import tl, triton

# gfx942 (MI300X) has 64 KiB of LDS per CU. We accept the default
# (BLOCK_KV=128, num_stages=2) tile only when *both* of these hold:
#
# 1. Occupancy gate. With waves_per_eu=2 and num_warps=4 we target two
#    workgroups co-resident on a CU -> per-WG LDS budget = 32 KiB. Triton
#    keeps Q in registers (loop-invariant) and the fp32 scores accumulator
#    in VGPRs (heavy VALU), so only the double-buffered KV tile is
#    expected to live in LDS. A 0.9 safety factor leaves headroom for any
#    LDS overhead the compiler may add.
#
# 2. Hardware ceiling. Defensive upper bound that also counts Q and
#    scores against the 64 KiB CU limit, in case a Triton version (older
#    or future) decides to spill them to LDS. False positives here only
#    shrink the tile; false negatives are JIT-aborts, so we lean
#    conservative.
_GFX942_CU_LDS_BYTES = 64 * 1024
_GFX942_PER_WG_LDS_BUDGET_BYTES = _GFX942_CU_LDS_BYTES * 9 // 20  # ~28.8 KiB


def _gfx942_default_tile_fits_lds(num_heads: int, head_size: int) -> bool:
    """Return True iff (BLOCK_KV=128, num_stages=2) fits in MI300X LDS."""
    BLOCK_KV = 128
    NUM_STAGES = 2
    kv_bytes = head_size * BLOCK_KV * NUM_STAGES
    scores_bytes = num_heads * BLOCK_KV * 4
    q_bytes = num_heads * head_size
    fits_occupancy = kv_bytes < _GFX942_PER_WG_LDS_BUDGET_BYTES
    fits_hardware = q_bytes + kv_bytes + scores_bytes <= _GFX942_CU_LDS_BYTES
    return fits_occupancy and fits_hardware


@triton.jit
def _fp8_mqa_logits_kernel(
    Q_ptr,  # fp8e4m3 [seq_len, H, D]
    KV_ptr,  # fp8e4m3 [seq_len_kv, D]
    kv_scales_ptr,  # fp32 [seq_len_kv]
    weights_ptr,  # fp32 [seq_len, H]
    cu_start_ptr,  # int32 [seq_len]
    cu_end_ptr,  # int32 [seq_len]
    logits_ptr,  # fp32 [seq_len, seq_len_kv]
    seq_len,
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    # strides
    stride_q_s: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_s: tl.int64,
    stride_kv_d: tl.constexpr,
    stride_w_s: tl.int64,
    stride_w_h: tl.constexpr,
    stride_logits_s: tl.int64,
    stride_logits_k: tl.int64,
    # block sizes
    BLOCK_KV: tl.constexpr,
):
    row_id = tl.program_id(0)
    # go from larger to smaller in terms of work
    # to reduce the tail effect
    row_id = tl.num_programs(0) - row_id - 1
    tl.assume(row_id >= 0)
    tl.assume(stride_q_s > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_q_d > 0)
    tl.assume(stride_kv_s > 0)
    tl.assume(stride_kv_d > 0)
    tl.assume(stride_w_s > 0)
    tl.assume(stride_w_h > 0)

    logits_row_ptrs = logits_ptr + row_id * stride_logits_s

    h_inds = tl.arange(0, NUM_HEADS)[:, None]
    d_inds = tl.arange(0, HEAD_SIZE)

    # load Q[BLOCK_Q, NUM_HEADS, HEAD_SIZE]
    q_ptrs = (
        Q_ptr + row_id * stride_q_s + h_inds * stride_q_h + d_inds[None, :] * stride_q_d
    )

    q_block = tl.load(q_ptrs, cache_modifier=".cg")
    w_ptrs = weights_ptr + row_id * stride_w_s + h_inds * stride_w_h
    w_block = tl.load(w_ptrs, cache_modifier=".cg").to(tl.float32)

    # Load start/end for each row in this block
    start_ind = tl.load(cu_start_ptr + row_id)
    end_ind = tl.load(cu_end_ptr + row_id)

    start_ind = tl.maximum(start_ind, 0)
    end_ind = tl.minimum(end_ind, seq_len_kv)
    shifted_end = end_ind - start_ind
    shifted_unmasked_end = shifted_end // BLOCK_KV * BLOCK_KV

    kv_col_offsets = tl.arange(0, BLOCK_KV) + start_ind
    kv_ptrs = (
        KV_ptr + kv_col_offsets[None, :] * stride_kv_s + d_inds[:, None] * stride_kv_d
    )

    kv_scales_ptrs = kv_scales_ptr + kv_col_offsets

    logits_ptrs = logits_row_ptrs + kv_col_offsets * stride_logits_k

    # Loop over KV tiles
    for _ in tl.range(0, shifted_unmasked_end, BLOCK_KV):
        kv_block = tl.load(kv_ptrs)
        kv_scales = tl.load(kv_scales_ptrs)

        # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
        scores = tl.dot(q_block, kv_block, input_precision="ieee")
        # Multiply by kv_scales (broadcast along rows)
        scores = scores * kv_scales[None, :]
        # ReLU
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        # [NUM_HEADS, BLOCK_KV] -> [BLOCK_KV, ]
        scores = tl.sum(scores, axis=0)
        tl.store(logits_ptrs, scores)

        kv_ptrs += BLOCK_KV * stride_kv_s
        kv_scales_ptrs += BLOCK_KV
        logits_ptrs += BLOCK_KV * stride_logits_k
        kv_col_offsets += BLOCK_KV

    # masked load
    kv_col_mask = kv_col_offsets < end_ind
    kv_block = tl.load(kv_ptrs, mask=kv_col_mask[None, :], other=0.0)
    kv_scales = tl.load(kv_scales_ptrs, mask=kv_col_mask, other=0.0)

    # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
    scores = tl.dot(q_block, kv_block, input_precision="ieee")
    # Multiply by kv_scales (broadcast along rows)
    scores = scores * kv_scales[None, :]
    # ReLU
    scores = tl.maximum(scores, 0.0)
    scores = scores * w_block
    # [NUM_HEADS, BLOCK_KV] -> [BLOCK_KV, ]
    scores = tl.sum(scores, axis=0)
    # masked store
    in_window = (kv_col_offsets >= start_ind) & (kv_col_offsets < end_ind)
    tl.store(logits_ptrs, scores, mask=in_window)


def fp8_mqa_logits_gfx942(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits on MI300X (gfx942) using the vendored kernel.

    Drop-in replacement for ``aiter.ops.triton.attention.fp8_mqa_logits.
    fp8_mqa_logits`` on MI300X. Selects ``(BLOCK_KV, num_stages)`` based on
    whether the default tile fits within the 64 KiB LDS budget of a gfx942
    CU (see module docstring).

    Args:
        q: Query tensor of shape ``[M, H, D]``, FP8 dtype.
        k_fp8: Key tensor of shape ``[N, D]``, FP8 dtype.
        kv_scales: K scales of shape ``[N]`` (or ``[N, 1]`` -- viewed as
            ``[N]``), float32.
        weights: Per-head weights of shape ``[M, H]``, float32.
        cu_starts: Start indices (inclusive) of shape ``[M]``, int32.
        cu_ends: End indices (exclusive) of shape ``[M]``, int32.

    Returns:
        Logits of shape ``[M, N]``, float32 -- positions outside
        ``[cu_starts[i], cu_ends[i])`` for row ``i`` are pre-filled with
        ``-inf`` so the caller can run a top-k without masking.
    """
    seq_len, num_heads, head_size = q.shape
    seq_len_kv = k_fp8.shape[0]
    assert num_heads & (num_heads - 1) == 0, (
        f"num_heads must be a power of two (got {num_heads})"
    )
    assert head_size & (head_size - 1) == 0, (
        f"head_size must be a power of two (got {head_size})"
    )

    # The kernel walks ``kv_scales`` as a 1-D contiguous array of size N
    # (it indexes by ``kv_scales_ptr + kv_col_offsets``). The vLLM caller
    # passes a ``[N, 4]`` uint8 view-cast-to-float32 which lands as
    # ``[N, 1]`` contiguous -- byte-identical to ``[N]`` -- but flatten
    # explicitly to keep the kernel's pointer arithmetic intent clear.
    kv_scales_1d = kv_scales.reshape(-1)

    # Initialise with -inf so positions outside [cu_starts, cu_ends) read
    # as ``-inf`` after the masked store path -- this matches AITER's
    # ``fp8_mqa_logits`` semantics and is what the top-k consumer expects.
    logits = torch.full(
        (seq_len, seq_len_kv),
        fill_value=-float("inf"),
        dtype=torch.float32,
        device=q.device,
    )

    if _gfx942_default_tile_fits_lds(num_heads, head_size):
        block_kv = 128
        num_stages = 2
    else:
        # DSv4 sparse indexer (NUM_HEADS=64, HEAD_SIZE=128) lands here:
        # default tile spills past gfx942's 64 KiB LDS budget. (64, 1)
        # needs ~33 KiB and clears the per-WG budget with margin.
        block_kv = 64
        num_stages = 1

    # heuristic for MFMA instruction shape, identical to AITER's choice
    matrix_instr_nonkdim = 32
    if seq_len <= 1024:
        matrix_instr_nonkdim = 16

    stride_q_s, stride_q_h, stride_q_d = q.stride()
    stride_kv_s, stride_kv_d = k_fp8.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()

    _fp8_mqa_logits_kernel[(seq_len,)](
        Q_ptr=q,
        KV_ptr=k_fp8,
        kv_scales_ptr=kv_scales_1d,
        weights_ptr=weights,
        cu_start_ptr=cu_starts,
        cu_end_ptr=cu_ends,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=stride_q_s,
        stride_q_h=stride_q_h,
        stride_q_d=stride_q_d,
        stride_kv_s=stride_kv_s,
        stride_kv_d=stride_kv_d,
        stride_w_s=stride_w_s,
        stride_w_h=stride_w_h,
        stride_logits_s=stride_logits_s,
        stride_logits_k=stride_logits_k,
        BLOCK_KV=block_kv,
        num_warps=4,
        num_stages=num_stages,
        waves_per_eu=2,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
    )

    return logits
```
