# Performance Analysis Report — GLM-5.1-FP8

> Generated via bypass route (HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass). framework=vllm, platform=mi300x, throughput_unit=tok/s, aggregation_scope=full_trace. Per-kernel roofline (bound/AI/efficiency) is computed analytically from captured operand shapes + measured kernel time (roofline_source=analytical).

## Executive Summary

| Metric | Value |
|--------|-------|
| Total GPU Time | 9135.011 ms |
| GPU Busy % | 99.98% |
| GPU Idle % | 0.02% |
| GPU MemCpy | 49.596 ms |
| Top Bottleneck Category | Other |
| Op-attribution Coverage | 99.99% |

## System-Level Signals

| Signal | % of total GPU time | Note |
|--------|---------------------|------|
| GPU idle | 0.02% | within 80% idle gate |
| Exposed communication | — | - |
| Exposed memcpy (device copy) | 0.54% | - |

## Top Hot Kernels

| Rank | Operation | Time (us) | GPU% | Eff% | AI | Bound | Category | Source File |
|------|-----------|-----------|------|------|----|-------|----------|-------------|
| 1 | vllm::all_reduce | 8512128.6 | 93.71% | — | — | — | Other | — |
| 2 | aiter::ck_moe_stage1 | 80591.1 | 0.89% | — | — | — | MoE | /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu |
| 3 | aiter::dynamic_per_token_scaled_quant | 55772.8 | 0.61% | 1.98% | 0.334 | memory_bound | Quantization | /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/quant_kernels.cu |
| 4 | aiter::ck_moe_stage2 | 45165.9 | 0.50% | — | — | — | MoE | /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu |
| 5 | _C::top_k_per_row_decode | 31679.9 | 0.35% | — | — | — | Other | — |
| 6 | vllm::rocm_aiter_sparse_attn_indexer | 28765.5 | 0.32% | — | — | — | Other | /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_fp8_mqa_logits.py |
| 7 | aiter::gemm_a8w8_blockscale_ck | 18672.5 | 0.21% | 100.00% | 15.917 | memory_bound | GEMM | /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale.cu |
| 8 | aiter::mla_decode_stage1_asm_fwd | 17499.6 | 0.19% | — | — | — | Other | — |
| 9 | aiter::moe_sorting_opus_fwd | 14891.7 | 0.16% | — | — | — | MoE | — |
| 10 | _C::fused_add_rms_norm | 14777.3 | 0.16% | 1.92% | 0.337 | memory_bound | Normalization | — |
| 11 | _C::rotary_embedding | 14243.2 | 0.16% | 100.00% | 0.250 | memory_bound | Elementwise | — |
| 12 | _C::rms_norm | 13894.8 | 0.15% | 2.05% | 0.337 | memory_bound | Normalization | /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/rmsnorm_kernels.cu |
| 13 | aten::mul | 13869.4 | 0.15% | 0.01% | 0.125 | memory_bound | Elementwise | — |
| 14 | aiter::gemm_a8w8_blockscale_ck | 12139.2 | 0.13% | 17.16% | 15.754 | memory_bound | GEMM | /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale.cu |
| 15 | aten::mm | 11299.6 | 0.12% | 13.43% | 7.511 | memory_bound | GEMM | — |
| 16 | aten::mm | 10347.2 | 0.11% | 4.93% | 14.511 | memory_bound | GEMM | — |
| 17 | aiter::mla_reduce_v1 | 10241.9 | 0.11% | — | — | — | Other | — |
| 18 | vllm::unified_mla_attention_with_output | 9007.0 | 0.10% | 0.56% | 0.348 | memory_bound | Quantization | — |
| 19 | record_param_comms | 8855.8 | 0.10% | — | — | — | Other | — |
| 20 | aten::cat | 8255.2 | 0.09% | — | — | — | Elementwise | — |

### P1: Other kernels

<!-- reasoning-candidate tier=compute rank=1 -->

**Data:**

| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | Efficiency | Bound | Args | Source File | Kernel Path (launcher) |
|-----------|-----------|------|------|-------|------------|------------|-------|------|-------------|------------------------|
| vllm::all_reduce | 8512128.6 | 93.71% | — | 2512 | — | — | — | (16,6144) bf16 | — | — |

### P3: Quantization kernels

<!-- reasoning-candidate tier=compute rank=3 -->

**Data:**

| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | Efficiency | Bound | Args | Source File | Kernel Path (launcher) |
|-----------|-----------|------|------|-------|------------|------------|-------|------|-------------|------------------------|
| aiter::dynamic_per_token_scaled_quant | 55772.8 | 0.61% | — | 9888 | 0.334 | 1.98% | memory_bound | (16,6144)<br>(768,128) bf16<br>(16,48) fp32 | /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/quant_kernels.cu | — |

### P7: Other kernels

<!-- reasoning-candidate tier=compute rank=7 -->

**Data:**

| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | Efficiency | Bound | Args | Source File | Kernel Path (launcher) |
|-----------|-----------|------|------|-------|------------|------------|-------|------|-------------|------------------------|
| aiter::mla_decode_stage1_asm_fwd | 17499.6 | 0.19% | — | 1248 | — | — | — | (16,16,576)<br>(1140445,1,1,576)<br>(17,) i32<br>(17,) i32<br>(32768,) i32<br>(16,) i32<br>(2,)<br>(305,) i32<br>(8495,8) i32<br>(606,1,16,512) fp32<br>(606,1,16,1) fp32<br>(16,16,512) bf16 | — | — |

### P8: MoE kernels

<!-- reasoning-candidate tier=compute rank=8 -->

**Data:**

| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | Efficiency | Bound | Args | Source File | Kernel Path (launcher) |
|-----------|-----------|------|------|-------|------------|------------|-------|------|-------------|------------------------|
| aiter::moe_sorting_opus_fwd | 14891.7 | 0.16% | — | 1200 | — | — | — | (16,8) i32<br>(16,8) fp32<br>(4216,) i32<br>(4216,) fp32<br>(264,) i32<br>(2,) i32<br>(16,6144) bf16 | — | — |

### P9: Normalization kernels

<!-- reasoning-candidate tier=compute rank=9 -->

**Data:**

| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | Efficiency | Bound | Args | Source File | Kernel Path (launcher) |
|-----------|-----------|------|------|-------|------------|------------|-------|------|-------------|------------------------|
| _C::fused_add_rms_norm | 14777.3 | 0.16% | — | 2496 | 0.337 | 1.92% | memory_bound | (16,6144) bf16<br>(16,6144) bf16<br>(6144,) bf16 | — | — |

### P12: Normalization kernels

<!-- reasoning-candidate tier=compute rank=12 -->

**Data:**

| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | Efficiency | Bound | Args | Source File | Kernel Path (launcher) |
|-----------|-----------|------|------|-------|------------|------------|-------|------|-------------|------------------------|
| _C::rms_norm | 13894.8 | 0.15% | — | 2512 | 0.337 | 2.05% | memory_bound | (16,6144) bf16<br>(16,6144) bf16<br>(6144,) bf16 | /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/rmsnorm_kernels.cu | — |

---

_Additional route-specific detail below (not part of the shared cross-route sections above)._

## Top Operations

| Rank | Category | GPU % | Time (ms) | Kernels |
|------|----------|-------|-----------|---------|
| 1 | Other | 94.93 | 8622.844 | 10065 |
| 2 | MoE | 1.63 | 148.243 | 4800 |
| 3 | Elementwise | 1.19 | 107.682 | 18896 |
| 4 | Quantization | 0.95 | 86.12 | 14880 |
| 5 | GEMM | 0.83 | 75.751 | 9952 |
| 6 | Normalization | 0.39 | 35.753 | 6256 |
| 7 | KVCacheStore | 0.08 | 7.069 | 1248 |

_Analytical roofline bound: 0 compute-bound, 10 memory-bound hot kernel(s)._

## Top 10 Kernels by Optimization Priority

_Priority = GPU% x (1 - efficiency): high-impact, low-efficiency kernels first. Full per-kernel metrics in the CSV linked below._

| # | kernel_id | Name | Category | GPU% | Bound | AI | Eff% | Priority | Suggestion |
|---|-----------|------|----------|------|-------|----|----|---------|------------|
| 1 | `k001` | vllm::all_reduce | Other | 93.71% | — | — | — | 93.71 | Profile the kernel for tile size and wave occupancy. |
| 2 | `k002` | aiter::ck_moe_stage1 | MoE | 0.89% | — | — | — | 0.89 | Optimize expert GEMM and routing; fuse gate/up projections. |
| 3 | `k003` | aiter::dynamic_per_token_scaled_quant | Quantization | 0.61% | memory_bound | 0.334 | 2.0% | 0.61 | Memory-bound: Fuse quantization into the adjacent GEMM epilogue and drop redundant per-tensor scaling passes. |
| 4 | `k004` | aiter::ck_moe_stage2 | MoE | 0.50% | — | — | — | 0.50 | Optimize expert GEMM and routing; fuse gate/up projections. |
| 5 | `k005` | _C::top_k_per_row_decode | Other | 0.35% | — | — | — | 0.35 | Profile the kernel for tile size and wave occupancy. |
| 6 | `k006` | vllm::rocm_aiter_sparse_attn_indexer | Other | 0.32% | — | — | — | 0.32 | Profile the kernel for tile size and wave occupancy. |
| 7 | `k008` | aiter::mla_decode_stage1_asm_fwd | Other | 0.19% | — | — | — | 0.19 | Profile the kernel for tile size and wave occupancy. |
| 8 | `k009` | aiter::moe_sorting_opus_fwd | MoE | 0.16% | — | — | — | 0.16 | Optimize expert GEMM and routing; fuse gate/up projections. |
| 9 | `k010` | _C::fused_add_rms_norm | Normalization | 0.16% | memory_bound | 0.337 | 1.9% | 0.16 | Memory-bound: Use a fused RMSNorm/LayerNorm and fold the residual/quant into the norm kernel. |
| 10 | `k007` | aiter::gemm_a8w8_blockscale_ck | GEMM | 0.21% | memory_bound | 15.9 | 100.0% | 0.16 | Memory-bound: Tune GEMM tile size / precision and fuse the epilogue where possible; vendor-library GEMMs (Tensile/rocBLAS) are not rewritable — tune via library config. |

## Compute Kernel Optimizations

_2 of 6 rewritable candidate(s) have a resolved editable source (auto-dispatchable to kernel-opt); the rest need a source first._

### P1: vllm::all_reduce (Other)

**Insight**: Other kernel consuming 93.71% of GPU time across 2512 launches.

**Action**: Profile the kernel for tile size and wave occupancy.

**Source**: unresolved — not auto-dispatchable (rewritable by classification, but no editable source was located for its launching op).

**Impact**: 93.71% of GPU time; bound=—, attainment=—, priority=93.71 (roofline_source=placeholder).

### P2: aiter::dynamic_per_token_scaled_quant (Quantization)

**Insight**: Quantization kernel consuming 0.61% of GPU time across 9888 launches.

**Action**: Fuse quantization into the adjacent GEMM epilogue and drop redundant per-tensor scaling passes.

**Source**: `/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/quant_kernels.cu` (via op_to_source); shapes captured: yes; task group `tg001`.

**Impact**: 0.61% of GPU time; bound=memory_bound, attainment=2.0%, priority=0.61 (roofline_source=analytical).

### P3: aiter::mla_decode_stage1_asm_fwd (Other)

**Insight**: Other kernel consuming 0.19% of GPU time across 1248 launches.

**Action**: Profile the kernel for tile size and wave occupancy.

**Source**: unresolved — not auto-dispatchable (rewritable by classification, but no editable source was located for its launching op).

**Impact**: 0.19% of GPU time; bound=—, attainment=—, priority=0.19 (roofline_source=placeholder).

### P4: aiter::moe_sorting_opus_fwd (MoE)

**Insight**: MoE kernel consuming 0.16% of GPU time across 1200 launches.

**Action**: Optimize expert GEMM and routing; fuse gate/up projections.

**Source**: unresolved — not auto-dispatchable (rewritable by classification, but no editable source was located for its launching op).

**Impact**: 0.16% of GPU time; bound=—, attainment=—, priority=0.16 (roofline_source=placeholder).

### P5: _C::fused_add_rms_norm (Normalization)

**Insight**: Normalization kernel consuming 0.16% of GPU time across 2496 launches.

**Action**: Use a fused RMSNorm/LayerNorm and fold the residual/quant into the norm kernel.

**Source**: unresolved — not auto-dispatchable (rewritable by classification, but no editable source was located for its launching op).

**Impact**: 0.16% of GPU time; bound=memory_bound, attainment=1.9%, priority=0.16 (roofline_source=analytical).

### P6: _C::rms_norm (Normalization)

**Insight**: Normalization kernel consuming 0.15% of GPU time across 2512 launches.

**Action**: Use a fused RMSNorm/LayerNorm and fold the residual/quant into the norm kernel.

**Source**: `/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/rmsnorm_kernels.cu` (via op_to_source); shapes captured: yes; task group `tg002`.

**Impact**: 0.15% of GPU time; bound=memory_bound, attainment=2.1%, priority=0.15 (roofline_source=analytical).

## Task Groups

_Rewritable candidates sharing one editable source collapse into a single dispatch (all observed shapes)._

| Group | Source | Kernels | GPU % | Time (ms) |
|-------|--------|---------|-------|-----------|
| tg001 | quant_kernels.cu | 1 | 0.614 | 55.773 |
| tg002 | rmsnorm_kernels.cu | 1 | 0.153 | 13.895 |

_18 hot kernel(s) are non-rewritable (vendor library / unresolved source) — see Detailed Analysis._

## Detailed Analysis

### k001: vllm::all_reduce (Other)

**Identification:** 93.71% GPU time, 2512 launches, reusable=True.

**Data:** device kernel `_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8RankDataES2_NS_11RankSignalsEPNS_6SignalEPT_ii`; duration 8512.13 ms.

**Source:** unresolved (shape provenance: torch_trace).

**Roofline:** bound=—, AI=—, attainment=—, priority=93.71 (roofline_source=placeholder).

**Suggested action:** Profile the kernel for tile size and wave occupancy.

### k002: aiter::ck_moe_stage1 (MoE)

**Identification:** 0.89% GPU time, 1200 launches, reusable=False, skip_reason=source file not resolved.

**Data:** device kernel `void ck::kernel_moe_gemm<ck::GridwiseMoeGemmBlockScale<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::Colum`; duration 80.59 ms.

**Source:** `/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu` (shape provenance: torch_trace).

**Roofline:** bound=—, AI=—, attainment=—, priority=0.89 (roofline_source=placeholder).

**Suggested action:** Optimize expert GEMM and routing; fuse gate/up projections.

### k003: aiter::dynamic_per_token_scaled_quant (Quantization)

**Identification:** 0.61% GPU time, 9888 launches, reusable=True.

**Data:** device kernel `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb0ELi64ELb0EEEvPT0_PfPKT_PKfliilPKii`; duration 55.77 ms.

**Source:** `/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/quant_kernels.cu` (shape provenance: torch_trace).

**Roofline:** bound=memory_bound, AI=0.334, attainment=2.0%, priority=0.61 (roofline_source=analytical).

**Suggested action:** Memory-bound: Fuse quantization into the adjacent GEMM epilogue and drop redundant per-tensor scaling passes.

### k004: aiter::ck_moe_stage2 (MoE)

**Identification:** 0.50% GPU time, 1200 launches, reusable=False, skip_reason=source file not resolved.

**Data:** device kernel `void ck::kernel_moe_gemm<ck::GridwiseMoeGemmBlockScale<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::Colum`; duration 45.17 ms.

**Source:** `/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu` (shape provenance: torch_trace).

**Roofline:** bound=—, AI=—, attainment=—, priority=0.50 (roofline_source=placeholder).

**Suggested action:** Optimize expert GEMM and routing; fuse gate/up projections.

### k005: _C::top_k_per_row_decode (Other)

**Identification:** 0.35% GPU time, 1248 launches, reusable=False, skip_reason=source file not resolved.

**Data:** device kernel `void vllm::topKPerRowDecode<512, true, false, false>(float const*, int const*, int*, int, int, int, int, int, float*, in`; duration 31.68 ms.

**Source:** unresolved (shape provenance: torch_trace).

**Roofline:** bound=—, AI=—, attainment=—, priority=0.35 (roofline_source=placeholder).

**Suggested action:** Profile the kernel for tile size and wave occupancy.

### k006: vllm::rocm_aiter_sparse_attn_indexer (Other)

**Identification:** 0.32% GPU time, 1248 launches, reusable=False, skip_reason=source file not resolved.

**Data:** device kernel `_gluon_deepgemm_fp8_paged_mqa_logits`; duration 28.77 ms.

**Source:** `/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_fp8_mqa_logits.py` (shape provenance: torch_trace).

**Roofline:** bound=—, AI=—, attainment=—, priority=0.32 (roofline_source=placeholder).

**Suggested action:** Profile the kernel for tile size and wave occupancy.

### k007: aiter::gemm_a8w8_blockscale_ck (GEMM)

**Identification:** 0.21% GPU time, 2496 launches, reusable=False, skip_reason=source file not resolved.

**Data:** device kernel `void ck::kernel_gemm_xdl_cshuffle_v3<ck::GridwiseGemmMultiD_ABScale_xdl_cshuffle_v3<ck::tensor_layout::gemm::RowMajor, c`; duration 18.67 ms.

**Source:** `/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale.cu` (shape provenance: torch_trace).

**Roofline:** bound=memory_bound, AI=15.9, attainment=100.0%, priority=0.16 (roofline_source=analytical).

**Suggested action:** Memory-bound: Tune GEMM tile size / precision and fuse the epilogue where possible; vendor-library GEMMs (Tensile/rocBLAS) are not rewritable — tune via library config.

### k008: aiter::mla_decode_stage1_asm_fwd (Other)

**Identification:** 0.19% GPU time, 1248 launches, reusable=True.

**Data:** device kernel `aiter::mla_a8w8_qh16_qseqlen1_gqaratio16_ps`; duration 17.50 ms.

**Source:** unresolved (shape provenance: torch_trace).

**Roofline:** bound=—, AI=—, attainment=—, priority=0.19 (roofline_source=placeholder).

**Suggested action:** Profile the kernel for tile size and wave occupancy.

### k009: aiter::moe_sorting_opus_fwd (MoE)

**Identification:** 0.16% GPU time, 1200 launches, reusable=True.

**Data:** device kernel `void aiter::opus_moe_sorting_entry<aiter::MoeSortingKernel<aiter::MoeSortingProblemEx<int, float, 2, true, false, false,`; duration 14.89 ms.

**Source:** unresolved (shape provenance: torch_trace).

**Roofline:** bound=—, AI=—, attainment=—, priority=0.16 (roofline_source=placeholder).

**Suggested action:** Optimize expert GEMM and routing; fuse gate/up projections.

### k010: _C::fused_add_rms_norm (Normalization)

**Identification:** 0.16% GPU time, 2496 launches, reusable=True.

**Data:** device kernel `std::enable_if<(((8)>(0)))&&_typeConvert<c10::BFloat16>::exists, void>::type vllm::fused_add_rms_norm_kernel<c10::BFloat`; duration 14.78 ms.

**Source:** unresolved (shape provenance: torch_trace).

**Roofline:** bound=memory_bound, AI=0.337, attainment=1.9%, priority=0.16 (roofline_source=analytical).

**Suggested action:** Memory-bound: Use a fused RMSNorm/LayerNorm and fold the residual/quant into the norm kernel.

### k011: _C::rotary_embedding (Elementwise)

**Identification:** 0.16% GPU time, 2496 launches, reusable=False, skip_reason=source file not resolved.

**Data:** device kernel `void vllm::rotary_embedding_kernel<c10::BFloat16, c10::BFloat16, false>(long const*, c10::BFloat16*, c10::BFloat16*, c10`; duration 14.24 ms.

**Source:** unresolved (shape provenance: torch_trace).

**Roofline:** bound=memory_bound, AI=0.25, attainment=100.0%, priority=0.16 (roofline_source=analytical).

**Suggested action:** Memory-bound: Fuse elementwise chains to cut intermediate memory traffic.

### k012: _C::rms_norm (Normalization)

**Identification:** 0.15% GPU time, 2512 launches, reusable=True.

**Data:** device kernel `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long`; duration 13.89 ms.

**Source:** `/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/rmsnorm_kernels.cu` (shape provenance: torch_trace).

**Roofline:** bound=memory_bound, AI=0.337, attainment=2.1%, priority=0.15 (roofline_source=analytical).

**Suggested action:** Memory-bound: Use a fused RMSNorm/LayerNorm and fold the residual/quant into the norm kernel.

### k013: aten::mul (Elementwise)

**Identification:** 0.15% GPU time, 2496 launches, reusable=False, skip_reason=source file not resolved.

**Data:** device kernel `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_inte`; duration 13.87 ms.

**Source:** unresolved (shape provenance: torch_trace).

**Roofline:** bound=memory_bound, AI=0.125, attainment=0.0%, priority=0.15 (roofline_source=analytical).

**Suggested action:** Memory-bound: Fuse elementwise chains to cut intermediate memory traffic.

### k014: aiter::gemm_a8w8_blockscale_ck (GEMM)

**Identification:** 0.13% GPU time, 1296 launches, reusable=False, skip_reason=source file not resolved.

**Data:** device kernel `void ck::kernel_gemm_xdl_cshuffle_v3<ck::GridwiseGemmMultiD_ABScale_xdl_cshuffle_v3<ck::tensor_layout::gemm::RowMajor, c`; duration 12.14 ms.

**Source:** `/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale.cu` (shape provenance: torch_trace).

**Roofline:** bound=memory_bound, AI=15.8, attainment=17.2%, priority=0.13 (roofline_source=analytical).

**Suggested action:** Memory-bound: Tune GEMM tile size / precision and fuse the epilogue where possible; vendor-library GEMMs (Tensile/rocBLAS) are not rewritable — tune via library config.

### k015: aten::mm (GEMM)

**Identification:** 0.12% GPU time, 1200 launches, reusable=False, skip_reason=vendor backend library (precompiled binary, no rewritable source).

**Data:** device kernel `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT16x16x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_D`; duration 11.30 ms.

**Source:** unresolved (shape provenance: torch_trace).

**Roofline:** bound=memory_bound, AI=7.51, attainment=13.4%, priority=0.12 (roofline_source=analytical).

**Suggested action:** Memory-bound: Tune GEMM tile size / precision and fuse the epilogue where possible; vendor-library GEMMs (Tensile/rocBLAS) are not rewritable — tune via library config.

### k016: aten::mm (GEMM)

**Identification:** 0.11% GPU time, 1248 launches, reusable=False, skip_reason=vendor backend library (precompiled binary, no rewritable source).

**Data:** device kernel `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x512_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB`; duration 10.35 ms.

**Source:** unresolved (shape provenance: torch_trace).

**Roofline:** bound=memory_bound, AI=14.5, attainment=4.9%, priority=0.11 (roofline_source=analytical).

**Suggested action:** Memory-bound: Tune GEMM tile size / precision and fuse the epilogue where possible; vendor-library GEMMs (Tensile/rocBLAS) are not rewritable — tune via library config.

### k017: aiter::mla_reduce_v1 (Other)

**Identification:** 0.11% GPU time, 1248 launches, reusable=False, skip_reason=source file not resolved.

**Data:** device kernel `_Z19kn_mla_reduce_v1_psI23MlaReduceKernelV1TraitsILi512ELi16ELi1EEfDF16bEv23MlaReduceKernelV1Params24MlaReduceKernelV1Co`; duration 10.24 ms.

**Source:** unresolved (shape provenance: torch_trace).

**Roofline:** bound=—, AI=—, attainment=—, priority=0.11 (roofline_source=placeholder).

**Suggested action:** Profile the kernel for tile size and wave occupancy.

### k018: vllm::unified_mla_attention_with_output (Quantization)

**Identification:** 0.10% GPU time, 1248 launches, reusable=False, skip_reason=source file not resolved.

**Data:** device kernel `_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel_HAS_BIAS_0_BLOCK_SIZE_M_16_BLOCK_SIZE_N_`; duration 9.01 ms.

**Source:** unresolved (shape provenance: torch_trace).

**Roofline:** bound=memory_bound, AI=0.348, attainment=0.6%, priority=0.10 (roofline_source=analytical).

**Suggested action:** Memory-bound: Fuse quantization into the adjacent GEMM epilogue and drop redundant per-tensor scaling passes.

### k019: record_param_comms (Other)

**Identification:** 0.10% GPU time, 17 launches, reusable=False, skip_reason=source file not resolved.

**Data:** device kernel `ncclDevKernel_Generic_2(ncclDevKernelArgsStorage<4096ul>)`; duration 8.86 ms.

**Source:** unresolved (shape provenance: torch_trace).

**Roofline:** bound=—, AI=—, attainment=—, priority=0.10 (roofline_source=placeholder).

**Suggested action:** Profile the kernel for tile size and wave occupancy.

### k020: aten::cat (Elementwise)

**Identification:** 0.09% GPU time, 1248 launches, reusable=False, skip_reason=source file not resolved.

**Data:** device kernel `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned `; duration 8.26 ms.

**Source:** unresolved (shape provenance: unresolved).

**Roofline:** bound=—, AI=—, attainment=—, priority=0.09 (roofline_source=placeholder).

**Suggested action:** Fuse elementwise chains to cut intermediate memory traffic.

## Appendix

- Framework: vllm
- Platform: mi300x
- Throughput unit: tok/s
- Aggregation scope: full_trace
- Events scanned: 6599622
- Attribution: 66064/66097 kernels linked to an op (99.99% of GPU time)

## Structured Metrics (CSV)

_Code-generated (no LLM). The Top-10 table above is a preview; these CSVs carry the full data._

- Per-kernel metrics (all hot kernels): `/shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/trace_analysis_proof/kernel-agent/runs/trace_analysis_proof/20260812T232620Z_bypass/bypass/kernel_metrics.csv`
- Category summary: `/shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/trace_analysis_proof/kernel-agent/runs/trace_analysis_proof/20260812T232620Z_bypass/bypass/kernel_summary.csv`
