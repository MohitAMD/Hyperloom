# Performance Analysis Report — GLM-5.1-FP8

> Generated via deterministic route (HYPERLOOM_TRACE_ANALYSIS_ROUTE=deterministic). Deterministic hot-kernel extraction from structured *_metrics.json / priority_data.json.

## Executive Summary

| Metric | Value |
|--------|-------|
| Total GPU Time | 10591.741 ms |
| GPU Busy % | 99.73% |
| GPU Idle % | 0.27% |
| GPU MemCpy | 17.688 ms |
| Top Bottleneck Category | Other |
| Op-attribution Coverage | — |

## System-Level Signals

| Signal | % of total GPU time | Note |
|--------|---------------------|------|
| GPU idle | 0.27% | within 80% idle gate |
| Exposed communication | 5.18% | - |
| Exposed memcpy (device copy) | 0.17% | - |

## Top Hot Kernels

| Rank | Operation | Time (us) | GPU% | Eff% | AI | Bound | Category | Source File |
|------|-----------|-----------|------|------|----|-------|----------|-------------|
| 1 | vllm::rocm_aiter_fused_allreduce_rmsnorm | 3944155.0 | 37.24% | 0.00% | — | — | Other | /usr/local/lib/python3.12/dist-packages/vllm/_aiter_ops.py |
| 2 | vllm::rocm_aiter_sparse_attn_indexer | 1439443.0 | 13.59% | 0.00% | — | — | Other | /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py |
| 3 | aiter::fmoe_fp8_blockscale_g1u1 | 571559.0 | 5.40% | 25.73% | — | compute | MoE | /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/py_itfs_cu/asm_fmoe.cu |
| 4 | aiter::fmoe_fp8_blockscale_g1u1 | 564374.0 | 5.33% | 80.60% | — | memory | MoE | /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/py_itfs_cu/asm_fmoe.cu |
| 5 | vllm::rocm_aiter_sparse_attn_indexer | 420868.0 | 3.97% | 0.00% | — | — | Other | /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py |
| 6 | vllm::rocm_aiter_sparse_attn_indexer | 289616.0 | 2.73% | 0.00% | — | — | Other | /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py |
| 7 | aiter::fused_allreduce_rmsnorm_quant_per_group | 176109.0 | 1.66% | 0.00% | — | — | Other | /usr/local/lib/python3.12/dist-packages/aiter/dist/device_communicators/custom_all_reduce.py |
| 8 | aiter::fused_allreduce_rmsnorm_quant_per_group | 176057.0 | 1.66% | 0.00% | — | — | Other | /usr/local/lib/python3.12/dist-packages/aiter/dist/device_communicators/custom_all_reduce.py |
| 9 | pseudo_mla_decode_fwd | 68198.0 | 0.64% | 0.00% | — | — | SDPA | /sgl-workspace/aiter/csrc/py_itfs_cu/asm_mla.cu |
| 10 | pseudo_mla_decode_fwd | 68198.0 | 0.64% | 0.00% | — | — | SDPA | /sgl-workspace/aiter/aiter/mla.py |
| 11 | aiter::gemm_a8w8_blockscale_ck | 115595.0 | 1.09% | 56.10% | — | compute | GEMM | /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale.cu |

### P1: Other kernels

<!-- reasoning-candidate tier=compute rank=1 -->

**Data:**

| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | Efficiency | Bound | Args | Source File | Kernel Path (launcher) |
|-----------|-----------|------|------|-------|------------|------------|-------|------|-------------|------------------------|
| vllm::rocm_aiter_fused_allreduce_rmsnorm | 3944155.0 | 37.24% | 11.17 | 2052 | — | 0.00% | — | (128,6144) bf16<br>(128,6144) bf16<br>(6144,) bf16 | /usr/local/lib/python3.12/dist-packages/vllm/_aiter_ops.py | vllm/_aiter_ops.py(786): _rocm_aiter_fused_allreduce_rmsnorm_impl |

### P2: Other kernels

<!-- reasoning-candidate tier=compute rank=2 -->

**Data:**

| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | Efficiency | Bound | Args | Source File | Kernel Path (launcher) |
|-----------|-----------|------|------|-------|------------|------------|-------|------|-------------|------------------------|
| vllm::rocm_aiter_sparse_attn_indexer | 1439443.0 | 13.59% | 4.08 | 312 | — | 0.00% | — | (8192,6144) bf16<br>(1122254,1,132) fp8<br>(8192,32,128) fp8<br>(8192,128) bf16<br>(8192,32) fp32<br>(8192,2048) int | /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py | vllm/v1/attention/ops/rocm_aiter_mla_sparse.py(645): rocm_aiter_sparse_attn_indexer |
| vllm::rocm_aiter_sparse_attn_indexer | 420868.0 | 3.97% | 1.19 | 2106 | — | 0.00% | — | (128,6144) bf16<br>(1122254,1,132) fp8<br>(128,32,128) fp8<br>(128,128) bf16<br>(128,32) fp32<br>(8192,2048) int | /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py | vllm/v1/attention/ops/rocm_aiter_mla_sparse.py(645): rocm_aiter_sparse_attn_indexer |
| vllm::rocm_aiter_sparse_attn_indexer | 289616.0 | 2.73% | 0.82 | 78 | — | 0.00% | — | (4096,6144) bf16<br>(1122254,1,132) fp8<br>(4096,32,128) fp8<br>(4096,128) bf16<br>(4096,32) fp32<br>(8192,2048) int | /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py | vllm/v1/attention/ops/rocm_aiter_mla_sparse.py(645): rocm_aiter_sparse_attn_indexer |

### P3: MoE kernels

<!-- reasoning-candidate tier=compute rank=3 -->

**Data:**

| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | Efficiency | Bound | Args | Source File | Kernel Path (launcher) |
|-----------|-----------|------|------|-------|------------|------------|-------|------|-------------|------------------------|
| aiter::fmoe_fp8_blockscale_g1u1 | 571559.0 | 5.40% | 3.51 | 300 | — | 25.73% | compute | (8192,6144) bf16<br>(8192,6144) fp8<br>(256,512,6144) fp8<br>(256,6144,256) fp8<br>(73720,) int<br>(73720,) fp32<br>(2304,) int<br>(2,) int<br>(8192,48) fp32<br>(256,4,48) fp32<br>(256,48,2) fp32 | /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/py_itfs_cu/asm_fmoe.cu | aiter/fused_moe.py(833): fused_moestage |

### P4: GEMM kernels

<!-- reasoning-candidate tier=compute rank=4 -->

**Data:**

| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | Efficiency | Bound | Args | Source File | Kernel Path (launcher) |
|-----------|-----------|------|------|-------|------------|------------|-------|------|-------------|------------------------|
| aiter::gemm_a8w8_blockscale_ck | 115595.0 | 1.09% | 0.42 | 312 | — | 56.10% | compute | (8192,6144) fp8<br>(2624,6144) fp8<br>(8192,48) fp32<br>(21,48) fp32<br>(8192,2624) bf16 | /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale.cu | aiter/ops/gemm_op_a8w8.py(756): gemm_a8w8_blockscale |

### P5: Other kernels

<!-- reasoning-candidate tier=compute rank=5 -->

**Data:**

| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | Efficiency | Bound | Args | Source File | Kernel Path (launcher) |
|-----------|-----------|------|------|-------|------------|------------|-------|------|-------------|------------------------|
| aiter::fused_allreduce_rmsnorm_quant_per_group | 176109.0 | 1.66% | 0.50 | 81 | — | 0.00% | — | (1,) fp32<br>(128,6144) bf16<br>(128,6144) bf16<br>(128,6144) bf16<br>(128,6144) fp8<br>(128,48) fp32<br>(6144,) bf16 | /usr/local/lib/python3.12/dist-packages/aiter/dist/device_communicators/custom_all_reduce.py | aiter/dist/device_communicators/custom_all_reduce.py(1610): fused_ar_rms_per_group_quant |
| aiter::fused_allreduce_rmsnorm_quant_per_group | 176057.0 | 1.66% | 0.50 | 3 | — | 0.00% | — | (1,) fp32<br>(4096,6144) bf16<br>(4096,6144) bf16<br>(4096,6144) bf16<br>(4096,6144) fp8<br>(4096,48) fp32<br>(6144,) bf16 | /usr/local/lib/python3.12/dist-packages/aiter/dist/device_communicators/custom_all_reduce.py | aiter/dist/device_communicators/custom_all_reduce.py(1610): fused_ar_rms_per_group_quant |

### P7: MoE kernels

<!-- reasoning-candidate tier=compute rank=7 -->

**Data:**

| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | Efficiency | Bound | Args | Source File | Kernel Path (launcher) |
|-----------|-----------|------|------|-------|------------|------------|-------|------|-------------|------------------------|
| aiter::fmoe_fp8_blockscale_g1u1 | 564374.0 | 5.33% | 0.90 | 2025 | — | 80.60% | memory | (128,6144) bf16<br>(128,6144) fp8<br>(256,512,6144) fp8<br>(256,6144,256) fp8<br>(9208,) int<br>(9208,) fp32<br>(288,) int<br>(2,) int<br>(128,48) fp32<br>(256,4,48) fp32<br>(256,48,2) fp32 | /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/py_itfs_cu/asm_fmoe.cu | aiter/fused_moe.py(833): fused_moestage |

### P8: SDPA kernels

<!-- reasoning-candidate tier=compute rank=8 -->

**Data:**

| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | Efficiency | Bound | Args | Source File | Kernel Path (launcher) |
|-----------|-----------|------|------|-------|------------|------------|-------|------|-------------|------------------------|
| pseudo_mla_decode_fwd | 68198.0 | 0.64% | 0.39 | 2028 | — | 0.00% | — | (125,16,576) fp8<br>(1122254,1,1,576) fp8<br>(126,) int<br>(126,) int<br>(256000,) int<br>(125,) int<br>(2,) long unsigned int<br>(305,) int<br>(8495,8) int<br>(606,1,16,512) fp32<br>(606,1,16,1) fp32<br>(125,16,512) bf16<br>() fp32<br>() fp32 | /sgl-workspace/aiter/csrc/py_itfs_cu/asm_mla.cu | aiter/mla.py(246): mla_decode_fwd |
| pseudo_mla_decode_fwd | 68198.0 | 0.64% | 0.39 | 2028 | — | 0.00% | — | (125,16,576) fp8<br>(1122254,1,1,576) fp8<br>(126,) int<br>(126,) int<br>(256000,) int<br>(125,) int<br>(2,) long unsigned int<br>(305,) int<br>(8495,8) int<br>(606,1,16,512) fp32<br>(606,1,16,1) fp32<br>(125,16,512) bf16<br>() fp32<br>() fp32 | /sgl-workspace/aiter/aiter/mla.py | aiter/mla.py(246): mla_decode_fwd |
