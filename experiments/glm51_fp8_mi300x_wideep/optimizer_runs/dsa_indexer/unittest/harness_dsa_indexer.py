#!/usr/bin/env python3
"""GEAK 4-mode harness for the DSA sparse-MLA indexer prefill logits kernel.

Target op:    vllm::rocm_aiter_sparse_attn_indexer  (GLM-5.1-FP8, deepseek-style DSA)
Target kernel: _fp8_mqa_logits_kernel  (@triton.jit)
Source:        vllm/v1/attention/ops/triton_fp8_mqa_logits.py
Launcher:      fp8_mqa_logits_gfx942  (grid=(seq_len,))  -- the gfx942/MI300X
               vendored path (ROCm/aiter#3257 tile fix) used when
               VLLM_ROCM_USE_AITER_MLA=1.

Reference vs kernel:
  * run_kernel -> the (possibly GEAK-patched) fp8_mqa_logits_gfx942. When GEAK
    drops a patched copy of triton_fp8_mqa_logits.py into GEAK_WORK_DIR we load
    THAT file by path so the microbench measures the candidate; otherwise we
    fall back to the installed vllm module (covers GEAK's in-place edit mode).
  * run_ref  -> a pure-torch fp32 re-implementation of the exact kernel math
    (per-head q.k, * kv_scale, relu, * weight, sum over heads), independent of
    the file under optimization.

Cost scales with KV length (long-context / prefill), not OSL / concurrency, so
this profiles at a cheap serving shape.

Contract: reads GEAK_WORK_DIR / GEAK_BENCHMARK_ITERATIONS; supports
--correctness / --benchmark / --full-benchmark / --profile; emits
GEAK_SHAPES_USED and GEAK_RESULT_LATENCY_MS.
"""
import argparse
import importlib.util
import math
import os
import sys

import torch

# ══════════════════════════════════════════════════════════════════════
# ██  FIXED BOILERPLATE                                              ██
# ══════════════════════════════════════════════════════════════════════

_GEAK_WORK_DIR = os.environ.get("GEAK_WORK_DIR", "") or os.environ.get("GEAK_REPO_ROOT", "")
if _GEAK_WORK_DIR and _GEAK_WORK_DIR not in sys.path:
    sys.path.insert(0, _GEAK_WORK_DIR)

WARMUP = int(os.environ.get("GEAK_WARMUP_ITERS", "20"))
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "50"))

# DSv4 / GLM-5.1 sparse-indexer shape (see kernel source line ~218):
#   NUM_HEADS=64, HEAD_SIZE=128, fp8 dtype torch.float8_e4m3fn.
_FP8 = torch.float8_e4m3fn
_KERNEL_BASENAME = "triton_fp8_mqa_logits.py"
_INSTALLED_KERNEL_PATH = (
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_fp8_mqa_logits.py"
)


def _find_kernel_file():
    """Locate the fp8_mqa_logits source, preferring a GEAK-patched copy.

    GEAK either (a) copies the kernel source into GEAK_WORK_DIR and edits the
    copy, or (b) edits the installed dist-packages file in place. Prefer (a) by
    scanning GEAK_WORK_DIR for the basename; else use the installed path.
    """
    roots = [r for r in (os.environ.get("GEAK_WORK_DIR", ""), os.environ.get("GEAK_REPO_ROOT", "")) if r]
    for root in roots:
        # Exact conventional location first.
        cand = os.path.join(root, "vllm", "v1", "attention", "ops", _KERNEL_BASENAME)
        if os.path.isfile(cand):
            return cand
        # Otherwise walk the work dir for the basename.
        for dirpath, _dirs, files in os.walk(root):
            if _KERNEL_BASENAME in files:
                return os.path.join(dirpath, _KERNEL_BASENAME)
    return _INSTALLED_KERNEL_PATH


def _load_kernel_launcher():
    """Import fp8_mqa_logits_gfx942 from the (patched-if-present) source file."""
    path = _find_kernel_file()
    spec = importlib.util.spec_from_file_location("_dsa_indexer_kernel_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"KERNEL_FILE={path}")
    return mod.fp8_mqa_logits_gfx942


_LAUNCHER = None


def _launcher():
    global _LAUNCHER
    if _LAUNCHER is None:
        _LAUNCHER = _load_kernel_launcher()
    return _LAUNCHER


# ══════════════════════════════════════════════════════════════════════
# ██  ADAPT — configuration + tensor creation + adapters             ██
# ══════════════════════════════════════════════════════════════════════

# (seq_len, seq_len_kv, num_heads, head_dim, dtype)
# CHEAP long-context shapes anchored on the real serving shape (~8k prefill,
# DSv4 NUM_HEADS=64 HEAD_SIZE=128). Smaller shapes give fast correctness; the
# 8k shapes are the representative optimization target.
ALL_CONFIGS = [
    (2048, 2048, 64, 128, _FP8),
    (4096, 4096, 64, 128, _FP8),
    (8000, 8000, 64, 128, _FP8),
    (8192, 8192, 64, 128, _FP8),
]


def setup_inputs(cfg):
    seq_len, seq_len_kv, num_heads, head_dim, dtype = cfg
    torch.manual_seed(42)
    dev = "cuda"

    # fp8 q [M,H,D] and k [N,D]; build in fp32 then cast (values are exactly
    # representable back in fp32 for a faithful reference).
    q_f = torch.randn(seq_len, num_heads, head_dim, device=dev, dtype=torch.float32) * 0.25
    k_f = torch.randn(seq_len_kv, head_dim, device=dev, dtype=torch.float32) * 0.25
    q = q_f.clamp(-448.0, 448.0).to(dtype)
    k_fp8 = k_f.clamp(-448.0, 448.0).to(dtype)

    # Per-K fp32 scales; vLLM passes a [N,4] uint8 view-cast-to-float32 ([N,1]).
    kv_scales = (torch.rand(seq_len_kv, 1, device=dev, dtype=torch.float32) * 0.5 + 0.5)

    # Per-(row,head) fp32 weights [M,H].
    weights = torch.randn(seq_len, num_heads, device=dev, dtype=torch.float32).abs()

    # Full window (start=0, end=seq_len_kv): exercises the full compute path
    # (dominant long-context cost) and avoids -inf boundary comparison noise.
    cu_starts = torch.zeros(seq_len, device=dev, dtype=torch.int32)
    cu_ends = torch.full((seq_len,), seq_len_kv, device=dev, dtype=torch.int32)

    return {
        "q": q,
        "k_fp8": k_fp8,
        "kv_scales": kv_scales,
        "weights": weights,
        "cu_starts": cu_starts,
        "cu_ends": cu_ends,
        # kept for the reference (upcast operands):
        "q_f32": q.to(torch.float32),
        "k_f32": k_fp8.to(torch.float32),
    }


def run_kernel(inputs):
    return _launcher()(
        inputs["q"],
        inputs["k_fp8"],
        inputs["kv_scales"],
        inputs["weights"],
        inputs["cu_starts"],
        inputs["cu_ends"],
    )


def run_ref(inputs):
    """Pure-torch fp32 reference of _fp8_mqa_logits_kernel's math.

    logits[i,j] = sum_h relu( (q[i,h,:].k[j,:]) * kv_scale[j] ) * weight[i,h]
    within [cu_starts[i], cu_ends[i]); -inf outside.
    """
    q = inputs["q_f32"]          # [M,H,D]
    k = inputs["k_f32"]          # [N,D]
    kv_scales = inputs["kv_scales"].reshape(-1)  # [N]
    weights = inputs["weights"]  # [M,H]
    cu_starts = inputs["cu_starts"]
    cu_ends = inputs["cu_ends"]

    seq_len, num_heads, _ = q.shape
    seq_len_kv = k.shape[0]

    # [M,H,N] = q[M,H,D] @ k^T[D,N]
    scores = torch.einsum("mhd,nd->mhn", q, k)
    scores = scores * kv_scales.view(1, 1, seq_len_kv)
    scores = torch.clamp(scores, min=0.0)               # relu
    scores = scores * weights.unsqueeze(-1)             # * weight[i,h]
    logits = scores.sum(dim=1)                          # sum over heads -> [M,N]

    # Apply the per-row window mask -> -inf outside [start,end).
    col = torch.arange(seq_len_kv, device=q.device).view(1, seq_len_kv)
    in_win = (col >= cu_starts.view(seq_len, 1)) & (col < cu_ends.view(seq_len, 1))
    logits = torch.where(in_win, logits, torch.full_like(logits, float("-inf")))
    return logits


def config_str(cfg):
    seq_len, seq_len_kv, num_heads, head_dim, dtype = cfg
    return f"M={seq_len} N={seq_len_kv} H={num_heads} D={head_dim} {dtype}"


# ══════════════════════════════════════════════════════════════════════
# ██  FIXED BOILERPLATE — benchmark & mode infrastructure            ██
# ══════════════════════════════════════════════════════════════════════

def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def check_correctness_val(out_ref, out_kernel):
    # fp8 operands -> fp32 accumulation; allow modest tolerance. Compare only
    # finite (in-window) positions so -inf==-inf sentinels don't skew ratios.
    finite = torch.isfinite(out_ref) & torch.isfinite(out_kernel)
    if finite.sum().item() == 0:
        return False, 1.0, 1.0
    r = out_ref[finite].double()
    k = out_kernel[finite].double()
    rtol, atol, max_err_ratio = 5e-2, 5e-2, 0.02
    isClose = torch.isclose(k, r, rtol=rtol, atol=atol)
    err_ratio = 0.0 if bool(isClose.all()) else (~isClose).sum().item() / r.numel()
    denom = (r * r + k * k).sum().item()
    cos_diff = 1 - 2 * (r * k).sum().item() / max(denom, 1e-12)
    return err_ratio <= max_err_ratio, err_ratio, cos_diff


def benchmark_kernel(inputs):
    def fn():
        run_kernel(inputs)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    lat = []
    for _ in range(ITERATIONS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        lat.append(s.elapsed_time(e))
    lat.sort()
    return lat[len(lat) // 2]


def mode_correctness(indices):
    print(f"Running correctness check on {len(indices)} configs...")
    all_pass = True
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = config_str(cfg)
        try:
            inputs = setup_inputs(cfg)
            out = run_kernel(inputs)
            ref = run_ref(inputs)
            passed, err_ratio, cos_diff = check_correctness_val(ref, out)
            status = "PASS" if passed else "FAIL"
            print(f"  [{idx}] {label}  err_ratio={err_ratio:.4f} cos_diff={cos_diff:.2e}  {status}")
            if not passed:
                all_pass = False
        except Exception as e:
            print(f"  [{idx}] {label}  ERROR: {e}")
            all_pass = False
        finally:
            torch.cuda.empty_cache()
    print(f"GEAK_SHAPES_USED={indices}")
    if not all_pass:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


def mode_benchmark(indices):
    print(f"Running benchmark on {len(indices)} configs...")
    latencies = []
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = config_str(cfg)
        try:
            inputs = setup_inputs(cfg)
            ms = benchmark_kernel(inputs)
            print(f"  {label}  {ms:.4f}ms")
            latencies.append(ms)
        except Exception as e:
            print(f"  {label}  ERROR: {e}")
        finally:
            torch.cuda.empty_cache()
    print(f"GEAK_SHAPES_USED={indices}")
    if latencies:
        geo = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
        print(f"GEAK_RESULT_LATENCY_MS={geo:.4f}")
    else:
        print("No successful benchmarks")
        sys.exit(1)


def mode_profile(indices):
    print(f"Running profile on {len(indices)} configs...")
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = config_str(cfg)
        try:
            inputs = setup_inputs(cfg)
            run_kernel(inputs)
            print(f"  {label}  OK")
        except Exception as e:
            print(f"  {label}  ERROR: {e}")
        finally:
            torch.cuda.empty_cache()
    print(f"GEAK_SHAPES_USED={indices}")


def main():
    parser = argparse.ArgumentParser(description="DSA indexer fp8_mqa_logits harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    total = len(ALL_CONFIGS)
    print(f"Total configs: {total}")
    if args.correctness:
        mode_correctness(_pick(ALL_CONFIGS, 25))
    elif args.benchmark:
        mode_benchmark(_pick(ALL_CONFIGS, 25))
    elif args.full_benchmark:
        mode_benchmark(list(range(total)))
    elif args.profile:
        mode_profile(_pick(ALL_CONFIGS, 5))


if __name__ == "__main__":
    main()
