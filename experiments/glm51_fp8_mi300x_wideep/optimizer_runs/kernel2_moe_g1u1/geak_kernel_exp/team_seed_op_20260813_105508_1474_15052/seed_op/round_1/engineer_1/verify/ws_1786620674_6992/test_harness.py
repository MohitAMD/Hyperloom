#!/usr/bin/env python3
"""GEAK per-kernel optimize harness for the GLM-5.1-FP8 MoE stage-1 GEMM (g1u1).

Target op:     aiter::fmoe_fp8_blockscale_g1u1  (GLM-5.1-FP8 routed-expert stage-1)
Production:    hand-written AMD ASM kernel (fmoe_bf16_blockscaleFp8_g1u1_vs_silutg),
               ~7.1% of GPU time @ ~26% roofline efficiency (compute-bound) in the
               phaseC2 decode trace -> real headroom, GEAK authors a Triton version.
Target kernel: _moe_g1u1_fp8_kernel  (@triton.jit)  in moe_fp8_blockscale_g1u1.py
Launcher:      moe_g1u1_fp8

Math (per routed (token,expert) pair; W1[e]=[2N,K], first N rows=gate, next N=up):
    gate = x @ Wgate[e]^T ;  up = x @ Wup[e]^T ;  out = silu(gate) * up
x is FP8 e4m3 with per-token per-128-K-group scales; W1 is FP8 e4m3 with [128x128]
block scales; accumulation FP32. Stage-1 does NOT apply the router weight.

GLM-5.1-FP8 shapes: hidden K=6144, moe_intermediate N=2048, E=256 experts,
top_k=8, fp8 weight_block_size=[128,128], act=silu. Decode token counts dominate.

Reference (run_ref): pure-torch fp32 dequant + GEMM + silu*mul -- independent of
the file under optimization -> trustworthy oracle.

Modes: --correctness / --benchmark / --full-benchmark / --profile.
Emits GEAK_SHAPES_USED and GEAK_RESULT_LATENCY_MS (geomean over cases, ms).
"""
import argparse
import importlib.util
import math
import os
import sys

import torch

_FP8 = torch.float8_e4m3fn
_KERNEL_BASENAME = "moe_fp8_blockscale_g1u1.py"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_kernel_file():
    local = os.path.join(_SCRIPT_DIR, _KERNEL_BASENAME)
    if os.path.isfile(local):
        return local
    roots = [r for r in (os.environ.get("GEAK_WORK_DIR", ""), os.environ.get("GEAK_REPO_ROOT", "")) if r]
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            if _KERNEL_BASENAME in files:
                return os.path.join(dirpath, _KERNEL_BASENAME)
    return local


def _load_kernel_mod():
    path = _find_kernel_file()
    spec = importlib.util.spec_from_file_location("_moe_g1u1_kernel_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"KERNEL_FILE={path}")
    return mod


_MOD = None


def _mod():
    global _MOD
    if _MOD is None:
        _MOD = _load_kernel_mod()
    return _MOD


# ══════════════════════════════════════════════════════════════════════
# ██  ADAPT — configuration + tensor creation                          ██
# ══════════════════════════════════════════════════════════════════════
# (num_tokens, top_k, num_experts, N=moe_intermediate, K=hidden)
GROUP = 128
ALL_CONFIGS = [
    (16, 8, 256, 2048, 6144),    # decode (primary — trace hot shape)
    (32, 8, 256, 2048, 6144),    # decode
    (64, 8, 256, 2048, 6144),    # decode (heavier batch)
    (128, 8, 256, 2048, 6144),   # small chunked-prefill
]

_WCACHE = {}


def _make_weights(E, N, K, dev):
    key = (E, N, K)
    if key in _WCACHE:
        return _WCACHE[key]
    torch.manual_seed(1234)
    # W1 = [E, 2N, K]  (gate rows [0,N), up rows [N,2N))
    w_bf = (torch.randn(E, 2 * N, K, device=dev, dtype=torch.bfloat16) * 0.10)
    w1_fp8 = w_bf.clamp(-448.0, 448.0).to(_FP8)
    w1_scale = (torch.rand(E, (2 * N) // GROUP, K // GROUP, device=dev, dtype=torch.float32) * 0.4 + 0.8)
    _WCACHE[key] = (w1_fp8, w1_scale)
    return _WCACHE[key]


def setup_inputs(cfg):
    num_tokens, top_k, E, N, K = cfg
    dev = "cuda"
    torch.manual_seed(42)

    a_bf = (torch.randn(num_tokens * top_k, K, device=dev, dtype=torch.bfloat16) * 0.25)
    a_fp8 = a_bf.clamp(-448.0, 448.0).to(_FP8)
    a_scale = (torch.rand(num_tokens, K // GROUP, device=dev, dtype=torch.float32) * 0.4 + 0.8)

    w1_fp8, w1_scale = _make_weights(E, N, K, dev)

    # random top_k distinct experts per token
    topk_ids = torch.stack([
        torch.randperm(E, device=dev, dtype=torch.int64)[:top_k]
        for _ in range(num_tokens)
    ]).to(torch.int32)

    return {
        "a_fp8": a_fp8, "a_scale": a_scale,
        "w1_fp8": w1_fp8, "w1_scale": w1_scale,
        "topk_ids": topk_ids,
        "N": N, "K": K, "E": E, "top_k": top_k, "num_tokens": num_tokens,
    }


def run_kernel(inp):
    return _mod().moe_g1u1_fp8(
        inp["a_fp8"], inp["a_scale"], inp["w1_fp8"], inp["w1_scale"], inp["topk_ids"],
        N=inp["N"], group_n=GROUP, group_k=GROUP,
    )


def run_ref(inp):
    """Pure-torch fp32 dequant + grouped GEMM + silu*mul (no router weight)."""
    a_fp8 = inp["a_fp8"]; a_scale = inp["a_scale"]
    w1_fp8 = inp["w1_fp8"]; w1_scale = inp["w1_scale"]
    topk_ids = inp["topk_ids"]
    N = inp["N"]; K = inp["K"]; top_k = inp["top_k"]
    num_tokens = inp["num_tokens"]

    # dequant activations: a_deq[r,k] = a_fp8[r,k] * a_scale[token(r), k//G]
    a = a_fp8.to(torch.float32)
    a_scale_exp = a_scale.repeat_interleave(GROUP, dim=1)[:, :K]           # [nt, K]
    a_scale_rows = a_scale_exp.repeat_interleave(top_k, dim=0)             # [nt*top_k, K]
    a_deq = a * a_scale_rows

    out = torch.empty((num_tokens * top_k, N), device=a_fp8.device, dtype=torch.float32)
    flat_experts = topk_ids.reshape(-1)                                   # [nt*top_k]

    # dequant per-expert weight on the fly (memory friendly)
    for r in range(num_tokens * top_k):
        e = int(flat_experts[r].item())
        w_e = w1_fp8[e].to(torch.float32)                                 # [2N, K]
        ws = w1_scale[e].repeat_interleave(GROUP, dim=0)[:2 * N, :]
        ws = ws.repeat_interleave(GROUP, dim=1)[:, :K]                    # [2N, K]
        w_deq = w_e * ws
        x = a_deq[r]                                                      # [K]
        proj = w_deq @ x                                                  # [2N]
        gate = proj[:N]; up = proj[N:2 * N]
        silu = gate * torch.sigmoid(gate)
        out[r] = silu * up
    return out.to(torch.bfloat16)


def config_str(cfg):
    num_tokens, top_k, E, N, K = cfg
    return f"T={num_tokens} topk={top_k} E={E} N={N} K={K}"


# ══════════════════════════════════════════════════════════════════════
# ██  FIXED BOILERPLATE — benchmark & mode infrastructure              ██
# ══════════════════════════════════════════════════════════════════════
WARMUP = int(os.environ.get("GEAK_WARMUP_ITERS", "20"))
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "50"))


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def check_correctness_val(out_ref, out_kernel):
    r = out_ref.double().reshape(-1)
    k = out_kernel.double().reshape(-1)
    finite = torch.isfinite(r) & torch.isfinite(k)
    if finite.sum().item() == 0:
        return False, 1.0, 1.0
    r = r[finite]; k = k[finite]
    rtol, atol, max_err_ratio = 5e-2, 5e-2, 0.05
    isClose = torch.isclose(k, r, rtol=rtol, atol=atol)
    err_ratio = 0.0 if bool(isClose.all()) else (~isClose).sum().item() / r.numel()
    denom = (r * r + k * k).sum().item()
    cos_diff = 1 - 2 * (r * k).sum().item() / max(denom, 1e-12)
    return err_ratio <= max_err_ratio, err_ratio, cos_diff


def benchmark_kernel(inp):
    def fn():
        run_kernel(inp)
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
            inp = setup_inputs(cfg)
            out = run_kernel(inp)
            ref = run_ref(inp)
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
            inp = setup_inputs(cfg)
            ms = benchmark_kernel(inp)
            print(f"  {label}  {ms:.4f}ms  (case={idx} GEAK_RESULT_LATENCY_MS={ms:.4f})")
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
            inp = setup_inputs(cfg)
            run_kernel(inp)
            print(f"  {label}  OK")
        except Exception as e:
            print(f"  {label}  ERROR: {e}")
        finally:
            torch.cuda.empty_cache()
    print(f"GEAK_SHAPES_USED={indices}")


def main():
    parser = argparse.ArgumentParser(description="GLM-5.1-FP8 MoE g1u1 stage-1 harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    total = len(ALL_CONFIGS)
    print(f"Total configs: {total}")
    if args.correctness:
        mode_correctness(list(range(total)))
    elif args.benchmark:
        mode_benchmark(_pick(ALL_CONFIGS, 3))
    elif args.full_benchmark:
        mode_benchmark(list(range(total)))
    elif args.profile:
        mode_profile(_pick(ALL_CONFIGS, 4))


if __name__ == "__main__":
    main()
