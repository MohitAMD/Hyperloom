#!/usr/bin/env python3
"""GEMM-isolated config sweep for _moe_g1u1_fp8_kernel."""
import importlib.util, itertools, json, math, os, sys, time
import torch, triton

WS = os.path.dirname(os.path.abspath(__file__)) + "/workspace"
sys.path.insert(0, WS)
spec = importlib.util.spec_from_file_location("kmod", WS + "/moe_fp8_blockscale_g1u1.py")
kmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(kmod)

sys.path.insert(0, WS)
import importlib.util as iu
hspec = iu.spec_from_file_location("hmod", WS + "/test_harness.py")
hmod = iu.module_from_spec(hspec); hspec.loader.exec_module(hmod)

GROUP = 128
CASES = hmod.ALL_CONFIGS


def bench(fn, warmup=15, iters=40):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    lat = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        lat.append(s.elapsed_time(e))
    lat.sort()
    return lat[len(lat)//2]


def run_case(ci, cfgs):
    cfg = CASES[ci]
    inp = hmod.setup_inputs(cfg)
    a_fp8 = inp["a_fp8"]; a_scale = inp["a_scale"]; w1 = inp["w1_fp8"]; w1s = inp["w1_scale"]
    topk_ids = inp["topk_ids"]; N = inp["N"]; K = inp["K"]; E = inp["E"]
    num_tokens, top_k = topk_ids.shape
    num_valid = num_tokens * top_k
    out = torch.zeros((num_valid, N), device=a_fp8.device, dtype=torch.bfloat16)
    ref = None
    align_cache = {}
    res = {}
    for c in cfgs:
        bm = c["BLOCK_M"]
        if bm not in align_cache:
            align_cache[bm] = kmod.moe_align_block_size(topk_ids, bm, E)
        sti, eids, ntpp = align_cache[bm]
        EM = sti.numel()
        extra = {k: v for k, v in c.items() if k not in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "num_warps", "num_stages")}
        grid = (triton.cdiv(EM, bm) * triton.cdiv(N, c["BLOCK_N"]),)

        def launch():
            kmod._moe_g1u1_fp8_kernel[grid](
                a_fp8, w1, out, a_scale, w1s, sti, eids, ntpp,
                N, K, EM, num_valid, top_k,
                a_fp8.stride(0), a_fp8.stride(1),
                w1.stride(0), w1.stride(1), w1.stride(2),
                out.stride(0), out.stride(1),
                a_scale.stride(0), a_scale.stride(1),
                w1s.stride(0), w1s.stride(1), w1s.stride(2),
                group_n=GROUP, group_k=GROUP,
                BLOCK_M=bm, BLOCK_N=c["BLOCK_N"], BLOCK_K=c["BLOCK_K"], GROUP_M=c["GROUP_M"],
                num_warps=c["num_warps"], num_stages=c["num_stages"], **extra)
        try:
            out.zero_(); launch(); torch.cuda.synchronize()
            if ref is None:
                ref = out.clone()
                ok = True
            else:
                d = (out.float() - ref.float()).abs()
                den = ref.float().abs() * 5e-2 + 5e-2
                ok = float((d > den).float().mean()) <= 0.05
            ms = bench(launch)
        except Exception as ex:
            print(f"  case{ci} {c} FAIL {type(ex).__name__}: {str(ex)[:120]}")
            continue
        res[json.dumps(c, sort_keys=True)] = (ms, ok)
        print(f"  case{ci} {c} -> {ms:.3f} ms ok={ok}", flush=True)
    del inp, a_fp8, a_scale, out
    torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "base"
    cfgs = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else [
        dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=128, GROUP_M=1, num_warps=4, num_stages=2)]
    allres = {}
    for ci in [int(x) for x in which.split(",")]:
        allres[ci] = run_case(ci, cfgs)
    print("DONE")
