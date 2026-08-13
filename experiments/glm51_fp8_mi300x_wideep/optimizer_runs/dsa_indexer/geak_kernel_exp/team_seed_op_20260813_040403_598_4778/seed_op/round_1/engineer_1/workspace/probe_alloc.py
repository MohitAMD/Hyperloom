#!/usr/bin/env python3
"""Private probe (NOT part of the patch): cost of -inf fill vs full-window check."""
import importlib.util, os
import torch

_SD = os.path.dirname(os.path.abspath(__file__))
spec2 = importlib.util.spec_from_file_location("_h", os.path.join(_SD, "test_harness.py"))
H = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(H)


def bench(fn, warmup=10, iters=40):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    lat = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize(); lat.append(s.elapsed_time(e))
    lat.sort(); return lat[len(lat)//2]


for idx in range(4):
    cfg = H.ALL_CONFIGS[idx]
    M, N = cfg[0], cfg[1]
    inp = H.setup_inputs(cfg)
    cs, ce = inp["cu_starts"], inp["cu_ends"]
    k8 = inp["k_fp8"]

    f_full = lambda: torch.full((M, N), -float("inf"), dtype=torch.float32, device="cuda")
    f_empty = lambda: torch.empty((M, N), dtype=torch.float32, device="cuda")

    def f_check():
        return bool((cs.eq(0).all() & ce.ge(N).all()).item())

    def f_check_min():
        # single fused reduce: amax(cu_starts) == 0 and amin(cu_ends) >= N
        return bool(torch.logical_and(cs.amax() <= 0, ce.amin() >= N).item())

    def f_checked_alloc():
        if f_check_min():
            return torch.empty((M, N), dtype=torch.float32, device="cuda")
        return torch.full((M, N), -float("inf"), dtype=torch.float32, device="cuda")

    def f_scrub():
        b = k8.view(torch.uint8)
        return torch.where(b == 128, b.new_zeros(()), b).view(torch.float8_e4m3fn)

    r = {}
    for name, fn in [("full", f_full), ("empty", f_empty), ("check", f_check),
                     ("check_min", f_check_min), ("checked_alloc", f_checked_alloc),
                     ("scrubK", f_scrub)]:
        r[name] = bench(fn)
    print(f"M=N={M}: " + "  ".join(f"{k}={v*1000:7.1f}us" for k, v in r.items()))
    del inp
    torch.cuda.empty_cache()
