#!/usr/bin/env python3
"""Private probe (NOT part of the patch): cheapest single-dispatch 0x80->0x00 scrub on k_fp8."""
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


for idx in (0, 2, 3):
    cfg = H.ALL_CONFIGS[idx]
    inp = H.setup_inputs(cfg)
    k8 = inp["k_fp8"]
    N, D = k8.shape
    b = k8.view(torch.uint8)
    ZERO = torch.zeros((), dtype=torch.uint8, device="cuda")

    variants = {}
    variants["where_newzeros"] = lambda: torch.where(b == 128, b.new_zeros(()), b)
    variants["where_const"] = lambda: torch.where(b == 128, ZERO, b)
    # 0x80 is the ONLY value with high bit set and low 7 bits zero.
    # bit trick: x - (x==128) is not right; use masked_fill
    def v_masked_fill():
        return b.masked_fill(b == 128, 0)
    variants["masked_fill"] = v_masked_fill
    # arithmetic, no bool tensor: (x != 128) as uint8 * x -> two ops
    variants["mul_ne"] = lambda: b * (b != 128)
    # in-place on a clone (clone + one kernel)
    def v_clone_ip():
        c = b.clone()
        c[c == 128] = 0
        return c
    variants["clone_index_put"] = v_clone_ip
    # view as int8: 0x80 == -128 == the int8 minimum -> clamp_min(-127)
    # this maps -128 -> -127 (0x81), NOT to 0. Instead: 0x80 as int8 is
    # the unique value where x == -128, so `torch.clamp(x, min=-127)`
    # would give 0x81 (a tiny negative fnuz value ~ -2^-10) rather than 0.
    i8 = k8.view(torch.int8)
    variants["clamp_min_i8"] = lambda: i8.clamp_min(-127)
    # torch.compile-free fused: use torch.where on int8
    variants["where_i8"] = lambda: torch.where(i8 == -128, i8.new_zeros(()), i8)
    # baseline: just a copy of the same size (lower bound on any pass)
    variants["clone_only"] = lambda: b.clone()

    out = []
    for name, fn in variants.items():
        try:
            out.append(f"{name}={bench(fn)*1000:6.1f}us")
        except Exception as e:
            out.append(f"{name}=ERR({type(e).__name__})")
    print(f"N={N} D={D} bytes={N*D}: " + "  ".join(out))
    del inp
    torch.cuda.empty_cache()
