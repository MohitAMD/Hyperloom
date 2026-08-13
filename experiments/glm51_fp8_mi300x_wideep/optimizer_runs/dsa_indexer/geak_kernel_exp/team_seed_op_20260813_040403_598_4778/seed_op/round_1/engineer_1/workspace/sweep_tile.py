#!/usr/bin/env python3
"""Private sweep harness (NOT part of the patch). Times the launcher's kernel
with explicit (BLOCK_KV, num_stages, nkd) so we don't burn 4 min per full run."""
import importlib.util, os, sys, math, time
import torch

sys.argv = [sys.argv[0], "--profile"]
_SD = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("_k", os.path.join(_SD, "triton_fp8_mqa_logits.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

spec2 = importlib.util.spec_from_file_location("_h", os.path.join(_SD, "test_harness.py"))
H = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(H)

kern = mod._fp8_mqa_logits_kernel


def run(inputs, block_kv, num_stages, nkd, alloc="full", logits=None):
    q = inputs["q"]; k_fp8 = inputs["k_fp8"]
    seq_len, num_heads, head_size = q.shape
    seq_len_kv = k_fp8.shape[0]
    kv_scales_1d = inputs["kv_scales"].reshape(-1)
    if logits is None:
        if alloc == "full":
            logits = torch.full((seq_len, seq_len_kv), -float("inf"), dtype=torch.float32, device=q.device)
        else:
            logits = torch.empty((seq_len, seq_len_kv), dtype=torch.float32, device=q.device)
    sq = q.stride(); sk = k_fp8.stride(); sw = inputs["weights"].stride(); sl = logits.stride()
    kern[(seq_len,)](
        Q_ptr=q, KV_ptr=k_fp8, kv_scales_ptr=kv_scales_1d, weights_ptr=inputs["weights"],
        cu_start_ptr=inputs["cu_starts"], cu_end_ptr=inputs["cu_ends"], logits_ptr=logits,
        seq_len=seq_len, seq_len_kv=seq_len_kv, NUM_HEADS=num_heads, HEAD_SIZE=head_size,
        stride_q_s=sq[0], stride_q_h=sq[1], stride_q_d=sq[2],
        stride_kv_s=sk[0], stride_kv_d=sk[1], stride_w_s=sw[0], stride_w_h=sw[1],
        stride_logits_s=sl[0], stride_logits_k=sl[1],
        BLOCK_KV=block_kv, num_warps=4, num_stages=num_stages, waves_per_eu=2,
        matrix_instr_nonkdim=nkd,
    )
    return logits


def bench(fn, warmup=8, iters=25):
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


if __name__ == "__main__":
    idx = int(os.environ.get("CASE", "2"))
    cfg = H.ALL_CONFIGS[idx]
    inputs = H.setup_inputs(cfg)
    ref = None
    print(f"case {idx}: {H.config_str(cfg)}")
    for (bkv, ns, nkd) in [(64,1,32), (64,1,16), (64,2,16), (128,2,32), (128,2,16), (128,1,16), (256,2,16)]:
        try:
            out = run(inputs, bkv, ns, nkd)
            torch.cuda.synchronize()
            if ref is None:
                ref = H.run_ref(inputs)
            ok, err, cos = H.check_correctness_val(ref, out)
            ms = bench(lambda: run(inputs, bkv, ns, nkd))
            # kernel-only (reuse buffer, no alloc)
            buf = torch.empty((cfg[0], cfg[1]), dtype=torch.float32, device="cuda")
            ms_k = bench(lambda: run(inputs, bkv, ns, nkd, logits=buf))
            print(f"  BLOCK_KV={bkv:3d} stages={ns} nkd={nkd:2d}  full={ms:8.4f}ms  kern-only={ms_k:8.4f}ms  err={err:.4f} {'PASS' if ok else 'FAIL'}")
            del buf
        except Exception as ex:
            print(f"  BLOCK_KV={bkv:3d} stages={ns} nkd={nkd:2d}  ERROR {type(ex).__name__}: {str(ex)[:160]}")
        torch.cuda.empty_cache()
