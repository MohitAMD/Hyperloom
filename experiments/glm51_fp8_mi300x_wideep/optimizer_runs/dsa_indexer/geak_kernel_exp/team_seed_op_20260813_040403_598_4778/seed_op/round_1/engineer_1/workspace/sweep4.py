#!/usr/bin/env python3
"""Private sweep #4 (NOT part of the patch): fine knobs around the current winner."""
import importlib.util, os
import torch

_SD = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("_k", os.path.join(_SD, "triton_fp8_mqa_logits.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
spec2 = importlib.util.spec_from_file_location("_h", os.path.join(_SD, "test_harness.py"))
H = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(H)
kern = mod._fp8_mqa_logits_kernel


def run(inputs, logits, block_kv, **kw):
    q = inputs["q"]; k_fp8 = inputs["k_fp8"]
    seq_len, num_heads, head_size = q.shape
    seq_len_kv = k_fp8.shape[0]
    sq = q.stride(); sk = k_fp8.stride(); sw = inputs["weights"].stride(); sl = logits.stride()
    kern[(seq_len,)](
        Q_ptr=q, KV_ptr=k_fp8, kv_scales_ptr=inputs["kv_scales"].reshape(-1), weights_ptr=inputs["weights"],
        cu_start_ptr=inputs["cu_starts"], cu_end_ptr=inputs["cu_ends"], logits_ptr=logits,
        seq_len=seq_len, seq_len_kv=seq_len_kv, NUM_HEADS=num_heads, HEAD_SIZE=head_size,
        stride_q_s=sq[0], stride_q_h=sq[1], stride_q_d=sq[2],
        stride_kv_s=sk[0], stride_kv_d=sk[1], stride_w_s=sw[0], stride_w_h=sw[1],
        stride_logits_s=sl[0], stride_logits_k=sl[1], BLOCK_KV=block_kv, **kw)


def bench(fn, warmup=8, iters=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    lat = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize(); lat.append(s.elapsed_time(e))
    lat.sort(); return lat[len(lat)//2]


idx = int(os.environ.get("CASE", "2"))
cfg = H.ALL_CONFIGS[idx]
inputs = H.setup_inputs(cfg)
ref = H.run_ref(inputs)
buf = torch.empty((cfg[0], cfg[1]), dtype=torch.float32, device="cuda")
print(f"case {idx}: {H.config_str(cfg)}")

TRIALS = [
    (128, dict(num_warps=4, num_stages=2, waves_per_eu=3, matrix_instr_nonkdim=16)),
    (128, dict(num_warps=4, num_stages=2, waves_per_eu=3, matrix_instr_nonkdim=16, kpack=1)),
    (128, dict(num_warps=4, num_stages=2, waves_per_eu=3, matrix_instr_nonkdim=16,
               schedule_hint="attention")),
    (128, dict(num_warps=4, num_stages=2, waves_per_eu=3, matrix_instr_nonkdim=16,
               schedule_hint="memory-bound-attention")),
    (128, dict(num_warps=8, num_stages=2, waves_per_eu=3, matrix_instr_nonkdim=16)),
]
for bkv, kw in TRIALS:
    try:
        run(inputs, buf, bkv, **kw); torch.cuda.synchronize()
        ok, err, _ = H.check_correctness_val(ref, buf)
        ms = bench(lambda: run(inputs, buf, bkv, **kw))
        print(f"  {kw}  {ms:8.4f}ms err={err:.4f} {'PASS' if ok else 'FAIL'}")
    except Exception as ex:
        print(f"  {kw}  ERROR {type(ex).__name__}: {str(ex)[:150]}")
