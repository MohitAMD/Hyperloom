#!/usr/bin/env python3
"""Private probe (NOT part of the patch): report compiled LDS/VGPR per tile config."""
import importlib.util, os
import torch

_SD = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("_k", os.path.join(_SD, "triton_fp8_mqa_logits.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
spec2 = importlib.util.spec_from_file_location("_h", os.path.join(_SD, "test_harness.py"))
H = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(H)
kern = mod._fp8_mqa_logits_kernel

cfg = (1024, 1024, 64, 128, torch.float8_e4m3fn)
inputs = H.setup_inputs(cfg)
q = inputs["q"]; k_fp8 = inputs["k_fp8"]
M, Hh, D = q.shape
N = k_fp8.shape[0]
logits = torch.empty((M, N), dtype=torch.float32, device="cuda")
sq = q.stride(); sk = k_fp8.stride(); sw = inputs["weights"].stride(); sl = logits.stride()

for (bkv, ns, nkd) in [(64,1,32),(64,1,16),(64,2,16),(128,1,16),(128,2,16),(128,2,32),(256,2,16)]:
    try:
        h = kern[(M,)](
            Q_ptr=q, KV_ptr=k_fp8, kv_scales_ptr=inputs["kv_scales"].reshape(-1), weights_ptr=inputs["weights"],
            cu_start_ptr=inputs["cu_starts"], cu_end_ptr=inputs["cu_ends"], logits_ptr=logits,
            seq_len=M, seq_len_kv=N, NUM_HEADS=Hh, HEAD_SIZE=D,
            stride_q_s=sq[0], stride_q_h=sq[1], stride_q_d=sq[2],
            stride_kv_s=sk[0], stride_kv_d=sk[1], stride_w_s=sw[0], stride_w_h=sw[1],
            stride_logits_s=sl[0], stride_logits_k=sl[1], BLOCK_KV=bkv,
            num_warps=4, num_stages=ns, waves_per_eu=2, matrix_instr_nonkdim=nkd)
        md = h.metadata
        print(f"BKV={bkv:3d} ns={ns} nkd={nkd:2d}  shared={md.shared:6d}  num_warps={md.num_warps}")
    except Exception as e:
        print(f"BKV={bkv:3d} ns={ns} nkd={nkd:2d}  ERROR {type(e).__name__}: {str(e)[:150]}")
