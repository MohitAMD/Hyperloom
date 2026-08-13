import os, sys, math, torch, importlib.util
sys.path.insert(0, "/tmp/tl_probe")
import test_harness as th

M = int(os.environ.get("M", "8000"))
cfg = (M, M, 64, 128, torch.float8_e4m3fn)
inp = th.setup_inputs(cfg)

import triton
import triton.language as tl

PREC = os.environ.get("PREC", "ieee")
BKV = int(os.environ.get("BKV", "64"))
NW = int(os.environ.get("NW", "4"))
NS = int(os.environ.get("NS", "1"))
WPE = int(os.environ.get("WPE", "2"))
NKD = int(os.environ.get("NKD", "32"))
HOIST = int(os.environ.get("HOIST", "0"))

@triton.jit
def kern(Q_ptr, KV_ptr, kv_scales_ptr, weights_ptr, logits_ptr,
         seq_len, seq_len_kv,
         NUM_HEADS: tl.constexpr, HEAD_SIZE: tl.constexpr,
         stride_q_s: tl.int64, stride_logits_s: tl.int64,
         BLOCK_KV: tl.constexpr, PREC: tl.constexpr, HOIST: tl.constexpr):
    row_id = tl.program_id(0)
    row_id = tl.num_programs(0) - row_id - 1
    tl.assume(row_id >= 0)
    h = tl.arange(0, NUM_HEADS)[:, None]
    d = tl.arange(0, HEAD_SIZE)
    q_ptrs = Q_ptr + row_id * stride_q_s + h * HEAD_SIZE + d[None, :]
    q_block = tl.load(q_ptrs, cache_modifier=".cg")
    w_block = tl.load(weights_ptr + row_id * NUM_HEADS + h).to(tl.float32)
    off = tl.arange(0, BLOCK_KV)
    kv_ptrs = KV_ptr + off[None, :] * HEAD_SIZE + d[:, None]
    sc_ptrs = kv_scales_ptr + off
    lg_ptrs = logits_ptr + row_id * stride_logits_s + off
    for _ in tl.range(0, seq_len_kv, BLOCK_KV):
        kv = tl.load(kv_ptrs)
        s = tl.load(sc_ptrs)
        acc = tl.dot(q_block, kv, input_precision=PREC)
        if HOIST == 1:
            acc = tl.maximum(acc, 0.0) * w_block
            out = tl.sum(acc, axis=0) * s
        else:
            acc = acc * s[None, :]
            acc = tl.maximum(acc, 0.0) * w_block
            out = tl.sum(acc, axis=0)
        tl.store(lg_ptrs, out)
        kv_ptrs += BLOCK_KV * HEAD_SIZE
        sc_ptrs += BLOCK_KV
        lg_ptrs += BLOCK_KV

def run():
    q = inp["q"]; k = inp["k_fp8"]
    logits = torch.empty((M, M), dtype=torch.float32, device="cuda")
    kern[(M,)](q, k, inp["kv_scales"].reshape(-1), inp["weights"], logits,
               M, M, 64, 128, q.stride(0), logits.stride(0),
               BLOCK_KV=BKV, PREC=PREC, HOIST=HOIST,
               num_warps=NW, num_stages=NS, waves_per_eu=WPE,
               matrix_instr_nonkdim=NKD)
    return logits

try:
    out = run()
    ref = th.run_ref(inp)
    ok, er, cd = th.check_correctness_val(ref, out)
    for _ in range(5): run()
    torch.cuda.synchronize()
    lat = []
    for _ in range(20):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); run(); e.record(); e.synchronize()
        lat.append(s.elapsed_time(e))
    lat.sort()
    ms = lat[len(lat)//2]
    tflops = 2*M*M*64*128/ (ms*1e-3) / 1e12
    print(f"RESULT PREC={PREC} BKV={BKV} NW={NW} NS={NS} WPE={WPE} NKD={NKD} HOIST={HOIST} -> {ms:.3f} ms  {tflops:.1f} TFLOPS  ok={ok} err={er:.4f}")
except Exception as ex:
    print(f"RESULT PREC={PREC} BKV={BKV} NW={NW} NS={NS} WPE={WPE} NKD={NKD} HOIST={HOIST} -> FAIL {type(ex).__name__}: {str(ex)[:200]}")
