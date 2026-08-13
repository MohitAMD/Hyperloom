import os, sys, torch
sys.path.insert(0, "/tmp/tl_probe")
import test_harness as th
import triton, triton.language as tl

M = int(os.environ.get("M", "8000"))
cfg = (M, M, 64, 128, torch.float8_e4m3fn)
inp = th.setup_inputs(cfg)

BKV = int(os.environ.get("BKV", "128"))
NW = int(os.environ.get("NW", "4"))
NS = int(os.environ.get("NS", "1"))
WPE = int(os.environ.get("WPE", "2"))
NKD = int(os.environ.get("NKD", "16"))
FNUZ = int(os.environ.get("FNUZ", "1"))

@triton.jit
def kern(Q_ptr, KV_ptr, sc_ptr, W_ptr, L_ptr, seq_len_kv,
         H: tl.constexpr, D: tl.constexpr, sq: tl.int64, sl: tl.int64,
         BLOCK_KV: tl.constexpr, FNUZ: tl.constexpr):
    row = tl.num_programs(0) - tl.program_id(0) - 1
    tl.assume(row >= 0)
    h = tl.arange(0, H)[:, None]; d = tl.arange(0, D)
    qb = tl.load(Q_ptr + row * sq + h * D + d[None, :], cache_modifier=".cg")
    wb = tl.load(W_ptr + row * H + h).to(tl.float32)
    if FNUZ:
        qb = qb.to(tl.float8e4b8, bitcast=True)
    off = tl.arange(0, BLOCK_KV)
    kvp = KV_ptr + off[None, :] * D + d[:, None]
    scp = sc_ptr + off
    lp = L_ptr + row * sl + off
    for _ in tl.range(0, seq_len_kv, BLOCK_KV):
        kv = tl.load(kvp)
        if FNUZ:
            kv = kv.to(tl.float8e4b8, bitcast=True)
        s = tl.load(scp)
        acc = tl.dot(qb, kv, input_precision="ieee")
        if FNUZ:
            acc = acc * (4.0 * s)[None, :]
        else:
            acc = acc * s[None, :]
        acc = tl.maximum(acc, 0.0) * wb
        tl.store(lp, tl.sum(acc, axis=0))
        kvp += BLOCK_KV * D; scp += BLOCK_KV; lp += BLOCK_KV

q = inp["q"]; k = inp["k_fp8"]
if FNUZ:
    # scrub negative-zero (0x80) which is NaN in fnuz
    qv = q.view(torch.uint8); kv_ = k.view(torch.uint8)
    print("neg-zero count q,k:", (qv == 0x80).sum().item(), (kv_ == 0x80).sum().item())
    q = torch.where(qv == 0x80, torch.zeros_like(qv), qv).view(torch.float8_e4m3fn)
    k = torch.where(kv_ == 0x80, torch.zeros_like(kv_), kv_).view(torch.float8_e4m3fn)

def run():
    logits = torch.empty((M, M), dtype=torch.float32, device="cuda")
    kern[(M,)](q, k, inp["kv_scales"].reshape(-1), inp["weights"], logits, M,
               64, 128, q.stride(0), logits.stride(0), BLOCK_KV=BKV, FNUZ=FNUZ,
               num_warps=NW, num_stages=NS, waves_per_eu=WPE, matrix_instr_nonkdim=NKD)
    return logits

tag = f"FNUZ={FNUZ} BKV={BKV} NW={NW} NS={NS} WPE={WPE} NKD={NKD}"
try:
    out = run(); ok, er, cd = th.check_correctness_val(th.run_ref(inp), out)
    for _ in range(5): run()
    torch.cuda.synchronize()
    lat = []
    for _ in range(20):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); run(); e.record(); e.synchronize(); lat.append(s.elapsed_time(e))
    lat.sort(); ms = lat[len(lat)//2]
    print(f"RESULT4 {tag} -> {ms:.3f} ms {2*M*M*64*128/(ms*1e-3)/1e12:.1f} TFLOPS ok={ok} err={er:.4f} cos={cd:.2e}")
except Exception as ex:
    print(f"RESULT4 {tag} -> FAIL {type(ex).__name__}: {str(ex)[:250]}")
