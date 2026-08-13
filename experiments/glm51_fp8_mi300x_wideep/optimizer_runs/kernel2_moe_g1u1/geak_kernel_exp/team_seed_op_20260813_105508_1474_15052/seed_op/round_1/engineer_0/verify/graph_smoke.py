import importlib.util, torch
WS="/shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/geak_kernel_exp/team_seed_op_20260813_105508_1474_15052/seed_op/round_1/engineer_0/verify/ws_1786621081_7759"
BASE="/shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/geak_kernel_exp/team_seed_op_20260813_105508_1474_15052/seed_op/baseline"
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
opt=load(WS+"/moe_fp8_blockscale_g1u1.py","opt")
ref=load(BASE+"/moe_fp8_blockscale_g1u1.py","ref")
GROUP=128; T,tk,E,N,K=16,8,256,2048,6144   # decode regime, smallest M
torch.manual_seed(42)
a=(torch.randn(T*tk,K,device="cuda",dtype=torch.bfloat16)*0.25).clamp(-448,448).to(torch.float8_e4m3fn)
asc=(torch.rand(T,K//GROUP,device="cuda",dtype=torch.float32)*0.4+0.8)
w=(torch.randn(E,2*N,K,device="cuda",dtype=torch.bfloat16)*0.10).clamp(-448,448).to(torch.float8_e4m3fn)
wsc=(torch.rand(E,(2*N)//GROUP,K//GROUP,device="cuda",dtype=torch.float32)*0.4+0.8)
tid=torch.stack([torch.randperm(E,device="cuda",dtype=torch.int64)[:tk] for _ in range(T)]).to(torch.int32)
call=lambda: opt.moe_g1u1_fp8(a,asc,w,wsc,tid,N=N,group_n=GROUP,group_k=GROUP)
# steady state first (JIT/autotune + inner graph build happen OUTSIDE capture)
for _ in range(3): eager=call()
torch.cuda.synchronize(); eager=call().clone(); torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
try:
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            gout=call()
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    print("OUTER_CAPTURE: OK  replay_matches_eager:", torch.equal(gout.clone(), eager))
except Exception as e:
    print("OUTER_CAPTURE_FAILED:", type(e).__name__, str(e)[:300])
# same probe on the pristine baseline for comparison
call_r=lambda: ref.moe_g1u1_fp8(a,asc,w,wsc,tid,N=N,group_n=GROUP,group_k=GROUP)
for _ in range(3): call_r()
torch.cuda.synchronize()
g2=torch.cuda.CUDAGraph(); s2=torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
try:
    with torch.cuda.stream(s2):
        with torch.cuda.graph(g2):
            call_r()
    torch.cuda.current_stream().wait_stream(s2); g2.replay(); torch.cuda.synchronize()
    print("BASELINE_OUTER_CAPTURE: OK")
except Exception as e:
    print("BASELINE_OUTER_CAPTURE_FAILED:", type(e).__name__, str(e)[:200])
