import importlib.util, torch
WS="/shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/geak_kernel_exp/team_seed_op_20260813_105508_1474_15052/seed_op/round_1/engineer_0/verify/ws_1786621081_7759"
s=importlib.util.spec_from_file_location("opt",WS+"/moe_fp8_blockscale_g1u1.py")
opt=importlib.util.module_from_spec(s); s.loader.exec_module(opt)
GROUP=128; T,tk,E,N,K=16,8,256,2048,6144
torch.manual_seed(42)
a=(torch.randn(T*tk,K,device="cuda",dtype=torch.bfloat16)*0.25).clamp(-448,448).to(torch.float8_e4m3fn)
asc=(torch.rand(T,K//GROUP,device="cuda",dtype=torch.float32)*0.4+0.8)
w=(torch.randn(E,2*N,K,device="cuda",dtype=torch.bfloat16)*0.10).clamp(-448,448).to(torch.float8_e4m3fn)
wsc=(torch.rand(E,(2*N)//GROUP,K//GROUP,device="cuda",dtype=torch.float32)*0.4+0.8)
tid=torch.stack([torch.randperm(E,device="cuda",dtype=torch.int64)[:tk] for _ in range(T)]).to(torch.int32)
call=lambda: opt.moe_g1u1_fp8(a,asc,w,wsc,tid,N=N,group_n=GROUP,group_k=GROUP)
# COLD graph cache: warm ONLY triton JIT by calling the eager align + kernel path directly
# (mimic a serving stack that captures before the align-graph key is ever populated)
opt._GRAPHS[("__block",)] = None
# force cold: clear caches, but pre-JIT triton via one call then wipe _GRAPHS
eager=call().clone(); torch.cuda.synchronize()
opt._GRAPHS.clear()
g=torch.cuda.CUDAGraph(); st=torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
try:
    with torch.cuda.stream(st):
        with torch.cuda.graph(g):
            gout=call()
    torch.cuda.current_stream().wait_stream(st)
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    print("COLD_CACHE_OUTER_CAPTURE: OK  matches_eager:", torch.equal(gout.clone(), eager))
except Exception as e:
    print("COLD_CACHE_OUTER_CAPTURE_FAILED:", type(e).__name__, str(e)[:250])
print("graphs_cached_after:", {k: (v is not None) for k,v in opt._GRAPHS.items()})
