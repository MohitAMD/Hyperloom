import importlib.util, torch, sys
WS="/shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/geak_kernel_exp/team_seed_op_20260813_105508_1474_15052/seed_op/round_1/engineer_0/verify/ws_1786621081_7759"
BASE="/shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/geak_kernel_exp/team_seed_op_20260813_105508_1474_15052/seed_op/baseline"
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
opt=load(WS+"/moe_fp8_blockscale_g1u1.py","opt")
ref=load(BASE+"/moe_fp8_blockscale_g1u1.py","ref")

def orig_align(topk_ids, block_m, E):
    return ref.moe_align_block_size(topk_ids, block_m, E)

torch.manual_seed(777)
bad=0; checked=0
# interleave shapes so cached graphs get reused with DIFFERENT content
shapes=[(16,8),(32,8),(64,8),(128,8)]
for it in range(40):
    T,tk = shapes[it % len(shapes)]
    E=256; bm=16
    topk = torch.stack([torch.randperm(E,device="cuda",dtype=torch.int64)[:tk] for _ in range(T)]).to(torch.int32)
    s_o,e_o,n_o = opt.moe_align_block_size(topk, bm, E)
    s_r,e_r,n_r = orig_align(topk, bm, E)
    n_ov=int(n_o.item()); n_rv=int(n_r.item())
    ok = (n_ov==n_rv)
    L=n_rv
    # compare only the live region; opt oversizes to EM_max
    ok = ok and torch.equal(s_o[:L].clone(), s_r[:L])
    ok = ok and torch.equal(e_o[:L//bm].clone(), e_r[:L//bm])
    # tail of opt buffer must be pad value
    ok = ok and bool((s_o[L:]==T*tk).all().item())
    checked+=1
    if not ok:
        bad+=1
        print(f"MISMATCH it={it} T={T} n_opt={n_ov} n_ref={n_rv}")
print(f"align_stress: checked={checked} bad={bad}")

# ---- anti-caching: does the FULL launcher respond to changed topk_ids? ----
GROUP=128
T,tk,E,N,K=16,8,256,2048,6144
torch.manual_seed(42)
a_bf=(torch.randn(T*tk,K,device="cuda",dtype=torch.bfloat16)*0.25)
a_fp8=a_bf.clamp(-448,448).to(torch.float8_e4m3fn)
a_scale=(torch.rand(T,K//GROUP,device="cuda",dtype=torch.float32)*0.4+0.8)
w_bf=(torch.randn(E,2*N,K,device="cuda",dtype=torch.bfloat16)*0.10)
w1=w_bf.clamp(-448,448).to(torch.float8_e4m3fn)
ws=(torch.rand(E,(2*N)//GROUP,K//GROUP,device="cuda",dtype=torch.float32)*0.4+0.8)
t1=torch.stack([torch.randperm(E,device="cuda",dtype=torch.int64)[:tk] for _ in range(T)]).to(torch.int32)
t2=torch.stack([torch.randperm(E,device="cuda",dtype=torch.int64)[:tk] for _ in range(T)]).to(torch.int32)
o1=opt.moe_g1u1_fp8(a_fp8,a_scale,w1,ws,t1,N=N,group_n=GROUP,group_k=GROUP).clone()
o2=opt.moe_g1u1_fp8(a_fp8,a_scale,w1,ws,t2,N=N,group_n=GROUP,group_k=GROUP).clone()
o1b=opt.moe_g1u1_fp8(a_fp8,a_scale,w1,ws,t1,N=N,group_n=GROUP,group_k=GROUP).clone()
r1=ref.moe_g1u1_fp8(a_fp8,a_scale,w1,ws,t1,N=N,group_n=GROUP,group_k=GROUP).clone()
r2=ref.moe_g1u1_fp8(a_fp8,a_scale,w1,ws,t2,N=N,group_n=GROUP,group_k=GROUP).clone()
print("differs_when_routing_changes:", not torch.equal(o1,o2))
print("repeatable_same_input:", torch.equal(o1,o1b))
print("matches_ref_t1:", torch.equal(o1,r1), " matches_ref_t2:", torch.equal(o2,r2))
print("nan_in_out:", bool(torch.isnan(o1).any().item()), bool(torch.isnan(o2).any().item()))
