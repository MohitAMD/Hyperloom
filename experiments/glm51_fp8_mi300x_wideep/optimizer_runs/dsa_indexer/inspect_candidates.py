import json, sys, glob
fs = sorted(glob.glob("/shared_inference/mdeopuja/Hyperloom/GLM-5.1-FP8/20260812T233301Z/kernel-agent/runs/*/*/kernel_candidates.json"))
f = fs[-1]
print("file:", f)
d = json.load(open(f))
cands = d if isinstance(d, list) else (d.get("hot_kernels") or d.get("candidates") or d.get("kernel_candidates") or [])
print("total candidates:", len(cands))
def gp(c): return c.get("gpu_pct") or 0
for c in sorted(cands, key=lambda x: -gp(x))[:14]:
    nm = str(c.get("name") or c.get("operation") or "")[:46]
    print("  %-46s gpu%%=%-7s reusable=%-5s kind=%-10s src_res=%s" % (
        nm, c.get("gpu_pct"), c.get("reusable_native_kernel"),
        c.get("kernel_kind"), c.get("source_resolution_method")))
# indexer specifically
for c in cands:
    nm = str(c.get("name") or c.get("operation") or "")
    if "indexer" in nm.lower():
        print("INDEXER:", nm, "| reusable=", c.get("reusable_native_kernel"),
              "| skip=", repr(c.get("skip_reason")), "| src=", str(c.get("source_file"))[:70],
              "| gpu%=", c.get("gpu_pct"))
