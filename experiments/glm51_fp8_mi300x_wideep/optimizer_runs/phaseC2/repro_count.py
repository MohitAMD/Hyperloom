import sys, time, gzip, json, resource, traceback
from pathlib import Path

f = Path(sys.argv[1])

def open_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)

RUNTIME_API_NAMES = {"hipeventsynchronize","hipdevicesynchronize","hipstreamsynchronize",
"hipgraphlaunch","hiplaunchkernel","hipmodulelaunchkernel","hipmemcpy","hipmemset",
"cudaeventsynchronize","cudadevicesynchronize","cudastreamsynchronize"}

def is_kernel_event(ev):
    cat = str(ev.get("cat") or ev.get("category") or "").lower()
    if cat != "kernel":
        return False
    name = str(ev.get("name") or ev.get("kernel_name") or "")
    if name.lower() in RUNTIME_API_NAMES:
        return False
    return True

t0 = time.time()
try:
    payload = open_json(f)
    print("open_json OK in %.1fs" % (time.time()-t0), flush=True)
    events = payload.get("traceEvents") if isinstance(payload, dict) else None
    print("traceEvents is list:", isinstance(events, list), "len:", (len(events) if isinstance(events, list) else None), flush=True)
    cnt = 0
    for ev in events or []:
        if isinstance(ev, dict) and is_kernel_event(ev):
            cnt += 1
    print("GPU kernel events:", cnt, flush=True)
except Exception as e:
    print("EXCEPTION after %.1fs:" % (time.time()-t0), repr(e), flush=True)
    traceback.print_exc()
print("peak RSS MB:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, flush=True)
print("total %.1fs" % (time.time()-t0), flush=True)
