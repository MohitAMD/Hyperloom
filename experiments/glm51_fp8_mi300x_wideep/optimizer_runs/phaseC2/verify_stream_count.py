import sys, time, resource
sys.path.insert(0, "/shared_inference/mdeopuja/Hyperloom/hyperloom/agents/kernel/tools")
from pathlib import Path
import tracelens_analysis as t

f = Path(sys.argv[1])
t0 = time.time()
n = t.count_gpu_kernel_events(f)
dt = time.time() - t0
print(f"count_gpu_kernel_events -> {n} in {dt:.1f}s")
print("peak RSS MB:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
