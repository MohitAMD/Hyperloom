import sys, time, gzip, resource
from pathlib import Path

_KERNEL_CAT_SENTINELS = (b'"cat": "kernel"', b'"cat":"kernel"')

def _stream_count_sentinels(trace_file, sentinels, max_events):
    opener = gzip.open if trace_file.suffix == ".gz" else open
    overlap = max(len(s) for s in sentinels) - 1
    def _count_before(data, limit):
        found = 0
        for pat in sentinels:
            start = 0
            while True:
                idx = data.find(pat, start)
                if idx == -1 or idx >= limit:
                    break
                found += 1
                start = idx + len(pat)
        return found
    total = 0
    carry = b""
    with opener(trace_file, "rb") as fh:
        while True:
            chunk = fh.read(8 << 20)
            if not chunk:
                total += _count_before(carry, len(carry))
                break
            data = carry + chunk
            limit = max(0, len(data) - overlap)
            total += _count_before(data, limit)
            if total >= max_events:
                return max_events
            carry = data[limit:]
    return total

f = Path(sys.argv[1])
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
t0 = time.time()
n = _stream_count_sentinels(f, _KERNEL_CAT_SENTINELS, cap)
print(f"streaming count -> {n} in {time.time()-t0:.1f}s  peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB")
