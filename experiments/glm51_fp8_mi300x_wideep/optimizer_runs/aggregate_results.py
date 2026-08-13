#!/usr/bin/env python3
"""Aggregate Phase B optimizer variant results.

Scans the session's runs/ tree for inferencex_result.json files (baseline,
explore, sweep, framework) and the fresh CONCURRENCY logs, and prints a table
of measured total/output tok/s + TPOT/TTFT per variant, tagged with the
EXTRA_VLLM_ARGS / knob deltas from each run's materialized config.
"""
import json, sys, glob, os, re
from pathlib import Path

SESS = sys.argv[1] if len(sys.argv) > 1 else None
if not SESS:
    # newest session dir
    cands = sorted(glob.glob("/shared_inference/mdeopuja/Hyperloom/GLM-5.1-FP8/2026*"), reverse=True)
    SESS = cands[0] if cands else None
sess = Path(SESS)
rows = []
for res in sess.rglob("inferencex_result.json"):
    # skip warmup_round measurements (only count real measured rounds)
    if "warmup_round" in res.parts:
        continue
    try:
        d = json.loads(res.read_text())
    except Exception:
        continue
    if not d.get("success"):
        continue
    phase = res.parts[len(sess.parts)+1]
    vname = ""
    for part in res.parts:
        if re.match(r"^v\d+_", part):
            vname = part
            break
    # find sibling config.yaml for knob context
    cfg = res.parent / "config.yaml"
    extra = ""
    if cfg.exists():
        t = cfg.read_text()
        m = re.search(r"EXTRA_VLLM_ARGS:\s*(.+)", t)
        if m: extra = m.group(1).strip().strip('"\'')
    rows.append({
        "phase": phase,
        "isl": d.get("isl"), "osl": d.get("osl"), "conc": d.get("conc"),
        "total_tok_s": d.get("total_token_throughput"),
        "out_tok_s": d.get("output_throughput"),
        "med_tpot": d.get("median_tpot_ms"), "mean_tpot": d.get("mean_tpot_ms"),
        "med_ttft": d.get("median_ttft_ms"),
        "completed": d.get("completed_requests"),
        "dur": d.get("duration_seconds"),
        "extra": (vname + " | " + extra) if vname else extra,
        "path": str(res.parent.relative_to(sess)),
    })
rows.sort(key=lambda r: (r["total_tok_s"] or 0), reverse=True)
R9 = 9957.68
print(f"session: {sess}")
print(f"{'phase':10} {'isl':>6} {'osl':>5} {'con':>4} {'total_tok/s':>11} {'out_tok/s':>9} {'medTPOT':>7} {'medTTFT':>8} {'cmpl':>5}  extra")
for r in rows:
    dpct = (100.0*((r['total_tok_s'] or 0)-R9)/R9) if r['total_tok_s'] else 0
    print(f"{r['phase'][:10]:10} {str(r['isl']):>6} {str(r['osl']):>5} {str(r['conc']):>4} "
          f"{(r['total_tok_s'] or 0):>11.2f} {(r['out_tok_s'] or 0):>9.2f} {(r['med_tpot'] or 0):>7.2f} "
          f"{(r['med_ttft'] or 0):>8.0f} {str(r['completed']):>5}  {r['extra']}")
if rows:
    best = rows[0]
    print(f"\nBEST total_tok/s = {best['total_tok_s']:.2f}  ({best['phase']} :: {best['extra'] or 'identity/baseline'})")
    print(f"vs R9 9957.68 -> delta {100.0*(best['total_tok_s']-R9)/R9:+.2f}%")
print(f"\n#variants(success) = {len(rows)}")
