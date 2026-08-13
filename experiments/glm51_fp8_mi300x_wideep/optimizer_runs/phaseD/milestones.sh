#!/bin/bash
LOG=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/phaseD/optimize_full.log
OUT=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/phaseD/milestones.log
touch "$OUT"
while true; do
  grep -aE "baseline landed|auto-enqueued initial roofline|trace flush settled|trace_gpu_kernel_events|Trace contains zero GPU|trace_analyze_failed|roofline.*(failed|produced|ok)|analysis_md|Auto-roofline|KERNEL phase|kernel-agent|GEAK|e2e cycle|cycle=|kept kernel|KEEP|microbench|Coordinator|EXIT=" "$LOG" 2>/dev/null \
    | grep -avF -f "$OUT" 2>/dev/null | while IFS= read -r line; do echo "$(date -u +%H:%MZ) | $line" >> "$OUT"; done
  # also snapshot session-level roofline/kernel-agent status files
  sleep 60
done
