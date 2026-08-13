# GEAK per-candidate benchmark cost shrink (Task "a")

## 1. Files changed

- `hyperloom/agents/kernel/tools/apply_and_bench.py` — turned the E2E driver into a
  **winner-gated, batched, cheap** confirm: added a microbench→E2E speedup gate,
  a cheap confirm shape, and dropped default reps 5→3. Backward-compatible
  (new params are keyword-only, default-off).
- `hyperloom/agents/kernel/tools/kernel_optimization.py` — ranking/KEEP already
  decides on the isolated microbench (`build_verification` scans micro_speedup;
  `make_proposal` KEEPs on it with E2E deferred). Added an **`e2e_confirm` gate
  annotation** on KEEP proposals so the orchestrator batches only qualifying
  winners into one serve instead of an E2E per candidate. Additive only — no
  decision/reasons changes.

Backups: `*.bak.1786564703` next to each edited file.

Not edited (already sufficient / out of ownership):
- `tracelens_analysis.py` already implements `HYPERLOOM_KERNEL_CANDIDATES_TOP_K`
  (default 100, non-positive = unbounded) — the candidate long-tail cap. No change needed.
- `_server_lifecycle.py` is already reuse-aware (round-1 `cleanup=false` persists,
  round-2 re-attaches). No change needed; the confirm should reuse the warm server.
- `coordinator_helpers.py` / `dispatcher.py` — the per-candidate E2E decision does
  NOT live there (no `apply_and_bench` call site in orchestrator code; it is driven
  by the kernel specialist), so no edits were required there.

### Key diffs (apply_and_bench.py)

```python
# reps default 5 -> 3 (warmup is already untimed)
reps: int = 3,
# new keyword-only knobs (default-off)
micro_speedups: list[float] | None = None,
confirm: bool = False,

# winner gate: drop candidates below the microbench->E2E gate BEFORE the serve
gate = _e2e_gate_min_speedup()            # HYPERLOOM_KERNEL_E2E_GATE_MIN_SPEEDUP, default 1.03
if micro_speedups is not None:
    pairs, gate_decisions = _gate_pairs(pairs, micro_speedups, gate)
    if not pairs:
        return {"status": "no_candidate_cleared_gate", ...}
# cheap confirm shape (decode-kernel TPOT is shape-portable)
if confirm:
    osl, conc, num_prompts = _resolve_confirm_shape(osl, conc, num_prompts)
```

`_resolve_confirm_shape` precedence: osl = `PROFILE_OSL` (`--profile-osl`) >
`HYPERLOOM_KERNEL_E2E_CONFIRM_OSL` > 128; conc = `HYPERLOOM_KERNEL_E2E_CONFIRM_CONC` > 8;
num_prompts capped to `HYPERLOOM_PROFILE_MAX_ITERS` (never below conc).

CLI additions: `--confirm`, `--profile-osl`, `--micro-speedups` (comma-separated,
aligned to `--pair` order). Survivors are all applied to the SAME patched server →
one batched confirm.

### Key diffs (kernel_optimization.py)

```python
_E2E_CONFIRM_GATE_MIN_SPEEDUP_DEFAULT = 1.03
def _e2e_confirm_gate_min_speedup() -> float: ...  # HYPERLOOM_KERNEL_E2E_GATE_MIN_SPEEDUP

# KEEP proposals now carry:
"e2e_confirm": micro_speedup >= gate,
"e2e_confirm_gate": gate,
```

## 2. New env vars

| Env | Default | Effect |
|-----|---------|--------|
| `HYPERLOOM_KERNEL_E2E_GATE_MIN_SPEEDUP` | 1.03 | Min isolated-microbench speedup for a candidate to earn an E2E confirm. Read by both apply_and_bench (drops sub-gate pairs) and kernel_optimization (annotates KEEP `e2e_confirm`). |
| `HYPERLOOM_KERNEL_E2E_CONFIRM_OSL` | 128 | Cheap-confirm output length (overridden by `PROFILE_OSL`/`--profile-osl` if set). |
| `HYPERLOOM_KERNEL_E2E_CONFIRM_CONC` | 8 | Cheap-confirm concurrency. |
| `HYPERLOOM_PROFILE_MAX_ITERS` | (unset) | Existing knob; when set, caps confirm `num_prompts` (fewer requests → shorter serve). |
| `HYPERLOOM_KERNEL_CANDIDATES_TOP_K` | 100 | Pre-existing (tracelens_analysis.py); caps the candidate long-tail. |

## 3. New cost model (one line)

was: full serve × N candidates; now: microbench × N + 1 batched cheap confirm.

## 4. Test results

`pytest hyperloom/agents/kernel/tests/ -k "apply_and_bench or kernel_optimization or verification"`
→ **92 passed, 805 deselected**. No regressions; no new tests were required (existing
suite already exercises make_proposal KEEP/REVERT/NEEDS_REVIEW and apply_and_bench helpers).
Import smoke passed for both edited files; helper sanity check confirmed gate default 1.03,
env override, pair gating (fail-open on missing signal), and confirm-shape resolution.

## 5. Follow-up to exercise on GPU

- The orchestration layer (kernel specialist / prompt) that today calls a full
  `apply_and_bench` per candidate should be switched to: (a) rank/keep on
  `micro_speedup`, then (b) collect all KEEP winners with `proposal.e2e_confirm == True`
  and invoke `apply_and_bench` ONCE with repeated `--pair` + `--micro-speedups --confirm`
  (or the `pairs=..., micro_speedups=..., confirm=True` kwargs). This is a
  prompt/wiring change owned outside the four files above.
- Confirm the batched cheap serve reuses the persistent warm server via
  `_server_lifecycle` (already reuse-aware) rather than a cold bring-up.
- Suggested first GPU smoke: 2–3 GEAK candidates at 32k/8k, verify only >1.03x
  microbench winners hit the single confirm serve and it completes in minutes.
