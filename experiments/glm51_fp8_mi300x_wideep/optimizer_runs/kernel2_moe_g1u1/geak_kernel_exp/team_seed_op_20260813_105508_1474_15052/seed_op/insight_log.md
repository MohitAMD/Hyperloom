# Insight Log — moe_fp8_blockscale_g1u1 (MoE grouped GEMM, Triton, gfx94x/95x)

Baseline geomean 15.0466 ms (case0 10.3734 / case1 14.3788 / case2 17.6388 / case3 19.4824 ms;
T=16/32/64/128, topk=8, E=256, N=2048, K=6144).

## Round 1 — cumulative verified geomean 3.3826x (integrated r1_d0 + r1_d1)

Per-case after integration: 2.5792 / 4.1482 / 5.6857 / 6.4364 ms
→ 4.02x / 3.47x / 3.10x / 3.03x. Arithmetic 3.4044x, weighted (total-time) 3.2825x.
Correctness: all 4 cases PASS, err_ratio 0.0000.
Integrated patch: `round_1/integrate/integrated_patch.diff`.

## Insight blackboard

1. **The dominant baseline cost was host-side, not GPU.** `moe_align_block_size` ran a 256-iteration
   Python/`.item()` loop over experts = 256 device→host syncs per call, costing a flat 6.9–12.7 ms
   that scaled with E, not with tokens. That is why the SMALLEST case gained most (3.43x on case0
   from the align rewrite alone vs 2.65x on case3).
2. **Vectorize then graph-capture, in that order.** Isolated align-only: 7.62 → 0.205 → 0.076 ms
   (case0); 12.67 → 0.199 → 0.129 ms (case3). ~100x. The vectorized rewrite gets you to ~0.2 ms,
   and the LAST ~0.1 ms is pure kernel-dispatch count — only CUDA-graph capture/replay removes it,
   no further op fusion helps once every op is <10 us.
3. **`torch.bincount` hides a device→host sync** (its internal `max()`). `scatter_add_` into a
   `zeros(E)` is the sync-free equivalent. Generalizable to any counting prologue.
4. **Oversizing outputs to a static `EM_max` is free.** Measured EM_max vs tight-EM grids: 2.960 vs
   2.965 ms (case0), 7.220 vs 7.232 ms (case3). The kernel's
   `pid_m*BLOCK_M >= num_tokens_post_padded` guard reads the device tensor and exits immediately, so
   the extra blocks cost nothing — and static shapes are what make graph capture possible.
5. **Graph-cache hygiene:** key the cache on shape/dtype/block_m/num_experts ONLY (never tensor
   identity/content), copy real inputs into the static input buffer each call. CAVEAT: graph outputs
   live in the graph's private pool, so two same-shape align calls alias the same buffers — safe only
   because the launcher consumes them immediately in a stream-ordered launch. Any future code holding
   an align result across another same-shape call MUST clone it.
6. **GEMM schedule: `num_stages=1` is the win, and the config space is now exhausted.** Isolated
   GEMM, BLOCK_M=16 / BLOCK_N=64 / BLOCK_K=128 / GROUP_M=1 / num_warps=4: ns=2 (baseline)
   2.98/4.59/6.45/7.33 → ns=1 2.52/3.88/5.60/6.13 ms = 1.15–1.20x. No neighbor of that config is
   within 10%. ns=0 measured essentially tied with ns=1 (4.4535 vs 4.4483 ms combined).
7. **CORRECTNESS TRAP: `BLOCK_K` must equal `group_k` (=128).** The k-loop uses `ks = kk` as the
   block-scale index, so BK=64 or BK=256 silently produce WRONG results (and are slower). `block_k`
   is now pinned to `group_k` in the launcher.
8. **The two lanes compose super-multiplicatively.** 2.9152 × 1.0439 = 3.043 expected, 3.383
   measured — removing the host sync/dispatch bubble exposes the GEMM as a larger share of wall
   time, so the kernel win lands more fully. Corollary: re-sweep compute knobs AFTER a host-side win;
   the optimum can shift (here it did not, ns=1 stayed best).
9. **Measurement discipline.** e2e run-to-run variance is ~1.5% at geomean level (one 15.13 vs 14.37
   outlier observed) — treat sub-2% e2e deltas as noise. The isolated-GEMM harness is far more stable
   and is the right screen for tile/schedule work. Integration repro spread was 0.12%.
10. **Bottleneck has fully shifted to the GEMM.** Post-patch, align is only 1.7–2.5% of wall time;
    ~97% is `_moe_g1u1_fp8_kernel`. All remaining headroom is in the GEMM lane.

## Confirmed dead-ends (do NOT re-explore)

- **"Shrink BLOCK_M to cut padded-FLOP waste" — DISPROVED.** BM=8 is 2.7x SLOWER (case0 6.77 vs
  2.52 ms), BM=4 3.3x slower, BM=32 4.6x slower, BM=64 2.3x slower. The 16x16x16 MFMA needs M>=16;
  below that Triton emits a degenerate layout costing far more than the padding saved. BLOCK_M=16 is
  a hard optimum. The padding waste is REAL but is NOT reachable through tile size — it must be
  attacked algorithmically (masked/contiguous grouped GEMM skipping empty experts, DeepGEMM-style, or
  a persistent kernel).
- BLOCK_N=32 (7.94 ms c0) and BLOCK_N=128 (2.85 ms c0) both worse than 64.
- BLOCK_K != 128: wrong results AND slower.
- num_warps=2 (2.81 c0) and 8 (17.89 c0) worse than 4.
- `waves_per_eu` 1/2/3/4/5/6/8: all worse than the default (best 2.61 vs 2.52).
- GROUP_M 1/2/4/8/16: flat (2.522–2.545), no lever.
- matrix_instr_nonkdim=32 (11.65 c0) far worse; nonkdim=16 and kpack=2 tie the default.

## Hypothesis ledger

| dir | specialty | expected | verified | verdict | lesson |
|---|---|---|---|---|---|
| r1_d0 | host_runtime | 2.8 | 2.9152 | confirmed | Vectorized sync-free align + CUDA-graph capture: ~100x on the align term; sync-free counting + static-shape oversizing are the enablers. |
| r1_d1 | compute | 1.25 | 1.0439 | partial | Only `num_stages=1` paid (1.15–1.20x isolated GEMM); the BLOCK_M premise was wrong and the tile space is exhausted. |
| integrate | — | — | 3.3826 | confirmed | Fully orthogonal, clean incremental stack, gains better than multiplicative. |

## Next-round steer

The GEMM is ~97% of wall time and its config space is closed. Round 2 must go ALGORITHMIC on the
grouped GEMM: (a) skip/compact empty-expert blocks so padded FLOPs aren't computed at all — at T=16,
topk=8, E=256 there are 128 assignments across 256 experts, so most BLOCK_M=16 tiles are near-empty
padding; (b) a persistent/stream-K style kernel to fix the tiny-grid occupancy at small T; (c) fuse
the g1u1 (gate/up) halves or improve fp8 block-scale operand staging (LDS/`tl.dot` scale handling).
A `deep_explore` round is justified if two specialist attempts on (a)/(b) plateau.

---

# Round 2 — cumulative verified geomean 9.1082x (winner: engineer r2_d0, standalone; no integrate)

Per-case: 0.9994 / 1.5672 / 2.0239 / 2.3325 ms → 10.38x / 9.17x / 8.72x / 8.35x vs baseline.
Incremental over round 1: **2.696x**. Correctness: all 4 PASS, err_ratio 0.0000, cos_diff 2.1e-10…1.3e-9.
Patches on disk: `round_2/engineer_0/best_patch.diff` (full, applies to pristine CANONICAL) and
`round_2/engineer_0/fp8_native_mfma_only.diff` (minimal delta over the R1 integrated patch).

## Insight blackboard (round-2 additions)

11. **THE WIN IS ONE BITCAST: OCP e4m3 → fnuz (`tl.float8e4b8`).** `torch.float8_e4m3fn` (OCP,
    exponent bias 7) and Triton's `tl.float8e4b8` (fnuz, bias 8) share the IDENTICAL 1/4/3 bit
    layout, so reinterpreting the bytes is an EXACT halving of every value — correct with a single
    `*4.0` fold into the scale (an exact power of two, hence bit-exactness: cos_diff 4.5e-10…1.3e-9,
    err_ratio 0.0000 on all 4 cases). This is a generalizable AMD-Triton fact, not a kernel-specific
    trick.
12. **Why it matters so much: Triton-on-AMD only feeds the native 8-bit MFMA from the fnuz types**
    (`supported_fp8_dtypes = fp8e4nv/fp8e5/fp8e5b16/fp8e4b8`). ANY gfx942 Triton kernel holding OCP
    fp8 tensors is *silently* running a ~161-VALU-per-MFMA software dequant to fp16. ISA diff
    (AMDGCN dump, case0): `v_mfma_f32_16x16x16_f16` ×32 → `v_mfma_f32_16x16x32_fp8_fp8` ×8 (native,
    2× the K per instruction); `v_cndmask_b32` 1750→0; `v_cmp_ne_u16` 1728→0; `v_lshlrev_b16` 192→0;
    `v_add_u16` 192→0; `v_perm_b32` 98→2; VGPR 176 (40 arch + 136 accum) → 104. Always dump the ISA
    and count MFMA opcodes before assuming a Triton fp8 kernel is using the fp8 MFMA.
13. **The one correctness guard needed: OCP `0x80` is −0.0 but fnuz `0x80` is NaN.** So
    `tl.where(b == 0x80, 0, b)` on the loaded bytes. `0x7F`/`0xFF` are OCP NaN but never occur for
    finite clamped inputs (byte census measured 0 occurrences). Verify with a byte census; never add
    a host-side elementwise fixup (that would cost more than the kernel).
14. **BOTTLENECK SHIFTED AGAIN: latency(C2 issue-wait) → MEMORY (expert-weight streaming). The
    compute lane is now CLOSED.** Post-patch: HBM 2.75 TB/s = **64 % of the MEASURED 4.32 TB/s
    achievable ceiling on this box** (do NOT use the 5.3 TB/s nameplate — I benchmarked it), L2 hit
    2.2 %, MFMA busy 5.8 %, VALU 21.3 %, `SQ_WAIT_INST_ANY/SQ_WAIT_ANY` 1.19 → 0.62. The kernel got
    2.7× faster while moving the SAME bytes, so effective BW rose 1.02 → 2.75 TB/s and memory became
    the limiter.
15. **Traffic is confirmed by two independent methods (0.8 % apart):** algorithmic 2.59 GB (103
    touched experts × 2N × K) vs `TCC_MISS_sum` × 128 B = 2.61 GB for case0. Per-case active-expert
    footprint 2.59 / 4.25 / 5.51 / 6.29 GB (103/169/219/250 of 256 experts touched). **Remaining
    headroom is exactly ~1.58×** (600 us roofline vs 949 us measured, case0) and it is ALL
    bandwidth efficiency, not work elimination.
16. **Occupancy sits exactly at the register ceiling:** VGPR 104 → waves/SIMD = min(8, 512/104) = 4
    (50 %). LDS = 0, scratch = 0, 4096 WGs vs 304 CUs (13× fill) — not a fill problem. Getting under
    86 VGPRs buys 5–6 waves, under 64 buys 8; only worth it if it converts into more loads in flight
    (measure BW, not occupancy).
17. **ORTHOGONALITY CONTRACT HELD AGAIN.** The r2_d0 patch touches ONLY the k-loop body, the
    `a_scale` load, and 4 launcher lines; `moe_align_block_size`, graph-cache keying and the
    static-buffer copy-in are byte-identical to round 1. `best_patch.diff` was verified to apply
    cleanly to the pristine CANONICAL and md5-reproduce the tested file. Keep enforcing this.
18. **PROCESS RISK — workspace drift.** `WORKSPACE/moe_fp8_blockscale_g1u1.py` is STILL byte-identical
    to `baseline/` after TWO rounds (md5 `17de246d…`). Round-3 engineers branching from CANONICAL
    would silently lose 9.1×. Promote `round_2/engineer_0/best_patch.diff` into WORKSPACE before
    round 3, or restate in every round-3 prompt that they must apply it first.

## Confirmed dead-ends (round-2 additions)

- **Tile/schedule space is CLOSED at this source — now swept twice, independently, and re-swept
  POST-MFMA-change.** 12 more variants after the fp8 bitcast (BLOCK_N 32/64/128/256, num_warps 2/4/8,
  num_stages 0/1/2/3, GROUP_M, kpack, matrix_instr_nonkdim, waves_per_eu, pid orders): all tie or
  lose, last 3 within 1 %. Do NOT issue another tile direction.
- **Tile compaction / skipping near-empty BLOCK_M=16 blocks — REJECTED on arithmetic, contra the
  r2_d0 report's own steer.** Each expert's weight columns are read exactly once across the 32
  disjoint N-blocks, and there are only ~1.0 (case0) to ~1.2 (case3) M-blocks per touched expert, so
  there is **no duplicated weight traffic to eliminate**; the 2.59–6.29 GB is the algorithmic
  minimum. With MFMA busy at 5.8 %, saved FLOPs are free FLOPs. Expected payoff ≈ 0. (This was true
  as a *FLOP* argument in round 1 too, and is now doubly dead as a *bandwidth* argument.)

## Hypothesis ledger (round 2)

| dir | specialty | expected | verified | verdict | lesson |
|---|---|---|---|---|---|
| r2_d0 | compute | 2.0 | 9.1082 (cum; 2.696 incremental) | confirmed | Bitcast OCP e4m3 → `tl.float8e4b8` unlocks `v_mfma_f32_16x16x32_fp8_fp8` and deletes a 161-VALU/MFMA software dequant; bit-exact, 4.5× over-delivery vs the 2.0 target. |

## Next-round steer (round 3) — bandwidth efficiency only

Ranked, all `memory`-lane:
1. **Kill the per-byte `tl.where(b==0x80,0,b)` on the two WEIGHT operands** (still 59 VALU/MFMA, in
   the load→MFMA path, stealing issue slots from loads-in-flight). Either sanitize `w1_fp8` once into
   a buffer cached on `(data_ptr, shape, dtype)` (6.4 GB of 192 GB; a shape-derived artifact,
   COMMANDMENT rule 5), or express the guard bitwise on packed 32-bit words. Keep the `a` guard — it
   is 1/8 the bytes and nearly free.
2. **Bypass L2 on the weight loads** — 2.2 % hit rate means 6.3 GB of single-use data is churning
   4 MB of L2 for nothing. Try `cache_modifier="cg"` / non-temporal (`sc0 sc1`) on the two `w_*`
   `tl.load`s; cheap, directly targets the 2.75 → 4.32 TB/s gap.
3. **Check weight-load width in the ISA** (`global_load_dwordx4` vs narrower). The uint8 `tl.where`
   may be forcing a narrow layout; opportunity 1 may fix this for free.
4. **VGPR reduction 104 → <86** to lift waves/SIMD 4 → 6 for HBM latency hiding (share the dual
   gate/up pointer arithmetic). Judge it by measured BW, not by occupancy.
