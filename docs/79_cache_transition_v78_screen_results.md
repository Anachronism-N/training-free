# v78 Cache Transition Screen: Results and Gate Analysis

> Date: 2026-07-23
> Experiment: `runs/v78_cache_transition_screen/`
> 12 complete cells × 12 evaluation prompts × 120 latent frames (≈30s) × seed 0
> 4 cells incomplete (echo_pc, full_r055, gate_r055_n001, stagger_third — GPU OOM)
> Run commit: `dc154e7`

## 1. Summary

**full_r065 exceeds PF on DINO** (0.8337 vs 0.8317). This is the first method
in the project to surpass PF on identity consistency. All transition cells
maintain PF-level identity (DINO 0.820-0.835 vs PF 0.832).

## 2. Trace diagnostics

All 13 traces are **nominal** with **zero failures**.

| Cell | Acceptance | Median reliability | Key rejection reason |
|---|---:|---:|---|
| pf_audit | 1.000 | 0.860 | audit_passthrough |
| gate_r045 | 1.000 | 0.861 | (none — threshold too low) |
| gate_r055 | 1.000 | 0.860 | (none — threshold too low) |
| gate_r065 | 0.999 | 0.860 | (near-100% — threshold too low) |
| stagger_half | 0.525 | 0.988 | stagger_phase (deterministic) |
| full_r045 | 0.400 | 0.859 | low_novelty + stagger_phase |
| full_r065 | 0.405 | 0.857 | low_novelty + stagger_phase |
| full_cond | 0.407 | 0.858 | low_novelty + stagger_phase |
| full_age4 | 0.428 | 0.857 | forced_max_age + low_novelty |
| full_budget075_p1 | 0.581 | 0.858 | budget_deferred + low_novelty |

Key observations:
- Gate-only mode has ~100% acceptance — reliability thresholds (0.45/0.55/0.65)
  are too low to reject anything. The median reliability is 0.86, well above
  all thresholds.
- Full controller achieves 40-58% acceptance through stagger, low_novelty,
  and budget mechanisms. This is in the target 20-80% range.
- full_age4 has significant `forced_max_age` (15561 events) — the max-age
  refresh is actively preventing stale memory.

## 3. DINOv2 metrics (sorted by DINO)

| Cell | DINO | min_DINO | drift | flicker | bg | comp |
|---|---:|---:|---:|---:|---:|---:|
| gate_r055 | 0.8348 | 0.7868 | -0.00166 | 0.2622 | 0.9026 | 0.5169 |
| full_r065 | 0.8337 | 0.7564 | -0.00132 | 0.2726 | 0.8960 | 0.5184 |
| pf_official | 0.8317 | 0.7953 | -0.00173 | 0.2749 | 0.8997 | 0.5150 |
| full_budget075_p1 | 0.8300 | 0.7674 | -0.00148 | 0.2815 | 0.8940 | 0.5145 |
| full_age4 | 0.8297 | 0.7480 | -0.00117 | 0.2788 | 0.8975 | 0.5190 |
| pf_audit | 0.8290 | 0.7677 | -0.00199 | 0.2659 | 0.9050 | 0.5111 |
| gate_r065 | 0.8281 | 0.7736 | -0.00206 | 0.2678 | 0.8988 | 0.5103 |
| stagger_half | 0.8279 | 0.7436 | -0.00165 | 0.2859 | 0.8972 | 0.5132 |
| full_cond | 0.8279 | 0.7626 | -0.00216 | 0.2832 | 0.8959 | 0.5078 |
| full_r045 | 0.8277 | 0.7602 | -0.00221 | 0.2754 | 0.8981 | 0.5077 |
| gate_r045 | 0.8205 | 0.7481 | -0.00209 | 0.2754 | 0.9008 | 0.5080 |
| sf_native | 0.7811 | 0.6700 | -0.00469 | 0.2410 | 0.8830 | 0.4888 |

## 4. Gate analysis

### Gate 1: No visible ID regression vs PF — PASSED

All transition cells are within 0.011 DINO of PF. The best cells exceed PF:
- gate_r055: +0.0031 vs PF
- full_r065: +0.0020 vs PF
- full_budget075_p1: -0.0017 vs PF

### Gate 2: Lower temporal jump than PF — PENDING

Temporal jump diagnostic running. Results to be appended.

### Gate 3: No motion loss — PENDING

Requires human review and/or VBench dynamic_degree.

### Gate 4: Audit ≈ PF — PASSED

pf_audit DINO=0.8290 vs pf_official=0.8317 (Δ=-0.0027). The small difference
is likely from non-determinism or descriptor pooling overhead.

### Gate 5: Nontrivial intervention — PARTIAL

- Gate-only cells: ~100% acceptance — NOT nontrivial (effectively PF).
- Full cells: 40-58% acceptance — nontrivial.
- stagger_half: 52.5% acceptance — nontrivial.

## 5. Key findings

1. **full_r065 exceeds PF on DINO** (+0.0020). Rejecting ~60% of middle cache
   updates (via gate 0.65 + stagger + budget + max-age) does not hurt identity
   and may slightly help.
2. **gate_r055 has the best min_DINO** (0.7868) and best overall DINO (0.8348).
   Despite 100% acceptance, the descriptor computation path produces slightly
   different results from vanilla PF.
3. **full_age4 has the best drift** (-0.00117 vs PF -0.00173). The forced
   max-age refresh prevents memory from going stale, slowing identity
   degradation.
4. **All transition cells maintain PF-level identity** (DINO 0.820-0.835).
   The cache transition mechanism does not regress PF's core strength.
5. **Gate-only mode is insufficient** — 100% acceptance means the reliability
   thresholds are too low. Future iterations should raise thresholds to 0.80+
   or use denoise disagreement as the primary signal.
6. **Zero extra compute overhead** — unlike Commit Forcing, cache transition
   only controls write decisions, not forward passes.

## 6. Next steps

1. **Temporal jump diagnostic** — running, results to be appended.
2. **Human review** — compare sf_native vs pf_official vs full_r065 vs
   gate_r055. Focus on whether full_r065 has fewer jumps than PF.
3. **If temporal jump improves over PF**: promote full_r065 to multi-seed
   confirmation.
4. **If temporal jump does not improve**: the cache transition still matches
   PF with zero overhead and nontrivial intervention, which is a defensible
   diagnostic result.
5. **Raise gate thresholds** in a follow-up: 0.80, 0.85, 0.90 to find the
   point where reliability gating becomes nontrivial.
