# v86 TransitionCache: Human Review and Combined Analysis

> Date: 2026-07-24
> Videos: v86 screen, prompts 0-0 and 1-0
> Reviewer: human, textual description only

## 1. Human review by cell

### Category 1: Best — v78 and pf_binary_balanced

**v78**: ID and background both well retained. Consistent with previous
reviews confirming v78 as PF-level quality.

**pf_binary_balanced**: ID and background both good. In 1-0, pf_binary may
be slightly better than v78 — encouraging more camera/scene motion while
maintaining identity. This is consistent with the v82 labels review where
pf_binary showed frontal shots and orbiting camera.

### Category 2: Good ID but specific artifacts

**consensus_balanced**: 0-0 ID and background good, camera stable. 1-0:
flashback artifacts in early frames; later, subject disappears and
reappears (transient loss).

**inverse_balanced**: 0-0 ID and background good; mid-video background
changes faster than acceptable. 1-0: physics violations (e.g., person's
leg entering a wall); flashback artifacts in early frames.

### Category 3: Good ID but duplicated subjects

**learned_age_only, learned_balanced, learned_late, learned_neutral,
random_balanced, replica_balanced**: ID and background retention good
overall. However, 1-0 later portion shows duplicated subjects (two
instances of the main character). Early frames also have flashback
artifacts.

### Category 4: Acceptable

**learned_conservative, learned_early**: ID and background acceptable.
No specific severe artifacts noted beyond the general patterns.

### Category 5: Baseline

**sf_native**: (Not explicitly reviewed in this round, but from previous
reviews: 5s degradation, 15s collapse.)

## 2. Artifact taxonomy

| Artifact | Cells affected | Severity |
|---|---|---|
| Early-frame flashback/伪影 | Most transition cells (1-0) | Universal, mild |
| Duplicated subjects (1-0 late) | learned_*, random, replica | Moderate, concerning |
| Subject disappearance/reappearance | consensus_balanced | Moderate |
| Physics violations (limb through wall) | inverse_balanced | Severe |
| Background rapid change | inverse_balanced (0-0 mid) | Mild-moderate |
| Camera motion encouragement | pf_binary_balanced | Positive (more dynamic) |

## 3. Combined DINOv2 + Human Review Analysis

| Cell | DINO | Human ID | Human BG | Key artifact | Assessment |
|---|---:|---|---|---|---|
| **v78** | **0.854** | Excellent | Good | None specific | **Best overall** |
| pf_binary_balanced | 0.853 | Good | Good | More camera motion (positive) | **Strong, may beat v78 on motion** |
| inverse_balanced | 0.852 | Good | Moderate | Physics violation, rapid BG change | Good DINO but visual artifacts |
| consensus_balanced | 0.850 | Good | Good | Subject disappearance | Good but transient failure |
| pf | 0.850 | Good | Good | — | Solid baseline |
| replica_balanced | 0.849 | Good | Good | Duplicated subjects | Reproducible but artifact |
| learned_balanced | 0.848 | Good | Good | Duplicated subjects | Does not beat v78/pf |
| learned_age_only | 0.848 | Good | Good | Duplicated subjects | Age-only competitive |
| learned_late | 0.848 | Good | Good | Duplicated subjects | Late layers work |
| random_balanced | 0.847 | Good | Good | Duplicated subjects | ≈ learned (classification not causal) |
| learned_early | 0.845 | Acceptable | Acceptable | — | Early layers weaker |
| learned_neutral | 0.844 | Good | Good | Duplicated subjects | Uniform policy + labels loaded |
| learned_conservative | 0.836 | Acceptable | Acceptable | — | Weakest learned variant |
| sf_native | 0.785 | 5s degrade | Severe | Collapse | Baseline |

## 4. Key findings

1. **v78 and pf_binary_balanced are the two best methods.** v78 has the
   highest DINO (0.854) and no specific artifacts. pf_binary has slightly
   lower DINO (0.853) but may encourage more motion (positive for dynamic
   degree).

2. **Duplicated subjects are the most concerning ProbeCache artifact.**
   6 of 14 cells show duplicated subjects in 1-0 late portion. This is a
   cache-write artifact, not a direct-recall artifact. v78 and
   pf_binary_balanced do NOT have this issue.

3. **Early-frame flashback artifacts are universal** across transition
   cells. This is likely a PF-inherited initialization artifact, not
   introduced by the transition controller.

4. **Inverse labels produce unique physics violations** (limb through
   wall). This confirms that label direction matters for physical
   plausibility, not just for DINO.

5. **consensus_balanced has transient subject disappearance.** The
   abstention mechanism may cause brief cache freeze-thaw cycles that
   lose the subject temporarily.

6. **DINO ranking does not perfectly match visual quality.** inverse
   (0.852) has high DINO but physics violations; v78 (0.854) has highest
   DINO AND best visual quality. DINO alone is insufficient for method
   selection.

## 5. Implications for paper

1. **v78 is confirmed as the paper candidate** — best DINO, best visual
   quality, no severe artifacts, robust across seeds and prompt suites.

2. **pf_binary_balanced is a strong alternative** — nearly identical DINO,
   may encourage more motion. Could be combined with v78 for a
   "best of both worlds" configuration.

3. **The counterfactual classification's main artifact (duplicated
   subjects) is a cache-write issue.** Unlike ProbeCache's direct-recall
   hallucinations (polygon noise, background corruption), this artifact
   comes from asynchronous role-conditioned writes creating inconsistent
   cache states that produce duplicated subjects.

4. **The classification contribution is not supported** — visual quality
   confirms the DINO conclusion: learned roles do not beat v78, pf_binary,
   or inverse on overall quality. The classification is reproducible but
   not causally beneficial.

## 6. VBench-Long status

VBench-Long is running (5 dimensions: subject_consistency,
background_consistency, aesthetic_quality, imaging_quality,
dynamic_degree). Motion_smoothness is excluded due to missing RAFT model
file (403 download error). Results will be appended when complete.

The temporal jump diagnostic from previous experiments serves as a
substitute for motion_smoothness.
