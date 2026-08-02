# 163: v155 Recovery, Head-Profile Evidence, and Next Decision

Date: 2026-08-02

## 1. Evaluation recovery

The frozen v155 manifest has 112 jobs (16 dimensions x 7 methods), not 63.
Its current status is 53/112. The valid MovieBench core-9 view is 53/63:
seven dimensions are complete, while subject and background consistency are
both 2/7.

The recovery path now:

- preserves the 53 completed jobs under their original frozen contracts;
- reports status without mutating contracts or results;
- resumes only genuinely missing jobs;
- loads DINO, DINOv2, CLIP, and DreamSim from shared offline caches;
- collects core-9 without treating annotation-dependent semantic dimensions
  as valid for arbitrary MovieBench prompts.

The offline smoke test initializes DINO, CLIP, and DreamSim without network
access. Recovery commands are in `docs/162_v155_run_results.md`.

## 2. What head profiling actually classified

| Evidence | Usable result | Claim boundary |
|---|---|---|
| v143 | zero admissible static head axes | rejects a single fixed taxonomy |
| v144 | camera 48, action 42, identity 32, scene 31, unresolved 207 | 57.5% unresolved; split agreement 0.4556; descriptors are not functional classes |
| v145 | 16 reproducible factor-axis and 51 state-specific candidates | continuous observational rankings only; static taxonomy is inadmissible |
| v147-v148 | ranked interventions have downstream effects; K is the strongest PF-independent axis | intervention specificity fails, so K susceptibility is a prior rather than a named role |
| v149-v151 | scalar policy tail and signed policy classes fail randomized/calibrated confirmation | do not route generation from these maps |
| v152 one-sided reanalysis | per-layer QK top-4 is a stable `History-Critical` candidate (112/120 overlap across seeds) | only the high tail is supported; the remaining 240 heads are `Default`, not `Recent-Critical` |

The only classification currently suitable for a generation intervention is
the frozen v152 top-4-per-layer map. Even that is a hypothesis under test, not
a proven functional taxonomy. PF Wave/Anchor/Veil and the weak semantic-factor
labels may be reported as descriptive cross-tabs, but not as causal role names.

## 3. How to apply the usable result

Use the v152 score only as a continuous, layer-relative candidate propensity
for dispersed history. v152's own oracle policy-choice gate failed, so this is
a final falsification experiment rather than a validated deployment rule. The
v155 allocation under test was:

```text
high QK-margin heads: sink1 + dispersed-history4 + recent4
all other heads:      sink1 + recent8
```

Every test must retain bottom-4 and count-matched random-4 maps. `Default` must
not be interpreted as a coherent recent-preferring class. The v145/v148 K-axis
can be a secondary prior or gate feature, but not another hard semantic label.
No broad profiling sweep is justified until trajectory-level membership works.

## 4. Preliminary v155 evidence

Seven dimensions are complete across all methods:

| Method | Flicker | Smooth | Overall | Dynamic | Aesthetic | Imaging |
|---|---:|---:|---:|---:|---:|---:|
| SF | .96804 | .98218 | .23314 | .64167 | .61629 | .68914 |
| QK top reservoir | .96346 | .98166 | .23824 | .72500 | .61661 | .70672 |
| QK bottom reservoir | .96127 | .97989 | .23741 | .76250 | .61965 | .71494 |
| QK random reservoir | .96423 | .98308 | .23794 | .70833 | .62167 | .71093 |
| all-head reservoir | .95468 | .97708 | .24142 | .83333 | .61824 | .69684 |

Top minus bottom has `+0.00083` overall consistency but `-0.03750` dynamic
degree. The frozen non-inferiority bound is `-0.03`, so the metric promotion
gate cannot pass even if missing subject/background scores favor top. Top
minus random is only `+0.00030` overall, while random is smoother and has
higher aesthetic and imaging quality. This is not robust membership evidence.

## 5. Next experiment decision

Do not launch a 128-prompt scale-up. First complete the ten offline core jobs
and the prepared 112-video blind review. They remain useful for diagnosing
identity/background retention and deciding whether the reservoir itself is
useful, even though the frozen metric promotion gate has already failed.

The v156 diagnostic implementation is now prepared; it must not be scaled
unless both objective and blind-review gates pass. After core-9 and human
review:

- If top does not beat both bottom and random on identity/background, stop the
  static QK membership axis. Keep any all-head reservoir benefit as a cache
  result, not a head-classification result.
- If human and subject/background results show a clear top-specific advantage,
  run the prepared v156 16-prompt exact-context alignment screen. At the
  frozen v152 frame-117 context it uses old anchors `[0,37,75,112]` plus recent
  `[113,114,115,116]`, exactly matching `uniform8`. It keeps top, bottom,
  random, all-head, all-recent, reservoir, and SF controls.
- v156 removes the extra sink and pending-frame duplicate K/V copies. Both
  profile and recent routes read and physically retain at most 8 FFE.
- The fixed anchors are exact only at the frame-117 profiling context and are
  underfilled early. They are not a rolling-uniform8 implementation; no claim
  beyond this frozen-context transfer test is allowed.

If the deterministic experiment also fails membership controls, terminate this
line. The next mechanism should be cache-policy or layer/timestep gating, using
continuous K/QK propensity as an input feature instead of another static map.

Implementation, run commands, proxy findings, GPU-operating constraints, and
the frozen v156 decision gates are in `docs/164_v156_profile_exact_experiment.md`.
