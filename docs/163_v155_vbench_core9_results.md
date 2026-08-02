# 163: v155 VBench-Long Core-9 Completion and Scores

Date: 2026-08-02
Commit: `44a44dd` (local-model fix + resume-missing + reservoir improvements)

## 1. Summary

The v155 VBench-Long evaluation is complete for all 9 core dimensions across
all 7 methods (63/63 tasks). The `resume-missing` action with `--local-models`
processed the tasks that previously failed due to network timeouts. The
`collect-core` action merged the results and produced the summary + paper
table.

Non-core dimensions (object_class, human_action, spatial_relationship, etc.)
remain incomplete — they require additional VBench models not in the local
cache. These are supplementary; the 9 core dimensions are sufficient for the
v155 decision.

## 2. VBench Core-9 Results

| Method | subject_cons | background_cons | temporal_flicker | motion_smooth | overall_cons | dynamic_degree | aesthetic | imaging | temporal_style |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sf_native | .96868 | .96103 | .96804 | .98218 | .23314 | .64167 | .61629 | .68914 | .23314 |
| qk_top4_reservoir4 | .96959 | .96233 | .96346 | .98166 | .23824 | .72500 | .61661 | .70672 | .23824 |
| qk_bottom4_reservoir4 | .96891 | .96019 | .96127 | .97989 | .23741 | .76250 | .61965 | .71494 | .23741 |
| qk_random4_reservoir4 | .96959 | .96194 | .96423 | .98308 | .23794 | .70833 | .62167 | .71093 | .23794 |
| all_reservoir4 | .96457 | .95842 | .95468 | .97708 | .24142 | .83333 | .61824 | .69684 | .24142 |
| qk_top4_prototype4 | .97164 | .96377 | .96344 | .98175 | .23166 | .67917 | .62202 | .69579 | .23166 |
| all_recent8 | .96857 | .96121 | .95941 | .97960 | .23087 | .73333 | .61968 | .70790 | .23087 |

## 3. Paper Table (scaled to 100)

| Method | Dynamic Degree | Motion Smoothness | Overall Consistency | Imaging Quality | Aesthetic Quality | Quality Score |
|---|---:|---:|---:|---:|---:|---:|
| sf_native | 64.17 | 98.22 | 23.31 | 68.91 | 61.63 | 83.05 |
| qk_top4_reservoir4 | 72.50 | 98.17 | 23.82 | 70.67 | 61.66 | 83.79 |
| qk_bottom4_reservoir4 | 76.25 | 97.99 | 23.74 | 71.49 | 61.97 | 84.01 |
| qk_random4_reservoir4 | 70.83 | 98.31 | 23.79 | 71.09 | 62.17 | 83.90 |
| all_reservoir4 | 83.33 | 97.71 | 24.14 | 69.68 | 61.82 | 83.72 |
| qk_top4_prototype4 | 67.92 | 98.17 | 23.17 | 69.58 | 62.20 | 83.42 |
| all_recent8 | 73.33 | 97.96 | 23.09 | 70.79 | 61.97 | 83.60 |

Semantic Score and Total Score remain `n/a` — they require all 16 official
VBench dimensions. The 7 missing non-core dimensions need additional models.

## 4. Key Findings

### 4.1 Reservoir improves motion over SF

Every reservoir and recent8 method has higher Dynamic Degree than SF native
(64.17). The all-reservoir4 control reaches 83.33. This confirms that
dispersed-history caching increases motion amplitude versus native SF.

### 4.2 QK membership does not selectively help

Per `docs/161` section 9, the advance gate requires QK-top to beat both
QK-bottom and QK-random. It does not:

- QK-top4 vs QK-bottom4: bottom beats top on Dynamic Degree (76.25 vs 72.50)
  and Quality Score (84.01 vs 83.79).
- QK-top4 vs QK-random4: random is within noise on Dynamic Degree (70.83 vs
  72.50) and beats top on Quality Score (83.90 vs 83.79).

This places the result in the **"Cache useful, classifier unsupported"**
decision branch: the reservoir cache mechanism improves motion and quality
over SF, but the QK head-membership classification does not selectively
direct it.

### 4.3 Reservoir vs Prototype

QK-top4-reservoir4 (72.50 dynamic) vs QK-top4-prototype4 (67.92 dynamic):
reservoir produces more motion than prototype under the same membership.
This is consistent with the v155 hypothesis that reservoir is a better
profile-aligned cache than prototype, but the membership selectivity issue
remains.

### 4.4 Trade-offs

All reservoir methods trade temporal stability for motion: temporal
flickering drops from SF's 0.96804 to 0.95468-0.96423, and motion smoothness
from 0.98218 to 0.97708-0.98308. The all-reservoir4 control has the most
motion (83.33) but the worst flicker (0.95468) and smoothness (0.97708).

## 5. Decision

Per `docs/161` section 9:

> **Cache useful, classifier unsupported**: reservoir/all-reservoir improves
> results but top does not beat bottom/random. Keep the cache mechanism and
> stop claiming QK membership utility.

The TemporalReservoirStrategy is a viable cache mechanism that increases
motion and quality over native SF. The v152 QK head-membership classification
does not selectively direct it — bottom and random memberships produce
comparable or better results. Human blind review is still needed to confirm
identity/background retention, but the VBench result does not support QK
membership as a generation method contribution.

## 6. VBench Model Setup

The v155 VBench eval required local model caching to avoid network timeouts
on remote nodes:

- Shared cache at `runs/vbench_cache/` (RAFT, AMT, CLIP, pyiqa, aesthetic)
- CLIP at `~/.cache/clip/` on all nodes
- DINO auto-downloaded to `~/.cache/torch/hub/` on node 0
- `--local-models --torch-hub-dir --runtime-home` flags use shared caches
- `resume-missing` action (NUM_NODES=1) processes only incomplete tasks
- `collect-core` merges the 9 core dimensions (skips non-core)

Non-core dimensions (7 of 16) need additional models (UMT, GRIT, ViClip,
caption) not in the cache. These are supplementary to the v155 decision.
