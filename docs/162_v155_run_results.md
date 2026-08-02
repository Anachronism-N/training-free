# 162: v155 Profile-Aligned Reservoir Transfer — Run Results

Date: 2026-08-02
Cluster: 4 nodes x 8x H20 (32 GPUs)
Commit: `c42f40a` on `codex/v98-correctness-fixes`

Status update: the ten missing core jobs were recovered offline and core-9 is
now 63/63 complete. The original 53/63 state below records the first run.
Final scores and decisions are in `docs/163_v155_vbench_core9_results.md` and
`docs/165_v155_final_analysis_and_v157_layer_gate_plan.md`.

## 1. Summary

v155 tests the v152 QK hypothesis with a profile-aligned cache:
`TemporalReservoirStrategy` (streaming Algorithm-R reservoir) replaces v154's
`TemporalPrototype4`. The reservoir stores exact dispersed-history frames,
directly matching the v152 QK score definition ("dispersed vs recent QK
compatibility").

**Generation fully succeeded**: 7 methods x 16 prompts = 112 videos (64 newly
generated + 48 reused from v154). Audit, blind preparation, and package all
passed. The review sheet has not yet been scored.

The frozen VBench manifest contains 16 dimensions x 7 methods = 112 jobs, of
which 53 are complete (47.3%). The scientifically valid MovieBench core-9
subset is 53/63 complete (84.1%): seven dimensions are complete and subject /
background consistency each have five missing jobs. The other seven official
semantic dimensions account for 49 missing jobs and cannot be treated as
network-only failures: several lack optional code dependencies, and the
arbitrary MovieBench prompts do not provide their required auxiliary labels.

## 2. Generation

### 2.1 Timeline

| Stage | Start | End | rc | Note |
|---|---|---|---|---|
| preflight | 15:42:44 | 15:42:49 | 0 | suite + map audit PASS |
| generate (4 nodes) | 15:43:06 | 15:55:20 | 0 | 64 new videos, ~12 min |
| audit | 15:55:54 | 15:55:54 | 0 | ok=true, failures=[] |
| blind | 15:55:57 | 15:55:57 | 0 | reviewer + private dirs |
| package | 15:56:01 | 15:56:01 | 0 | tarball created |

### 2.2 Seven methods

| Method | Membership | label-10 | label-11 | Source |
|---|---|---|---|---|
| sf_native | none | SF | SF | reused v154 |
| ours_qk_top4_reservoir4 | top4/layer | Reservoir4 | recent8 | new |
| ours_qk_bottom4_reservoir4_control | bottom4/layer | Reservoir4 | recent8 | new |
| ours_qk_random4_reservoir4_control | random4/layer | Reservoir4 | recent8 | new |
| ours_all_reservoir4_control | all heads | Reservoir4 | Reservoir4 | new |
| ours_qk_top4_prototype4_reference | top4/layer | Prototype4 | recent8 | reused v154 |
| ours_all_recent8_reference | all heads | recent8 | recent8 | reused v154 |

`V155_REUSE_V154_ROOT` set; 3 methods reused, 4 newly generated (64 videos).

### 2.3 Blind review

`blind_review/reviewer/` contains 112 anonymous videos + `v155_review_sheet.csv`.
`blind_review/private/v155_blind_key.json` holds the method mapping.

## 3. VBench-Long

### 3.1 Actual completion state

The frozen 16-dimension status is 53/112. The valid core-9 view is:

| Dimension | Complete | Missing |
|---|---|---|
| temporal_flickering | 7/7 | — |
| motion_smoothness | 7/7 | — |
| dynamic_degree | 7/7 | — |
| imaging_quality | 7/7 | — |
| overall_consistency | 7/7 | — |
| aesthetic_quality | 7/7 | — |
| temporal_style | 7/7 | — |
| subject_consistency | 2/7 | 5 methods |
| background_consistency | 2/7 | 5 methods |

7 of 9 dimensions fully complete across all 7 methods. The 4 key decision
dimensions (dynamic_degree, temporal_flickering, motion_smoothness,
imaging_quality) plus overall_consistency, aesthetic_quality, and
temporal_style are all fully available.

### 3.2 Failure root causes

The ten missing core jobs failed while resolving DINO or DreamSim models,
including:
```
urllib.error.URLError: <urlopen error [Errno 110] Connection timed out>
```

The recovery code now enables VBench's local-model mode, points `torch.hub` at
the shared DINOv2 cache, validates the local CLIP checkpoint, and links DINO
and DreamSim's nested DINO cache into shared storage. It also adds read-only
`status`, missing-only resume, and dimension-subset collection. Existing 53
markers remain valid under their original job contracts; only unfinished jobs
receive the offline-cache wrapper contract.

The other 49 failures are separate:

| Dimension group | Jobs | Cause |
|---|---:|---|
| object_class / multiple_objects / color / spatial_relationship | 28 | optional Detectron2 stack absent |
| scene | 7 | optional FairScale stack absent |
| human_action | 7 | installed timm API incompatible with VBench model constructor |
| appearance_style | 7 | MovieBench mapping lacks required `auxiliary_info` labels |

More importantly, object/action/color/spatial/scene/style scoring is defined
against benchmark-specific annotations. Installing packages would not make
those scores valid for this arbitrary 16-prompt MovieBench subset. They remain
in the frozen manifest for auditability, but are excluded from the core-9
decision table.

### 3.3 Recovery path

Run the ten missing core jobs on any node that can see the shared repository:

```bash
export V155_VBENCH_DIMENSIONS=subject_consistency,background_consistency,temporal_flickering,motion_smoothness,overall_consistency,dynamic_degree,aesthetic_quality,imaging_quality,temporal_style
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v155_vbench_long.sh status
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v155_vbench_long.sh resume-missing
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v155_vbench_long.sh collect-core
```

`resume-missing` defaults to `V155_LOCAL_MODELS=1`; it does not require proxy
access. `collect-core` writes `vbench_core9_summary.*`,
`v155_vbench_core9_analysis.*`, and `paper_table_core9/`. The official Quality
composite is available there; Semantic and Total remain `n/a` because the full
nine-dimension semantic contract is intentionally incomplete.

## 4. Decision status

Per `docs/161` section 9, the promotion gate requires both blind review and
core-9 VBench. Blind review is prepared but not scored. Seven core dimensions
are complete, while subject/background consistency still require ten offline
jobs. The frozen metric promotion gate already cannot pass because
top-reservoir is `0.0375` below bottom-reservoir on dynamic degree, beyond the
`0.03` non-inferiority bound. Completing core-9 and human review is still
required to diagnose identity/background retention and decide whether the
cache mechanism, rather than the head classifier, should be retained.

The v155 hypothesis (reservoir directly matches QK score definition) is a
stronger test than v154's Prototype4 mismatch. The seven complete dimensions
are analyzed in `docs/163`; collection remains necessary for the core-9
aggregate and final diagnostic table.

## 5. Preserved artifacts

```text
runs/v155_profile_aligned_moviebench16/full7/
|-- videos/                    # 112 MP4 files (64 new + 48 reused)
|-- published_manifest.json
|-- blind_review/
|   |-- reviewer/              # 112 anonymous videos + review sheet
|   `-- private/               # blind key
|-- contracts/
|-- diagnostics/
|-- metrics/vbench_long_parts/ # 53/112 frozen-manifest task results
`-- v155_diagnostics.tar.gz
```

Key results copied to `docs/results/v155_profile_aligned_moviebench16/`.

## 6. Supervisor notes

The v155 supervisor (`v155_supervisor.sh`) coordinated the full pipeline with
GPU occupy release/re-acquire. Generation needed no retries. VBench eval
failed on all 4 nodes (3 retries each) due to network timeouts. The supervisor
correctly re-occupied all nodes after the eval phase and proceeded to collect
(which failed on missing results). GPU occupy maintained throughout.
