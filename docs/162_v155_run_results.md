# 162: v155 Profile-Aligned Reservoir Transfer — Run Results

Date: 2026-08-02
Cluster: 4 nodes x 8x H20 (32 GPUs)
Commit: `c42f40a` on `codex/v98-correctness-fixes`

## 1. Summary

v155 tests the v152 QK hypothesis with a profile-aligned cache:
`TemporalReservoirStrategy` (streaming Algorithm-R reservoir) replaces v154's
`TemporalPrototype4`. The reservoir stores exact dispersed-history frames,
directly matching the v152 QK score definition ("dispersed vs recent QK
compatibility").

**Generation fully succeeded**: 7 methods x 16 prompts = 112 videos (64 newly
generated + 48 reused from v154). Audit, blind review, and package all passed.
VBench-Long completed 53/63 tasks (84%): 7 of 9 dimensions fully complete
across all 7 methods. The remaining 10 tasks (subject/background_consistency
on several methods) failed due to network timeouts during model download on
remote nodes, not a generation or split issue. Collect cannot merge.

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

### 3.1 Completed dimensions

53/63 tasks done (84%). 9 dimensions evaluated:

| Dimension | Complete | Missing |
|---|---|---|
| temporal_flickering | 7/7 | — |
| motion_smoothness | 7/7 | — |
| dynamic_degree | 7/7 | — |
| imaging_quality | 7/7 | — |
| overall_consistency | 7/7 | — |
| aesthetic_quality | 7/7 | — |
| temporal_style | 7/7 | — |
| subject_consistency | 1/7 | 6 methods |
| background_consistency | 3/7 | 4 methods |

7 of 9 dimensions fully complete across all 7 methods. The 4 key decision
dimensions (dynamic_degree, temporal_flickering, motion_smoothness,
imaging_quality) plus overall_consistency, aesthetic_quality, and
temporal_style are all fully available.

### 3.2 Failure root cause

The 10 missing tasks fail with:
```
urllib.error.URLError: <urlopen error [Errno 110] Connection timed out>
```

VBench's subject/background_consistency dimensions auto-download models (DINO
for subject, CLIP for background) via `torch.hub` or `clip` package. On remote
nodes (1-3), these downloads time out (no proxy configured or network
restriction). Node 0 has DINO/CLIP cached from v154 and succeeds on some
tasks. The split_clip fix from `vbench_long_split_cache.py` was applied
successfully (split stage passed on all nodes) — the failure is purely a
network/model-download issue.

### 3.3 Recovery path

To complete the 10 missing tasks:
1. Set `HTTP_PROXY`/`HTTPS_PROXY` on all nodes before eval; OR
2. Pre-download DINO to `~/.cache/torch/hub/` and CLIP to `~/.cache/clip/`
   on nodes 1-3 (CLIP already done; DINO still needed); OR
3. Run eval on node 0 only (NUM_NODES=1) — node 0 has all models cached.

Then re-run `collect` on node 0.

## 4. Decision status

Per `docs/161` section 9, the promotion gate requires both blind review and
VBench. Blind review is prepared but not scored (human task). VBench has 7/9
dimensions complete — the key decision dimensions are fully available but
subject/background_consistency is incomplete.

The v155 hypothesis (reservoir directly matches QK score definition) is a
stronger test than v154's Prototype4 mismatch. The aggregate VBench scores
for the 7 complete dimensions will be available once the missing tasks are
re-run and collect succeeds.

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
|-- metrics/vbench_long_parts/ # 53/63 VBench task results
`-- v155_diagnostics.tar.gz
```

Key results copied to `docs/results/v155_profile_aligned_moviebench16/`.

## 6. Supervisor notes

The v155 supervisor (`v155_supervisor.sh`) coordinated the full pipeline with
GPU occupy release/re-acquire. Generation needed no retries. VBench eval
failed on all 4 nodes (3 retries each) due to network timeouts. The supervisor
correctly re-occupied all nodes after the eval phase and proceeded to collect
(which failed on missing results). GPU occupy maintained throughout.
