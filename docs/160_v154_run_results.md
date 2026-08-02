# 160: v154 History-Critical Validation — Run Results

Date: 2026-08-02
Cluster: 4 nodes x 8x H20 (32 GPUs)
Commit: `6b69c63` on `codex/v98-correctness-fixes`

## 1. Summary

v154 validates the v152 QK history-critical head map across 16 diverse
MovieBench prompts. Eight paired methods (SF native, QK-top, QK-bottom,
random, all-recent8, all-prototype4, legacy membership, legacy reference)
each generated 16 thirty-second videos — 128 videos total on 32 GPUs.

**Generation fully succeeded**: all 128 videos generated, audit passed
(contract ok=true, 128 publish markers, no failures), blind review prepared.
VBench-Long evaluation completed 56/64 tasks (87.5%): 6 of 8 dimensions are
fully complete across all 8 methods. The remaining 8 tasks (subject_consistency
for 6 methods, background_consistency for 2 methods) fail because a metadata
file is visible to two VBench clip-enumeration paths. Collect cannot merge
because it requires all 64 tasks.

## 2. Generation

### 2.1 Timeline

| Stage | Start | End | rc | Note |
|---|---|---|---|---|
| preflight | 00:38:42 | 00:38:46 | 0 | suite + map audit PASS |
| generate (4 nodes) | 00:39:04 | 01:01:56 | 0 | 128 videos, ~22 min |
| audit | 01:02 | 01:02 | 0 | 128 videos, 128 markers, ok=true |
| blind | 01:02 | 01:03 | 0 | reviewer + private dirs created |
| package | 01:03 | 01:03 | 0 | tarball created |

### 2.2 Eight methods

| Method | Membership | label-10 route | label-11 route |
|---|---|---|---|
| sf_native | none | SF native | SF native |
| ours_qk_top4 | QK top4/layer | Prototype4 | recent8 |
| ours_qk_bottom4_control | QK bottom4/layer | Prototype4 | recent8 |
| ours_qk_random4_control | random4/layer | Prototype4 | recent8 |
| ours_all_recent8_control | QK map | recent8 | recent8 |
| ours_all_prototype4_control | QK map | Prototype4 | Prototype4 |
| ours_legacy_membership | old-v98 304/56 | Prototype4 | recent8 |
| ours_legacy_reference | old-v98 304/56 | Prototype4 | Retrieval1(age<=24) |

v125 reuse root not found; all 128 videos generated fresh. Each video is
120 latent frames (~30 seconds), seed 0.

### 2.3 Blind review

`blind` created 128 anonymous hardlinks with per-prompt randomized method
order:
- `blind_review/reviewer/videos/` — 128 anonymous videos
- `blind_review/reviewer/v154_review_sheet.csv` — review template
- `blind_review/private/v154_blind_key.json` — method-to-anonymous mapping

Reviewer only accesses `reviewer/`. After filling the sheet, run
`analyze_v154_blind_review.py` for per-prompt W/T/L and bootstrap CIs.

## 3. VBench-Long

### 3.1 Model setup

VBench required pretrained models not present on the cluster. A shared cache
was created at `runs/vbench_cache/` (1.4 GB) containing:
- RAFT (`raft-things.pth`, copied from repo root)
- AMT (`amt-s.pth`, downloaded from HuggingFace)
- CLIP (`ViT-B-32.pt`, `ViT-L-14.pt`, downloaded from OpenAI)
- pyiqa (`musiq_spaq_ckpt`, downloaded from GitHub)
- aesthetic (`sa_0_4_vit_l_14_linear.pth`, downloaded from LAION)

`VBENCH_CACHE_DIR` was set to the shared path so all 4 nodes use one copy.
CLIP models were also copied to `~/.cache/clip/` on each node (VBench's
`clip` package uses that path). DINO (for subject_consistency) auto-downloads
from `torch.hub` to `~/.cache/torch/hub/` per node.

### 3.2 Completed dimensions

56/64 tasks done (87.5%). Dimensions fully complete across all 8 methods:

| Dimension | Status | Models used |
|---|---|---|
| temporal_flickering | 8/8 PASS | RAFT + AMT |
| motion_smoothness | 8/8 PASS | RAFT + AMT |
| dynamic_degree | 8/8 PASS | RAFT |
| imaging_quality | 8/8 PASS | pyiqa |
| overall_consistency | 8/8 PASS | CLIP ViT-L-14 |
| aesthetic_quality | 8/8 PASS | aesthetic + CLIP |
| background_consistency | 6/8 | CLIP (missing ours_qk_top4, ours_legacy_reference) |
| subject_consistency | 2/8 | DINO (sf_native and ours_all_recent8_control) |

### 3.3 Missing tasks and root cause

The 8 missing tasks fail with:
```
Exception: .../split_clip/.v129_split_manifest.json should be a path
that contains video clips or a path of a video file!
```

All split directories contain the expected 16 prompt folders and 240 clips.
The actual failure is that `.v129_split_manifest.json` was written inside the
`split_clip` root. VBench subject/background utilities inspect the first root
entry as if it were a clip directory; filesystem enumeration can therefore
select the JSON file and fail nondeterministically. The repaired pipeline moves
the unchanged manifest beside `split_clip`, requires a directory-only clip
root, and resumes the 56 valid results without regenerating videos.

### 3.4 Available per-task results

Each completed task has `done.json`, `results.json`, `prompt_mapping.json`,
and `run.log` under `metrics/vbench_long_parts/<method>/<dimension>/`. These
56 result sets are preserved and can be manually merged or the split issue
can be fixed and the 8 missing tasks re-run.

## 4. Decision status

Per `docs/159` section 9, the promotion gate requires both blind review and
VBench. The blind review is prepared but not yet scored (human task). VBench
has 6/8 dimensions complete — the 4 key dimensions for the decision
(dynamic_degree, temporal_flickering, motion_smoothness, imaging_quality)
are fully available. The collect merge is blocked by 8 missing tasks.

Recommended next steps:
1. Fix the split_clip directory issue and re-run the 8 missing VBench tasks;
2. Complete the human blind review;
3. Run `analyze_v154_blind_review.py` and `collect` to produce the final
   comparison.

## 5. Preserved artifacts

```text
runs/v154_history_critical_moviebench16/full8/
|-- videos/                    # 128 MP4 files (8 methods x 16 prompts)
|-- published_manifest.json    # frozen generation contract
|-- blind_review/
|   |-- reviewer/              # 128 anonymous videos + review sheet
|   `-- private/               # blind key (method mapping)
|-- contracts/                 # frozen run contract
|-- diagnostics/               # per-cell policy + role-event traces
|-- metrics/vbench_long_parts/ # 56/64 VBench task results
`-- v154_diagnostics.tar.gz    # package tarball
```

Raw videos and runs are gitignored. Key results copied to
`docs/results/v154_history_critical_moviebench16/`.

## 6. Supervisor notes

The v154 supervisor (`v154_supervisor.sh`) coordinated the full pipeline:
generation (4-node GPU), audit, blind, package, VBench (prepare, split,
preflight, eval, collect). GPU occupy was released for GPU stages and
re-acquired for CPU stages. The VBench re-run (`v154_vbench_rerun.sh`)
handled the model dependency fix and resumable eval. No generation retries
were needed; VBench eval required model setup and a re-run.
