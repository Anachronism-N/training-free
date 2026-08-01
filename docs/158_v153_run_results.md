# 158: v153 History-Critical Transfer — Run Results

Date: 2026-08-01
Cluster: single node (node 0: 29.232.229.115), GPU 0-6 (7 GPUs)
Commit: `139de01` on `codex/v98-correctness-fixes`

## 1. Summary

v153 tests whether the v152 one-sided QK history-critical head map transfers
to actual 30-second generation. Seven single-video cells ran in parallel on
one node. All seven completed with no structural errors (no polygon noise, no
early termination, no contract violation). Per `docs/157` section 7, this
clears the generation-transfer gate and supports proceeding to a 16-prompt
comparison — pending human blind review of the videos.

This is a one-video screen, not a statistical conclusion. It only verifies
that the QK-top membership routes through the PyramidKV history-polarity path
without breaking generation.

## 2. Configuration

- 7 cells, same Qwen MovieBench prompt (index 0), seed 0, 120 latent frames
  (~30 seconds);
- Pyramid-Forcing inference with `--pyramidkv_history_polarity`;
- checkpoint: `/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt`;
- conda env: `longlive`;
- head maps frozen in `configs/head_maps/v152_qk_history_critical_*.csv`;
- CUDA extension `pyramidkv_scatter_v30` compiled on first run (501s, now
  cached).

## 3. Manifest Fix

The frozen manifest `v152_qk_history_critical_manifest.json` committed in
`139de01` carried a `pf_labels_sha256` from a different machine. On this
cluster `third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv`
has a different SHA256, so `preflight` failed with "frozen artifact mismatch".

Fix: regenerated the manifest on this machine via
`python scripts/analyze_v152_one_sided_history_critical.py` (no `--check`).
Only the `pf_labels_sha256` field changed (2 lines); the three head-map CSVs
and their hashes are unchanged. The check now passes.

## 4. Run Timeline

Supervisor log: `/tmp/v153_supervisor/supervisor.log`.

| Stage | Start | End | rc | Note |
|---|---|---|---|---|
| 1 preflight | 22:39:10 | 22:39:14 | 0 | map audit PASS, overlap=112/120 |
| 2 screen | 22:39:19 | 22:54:11 | 0 | 7 cells in parallel, ~15 min |
| 3 package | 22:54:20 | 22:54:23 | 0 | tarball 532 KB |

`screen` wall time includes the one-time CUDA extension compilation (501s).
Actual generation was ~7 min once the extension was cached.

## 5. Cells and Results

All seven cells completed (`ok=true`, `failures=[]`):

| Cell | GPU | Map | label10 | label11 | Status |
|---|---|---|---|---|---|
| `qk_top4_prototype4_default_recent8` | 0 | qk_top4 | 120 | 240 | completed |
| `qk_bottom4_control_prototype4_default_recent8` | 1 | qk_bottom4 | 120 | 240 | completed |
| `qk_random4_control_prototype4_default_recent8` | 2 | random4 | 120 | 240 | completed |
| `legacy_v98_membership_prototype4_default_recent8` | 3 | legacy 304/56 | 304 | 56 | completed |
| `qk_top4_all_recent8_control` | 4 | qk_top4 | — | — | completed |
| `qk_top4_all_prototype4_control` | 5 | qk_top4 | — | — | completed |
| `legacy_v98_prototype4_retrieval1_age24_reference` | 6 | legacy 304/56 | 304 | 56 | completed |

Each cell produced a 30-second MP4 (~3.5-3.9 MB), 2400 policy records, and
200 role-event records (360 for the reference cell). The reference cell
(`legacy_v98_prototype4_retrieval1_age24_reference`) uses the known-working
v98 Retrieval1+age24 link as a sanity check.

## 6. Map Audit

The screen log records the full map audit:

- `qk_top4`: 120 history-critical (label 10), 240 default (label 11), 4 per
  layer, PF cross-tab Anchor 24 / Veil 19 / Wave 77;
- `qk_bottom4_control`: same counts, reverse membership, Anchor 84 / Veil 5
  / Wave 31;
- `random4_control`: count-matched, Anchor 59 / Veil 7 / Wave 54;
- `legacy`: 304/56 absolute-sign map.

None of the QK maps replicate the PF Anchor set, confirming the v152 finding
that QK history-criticality is a distinct axis from PF temporal logit class.

## 7. Decision (per docs/157 section 7)

- No non-reference cell produced polygon noise, static frames, or early
  termination → **not a "return to implementation audit" outcome**.
- The QK-top candidate ran cleanly through the PyramidKV history-polarity
  path with `exclusive_owner=true` and `legacy_pf_labels=false`.
- Scientific promotion requires human blind review comparing QK-top against
  bottom/random/reference on identity, background, motion amplitude, and
  late-half drift. This run produced the videos for that review but does not
  score it.

Recommended next step: 16-prompt comparison of SF, QK-top, bottom, random,
and the known reference, as described in `docs/157` section 7 "推进".

## 8. Preserved Artifacts

```text
docs/results/v153_history_critical_transfer/
|-- v153_diagnostics.tar.gz          # full package (532 KB)
|-- contracts/all.json               # frozen run contract
|-- status/                          # 7 .done.json + all.node0.summary.json
`-- diagnostics/                      # 21 JSON files (policy, role_event, video)
```

Raw videos and logs remain in `runs/v153_history_critical_transfer_1video/`
(not committed; `runs/` is gitignored).

## 9. Supervisor Notes

The supervisor (`/apdcephfs_gy2/share_303214315/cedricnie/v153_supervisor.sh`)
coordinated GPU occupy release/re-acquire around the `screen` stage, retried
on failure, and re-occupied all nodes at the end. The occupy check uses
`nvidia-smi` memory (>500 MB) instead of `pgrep` to avoid the false-positive
shell-matching bug seen in v151/v152. No retries were needed.
