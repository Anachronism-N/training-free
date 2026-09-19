# Compact experiment results

This directory tracks source-bound reports, decisions, metric summaries, completion receipts, and provenance needed to audit completed campaigns.

Large generated media, model weights, VBench clips, tensor dumps, and raw traces remain on experiment storage and are intentionally excluded from Git.

- `v209_generation_86f10607/`: completed original v209 Self-Forcing protocol/budget campaign. The frozen recommendation is `freeze_both_sf_protocol_controls_before_addon_screen`.
- `v209_generation_d8a6a88f/`: completed source-bound native v209 Screen32 rerun with 150/150 formal shards, 45/45 VBench jobs, temporal diagnostics, budget analysis, and explicit Gate-bypass provenance.
- `v210_generation_789d8604_eval_08fe9d72/`: completed v210 LPHC Screen8 campaign and evaluation. The frozen recommendation is `stop_v210_no_eligible_lphc_candidate`.
- `v211_generation_9a1c352b_eval_3093c028/`: completed sink-preserving LPHC Screen8 campaign and evaluation. Both candidates were directionally positive, but the frozen recommendation is `stop_v211_no_eligible_lphc_candidate`.
