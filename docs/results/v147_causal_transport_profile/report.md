# v147 Causal Transport Profiling Results

## Integrity and mechanism gates

- `g0_native_replay_parity`: **PASS**
- `g1_ranked_heads_have_reproducible_downstream_effect`: **PASS**
- `g2_q_retrieval_rescues_ranked_heads`: **FAIL**
- `g3_value_shift_is_non_degenerate`: **PASS**
- `g4_q_retrieval_is_head_selective`: **FAIL**

- Native replay maximum relative RMS: `0`
- Source ranking: `full_semantic / k_shift`
- Profiles: `64`; prompts: `32`; seeds per prompt: `2`

## Interpretation

`G1` is required before assigning a functional role to the v145 ranking.
`G2` is required before turning Q-retrieval into a proposed cache design.
QK-V correspondence is a transport-alignment diagnostic, not optical flow
and not an independently validated motion-head label.

## Artifacts

- `profile_audit.csv`
- `downstream_observations.csv.gz`
- `downstream_comparisons.csv`
- `layer_band_effects.csv`
- `qkv_head_observations.csv.gz`
- `qkv_group_comparisons.csv`
- `report.json`
