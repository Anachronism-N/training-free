# v152 Online State-Conditioned Policy Profiling

- Profiles: `128`
- Prompts: `64`
- Native replay maximum relative RMS: `0`

## Gates

```json
{
  "g0_native_replay_and_contract": true,
  "g1_oracle_policy_choice": false,
  "g2_qk_policy_choice": false,
  "g3_qk_beats_count_matched_random": false,
  "g4_qk_matches_oracle": true,
  "g5_online_qk_candidate_confirmed": false,
  "g6_old_mass_baseline": false
}
```

## Qualifying Contexts

```json
{
  "old_mass_policy_choice": [],
  "online_qk_candidate": [],
  "oracle_policy_choice": [],
  "qk_beats_random": [],
  "qk_oracle_alignment": [
    "noisy_f117_t1000",
    "noisy_f117_t750",
    "noisy_f117_t500",
    "noisy_f117_t250"
  ],
  "qk_policy_choice": []
}
```

The oracle uses native full-history policy errors and is not deployable. Only a passing QK branch is eligible for trajectory-level routing. Its shared candidate-bank cost must still be measured.
