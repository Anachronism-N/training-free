# v151 Signed Policy / Low-Tail Confirmation

## Integrity

- Profiles: `64`
- Prompts: `32`
- Native replay maximum relative RMS: `0`
- Probe/context calibration pass rate: `63/128`
- Intact contexts: `[]`
- Invalid contexts: `['noisy_t1000', 'noisy_t250', 'noisy_t500', 'noisy_t750']`

## Gates

```json
{
  "g0_native_replay": true,
  "g1_scalar_low_tail_susceptibility": false,
  "g2_scalar_policy_leverage": false,
  "g3_scalar_intervention_specificity": false,
  "g4_scalar_candidate_confirmed": false,
  "g5_signed_source_screen": true,
  "g6_signed_group_effect": {
    "leverage": false,
    "susceptibility": false
  },
  "g7_signed_intervention_specificity": false,
  "g8_signed_candidate_confirmed": false
}
```

The scalar and signed branches are separate hypotheses. A passing one-step gate does not establish improved long-video generation.
Contexts that fail calibration integrity remain in diagnostic CSVs but cannot satisfy any confirmation gate.
