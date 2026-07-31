# v150 Policy-Group Confirmation Results

## Integrity

- Suite: `v150_policy_group_core`
- Profiles: `64`
- Prompts: `32`
- Native replay maximum relative RMS: `0`
- Probe/context calibration pass rate: `41/66`
- Calibration clipped / degenerate layers: `0 / 0`

## Gates

```json
{
  "g0_native_replay": true,
  "g1_count_matched_group_effect": {
    "leverage": false,
    "susceptibility": false
  },
  "g2_intervention_specificity": {
    "leverage": false,
    "susceptibility": false
  },
  "g3_policy_group_confirmed": {
    "leverage": false,
    "susceptibility": false
  }
}
```

## Interpretation Boundary

A policy group is confirmed only when top4 beats bottom4, middle4, and the eight-map random ensemble in one intact context. Core confirmation additionally requires policy contrast to exceed both K- and V-shift separations.

These are one-step frame-117 causal measurements. They do not establish trajectory-level video quality.
