# v150 Policy-Group Confirmation Results

## Integrity

- Suite: `v150_policy_group_strength`
- Profiles: `32`
- Prompts: `16`
- Native replay maximum relative RMS: `0`
- Probe/context calibration pass rate: `41/66`
- Calibration clipped / degenerate layers: `0 / 0`

## Gates

```json
{
  "g0_native_replay": true,
  "g1_group_effect_at_multiple_targets": {
    "leverage": false,
    "susceptibility": false
  },
  "g2_target_response_sanity": {
    "leverage": false
  },
  "g3_strength_robust_policy_group": {
    "leverage": false
  }
}
```

## Interpretation Boundary

A policy group is confirmed only when top4 beats bottom4, middle4, and the eight-map random ensemble in one intact context. Core confirmation additionally requires policy contrast to exceed both K- and V-shift separations.

These are one-step frame-117 causal measurements. They do not establish trajectory-level video quality.
