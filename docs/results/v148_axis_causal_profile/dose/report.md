# v148 Axis-Matched Causal Profiling Results

## Integrity

- Suite: `v148_axis_dose`
- Profiles: `32`
- Prompts: `16`
- Native replay maximum relative RMS: `0`
- Shift interventions non-degenerate: `True`

## Gates

```json
{
  "g0_native_replay_parity": true,
  "g1_positive_rank_separation_at_multiple_doses": {
    "k": false,
    "policy": true,
    "v": true
  },
  "g2_dose4_exceeds_dose1": {
    "k": false,
    "policy": false,
    "v": true
  }
}
```

## Claim boundary

Dose effects compare top-k and bottom-k at equal head count. Absolute perturbation growth with head count is not evidence.

The analyzer reports perturbation sensitivity, not video quality. A passing
axis must still be tested in a trajectory-level method experiment.
