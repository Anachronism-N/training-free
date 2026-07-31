# v148 Axis-Matched Causal Profiling Results

## Integrity

- Suite: `v148_axis_core`
- Profiles: `64`
- Prompts: `32`
- Native replay maximum relative RMS: `0`
- Shift interventions non-degenerate: `True`

## Gates

```json
{
  "g0_native_replay_parity": true,
  "g1_axis_matched_causal_effect": {
    "k": true,
    "policy": true,
    "v": true
  },
  "g2_pf_independent_effect": {
    "k": true,
    "policy": false,
    "v": false
  },
  "g3_intervention_specificity": {
    "k": false,
    "policy": false,
    "v": false
  }
}
```

## Claim boundary

A functional axis requires the same denoising context to pass top-vs-bottom and top-vs-two-random controls. Independence from PF additionally requires the within-layer, same-PF-label pair.

The analyzer reports perturbation sensitivity, not video quality. A passing
axis must still be tested in a trajectory-level method experiment.
