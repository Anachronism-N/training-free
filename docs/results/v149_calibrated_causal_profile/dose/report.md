# v149 Calibrated Causal Profiling Results

## Integrity

- Suite: `v149_calibrated_dose`
- Profiles: `32`
- Prompts: `16`
- Native replay maximum relative RMS: `0`
- Calibration maximum relative target error: `1`
- Calibration clipped layers: `2`
- Calibration degenerate layers: `8`

## Gates

```json
{
  "g0_integrity_and_calibration": false,
  "g1_positive_separation_at_multiple_doses": {
    "leverage": {
      "k": false,
      "policy": true,
      "v": false
    },
    "susceptibility": {
      "k": true,
      "policy": true,
      "v": true
    }
  }
}
```

## Interpretation Boundary

`susceptibility` is the raw projected local replacement before calibration. `leverage` is the final x0 effect after every layer has been calibrated to the same projected relative RMS.

These are one-step downstream causal measurements at frame 117, not trajectory-level video quality or a validated cache method.
