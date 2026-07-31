# v149 Calibrated Causal Profiling Results

## Integrity

- Suite: `v149_calibrated_core`
- Profiles: `64`
- Prompts: `32`
- Native replay maximum relative RMS: `0`
- Calibration maximum relative target error: `1`
- Calibration clipped layers: `0`
- Calibration degenerate layers: `4`

## Gates

```json
{
  "g0_integrity_and_calibration": false,
  "g1_matched_axis_effect": {
    "leverage": {
      "k": false,
      "policy": false,
      "v": false
    },
    "susceptibility": {
      "k": true,
      "policy": false,
      "v": false
    }
  },
  "g2_pf_independent_effect": {
    "leverage": {
      "k": false,
      "policy": false,
      "v": false
    },
    "susceptibility": {
      "k": true,
      "policy": true,
      "v": false
    }
  },
  "g3_intervention_specificity": {
    "leverage": {
      "k": false,
      "policy": false,
      "v": false
    },
    "susceptibility": {
      "k": false,
      "policy": true,
      "v": false
    }
  }
}
```

## Interpretation Boundary

`susceptibility` is the raw projected local replacement before calibration. `leverage` is the final x0 effect after every layer has been calibrated to the same projected relative RMS.

These are one-step downstream causal measurements at frame 117, not trajectory-level video quality or a validated cache method.
