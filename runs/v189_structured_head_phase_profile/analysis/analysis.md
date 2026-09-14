# v189 Structured Head x Phase Profile

- Recommendation: `advance_head_phase_maps_to_causal_screen`
- Generation candidates: `['retrieval']`
- Holdout prompts were not used for classification.
- Classification is per denoising call; cross-call consistency is not a gate.

| Operator | Joint cells | Head-only cells | Phase/layer-only cells | Phase-selective cells | Per-call joint |
|---|---:|---:|---:|---:|---|
| landmark | 0 | 0 | 0 | 0 | [0, 0, 0, 0] |
| retrieval | 66 | 68 | 0 | 36 | [20, 15, 15, 16] |

The `compatible` map is the primary generation candidate. `phase_selective` and `top12_discovery` are preregistered mechanism/threshold controls, not alternative claims selected after video review.
