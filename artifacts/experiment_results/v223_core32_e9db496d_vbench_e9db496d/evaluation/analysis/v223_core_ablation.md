# v223 fixed-seed core ablation

Uniform 32-prompt subset, not independent confirmation. Retain all contrasts. Dose is 0.02 vs 0.10 with clipping unchanged; phase is e1 vs full at 0.02. Full phase increases both intervention exposure and compute; not dose/FLOP matched. No unclipped, direct-union, or clipping-only ablation was performed.

All displayed differences use a 0-100 scale. All 32 prompts are retained.

| Frozen Ours minus control | Imaging delta | Imaging 95% CI | Quality delta | DD delta |
|---|---:|---|---:|---:|
| sf_fifo21 | +0.1206 | [-0.2864, +0.5084] | -0.0552 | -1.0417 |
| strong_e1 | -0.1097 | [-0.5739, +0.3225] | -0.1151 | -0.8333 |
| phase_full | -0.0409 | [-0.4385, +0.3381] | -0.1423 | -1.0417 |

All windows, metrics and automatic flags are retained in JSON/CSV. No new manual review requested.
SF and frozen Ours reuse original v219 videos; this batch is not a paired latency benchmark.
