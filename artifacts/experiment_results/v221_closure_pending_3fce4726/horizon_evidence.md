# LPHC fixed-seed horizon closure

Status: v220_pending

No new generation or seed search. No automatic acceptance verdict.

| Cohort | Method | Imaging | Quality | Subject | Dynamic |
|---|---|---:|---:|---:|---:|
| v219 | sf_fifo21 | 0.690480 | 81.617819 | 0.969634 | 0.426042 |
| v219 | ours_correct | 0.694270 | 81.551842 | 0.969252 | 0.413542 |
| v219 | ours_random | 0.693825 | 81.736749 | 0.969944 | 0.430208 |
| v219 | pooled_correct | 0.692944 | 81.616263 | 0.969645 | 0.412500 |

## Frozen primary
- v219 full/imaging_quality: +0.003790, CI [-0.0007757851526296464, 0.008777501928759729]; includes_zero.

## Mechanism controls

| Control | Imaging delta | 95% CI |
|---|---:|---|
| ours_random | +0.000446 | [-0.002870, 0.003650] |
| pooled_correct | +0.001327 | [-0.001292, 0.004266] |

## Descriptive horizon interaction

| Metric | Effect at 60s minus effect at 30s | 95% CI |
|---|---:|---|

## Automatic risks
- v219: 10/64 flagged, automatic pass=False; not confirmed failures.

Requested diagnostic pairs: 0; not a user study.

- Compact receipt checks, not a new media or raw VBench evaluation.
- Same 64 prompts and one seed per prompt across horizons; not 128 independent prompts.
- Same seed does not guarantee identical 30s/60s rollout prefixes.
- Cross-horizon interactions are descriptive and cannot replace the frozen endpoint.
- Keep random/pooled controls; do not select a best seed per method or omit unfavorable samples.
- Raw metrics use native 0-1 scale; official Quality uses 0-100. Core-9 is not full Total/Semantic.
- Automatic risk flags are not confirmed visual artifacts. No automatic submission decision.
- Wall times include shared-load effects; sampled process memory is not a CUDA allocator peak.
