# LPHC fixed-seed horizon closure

Status: 30s_60s_evidence_available

No new generation or seed search. No automatic acceptance verdict.

| Cohort | Method | Imaging | Quality | Subject | Dynamic |
|---|---|---:|---:|---:|---:|
| v219 | sf_fifo21 | 0.690480 | 81.617819 | 0.969634 | 0.426042 |
| v219 | ours_correct | 0.694270 | 81.551842 | 0.969252 | 0.413542 |
| v219 | ours_random | 0.693825 | 81.736749 | 0.969944 | 0.430208 |
| v219 | pooled_correct | 0.692944 | 81.616263 | 0.969645 | 0.412500 |
| v220 | sf_fifo21 | 0.664866 | 79.380911 | 0.966096 | 0.311458 |
| v220 | ours_correct | 0.667157 | 79.477319 | 0.966811 | 0.307812 |

## Frozen primary
- v219 full/imaging_quality: +0.003790, CI [-0.0007757851526296464, 0.008777501928759729]; includes_zero.
- v220 full/imaging_quality: +0.002290, CI [-0.004212583454391766, 0.008957997553927503]; includes_zero.

## Mechanism controls

| Control | Imaging delta | 95% CI |
|---|---:|---|
| ours_random | +0.000446 | [-0.002870, 0.003650] |
| pooled_correct | +0.001327 | [-0.001292, 0.004266] |

## Descriptive horizon interaction

| Metric | Effect at 60s minus effect at 30s | 95% CI |
|---|---:|---|
| quality_without_dynamic_degree | +0.094277 | [-0.39755359936574997, 0.696085991441222] |
| official_quality_score | +0.162386 | [-0.3399298235007784, 0.6971734696954609] |
| identity_background | +0.000843 | [-0.000670110246023392, 0.002701237451979047] |
| temporal_mechanics | +0.001015 | [-0.002956646350217257, 0.005666327693820669] |
| semantic_alignment | -0.001871 | [-0.0066120375576914185, 0.002707202714324616] |
| visual_quality | -0.001130 | [-0.007495387352989453, 0.005363544743430475] |
| dynamic_degree | +0.008854 | [-0.028124999999999997, 0.044791666666666674] |
| subject_consistency | +0.001097 | [-0.0011045089507071037, 0.003928853857966836] |
| imaging_quality | -0.001500 | [-0.008422363301545083, 0.005105914519963942] |

## Automatic risks
- v219: 10/64 flagged, automatic pass=False; not confirmed failures.
- v220: 18/64 flagged, automatic pass=False; not confirmed failures.

Requested diagnostic pairs: 0; not a user study.

- Compact receipt checks, not a new media or raw VBench evaluation.
- Same 64 prompts and one seed per prompt across horizons; not 128 independent prompts.
- Same seed does not guarantee identical 30s/60s rollout prefixes.
- Cross-horizon interactions are descriptive and cannot replace the frozen endpoint.
- Keep random/pooled controls; do not select a best seed per method or omit unfavorable samples.
- Raw metrics use native 0-1 scale; official Quality uses 0-100. Core-9 is not full Total/Semantic.
- Automatic risk flags are not confirmed visual artifacts. No automatic submission decision.
- Wall times include shared-load effects; sampled process memory is not a CUDA allocator peak.
