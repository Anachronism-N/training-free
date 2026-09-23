# v224 final evidence

Write the narrow method/results draft now; no new generation or seed search.
Core ablation does not establish alpha=0.02 or phase-0 as a quality optimum. Keep all controls.
Nine raw dimensions plus Quality are exported; not full Semantic/Total. Temporal style duplicates alignment in this evaluator.
Missing source artifacts: []

Raw saved metric arithmetic checked; no local model/video rerun. Generation-to-evaluation receipts are bound by their recorded hashes. Raw traces and videos are not uploaded; local checks do not verify their actual bytes.

## Reassessment differences (full video)

| Metric | Mean difference of paired effects | Maximum absolute per-prompt difference |
|---|---:|---:|
| quality_without_dynamic_degree | +0.004271 | 0.052454 |
| official_quality_score | +0.036322 | 1.078095 |
| identity_background | +0.004914 | 0.072702 |
| temporal_mechanics | +0.000059 | 0.005703 |
| semantic_alignment | -0.005563 | 0.129927 |
| visual_quality | +0.007160 | 0.134299 |
| dynamic_degree | +0.416667 | 13.333333 |
| subject_consistency | +0.001438 | 0.070384 |
| imaging_quality | -0.001712 | 0.137178 |

All differences above use a 0-100 scale. These are repeated-score diagnostics, not treatment effects.
Run audit_v224_reused_clips.py on existing server files to separate changed preprocessing from evaluator variation.
