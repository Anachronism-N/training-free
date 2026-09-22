# v219 results and v220 decision

Date: 2026-09-22. Results upload: `e31c8527`.
Branch: `worktree-v210-lphc`. v220 is still running; do not retune its frozen method.

## Evidence and checks

- v219: `artifacts/experiment_results/v219_mechanism64_38a06347_vbench_45e79ec1`.
- v216: `artifacts/experiment_results/v216_confirmation80_64f5c72a_vbench_45e79ec1`.
- v219 used 64 fixed MovieGen prompts, approximately 30 seconds, seed `21600 + source_index`, and four methods (256 videos).
- The frozen choice remains `headwise_correct`, phase 0 only, correction strength 0.02; the primary endpoint remains full-video imaging quality.
- `export_lphc_paper_evidence.load_bundle(..., "v219")` passed 261 bound-file checks, including generation identities and trace-audit receipts. All 477 files listed in the publication manifest passed hash checks.
- Both v219 `sf_upstream_gate` and `gate0` passed and are bound to this input manifest. This supports baseline/zero-correction parity for the checked cases, not every possible execution or PF runtime parity.
- Cross-campaign parent selection, development evidence, evaluator, model/checkpoint, prompt file, inference runtime hashes, method specifications and config hashes matched. Random differs only in retrieval mode; pooled differs only in descriptor mode.
- `summarize_v216_closure.summarize` checked 336 uploaded files. v216 still lacks its two compact gate receipts (`decisions/sf_upstream_gate.json`, `decisions/gate0.json`). Export the originals from the server; do not regenerate videos or synthesize passing receipts.
- These are compact-artifact and arithmetic checks, not a local rerun of inference, raw VBench evaluation, or visual review.

## v219 alone

All displayed metrics below are on a 0-100 scale. Quality is the implemented official Quality aggregate, not Semantic or Total.

| Method | Imaging | Quality | Subject consistency | Dynamic degree |
|---|---:|---:|---:|---:|
| SF FIFO21 | 69.0480 | 81.6178 | 96.9634 | 42.6042 |
| LPHC headwise | 69.4270 | 81.5518 | 96.9252 | 41.3542 |
| LPHC random history | 69.3825 | 81.7367 | 96.9944 | 43.0208 |
| LPHC pooled descriptor | 69.2944 | 81.6163 | 96.9645 | 41.2500 |

| Paired contrast | Imaging difference | Bootstrap 95% CI |
|---|---:|---|
| Headwise - SF | +0.3790 | [-0.0776, +0.8778] |
| Headwise - random | +0.0446 | [-0.2870, +0.3650] |
| Headwise - pooled | +0.1327 | [-0.1292, +0.4266] |

The direction of the SF imaging comparison repeats, but all three single-seed intervals include zero. Imaging improves on 36/64 prompts. Content retrieval and the head-preserving descriptor have not demonstrated an independent advantage in this replication. This is lack of superiority evidence, not proof that the controls are equivalent.

Ours - SF Quality is -0.0660 points, CI [-0.2799, +0.1422]. Dynamic degree is -1.2500 points, CI [-3.2292, +0.8333]. Neither overall-quality nor motion improvement is established. Subject consistency also does not support an identity-improvement claim.

## Same-prompt, two-seed aggregation

Use only the 64 source indices shared by v216 and v219. For each prompt, average its two paired Ours-minus-SF differences; bootstrap these 64 prompt averages using the existing `export_lphc_paper_evidence.joint_seeds` implementation. Do not treat 128 prompt-seed pairs as 128 independent prompts, and do not combine the entire 80-prompt v216 cohort with 64 v219 prompts as 144 independent samples.

| Full-video endpoint | v216 on same 64 | v219 on same 64 | Two-seed mean | Prompt-cluster 95% CI |
|---|---:|---:|---:|---|
| Imaging quality | +0.2349 | +0.3790 | +0.3069 | [+0.0388, +0.6109] |
| Official Quality | +0.0428 | -0.0660 | -0.0116 | [-0.1525, +0.1358] |
| Subject consistency | -0.0130 | -0.0382 | -0.0256 | [-0.1047, +0.0409] |
| Dynamic degree | -0.5208 | -1.2500 | -0.8854 | [-2.6563, +0.8333] |

This gives a positive, modest imaging signal for the frozen method across these two seeds. The interval excludes zero for this prompt-cluster analysis; it does not establish significance for either individual seed, generalize over arbitrary seeds, certify subjective visibility, or establish overall superiority. The missing v216 gate receipts remain an evidence-completeness issue and are not silently waived by this arithmetic.

The secondary internal `visual_quality` composite also has a positive two-seed interval, but must not replace the frozen imaging endpoint or be presented as a new official VBench score. There is no v216 matched random/pooled arm, so this aggregation cannot establish those mechanisms.

## Long-time behavior, risks and costs

v219 imaging differences are approximately +0.0031 points in the early half and +0.7079 in the late half. The within-prompt late-minus-early advantage is +0.7048, CI [-0.1407, +1.5894]. This is a descriptive reason to finish v220, not evidence that late improvement is already established or a new primary endpoint.

The automatic temporal guard flags 10/64 Ours videos against SF, above its allowed count of two:

- Late-motion-collapse flags: source 21, 33, 89, 122.
- Temporal-discontinuity flags: source 26, 85.
- Edge-density flags: source 50, 84, 98, 106.

These are Farneback/appearance heuristics, not ten confirmed artifacts. Do not claim that all videos passed safety or silently delete flagged prompts. The existing report limits new review to the two diagnostic pairs at sources 84 and 50; this is not a preference study or a complete motion-risk audit. Keep the other flags unresolved unless inspected. Videos are under:

```text
/apdcephfs_gy2/share_302533218/cedricnie/v219_runs/v219_38a06347_mechanism64/evaluation/vbench_comparison/published/
```

The two selected pair filenames are `000042-0.mp4` (source 84) and `000025-0.mp4` (source 50), under both `ours_correct/` and `sf_fifo21/`.

Maximum sampled process memory is 33,244 MiB for Ours versus 26,776 MiB for SF (+24.2%). Wall-time summaries are inconsistent as speed estimators: ratio of method medians is about 1.057, while the median paired ratio is 0.870. These different statistics need not agree under heterogeneous/shared load. Do not claim acceleration; neither isolates steady-state DiT compute.

## Paper decision

Start the method section, related work, experimental protocol and narrow results draft now. Do not yet freeze a strong efficacy narrative or claim that three independent innovations have been validated.

The defensible current method is **bounded historical readout correction while preserving the native local cache**. Head-preserving retrieval and phase-0 application are implementation choices with incomplete independent causal evidence. v219 neither validates dynamic head-by-phase routing nor proves all earlier routing approaches ineffective; it did not test them.

A supportable result statement is: on the shared 64-prompt, two-seed 30-second evaluation, LPHC provides a small imaging-quality gain, while overall Quality, identity and motion gains are not established. Include the random and pooled controls. Four-page length is a reason to narrow claims, not omit contrary controls.

## Next actions

1. Finish the already running v220 (same frozen method, SF/Ours, 64 prompts, approximately 60 seconds). No new parameter sweep or rerun is requested by this analysis.
2. Keep the frozen full-video imaging endpoint; report Quality, subject/background consistency, dynamic degree, early/late behavior, automatic-risk counts and costs alongside it. Do not promote late-half imaging to primary after seeing the result.
3. Export the two missing v216 gate receipts and, when practical, inspect only the existing two v219 diagnostic pairs. Do not launch broad manual review or regenerate the 256 completed videos.
4. If v220 supports imaging benefit without unacceptable degradation, finalize a narrow local-preserving correction paper. It still cannot establish retrieval superiority without matched controls at that horizon.
5. If v220 is flat or worse, do not immediately add CF/DeepForcing runs to imply stronger method evidence. First decide between a limited-effect short paper and further method development. A higher mean from random in this table is not permission to relabel it as the confirmed method.

No inference code, runtime configuration, frozen selection or v220 job was changed for this review.
