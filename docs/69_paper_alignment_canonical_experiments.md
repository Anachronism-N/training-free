# Paper alignment and canonical experiments

Status: implementation-ready, pending server generation and review
Date: 2026-07-22

This document is the current decision record for the paper direction and the
first canonical experiment package. It supersedes older scripts as the default
entry point, but it does not erase negative results or previous designs.

## 1. Research target and task priority

The target is a top-tier conference paper on **training-free long-horizon video
extrapolation**.

The task order is fixed:

1. **Primary: one prompt, direct long-video extrapolation.** Preserve subject
   identity, characteristic objects, background layout, motion, and camera
   continuity as autoregressive generation moves far beyond the native horizon.
2. **Secondary: prompt/scene switching.** Support A-B-A return and segmented
   control without contaminating the current scene. A long prompt may be
   represented as multiple controlled segments, but this task must not replace
   the primary continuous-generation result.

The first gate is deliberately small: three complex prompts, one seed, and 30
seconds per video. We inspect videos before computing metrics. Larger prompt
suites, more seeds, and 60/120-second experiments begin only after this gate.
The three-prompt scores are engineering triage, not a paper benchmark result or
a statistically supported claim.

## 2. Current idea

Working name: **Scope-Conditioned Evidence-Gated Historical Recall**.

The base generator is Self-Forcing. Our vendored inference contains an optional
memory bridge, but disabling it follows the native SF working-cache/output path.
The method maintains a bounded side archive of clean, pre-RoPE K/V frames and
computes a separate historical-memory attention output. The native working cache
remains intact.
Historical information is fused only when the current query provides sufficient
evidence; otherwise the path abstains and returns native SF output.

There are two explicit scopes:

- `intra_episode`: for one-prompt long generation, retrieve sufficiently old
  frames from the current continuous episode while excluding recent frames.
- `cross_episode`: for A-B-A return, retrieve from an admitted older episode,
  reject the immediately previous scene, and preserve scene separation.

The paper-facing question is not merely "which tokens should a cache retain?"
It is:

> When is an additional historical intervention useful, which history supports
> it, and where in the model can it be applied without overriding native motion?

The current implementation factorizes evidence along the following axes, but
only some are active decisions today:

| Axis | Current implementation | Claim status |
|---|---|---|
| Scope | Explicit continuity vs return recall | Core method candidate |
| History/content | Online Q-K retrieval, age/episode constraints, confidence and abstention | Core method candidate |
| Head | Online role evidence from K/V persistence and query stability; all-head control retained | Candidate; keep only if video and logs support it |
| Layer | Conservative fixed active band, with per-layer diagnostics | Not yet a classifier or contribution |
| Denoising timestep | Every call is tagged by `attention_call_index`; per-call diagnostics are emitted | Not yet routed or claimed |
| CFG branch | Not used in the current few-step SF path | Explicitly excluded from the main claim |

This staging is intentional. The first server run must show whether layer,
head, and denoising-call behavior is meaningfully heterogeneous. We will not add
weights or a classifier solely to make the method appear more complex.

## 3. Distinction from validated related work

The following comparison is a claim boundary, not an attempt to minimize prior
work.

| Work | Validated idea used or studied | Difference in our current implementation |
|---|---|---|
| [Self-Forcing](https://github.com/guandeh17/Self-Forcing) | Autoregressive few-step video diffusion base | Base model and native control; not our contribution |
| [Pyramid-Forcing](https://github.com/if-lab-pku/Pyramid-Forcing) | Offline head profiling and fixed per-head native KV retention policies | PF changes native cache composition. Ours uses online intervention evidence and a separate recall branch. PF is run from its official config and reported as a strong baseline. |
| [Echo-Forcing](https://github.com/mingqiangWu/Echo-Forcing) | Hierarchical temporal memory, scene recall, and difference-aware decay | Echo is used as an official long-video and scene-recall baseline. Ours uses constrained Q-K historical readout on native SF and does not claim Echo's scene-memory mechanisms. |
| [LongLive-RAG](https://github.com/qixinhu11/LongLive-RAG) | Temporary retrieval view over offloaded history | Supports the general motivation for a separate recall path; our scope/admission/fusion implementation is independent. |
| [MemRoPE](https://github.com/YoungRaeKimm/MemRoPE) | Position-safe memory reuse | Motivates explicit positional auditing. The current main cell uses no additional memory position transform and records this choice. |
| [SWIFT](https://github.com/ShanwenTan/SWIFT) | Prompt-adaptive semantic injection/cache | High-overlap related work that constrains broad semantic-memory claims; see `docs/65_swift_collision_audit.md`. |

Additional provenance and license details remain in
`docs/64_related_work_code_provenance_and_claims.md`.

Safe provisional contribution language:

1. A scope-conditioned formulation that separates continuity recall from scene
   return rather than treating all old context identically.
2. A fail-closed historical intervention on native SF with explicit episode,
   age, retrieval, alignment, and optional online head evidence.
3. A diagnostic protocol that measures where the intervention acts across
   layer, head, and denoising calls and drops unsupported routing axes.

These are hypotheses until the canonical experiments establish quality and
causal evidence. We must not claim that generic retrieval, head-aware caching,
scene memory, or memory decay originated in this project.

## 4. Required models and locations

Each vendored implementation resolves the Wan model relative to its own working
directory. Place real directories or symbolic links at all three locations:

```text
third_party/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Pyramid-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Echo-Forcing/wan_models/Wan2.1-T2V-1.3B/
```

Place the Self-Forcing DMD checkpoint at:

```text
third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Echo-Forcing/checkpoints/self_forcing_dmd.pt
```

The files may point to a single shared model/checkpoint store. All cells must use
the same checkpoint content; record checksums on the server if separate copies
are used. Pyramid-Forcing also requires its tracked head-label/config assets,
including `configs/head_configs/best_labels.csv` as referenced by its YAML.

Metric dependencies are loaded by the evaluator:

- DINOv2 ViT-L/14 through `torch.hub` or the torchvision fallback.
- OpenCLIP ViT-L/14. Set `CLIP_CHECKPOINT` to a local file if the server is
  offline; its default is `~/.cache/clip/ViT-L-14.pt`.
- torchvision RAFT-Large, LPIPS-VGG, and InsightFace `buffalo_l`.
- VBench-Long under `VBENCH_ROOT`, defaulting to
  `$ROOT/../research_sprint/bench_baselines/VBench`.

## 5. Canonical generation commands

The scripts default to the existing server root and `longlive` conda
environment. Override `REPO_ROOT`, `CONDA_SH`, or `CONDA_ENV` when needed.

### 5.1 Primary single-prompt matrix

Five GPUs in parallel:

```bash
bash scripts/run_paper_single_prompt_30s.sh 0 1 2 3 4
```

One GPU sequentially:

```bash
PARALLEL=0 bash scripts/run_paper_single_prompt_30s.sh 0 0 0 0 0
```

Cells:

```text
sf_native
sf_pyramid_forcing
sf_echo_forcing
ours_all_heads
ours_role
```

The prompt file is `prompts/hrem_v2_single_long_complex_3.txt`; 120 latent
frames correspond to approximately 30 seconds at the current decoding rate.
Echo receives `prompts/paper_single_long_echo_3.txt`, which contains the same
wording with punctuation semicolons changed to periods because Echo reserves
the first semicolon as its subtitle delimiter. The preflight validator rejects
truncated/control-bearing plain prompts.
Outputs are immutable reusable baseline assets under
`runs/paper_single_30s_s0/`. Do not regenerate a completed baseline merely for
a later method ablation. Each cell writes `run_metadata.txt`; skip logic checks
the prompt hash, config fingerprint, seed, and frame count. Our cells additionally
require the current method commit, while frozen baseline cells can be reused
across documentation-only commits.

### 5.2 Secondary A-B-A scene-switch matrix

Four GPUs in parallel:

```bash
bash scripts/run_paper_scene_switch_30s.sh 0 1 2 3
```

One GPU sequentially:

```bash
PARALLEL=0 bash scripts/run_paper_scene_switch_30s.sh 0 0 0 0
```

Cells:

```text
sf_segmented_reset
sf_echo_forcing
ours_all_heads
ours_role
```

`sf_segmented_reset` is an SF control using this repository's equal-third
`||` prompt scheduler and native-cache reset. It must not be described as an
upstream Self-Forcing feature. It reads `prompts/paper_scene_switch_sf_3.txt`.
Echo uses its official duration/transition syntax from
`prompts/paper_scene_switch_echo_3.txt`, with equivalent 10s A, 10s B, and 10s
A-return content. The preflight strips Echo markers and requires every scene
description to exactly match the SF file. Neither file contains subtitle labels,
so no baseline receives rendered text overlays.

## 6. Review-first evaluation protocol

Both generation scripts create `blind_review/` automatically:

```text
manifest_public.json   reviewer-visible prompts and randomized labels
scorecard.csv          scores to freeze before unblinding
key_private.json       method mapping; keep hidden until scores are final
prompt_00/A.mp4 ...    hard links (relative symlink fallback), not copied videos
```

For every candidate, review the full video and record:

- subject face, clothing, geometry, and characteristic object identity;
- background/layout drift and irreversible scene contamination;
- motion magnitude, naturalness, freezing, looping, and duplicate subjects;
- camera direction and continuity;
- first visible failure time;
- prompt alignment and overall rank.

After freezing the scorecard, run:

```bash
HUMAN_REVIEW_DONE=1 bash scripts/run_paper_metrics.sh single 0
HUMAN_REVIEW_DONE=1 bash scripts/run_paper_metrics.sh scene 0
```

Set `RUN_VBENCH=0` for a quick comprehensive/return-only pass. The default also
runs the six VBench-Long dimensions: subject consistency, background
consistency, aesthetic quality, imaging quality, motion smoothness, and dynamic
degree.

The comprehensive evaluator additionally reports DINO consistency and drift,
RAFT motion smoothness, ArcFace identity, LPIPS flicker, CLIP alignment,
background consistency, and loop/repetition evidence. For A-B-A videos,
`aba_return.json` separately compares A1-A2 return similarity against A-B scene
separation.

## 7. Debug evidence to return

Our cells emit both human-readable logs and JSONL traces. Each accepted recall
records:

- recall scope, current/allowed/selected episodes, selected frame ages;
- layer and `attention_call_index`;
- selected indices, confidence, retrieval margin, and retrieval entropy;
- role evidence, per-head gate, active fraction, and calibration validity;
- native-memory alignment, effective weight, and delta/native RMS;
- archive checksums and boundary preservation.

`scripts/analyze_hrem_v2_debug.py` creates:

```text
per_layer
per_attention_call
per_layer_attention_call
```

These tables determine whether layer or timestep routing is worth implementing.
Useful server checks:

```bash
grep -E '\[HREMv2\]' runs/paper_single_30s_s0/logs/ours_role.log | tail -n 200
grep -E 'scene|recall|candidate|selected' runs/paper_scene_30s_s0/logs/sf_echo_forcing.log | tail -n 200
```

Return the following package for the next review:

```text
blind_review/scorecard.csv
metrics/**/*.json
traces/*.jsonl
traces/*_diagnosis.json
logs/*.log
*/run_metadata.txt
```

Do not send generated videos through Git; keep their paths stable on the server.

## 8. Go/no-go decisions after the first run

### Primary method gate

Continue the historical-recall direction only if `ours_all_heads` or
`ours_role` visibly improves long-range identity/layout in at least two of three
single prompts without systematic motion freezing, ghosting, or duplicate
subjects. Metrics should agree in direction on subject/ID/drift while dynamic
degree and smoothness do not collapse.

### Head-routing gate

- Keep the head-aware story if `ours_role` is at least as good as
  `ours_all_heads`, changes a nontrivial subset of heads, and active identities
  are reasonably stable across denoising calls.
- If `ours_role` is worse or the gate is nearly all-on/all-off, use the result as
  a negative ablation and simplify the method. Do not hide it or retune after
  looking at only one prompt.

### Layer/timestep gate

- Add layer or denoising-call routing only if the factorized diagnosis shows a
  stable, repeated intervention/evidence difference across prompts, followed by
  a controlled quality ablation. Delta RMS alone is not a quality metric.
- If differences are weak or inconsistent, retain a conservative fixed band and
  report the diagnostic rather than claiming a new classifier.

### Strong-baseline gate

PF and Echo quality must be attributed to their methods. If our native-SF method
is far behind both, the next iteration should first improve the memory
intervention itself. Combining our method with PF can be a later compatibility
experiment, but it cannot serve as evidence that our native-SF contribution is
effective.

## 9. Academic-integrity rules

1. Keep original repositories, paper names, URLs, licenses, and exact borrowed
   mechanisms in the provenance ledger.
2. Separate inspiration, reimplementation, direct vendored code, and our own
   modifications in both code comments and paper text.
3. Do not rename PF's head classification, Echo's scene memory, or another
   method's decay/retrieval rule and present it as a new algorithm.
4. Report official baseline configurations and local deviations. In particular,
   label our scene scheduler as local and disable local HeadRole prototypes in
   the Echo baseline.
5. Preserve negative ablations and raw logs. Claims are promoted only after
   video, metric, and mechanism evidence agree.

## 10. Provisional paper story

Provisional title:

> **Evidence-Gated Historical Recall for Training-Free Long-Horizon Video Generation**

Story outline:

1. Native long autoregressive generation forgets old identity and layout, while
   blindly retaining or reinjecting history can suppress motion or contaminate a
   new scene.
2. Long-video history has different semantic scopes: continuous identity support
   and explicit scene return should not share an unconstrained recall rule.
3. A bounded, separate, fail-closed recall intervention can recover supported
   history while preserving the native working cache.
4. Online intervention diagnostics reveal which heads/layers/denoising calls can
   use history safely; unsupported axes are removed rather than assumed.
5. The method is evaluated first on single-prompt long extrapolation, then on
   scene switching, against native SF and strong PF/Echo references.

This story remains provisional until the first canonical server package is
reviewed.
