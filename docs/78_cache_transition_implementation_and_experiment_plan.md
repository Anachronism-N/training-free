# Trust-Conditioned Cache Transition: Implementation and Server Plan

> Status: code complete; static checks passed; GPU inference not run on the
> current machine.
>
> Depends on: `docs/77_commit_forcing_v76_screen_results.md`

## 1. Decision after v76

The v76 result rejects multiscale/trajectory Commit Forcing, not persistent
multiscale attention memory in general. Commit Forcing intervenes through an
occasional correction pass. PF and Echo instead keep long-term state inside
native self-attention on every forward pass.

The next paper line therefore uses PF as the strongest implementation base and
tests a different research question:

> Long autoregressive generation can fail when an unreliable generated block is
> synchronously promoted into persistent attention memory. Can training-free,
> head-wise control of when middle memory changes improve continuity without
> losing PF's identity retention?

PF and Echo must remain explicitly cited. Sink/middle/recent composition,
cyclic/stride/merge, rolling sinks, and query-conditioned recall are not claimed
as our inventions.

## 2. Implemented method

The v1 implementation controls only PF's middle cache:

- **Sink/anchor:** unchanged from PF.
- **Recent cache:** always updated through PF's original path.
- **Middle candidate:** the clean K/V block that PF would normally send to its
  cyclic, stride, or merge strategy.
- **Active middle memory:** the existing strategy state. A rejected candidate
  leaves it unchanged.

For each layer/head, the controller pools normalized K and V descriptors. It
measures:

1. `shock`: cosine distance between the clean candidate and the last committed
   descriptor.
2. `denoise_disagreement`: cosine distance between the last noisy-pass
   descriptor and the clean descriptor for the same AR block.
3. `reliability = exp(-w_s * shock - w_d * denoise_disagreement)`.
4. `novelty = shock`.
5. `age`: AR blocks since the last accepted middle update.

The controller supports:

- `audit`: compute/log diagnostics, accept every update.
- `gate`: reject low-reliability or low-novelty candidates.
- `stagger`: update only a deterministic subset of heads per block.
- `full`: combine gate, stagger, a per-layer/head budget, warmup, and forced
  max-age refresh.

This is an active/candidate state machine at the descriptor and write-decision
level. It does not keep a second full K/V bank and does not yet blend two
attention outputs.

## 3. Code map

- `third_party/Pyramid-Forcing/pyramidkv/transition.py`
  - transition state, metrics, decisions, JSONL trace.
- `third_party/Pyramid-Forcing/pyramidkv/adaptive_cache.py`
  - intercepts only clean middle-strategy writes.
  - disables incompatible all-head C++ fast paths when transition control is on.
- `third_party/Pyramid-Forcing/pipeline/pyramidkv_config.py`
  - configuration schema and pipeline mapping.
- `third_party/Pyramid-Forcing/pipeline/causal_inference.py`
  - primary few-step PF inference integration.
- `third_party/Pyramid-Forcing/pipeline/causal_diffusion_inference.py`
  - alternate diffusion inference integration.
- `third_party/Pyramid-Forcing/inference.py`
  - CLI controls.
- `scripts/summarize_cache_transition_trace.py`
  - strict trace validation and aggregate diagnostics.

Commit Forcing closure controls are in:

- `src/lifecycle_kv/commit_forcing.py`
  - block interval, global strength, per-timestep strength, and ramp.
- `third_party/Self-Forcing/pipeline/causal_inference.py`
  - interpolation between native and corrected noisy inputs.

All new behavior is disabled by default. Original PF and v74/v76 configurations
retain their previous behavior.

## 4. Required models

Expected defaults:

```text
third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Echo-Forcing/checkpoints/self_forcing_dmd.pt
```

All three should resolve to the same Self-Forcing generator checkpoint unless a
method's upstream instructions explicitly require a different artifact. The run
scripts fail before launch if any repository, config, prompt file, or checkpoint
is missing.

## 5. Execution order

Run from the repository root on the 16-H20 server.

### 5.1 Pull and make scripts executable

```bash
git pull --ff-only
chmod +x scripts/run_v77_commit_closure_16gpu.sh
chmod +x scripts/run_v78_cache_transition_16gpu.sh
chmod +x scripts/postprocess_v77_v78.sh
```

### 5.2 Main-line smoke test first

```bash
bash scripts/run_v78_cache_transition_16gpu.sh smoke
```

This launches PF baseline, audit, gate, and full on four GPUs with the smoke
prompt suite.

Do not start the 16-cell screen unless:

- every method produces the expected video count;
- every transition trace contains all 30 layers;
- `pf_transition_audit` has acceptance rate `1.0`;
- no trace has non-finite values or per-head length mismatches;
- logs contain `[CacheTransition]` summaries and no fallback exception.

### 5.3 Main 16-GPU screen

```bash
bash scripts/run_v78_cache_transition_16gpu.sh screen
```

The screen contains:

- SF, official PF, and dedicated-GPU Echo baselines;
- PF audit;
- gate thresholds `0.45/0.55/0.65`;
- novelty on/off;
- half and one-third stagger schedules;
- full variants over reliability, max age, CFG branch, and commit budget.

Echo must run on an otherwise empty H20. Its earlier failure happened when the
GPU already held about 50 GiB from another process.

### 5.4 Commit Forcing closure

This is lower priority than v78. Run it to close the alternatives listed in
docs/77:

```bash
bash scripts/run_v77_commit_closure_16gpu.sh smoke
bash scripts/run_v77_commit_closure_16gpu.sh screen
```

It tests unreliable-block triggers, every-2/every-3-block correction, global
correction strength, t500-only correction, weaker t250 correction, ramp-in, and
one combined low-frequency/low-strength cell.

Do not expand Commit Forcing further unless a v77 cell clearly beats v74 on
both identity and temporal jump.

## 6. Human review and metrics

Human review must be frozen before metrics:

1. Create randomized blind-review material with the existing
   `scripts/prepare_blind_review.py`.
2. Score ID persistence, style drift, darkening, motion freezing, jumps,
   duplicated subjects, and high-motion artifacts.
3. Freeze `blind_review/scorecard.csv`.
4. Run:

```bash
HUMAN_REVIEW_DONE=1 \
RUN_ROOT=/absolute/path/to/runs/v78_cache_transition_screen \
TRACE_KIND=transition \
bash scripts/postprocess_v77_v78.sh
```

For v77, set `TRACE_KIND=commit` and use its run root.

The postprocessor checks video completeness, summarizes traces, computes the
comprehensive DINO-based metrics and temporal jump, then runs these VBench-Long
dimensions:

```text
subject_consistency
background_consistency
aesthetic_quality
imaging_quality
motion_smoothness
dynamic_degree
```

## 7. Trace interpretation

Each `cache_transition` event records:

- layer, CFG branch, block, PF head label;
- per-head commit mask and rejection reason;
- reliability, shock, denoise disagreement, novelty, and pre-decision age.

Important failure patterns:

| Observation | Interpretation | Next action |
|---|---|---|
| Audit acceptance below 100% | implementation bug | stop |
| Gate acceptance above 95% | controller is effectively PF | raise threshold or novelty |
| Gate acceptance below 5% | controller is freezing middle memory | lower threshold |
| `low_reliability` dominates with high denoise disagreement | clean/noisy state is unstable | lower denoise weight |
| `low_reliability` dominates with high shock only | generated state changes quickly | compare motion-heavy prompts |
| `low_novelty` dominates | middle writes are redundant | increase min interval before stronger gating |
| `forced_max_age` dominates | gate is too strict | lower threshold or increase max age |
| Stable heads update much more than oscillating heads | role behavior may be inverted | inspect acceptance by label |
| Good ID but worse jump | updates still change too synchronously | reduce budget or increase stagger period |
| Good jump but frozen motion | middle state is too stale | increase budget or reduce max age |

The target screen region is roughly 20-80% acceptance, with no single forced
fallback dominating. This is a diagnostic range, not a claimed optimum.

## 8. Go/no-go criteria

Promote the line to multi-seed confirmation only if at least one controlled PF
cell satisfies all of:

1. no visible regression in 30-second ID retention relative to PF;
2. lower temporal jump or fewer human-observed transitions than PF;
3. no meaningful loss in dynamic degree/motion;
4. audit is metric-equivalent to PF;
5. trace shows a nontrivial intervention, not near-0% or near-100% acceptance.

If the full controller fails but stagger-only improves jumps, simplify the paper
line to asynchronous cache transition. If gate-only helps but staggering does
not, focus on reliability-gated state promotion. If neither improves PF, retain
PF/EF as baselines and do not claim this mechanism.

## 9. Known limitations before server validation

- The current machine has no PyTorch/GPU runtime, so tensor unit tests and real
  inference have not been executed here.
- Transition cells intentionally disable MegaCache and all-head C++ strategy
  updates because those paths cannot consume per-head commit masks.
- Descriptor pooling adds clean/noisy-pass overhead and a small GPU-to-CPU sync
  at each clean decision.
- v1 uses K/V descriptors, not query-conditioned transition risk.
- A rejected candidate is discarded; delayed full-candidate handoff is not yet
  implemented.

These points must be checked from server logs before interpreting video quality.
