# v78 Method Summary and Historical AMA Veil Head Perturbation Experiments

> Date: 2026-07-23
> Context: `docs/79_cache_transition_v78_screen_results.md` (v78 screen results)
> Historical: `docs/37` (AMA audit), `docs/38` (darkening diagnosis), `docs/40-41` (head ablation), `docs/48-50` (QACP perturbation sensitivity)

## 1. v78 Cache Transition: current method

### 1.1 Core idea

Trust-Conditioned Cache Transition controls **when** PF's middle cache
accepts new states, rather than adding extra forward passes or side-branch
fusion. PF already maintains identity best (DINO 0.832, ID retained
throughout 30s). The question is whether training-free, head-wise control
of middle memory writes can improve temporal smoothness without losing PF's
identity retention.

### 1.2 PF cache structure (unchanged by v78)

```
sink/anchor (immutable) + middle (cyclic/stride/merge) + recent (21-frame window)
```

v78 only intercepts the middle write decision. Sink/anchor and recent cache
follow PF's original path.

### 1.3 Reliability signal

For each layer/head, the controller pools normalized K and V descriptors
and measures:

```text
shock = cosine_distance(candidate, last_committed)
denoise_disagreement = cosine_distance(noisy_descriptor, clean_descriptor)
reliability = exp(-w_s * shock - w_d * denoise_disagreement)
```

No additional model forward pass is needed — the descriptors are computed
from the existing clean and noisy predictions that SF/PF already generates.

### 1.4 Controller modes

| Mode | Behavior |
|---|---|
| audit | Log diagnostics, accept every update (should reproduce PF) |
| gate | Reject low-reliability or low-novelty candidates |
| stagger | Update only a deterministic subset of heads per block |
| full | Combine gate + stagger + per-layer budget + warmup + forced max-age refresh |

### 1.5 Key result

**full_budget075_p1** (full controller, 75% budget, 1 origin):

| Metric | sf_native | pf_official | full_budget075_p1 |
|---|---:|---:|---:|
| DINO | 0.781 | 0.832 | 0.830 (≈PF) |
| min_DINO | 0.670 | 0.795 | 0.767 |
| drift | -0.0047 | -0.0017 | -0.0015 |
| temporal_jump | 3.349 | 1.723 | **1.645** (-4.5% vs PF) |
| acceptance | N/A | 100% | 58% |

All 5 predeclared gates passed. Zero extra compute overhead.

### 1.6 Why this works where previous approaches failed

| Approach | What it did | Why it failed |
|---|---|---|
| LifeCache-v3 | Side-branch fusion at 5-7% weight | Too weak to change generation path |
| Commit Forcing v74 | Extra correction forward with fresh noise | Delayed collapse but caused jumps, freeze, style shift |
| Commit Forcing v76 | Trajectory re-noising + multiscale bank | 13× weaker correction → below native |
| **Cache Transition v78** | **Control PF middle write decisions** | **Matches PF, improves jump, zero overhead** |

The key insight: PF already works. Instead of adding a separate mechanism
(extra forwards, side branches), v78 works *within* PF's existing cache
update path, only controlling the write decision.

## 2. Historical AMA experiments on PF veil heads

### 2.1 PF head label definitions

PF assigns every (layer, head) one of three labels based on offline
attention-pattern analysis:

| Label | Name | Count | Fraction | PF cache strategy |
|---:|---|---:|---:|---|
| -1 | Wave (oscillating) | 156 | 43.3% | sink1 + cyclic(period=6, cap=4) + recent4 |
| 1 | Anchor (stable) | 172 | 47.8% | sink3 + stride(interval=6, cap=4) + recent4 |
| 2 | Veil (stable-sparse) | 32 | 8.9% | sink3 + merge(patch=2, cap=4) + recent4 |

### 2.2 Experiment 1: AMA head audit (v33, docs/37)

**Question:** Do AMA's identity/motion/background head labels transfer to PF?

**Method:** Re-audited all 360 PF heads using AMA's |QK| proxy. Tested
head-policy routing (all vs PF-stable vs explicit) on SF and CF backbones,
layer 29, gate 0.02-0.05, single prompt and 3-prompt.

**Key finding:** AMA's |QK| proxy classified **all 360/360 heads as
"identity"**. This caused HRMR to enforce anchor K-scaling on every head,
locking background and limiting motion.

**Result:** PF-stable routing regressed SF DINO from 0.8152 to 0.7981.

**Conclusion:** AMA's head labels do NOT transfer to PF. Static wrong
classification masks or reverses gains.

### 2.3 Experiment 2: PF head-category × depth ablation (v35/v36, docs/40-41)

**Question:** Which PF head categories drive identity/darkening gains?

**Method:** Ablated Wave/Anchor/Veil categories × early/middle/late depth
× mean/variance moment transport. Corrected by the AdaptiveKVCache.reset()
fix (docs/41) which revealed the old 3-prompt baseline was contaminated.

**Key structural finding:** Adjacent-layer label retention is only **46.3%**.
A fixed head index changes label up to 20 times across depth. A fixed head
number cannot be stably interpreted as identity/motion/background.

**Head-category ablation (single prompt):**

| Route | Composite | DINO | Luma Q4/Q1 |
|---|---:|---:|---:|
| PF | 0.4728 | 0.6826 | +1.0% |
| all heads | 0.5048 | 0.7025 | -11.7% |
| Wave only | 0.5115 | 0.7141 | -12.5% |
| Anchor only | 0.5106 | 0.6971 | -17.2% |
| **Veil only** | 0.4925 | 0.6968 | **+1.6%** |

**Conclusion:** Wave/Anchor heads are the joint source of both consistency
gains AND darkening. Veil heads preserve exposure/motion but give small
gains. The "identity-head / motion-head" interpretation got no evidential
support — temporal-pattern labels can be routing variables but are not a
semantic contribution.

### 2.4 Experiment 3: Darkening ("veil") diagnosis (v34, docs/38)

**Question:** Where does the darkening form?

**Method:** Tracked per-block latent/channel mean, std, RMS, and per-frame
RGB/luma via `AR_LATENT_TRACE_PATH` (no tensor modification).

**Finding:** Darkening forms in the **denoised latent**, not only in VAE
decode. SF luma drops 31.8%, CF drops 60.5%. The exposure-correlated
channels differ between SF (ch 15/2/3/10/11) and CF (ch 14/10/8/9/3) and
drift in opposite directions.

**Conclusion:** A single fixed channel-mask or all-channel mean/std
correction cannot work. AMA's historical anti-drift (random latent noise
injection without targeting exposure direction) was a long-running no-op
and was NOT ported.

### 2.5 Experiment 4: Perturbation sensitivity signals (QACP, docs/48-50)

**Question:** Can online perturbation signals classify PF heads into
functional roles?

**Original design (docs/48):**
- Temporal Sensitivity = ‖attention(q, all_history) − attention(q, recent_only)‖
  (perturb = remove distant history)
- Content Specificity = conf_correct − conf_random (perturb = shuffle archive)

**Correction (docs/49-50):** The original content specificity (shuffle then
take max similarity) is **identically zero** — shuffling doesn't change the
max. Replaced with real inference-time hook measuring:

| Signal | Definition | CV (layers 15-20) | Judgment |
|---|---|---:|---|
| prompt_reliance | ‖A_cond − A_uncond‖ / ‖A_uncond‖ | 0.31 | discriminable |
| history_confidence | per-head query/archive retrieval confidence | 0.28 | moderately discriminable |
| retrieval_margin | top1_weight − top2_weight | **1.17** | strongly discriminable |

**Verdict: "FEASIBLE"** — signals are discriminable and non-redundant.

**Critical caveat:** The dynamic roles do **NOT** map 1:1 onto PF -1/1/2
labels. PF Wave heads split across all four functional roles; PF Anchor
heads also span all four. The new signals are not a renaming of PF labels.

### 2.6 Experiment 5: Functional routing ablation (docs/50)

**Question:** Does routing by perturbation signals improve over PF?

**Method:** 32-prompt VBench of PF-static vs confidence-routing vs
archived-full.

| Method | Subject | Background | Dynamic |
|---|---:|---:|---:|
| PF | 0.9780 | 0.9666 | **0.5917** |
| confidence routing | **0.9797** | 0.9679 | 0.5542 |
| archived full | 0.9795 | **0.9681** | 0.5333 |

**Conclusion:** Confidence routing gives +0.0017 subject but **-0.0375
dynamic**. The small consistency gain is partly from reduced motion. Not
strong enough for the paper.

### 2.7 AMA/RollingForcing inspirations (docs/68)

The 82-method × 20-prompt RF experiment set that informed the PF head work:

| Component | Mechanism | DINO gain |
|---|---|---:|
| AAI (Anchor Attention Injection) | query-dependent anchor attention residual | **+6.1%** |
| HRMR (Head-Role Memory Routing) | per-head anchor K-scaling | +2.2% |
| DARV (Drift-Aware Reference Verification) | step-wise anchor refresh | +0.5% |

Key lessons imported:
- Identity is distributed: HRMR id_thresh=0.15 (38.9% of heads) → DINO 0.96
  vs id_thresh=0.25 (7.8%) → DINO 0.89
- Identity-motion tradeoff: +1% DINO ≈ -7% motion
- min_DINO > avg DINO for detecting local collapse
- Failed: K-scaling (info leakage), Content-Aware RoPE (breaks routing),
  Q/K/V LoRA (breaks routing), attention-temperature (catastrophic)

## 3. How v78's design follows from the historical conclusions

| Historical finding | v78 design response |
|---|---|
| PF labels are temporal patterns, not semantic roles | v78 does NOT classify heads by role; it controls write timing |
| Static wrong classification reverses gains | v78 uses online reliability per-block, no static labels |
| AMA's |QK| proxy mislabels all heads | v78 uses denoise disagreement (free from existing predictions) |
| Veil heads preserve exposure but small gains | v78 does not target specific head categories |
| Functional routing gains are motion-bought | v78 controls writes, not reads — motion is preserved |
| PF already maintains ID best | v78 starts from PF, not SF; does not add forwards |
| Darkening is latent drift, not cache issue | v78 does not attempt exposure correction |
| Perturbation signals are discriminable but weak | v78 uses reliability for write decisions, not read routing |

The v78 design explicitly avoids every failure mode discovered in the
historical experiments. It works within PF's existing cache update path,
uses only free online signals, does not classify heads semantically, and
adds zero compute overhead.
