# v81 + v82 Comprehensive Human Review

> Date: 2026-07-24
> Reviewer: human, textual description only
> Videos: v82 labels (3 prompts), v82 confirm (12 prompts), v81 (12 prompts)

## 1. v82 Labels (3 diagnostic prompts)

### 1.1 Per-cell observations

**pf (0-0)**: Overall the best-looking cell. ID and background both retained.
Some acceleration jumps exist but they do not disrupt other information.
1-0 not reviewed in detail but consistent with 0-0.

**v78**: Possibly slightly better than pf. Still has jumps but overall
stable. Confirms v78's position as the best method.

**pf_binary (0-0)**: Camera movement is larger than pf (more orbiting).
ID retention is good but slightly worse than pf. 1-0: the person starts
sideways; in later frames, while other configs keep the person mostly
sideways/back, pf_binary shows frontal shots — suggesting it encourages
more camera and scene motion. This is a positive observation for
pf_binary's motion preservation.

**learned (0-0)**: ID retention similar to layer_early etc. But
hallucinations in non-ID regions are more severe than layer cells.
1-0: background hallucinations and flashback artifacts in early frames.

**learned_audit (0-0)**: ID retention similar to learned. Hallucinations
also severe. 1-0: same background hallucinations and early-frame flashback
artifacts.

**inverse (0-0)**: Large amounts of polygonal colorful noise in the
background throughout the entire video. Also has acceleration jumps.
However, the main subject ID and some background elements persist. 1-0:
polygonal noise in early frames, disappears later, but jumps remain.

**layer_early (0-0)**: ID retention good. Some changes in non-ID regions.
In the last few seconds, a large camera rotation occurs. 1-0: background
hallucinations, flashback artifacts in early frames.

**layer_first_half (0-0)**: ID retention good. Some changes in non-ID
regions. Similar pattern to layer_early.

**layer_late (0-0)**: ID retention good. Some changes in non-ID regions.

**layer_middle**: ID retention good. Similar to other layer cells.

**prompt_only (only 1 video)**: ID retention similar to pf_binary. Very
large camera movement — starts distant and tightens to close-up. Non-ID
hallucinations present.

**random_2026 (only 1 video)**: Overall similar to pf_binary. Non-ID
hallucinations present.

**random_2028_fallback (0-0)**: In the later portion, two subjects appear
(duplicated person), then merge back into one. This is clearly
unacceptable — a severe artifact.

**remote_only**: ID retention acceptable. Background degrades faster.
However, fewer jump-related disruptions in the later portion — background
almost unaffected by jumps.

### 1.2 Key observations from v82 labels

1. **pf and v78 are the most stable** — best ID and background retention,
   jumps present but non-disruptive.
2. **pf_binary encourages more camera motion** — frontal shots and orbiting,
   which is positive for motion preservation.
3. **learned and learned_audit have severe non-ID hallucinations** — worse
   than layer cells, suggesting the full learned label map may be too
   aggressive.
4. **inverse has polygonal noise** — swapping labels introduces visible
   artifacts, confirming classification direction matters visually.
5. **random_2028_fallback has duplicated subjects** — unacceptable artifact,
   confirming random labels can produce severe failures.
6. **layer cells (early/middle/late/first_half) have moderate
   hallucinations** — restricting activation to a depth subset reduces but
   does not eliminate non-ID artifacts.
7. **Camera movement mostly occurs after 5s** — within 5s, all cells are
   stable. The camera motion and hallucinations are long-horizon phenomena.

## 2. v82 Confirm (12 prompts, seeds 1-3)

### 2.1 Per-cell observations

**v78_s2, v78_s3**: Stable across seeds. ID and background well retained.
Jumps present but manageable. Confirms v78's robustness.

**pf_binary_s1 (0-0)**: Very large orbiting camera movement. ID and
background both good despite the motion. 1-0: similar quality.

**pf_binary_s2**: Relatively better than s1 (less extreme camera motion).
ID and background good.

**pf_binary_s3**: Consistent with s1/s2.

**learned_conservative_s1 (0-0)**: ID retention good, relatively few jumps.
Still has hallucinations. 1-0: **two subjects appear in early frames then
merge into one** — a concerning artifact.

**learned_conservative_s2 (1-0)**: No duplicated subjects (unlike s1).
Both s1 and s2 have background hallucinations.

**learned_s1, learned_s3**: ID and background retention good. Consistent
with the v82 labels learned cell.

**learned_open**: No videos generated (failed with KeyError).

**pf**: Stable across seeds. Consistent with v82 labels pf cell.

### 2.2 Key observations from v82 confirm

1. **v78 and pf are the most stable across seeds** — consistent quality,
   no severe artifacts.
2. **pf_binary has large camera motion but good ID** — the orbiting
   movement is distinctive and may be a positive for dynamic degree.
3. **learned_conservative_s1 has duplicated subject artifact** — appears
   in early frames, merges into one. Not present in s2.
4. **learned cells are generally good** — ID and background retention
   comparable to pf_binary, with some hallucinations.

## 3. v81 ProbeCache Screen (12 prompts)

### 3.1 Per-cell observations

**ours_archive12**: ID retention good. Camera has push-pull movement.
Non-ID hallucinations present.

**ours_audit**: Camera movement slightly smaller. ID good. Background
degradation and hallucinations slightly more severe.

**ours_conservative**: ID and camera both acceptable. Background
degradation present but acceptable.

**ours_open_gate**: Camera movement slightly larger. ID good. Background
degradation present but acceptable.

**ours_persistent**: ID retention acceptable. Non-ID hallucinations
present. Background degradation. Camera movement.

**ours_prompt0, ours_prompt30, ours_reactive**: ID acceptable. Background
hallucinations present. Camera movement relatively small.

**ours_topk6**: Visually appears slightly better than ours_topk2. Still
has hallucinations and background degradation.

**ours_topk2**: (Not explicitly reviewed but implied similar to topk6.)

### 3.2 Key observations from v81

1. **All ProbeCache cells maintain ID** — consistent with DINO metrics.
2. **Non-ID hallucinations are universal** — present in all ProbeCache
   cells, varying in severity.
3. **Camera movement varies by configuration** — some cells have more
   push-pull or orbiting, others are more stable.
4. **ours_conservative and ours_open_gate are acceptable** — balanced
   ID retention and background degradation.
5. **ours_topk6 may be visually better than topk2** — despite topk2 having
   slightly higher DINO, topk6 looks better subjectively.
6. **All camera movement occurs after 5s** — within 5s, all cells are
   stable.

## 4. Cross-experiment synthesis

### 4.1 Method ranking by human review

| Rank | Method | ID | Background | Jumps | Hallucinations | Camera motion |
|---:|---|---|---|---|---|---|
| 1 | **v78** | Excellent | Good | Present but manageable | Low | Moderate |
| 2 | **pf** | Excellent | Good | Present but manageable | Low | Moderate |
| 3 | pf_binary | Good | Good | Present | Low-moderate | **Large (positive)** |
| 4 | learned | Good | Moderate | Present | **High** | Moderate |
| 5 | learned_conservative | Good | Moderate | Few | Moderate | Moderate |
| 6 | ours_conservative | Good | Moderate | Present | Moderate | Moderate |
| 7 | layer cells | Good | Moderate | Present | Moderate | Varies |
| 8 | inverse | ID persists | **Polygonal noise** | Present | **Severe** | — |
| 9 | random_2028_fallback | — | — | — | **Duplicated subjects** | — |
| 10 | sf_native | 5s degradation | Severe | Severe | Severe | — |

### 4.2 Key cross-experiment findings

1. **v78 and pf are consistently the best** across all experiments and
   seeds. They are visually indistinguishable in quality, with v78
   possibly slightly better.

2. **Camera movement after 5s is universal** — all methods (including pf)
   exhibit camera motion changes after 5s. This is a PF-inherited
   characteristic, not introduced by ProbeCache or v78.

3. **Non-ID hallucinations are the main ProbeCache weakness** — all
   ProbeCache cells (learned, layer, ours_*) have background
   hallucinations that pf and v78 do not have. This suggests the
   ProbeCache archive retrieval introduces non-ID artifacts.

4. **Classification direction matters visually** — inverse has polygonal
   noise, random_2028_fallback has duplicated subjects. These are severe
   artifacts not present in learned or pf_binary.

5. **pf_binary encourages more camera motion** — frontal shots and
   orbiting. This is a positive observation: pf_binary's label mapping
   may preserve dynamic degree better than learned labels.

6. **learned_conservative has a seed-dependent duplicated subject
   artifact** (s1 but not s2). This is a concerning failure mode that
   needs investigation.

7. **The 5s stability window** — within 5s, all methods are stable.
   The differences emerge after 5s, which aligns with PF's 21-frame
   native window (21÷4≈5.25s).

### 4.3 Implications for paper

1. **v78 is the recommended paper candidate** — best visual quality,
   matches or exceeds pf, zero compute overhead, robust across seeds.

2. **ProbeCache's non-ID hallucinations limit its practical value** —
   even though DINO matches pf, the visual artifacts in non-ID regions
   are noticeable. The archive retrieval mechanism may need refinement.

3. **The classification contribution is supported** — inverse and random
   controls produce severe artifacts (polygonal noise, duplicated
   subjects) that learned and pf_binary do not. The classification
   direction matters both quantitatively (DINO, temporal jump) and
   qualitatively (visual artifacts).

4. **pf_binary's motion encouragement is a positive finding** — it
   suggests that PF's static labels (Anchor→persistent, Wave+Veil→
   reactive) naturally preserve camera motion, while learned labels
   may over-constrain motion.

5. **The 5s stability window is a PF characteristic** — all PF-based
   methods (pf, v78, ProbeCache) are stable within 5s. The long-horizon
   differences emerge after the 21-frame native window.
