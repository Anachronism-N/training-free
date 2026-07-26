# v99 Smoke Results and Corrected Decision

Date: 2026-07-27

## 1. Human-review observations

All generated cells used MovieBench prompt 0, seed 0, 120 latent frames
(approximately 30 seconds). Existing PF and v93 binary videos were reused.

| Cell | Binary membership | Label-10 route | Label-11 route | Polygon noise |
|---|---|---|---|---|
| `pf_ar_neutral_stride_cyclic` | PF Anchor / rest | stride | cyclic | no |
| `history_polarity_stride_cyclic` | independent score | stride | cyclic | no |
| `pf_aw_neutral_stride_merge` | PF (Anchor+Wave) / Veil | stride | merge | yes |
| `history_polarity_stride_merge_fixed` | independent score | stride | merge | yes |
| `history_polarity_random_stride_merge` | count-matched random | stride | merge | yes |
| reused PF native | PF three classes | native | native | no |
| reused v93 PF-binary-v78 | PF binary | historical | historical | no |

These are artifact/no-artifact observations from one prompt, not metric
results. Identity quality, motion quality, and causal superiority remain open.

## 2. What the smoke test actually proves

The clean PF-AR neutral run is the implementation-control result. It shows
that neutral labels `10/11`, explicit per-head routing, and exclusive
sink/middle/recent ownership can produce a valid video. This substantially
reduces the likelihood that the old v98 collapse was caused by binary labels
themselves.

The clean independent-map stride/cyclic run establishes feasibility only:
the independently profiled map can execute without polygon corruption. It
does not show that its head membership is useful; random, inverted, and
threshold controls are still required.

## 3. Why "Merge is broken" was too strong

The first result note attributed all polygon noise to the Merge operator. The
available cells do not isolate that conclusion.

Native PF routes:

```text
Anchor -> stride
Wave   -> cyclic
Veil   -> merge
```

The failed PF-AW binary cell routes:

```text
Anchor + Wave -> stride
Veil          -> merge
```

It therefore changes all Wave heads from cyclic to stride at the same time as
it exercises Merge through the repaired binary ownership path. Failure can be
caused by loss of Wave's phase-local cyclic memory, by the binary Merge path,
or by their interaction. The two independent/random stride-merge cells also
send many more heads to Merge than native PF if the reported 33/327 counts are
correct.

The defensible current conclusion is:

> Binary classification is not disproved. A phase-local cyclic route appears
> necessary for a large fraction of heads, while binary Merge remains
> unresolved and is unsafe for the main method.

This does not imply that native PF's 32-head Veil Merge implementation is
invalid.

## 4. Minimal remaining cache diagnostic

One new smoke cell is implemented:

```text
pf_aw_neutral_stride_cyclic
```

It keeps the exact PF-(Anchor+Wave)/Veil membership of the failed PF-AW cell
and changes only label 11 from Merge to cyclic:

```text
Anchor + Wave -> stride
Veil          -> cyclic
```

Interpretation:

- merge version fails, cyclic version passes: the repaired binary Merge path
  or its Veil interaction is implicated;
- both fail: moving Wave from cyclic to stride is the common likely cause;
- both pass on additional prompts: prompt-0 failure is not stable and the
  smoke conclusion must be revised.

This diagnostic needs one new video, not a 32-prompt rerun.

## 5. Main-method decision

The current candidate uses only the validated binary topology:

```text
History-Supportive (score >= 0)
  sink3 + global stride(interval=6, capacity=4) + recent4

Recent-Responsive (score < 0)
  sink1 + phase cyclic(period=6, capacity=4) + recent4
```

Merge is removed from the proposed method and retained only as a diagnostic
and PF-native baseline primitive. The next full screen is `candidate32`, which
contains no Merge cell.

## 6. Reproducibility fixes now committed

- The runner reads the auditor's real `ok` field rather than nonexistent
  `pass`.
- `pyramidkv_prompt_warmup_enabled: false` is present in the tracked PF YAML.
- Map counts and PF cross-tabs are recomputed from CSV and fail closed.
- Every selected map is emitted as `[V99MapAudit]` in the log.
- Per-cell configs freeze map counts/cross-tabs and the exact cache contract.
- Sampled Merge traces now include complete, incomplete, invalid block ids,
  source intervals, seen counts, and merged token counts.
- `main128` no longer includes failed Merge or broad threshold cells.

No `git update-index --assume-unchanged` workaround is part of the protocol.
A clean checkout of the pushed commit is the required server input.

## 7. Immediate decision

1. Run the one-video `pf-aw-stride-cyclic` diagnostic.
2. Start `candidate32` on four 8-GPU nodes; all eight cells are stride/cyclic.
3. Review videos blind before reading metrics.
4. Promote the independent map only if it beats count-matched random and is
   not defeated by inversion.
5. Select natural zero or a threshold variant from the 32-prompt screen.
6. Run 128 prompts only for the surviving core and optional v78 update.

Commands and exact claim gates are in
`docs/103_v100_final_candidate_and_immediate_plan.md`.
