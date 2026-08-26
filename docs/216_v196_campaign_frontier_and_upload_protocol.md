# v196: Evidence-Ladder Frontier and Upload Protocol

> Date: 2026-08-26
> Status: code ready; no GPU experiment is added in this stage

## 1. Current synchronized conclusion

After fetching every remote branch, the newest experiment commit remains
`b026a896281c6aeb59012ae426818378949a17b5`. Neither the experiment branch nor
`main` contains a v189, v190, v191, v192, v194, or v195 result artifact.

The local evidence frontier is therefore **v189 missing**. This does not prove that
the server has not run v189. It means the repository does not contain enough small
artifacts to verify it.

Writing another downstream generation experiment at this point would be unsafe:
v190 through v195 are conditional, and each changes meaning depending on the first
failed gate. v196 addresses the missing campaign state before any v197 or ABA code is
written.

## 2. What v196 does

`scripts/inspect_v196_campaign_frontier.py` inspects the fixed ladder:

```text
v189 profile
  -> v190 classifier-holdout causal screen
  -> v191 unseen-128 confirmation
  -> v192 seed/length robustness
  -> v194 Causal-checkpoint generation transfer
  -> v195 cross-checkpoint mechanism profile
```

For each final artifact it reports one of:

- `missing`: the final decision/analysis file is absent;
- `invalid`: JSON structure, recommendation, gate consistency, referenced path, or
  SHA binding failed;
- `passed`: a confirmatory stage passed;
- `failed`: a complete stage rejected the route;
- `complete`: v195 finished and produced a mechanism interpretation;
- `blocked`: an earlier stage is unresolved or failed.

Checking only filenames is insufficient. v196 verifies every available SHA-bound
comparison manifest, VBench summary, temporal diagnostic, temporal contract, input
manifest, profile audit, and source cell-score table referenced by a decision. It
also recursively verifies both v192 scope reports and requires the final v194
decision to bind its camera-compensated motion report and motion provenance.

## 3. Advancement rules

The controller encodes the following rules:

1. v189, v190, v191, or v192 failure stops the frozen route. Downstream files are
   marked stale rather than accepted.
2. v194 success **or failure** advances to v195. The profile is needed to distinguish
   classifier failure from shadow/generation-objective mismatch.
3. v195 never authorizes a GPU experiment automatically. It emits a scientific
   design branch:

| v194 | v195 mechanism | Next design |
|---|---|---|
| pass | exact head identity | freeze the route, review it, then design prompt-switch AB/ABA |
| pass | phase/layer only | run a coarse phase/layer transfer ablation before prompt switching |
| pass | operator only/unsupported | reduce or reject the head-classification claim |
| fail | profile structure transfers | diagnose shadow-versus-generation mismatch |
| fail | profile unsupported | stop the current cross-checkpoint route |

No broad human review is requested by v196.

## 4. Run on the server

After pulling the commit:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

bash scripts/run_v196_campaign_frontier.sh show
```

This writes:

```text
runs/v196_campaign_frontier/campaign_state.json
runs/v196_campaign_frontier/campaign_state.md
runs/v196_campaign_frontier/next_commands.txt
```

To print only the authorized stage commands:

```bash
bash scripts/run_v196_campaign_frontier.sh next
```

Lines containing `NODE_RANK=<0|1|2|3>` are templates to launch concurrently on the
four nodes. `next_commands.txt` is intentionally not an executable shell script, so
the placeholder cannot accidentally be interpreted as shell redirection.

## 5. Package evidence for review

After the current stage finishes, run:

```bash
bash scripts/run_v196_campaign_frontier.sh package
```

The package contains only JSON, Markdown, CSV, and TXT files no larger than 8 MiB:

```text
runs/v196_campaign_frontier/v196_campaign_evidence.zip
runs/v196_campaign_frontier/evidence_manifest.json
```

The manifest records path, size, and SHA-256 for every included file. MP4 videos,
`.pt` profiles, model checkpoints, and large logs are excluded. If a decision is
invalid, return its named source files or the corresponding existing diagnostic
archive separately.

The minimum files to push are:

```text
runs/v196_campaign_frontier/campaign_state.json
runs/v196_campaign_frontier/campaign_state.md
runs/v196_campaign_frontier/evidence_manifest.json
```

Push the small decision/analysis files listed in `evidence_manifest.json` as well.
The zip is convenient for direct transfer but does not need to be committed when the
contained files are already pushed.

## 6. Current next command

On the local checkout, v196 reports:

```text
frontier.kind = run_stage
frontier.key  = v189
reason        = v189 final artifact is absent
```

If the server also reports v189, run the generated v189 commands. They perform the
128-prompt Landmark/Retrieval shadow profile, audit all 368,640 records, analyze the
fixed 64/32/32 split, and package the small artifacts. The runbook then invokes the
CPU-only v197 cross-split Head x Phase structure audit; this adds no video generation
or human review and does not change the frozen v189 map. If the server reports a
later frontier, follow that report instead; server-side result files may be newer
than the Git repository.

## 7. Why ABA remains deferred

The current classifier story is only meaningful if at least v191 confirms the frozen
Head x Phase route, and cross-checkpoint portability is only meaningful after v194
and v195. Implementing prompt-sensitive reset or ABA recall before those gates would
mix a new scene-boundary mechanism with an unconfirmed single-prompt method.

The controller explicitly routes to prompt-switch design only after both generated-
video transfer and exact-head mechanism transfer pass. This keeps the next paper
claim attributable and prevents another large, ambiguous experiment grid.
