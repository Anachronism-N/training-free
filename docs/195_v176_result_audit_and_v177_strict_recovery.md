# v176 Result Audit and v177 Strict-Superset Recovery

## 1. Executive decision

Remote commit `079a7263` completed all 128 prompts and reported four Coverage
heads (`L0H10`, `L5H3`, `L6H6`, `L8H6`). Those memberships are **not valid
generation candidates**. The run violated its defining fair-teacher contract,
so `generation_ready=true` in the uploaded v176 report is superseded.

Do not run v176 matched generation. Do not cite its four-head result.

v177 fixes the teacher boundary, restores fail-fast auditing, and writes a new
artifact version and directory. Its statistical thresholds are frozen before
the rerun; they were not changed after seeing v176.

## 2. What completed correctly in v176

- 16/16 shards, 128/128 prompts and 184,320 records were produced.
- Every prompt/layer had 48 records from denoising calls 0/1/2/3.
- The active generation path remained Recent, so shadow readouts did not alter
  the profiled latent trajectory.
- Candidate budgets remained at most 9 frame equivalents and the nominal
  teacher remained at most 17.
- Discovery, validation and untouched-generation prompt ids were separated.

These properties make the run useful for debugging, but not for selecting
heads because the teacher itself was incomplete.

## 3. Decisive v176 failure

The uploaded runtime changed the mandatory subset assertion to a warning and
the offline loader changed the corresponding error to a print. Across the 16
logs there are 4,668 actual superset violation events:

| Candidate | Violations |
|---|---:|
| Coverage | 768 |
| Episode | 3,900 |

Every shard contains violations (240 to 324 each). At the first sampled state,
for example, Coverage selected historical frames 5 and 6 and Episode selected
frame 6, while Union omitted them.

The uploaded audit still passed because runtime metadata set
`candidate_physical_superset_verified=true` after warning, irrespective of the
failures. Therefore the four reported heads compare candidates against a
different reference support and cannot satisfy the fair-teacher hypothesis.

## 4. Root cause

Candidates have equal total budgets but different recent windows:

| Readout | Contents |
|---|---|
| Recent | sink1 + recent8 |
| Coverage | sink1 + reservoir4 + recent4 |
| Episode | sink1 + reservoir2 + motion-pair2 + recent4 |
| Union | physical/representation union of all above |

v176 collected Union's reservoir and motion banks using a boundary derived
from `recent8`. The corresponding candidate banks use `recent4`. Frames that
were old enough for Coverage/Episode but still fell inside Union's recent8
interval were filtered out of Union's middle-bank collection. Three-frame AR
updates made this mismatch immediately visible.

The first implementation also checked only physical frame ids. An anchor may
apply dynamic RoPE while the same physical frame in Recent uses its saved
position. These are different K representations. A correct teacher must cover
`(physical frame id, representation family)`, not merely the frame id.

## 5. v177 correction

v177 is artifact version 3 with method
`strict_superset_residual_cache_compatibility`.

1. Coverage reservoir, Episode reservoir and motion pair are collected with
   their candidate `recent4` eligibility boundary even inside Union. The
   boundary is derived from the actual three-frame update extent, fixing the
   old two-frame offset in candidate middle-bank eligibility as well.
2. Union keeps Recent's `recent8` dynamic tokens and all necessary anchor
   representations. The maximum remains 17 frame equivalents.
3. Runtime constructs `(physical frame, representation family)` identities
   for every policy, layer, batch item and head. It distinguishes saved,
   time-mapped and dynamic-RoPE representations. Duplicate anchors from
   different banks must also have exactly equal K/V/position tensors.
4. A missing identity, malformed source code or duplicate identity raises a
   `RuntimeError`; it can no longer be downgraded to a warning.
5. Every record stores 36 successful subset checks (3 candidates x 12 heads),
   zero failures and the `v177` verification contract.
6. The offline loader independently repeats the representation-aware subset
   test for trace layers 0/10/20/29.
7. v177 preserves v176's frozen split seed (`1762026`) so the untouched
   generation prompts remain untouched, while using a separate output root.

## 6. Required execution order

First run only the one-prompt smoke on node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
NODE_RANK=0 bash scripts/run_v177_strict_superset_rccp_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v177_strict_superset_rccp_32gpu.sh smoke
```

The smoke must satisfy all of the following automatically:

- 1,440 records (1 prompt x 30 layers x 48 records);
- profile version 3 and contract `v177`;
- exactly 36 subset checks per record and zero failures;
- representation-subset flag is true for every record;
- no `teacher is not a cache-representation superset` text in the log;
- no traceback.

Do not start the full run if smoke fails. Push the smoke log and audit instead.

After smoke passes, run all nodes (`NODE_RANK=0,1,2,3`):

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v177_strict_superset_rccp_32gpu.sh profile128
```

Then on node 0:

```bash
NODE_RANK=0 bash scripts/run_v177_strict_superset_rccp_32gpu.sh status
NODE_RANK=0 bash scripts/run_v177_strict_superset_rccp_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v177_strict_superset_rccp_32gpu.sh analyze
NODE_RANK=0 bash scripts/run_v177_strict_superset_rccp_32gpu.sh package
```

## 7. Decision after v177

- **No supported nonlocal heads:** reject static operator compatibility on this
  model and stop this classifier line. Do not tune thresholds post hoc.
- **Supported heads:** inspect automatic stability and salience fields, then
  run matched routing versus all-Recent and all four layer/count-matched hard
  negatives on the untouched 32 prompts with the v178 runner documented in
  `docs/196_v177_v178_execution_and_decision.md`. Manual review is unnecessary
  before metrics identify a causal winner.
- **Matched fails hard negatives:** local residual compatibility does not
  transfer to trajectory quality; reject membership as a generation rule.
- **Matched wins:** only then claim an operator-aligned self-profiled cache
  routing mechanism and expand generation evaluation.

This remains distinct from PF's temporal-QK three-way taxonomy and
stride/cyclic/merge routing. The candidate idea is residual-space compatibility
between a head and a cache operator, but v177 profiling and held-out causal
generation are required before calling it an effective method.
