# v169 Review Conclusion and v170 Matched Attribution

## 1. Frozen conclusion after review

The v169 minimal blind review is complete. Both reviewed pairs preferred the
frozen v166 multi-scale-motion selector after unblinding.

| Prompt | Blind pair | Human conclusion | Query-weighted VBench delta |
|---:|---|---|---:|
| 15 | V001=v166, V002=Query | V001 better; V002 has larger identity change; both nearly stop moving after 5 s | Quality -1.5524, Dynamic -0.2000 |
| 3 | V003=v166, V004=Query | V003 better; V004 changes crew count | Quality +2.1755, Dynamic +0.2667 |

Prompt 15 agrees with VBench. Prompt 3 is a direct metric-human
contradiction: the largest metric upside is a visual regression under blind
review. The aggregate v169 Query-weighted deltas versus v166 were also small
and unstable:

- official Quality: +0.1465;
- Dynamic Degree: +0.0208;
- identity/background: -0.000860;
- temporal mechanics: -0.000542;
- semantic alignment: -0.000843;
- paired Quality: 7 positive and 9 negative prompts, median -0.0184, 95%
  bootstrap CI [-0.3206, 0.6436].

Therefore Query-weighted recall is **not promoted**. More hand-picked review is
not an appropriate way to rescue it.

## 2. Why v170 is necessary

The v169 reference videos were reused from v168 while Query-weighted videos
were newly generated. Method, GPU/node, run time and execution order were not
fully separated. The two reviewed contradictions make another selector search
premature: first determine whether the measured v169 differences are policy
effects or run variance.

v170 changes no cache budget and proposes no new paper method. It is a matched
causal attribution experiment between:

- v166: `reservoir2_multiscalemotion1`;
- candidate: `reservoir2_multiscalequeryweighted1`.

Both use the same Middle10 layer map. Active layers use `sink1 + reservoir2 +
one recalled atomic pair2 + recent4`; all other layers use `sink1 + recent8`.
Only the archive-pair ranking rule differs.

This must still be described as a **layer policy**, not a validated semantic
head taxonomy.

## 3. Matched 32-GPU design

The experiment uses 16 fixed MovieBench prompts, four nodes and eight GPUs per
node. Each prompt receives two independent GPU lanes. Within each lane, v166
and Query-weighted run sequentially on the same GPU. The second lane reverses
the order, and prompt parity swaps both orders again.

| Quantity | Frozen value |
|---|---:|
| Prompts | 16 |
| Policies | 2 |
| Replicas per policy | 2 |
| New 30-second videos | 64 |
| Nodes x GPUs | 4 x 8 |
| Videos per GPU | 2, sequential |

For prompt `p`, let `dA = QueryA - v166A` and `dB = QueryB - v166B`.

- matched policy effect: `(dA + dB) / 2`;
- same-policy replica noise: `(abs(v166A-v166B) + abs(QueryA-QueryB)) / 2`;
- lane disagreement: `abs(dA-dB) / 2`;
- query-first and query-second strata expose order effects.

This design does not make GPU A and GPU B identical. It makes each policy
comparison local to one GPU and balances which policy runs first.

## 4. Full active-layer mechanism audit

v169 traced all heads of one representative active layer for its final audit.
Raw logs showed replicated decisions across the 12 heads in that layer, but
the other nine active layers were not independently checked.

v170 traces head 0 for every active layer `10..19`. This reduces trace volume
by a factor of 12 while expanding layer coverage by a factor of 10. The audit
requires, for all four replicas:

1. all 16 prompt traces and all ten active layers;
2. only the requested trace head;
3. exact independent score and selector recomputation;
4. exercised archive retrieval and multi-candidate selection;
5. atomic pair reads and zero read-budget violations;
6. zero cache-contract failures.

Selector changes are required in aggregate, not in every layer. A layer with
no counterfactual change is reported rather than falsely marked broken.

`PYRAMIDKV_POLICY_TRACE_HEADS` is optional. Its empty default preserves all
historical behavior; v170 explicitly sets it to `0`.

## 5. Automated decision rule

No new manual review is requested by default. After core-9 VBench-Long:

1. both order-balanced lanes must have nonnegative mean deltas for official
   Quality, identity/background, temporal mechanics and Dynamic Degree;
2. their matched mean effects must be nonnegative on the same axes;
3. the mean Quality effect must exceed mean same-policy replica noise;
4. the full-layer mechanism gate must pass.

Passing means only that Query weighting survives causal debugging and may
enter another development experiment. It does not establish paper-level
superiority. Failing means reject Query weighting without additional manual
review and return to mechanism design around the frozen v166 reference.

Bootstrap intervals and per-prompt signs are always reported, but the
16-prompt suite is adaptive development evidence and cannot be presented as
an unbiased final benchmark.

## 6. Scope boundary

v170 deliberately does not run PF, SF, ABA, new tricks or broad ablations.
Existing videos remain valid for historical comparisons. The immediate goal
is to repair attribution before spending compute on another idea. If v170
rejects Query weighting, the next generation-side work should target the
observed late-motion collapse with an independently motivated state/update
mechanism, rather than another archive-ranking heuristic.
