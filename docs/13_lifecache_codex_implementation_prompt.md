# Codex implementation prompt for LifeCache-v1

Use this prompt when asking Codex or a coding agent to start implementing the LifeCache-v1 prototype.

---

You are working in repository `Anachronism-N/training-free`.

Read these documents first:

- `docs/11_lifecache_v1_design.md`
- `docs/12_lifecache_experiment_plan.md`
- `docs/09_lifecache_mechanism_map.md`
- `docs/08_full_third_party_inventory.md`

Goal:

Implement a prototype of **LifeCache-v1: Token-level Head-aware Cache Recall for Training-free Long AR Video Generation**.

Important constraints:

1. Self-Forcing and Causal-Forcing are AR / sliding-window video generators. Do not treat them as independent chunk-by-chunk generators.
2. Do not store every historical token with a Python object or metadata entry.
3. Do not implement a heavy external memory generator.
4. The memory bank should store selected token-level K/V payloads and summaries only.
5. Generation must be affected through active self-attention K/V composition.
6. The base generator must remain frozen.

Core data structure:

```python
@dataclass
class TokenSet:
    set_id: str
    chunk_id: int
    frame_ids: list[int]
    layer_id: int
    head_group: str
    k: torch.Tensor
    v: torch.Tensor
    token_indices: torch.Tensor
    k_summary: torch.Tensor
    prompt_summary: torch.Tensor | None
    visual_summary: torch.Tensor | None
    importance_score: torch.Tensor
    motion_score: torch.Tensor | None
    quality_score: float
    access_count: int
    last_used_step: int
```

Prototype modules to implement under `src/lifecycle_kv/`:

```text
tokenset.py
bank.py
compression.py
recall.py
anchor.py
motion.py
head_roles.py
active_cache.py
instrumentation.py
```

Implementation phases:

## Phase 0: Cache instrumentation

Add hooks to inspect Self-Forcing K/V cache shapes, clean-refresh boundaries, and attention masses.

Deliverables:

- `cache_trace.jsonl` or `cache_trace.pt`
- logs of K/V shape, layer/head ids, recent cache span, and attention to regions

## Phase 1: Compression-only bank

Implement:

- eviction from recent cache;
- Attention Participation Top-k compression;
- TokenSet storage in CompressedBank.

Do not implement recall yet. Verify token counts and memory use.

## Phase 2: Token-level recall

Implement:

- Q-summary extraction;
- TokenSet-level retrieval;
- token-level Q-K top-k recall;
- active K/V concatenation.

Start with all-head shared recall.

## Phase 3: Anchor and motion cache

Implement:

- fixed first anchor;
- dynamic anchor score;
- latent-delta motion score;
- dynamic-K motion score;
- MotionCache.

## Phase 4: Head-aware active cache

Implement:

- Pyramid/Forcing-KV head label loader;
- head-specific region budget;
- head-specific active-cache composition;
- optional region bias.

Expected active cache:

```text
Layout heads:
  fixed anchor + dynamic anchor + scene/query recall + recent + current

Motion heads:
  motion tokens + recent + tiny anchor + current

Recall/semantic heads:
  query-recalled tokens + anchor + recent + current

Generic heads:
  recent + small anchor + current
```

Ablations to support:

```text
no compressed bank
AP Top-k compression
Head-group-aware compression
Key-token compression
no recall
chunk-level recall
token-level Q-K recall
all-head shared cache
head-aware active cache
no motion cache
latent-delta motion
dynamic-K motion
region budget only
region budget + region bias
```

First runnable experiment:

```text
Base: Self-Forcing
Length: 30s or 60s
Prompts: scene revisit and long motion continuation
Compression: AP Top-k
Recall: token-level Q-K recall
Anchor: fixed first + dynamic anchor
Motion: latent-delta only
Head: all-head shared first, then Pyramid/Forcing-KV prior labels
```

Success criteria:

1. LifeCache runs without memory explosion.
2. Recalled tokens are actually attended to.
3. Scene revisit or subject consistency improves over vanilla.
4. Motion does not collapse compared to vanilla.
5. Memory/time overhead is reported.
