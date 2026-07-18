from __future__ import annotations

import torch
import torch.nn.functional as F

from lifecycle_kv.attention_fusion import fuse_parallel_attention
from lifecycle_kv.cache_types import HeadRole
from lifecycle_kv.compression import qk_proxy_scores
from lifecycle_kv.instrumentation import CacheTraceEvent, CacheTraceWriter
from lifecycle_kv.latent_trace import frame_statistics, tensor_statistics
from lifecycle_kv.oracle import slice_clean_block_frames
from lifecycle_kv.recall import token_qk_scores
from lifecycle_kv.runtime import LifeCacheRuntime, LifeCacheRuntimeConfig
from lifecycle_kv.tokenset import CacheRegion, TokenSet


def _token_set(set_id: str, *, frame: int = 0, tokens: int = 4) -> TokenSet:
    k = torch.randn(tokens, 2, 4)
    return TokenSet(
        set_id=set_id,
        chunk_id=0,
        frame_ids=[frame],
        layer_id=1,
        head_group="layout",
        k=k,
        v=torch.randn_like(k),
        token_indices=torch.arange(tokens),
        k_summary=k.mean(dim=0),
        region=CacheRegion.COMPRESSED,
        rope_mode="pre_rope",
        frame_positions=torch.full((tokens,), frame, dtype=torch.long),
        spatial_positions=torch.arange(tokens),
    )


def test_chunked_qk_scores_match_materialized_formula() -> None:
    torch.manual_seed(0)
    q = torch.randn(7, 3, 5)
    k = torch.randn(11, 3, 5)
    qn = F.normalize(q.float(), dim=-1)
    kn = F.normalize(k.float(), dim=-1)
    expected = torch.einsum("qhd,khd->qkh", qn, kn).max(dim=0).values.mean(dim=-1)

    assert torch.allclose(token_qk_scores(q, k, query_chunk_size=2), expected)

    proxy_expected = expected.clamp_min(0)
    proxy_expected = proxy_expected / proxy_expected.sum().clamp_min(1e-8)
    assert torch.allclose(qk_proxy_scores(q, k, query_chunk_size=2), proxy_expected)


def test_no_compression_preserves_rope_metadata() -> None:
    rt = LifeCacheRuntime(LifeCacheRuntimeConfig(
        enabled=True,
        trace_only=False,
        compression="none",
        capture_clean_only=False,
    ))
    k = torch.randn(6, 2, 4)
    frame_positions = torch.tensor([2, 2, 2, 3, 3, 3])
    spatial_positions = torch.tensor([0, 1, 2, 0, 1, 2])
    stored = rt.on_kv_evicted(
        layer_id=1,
        head_group="layout",
        evicted_k=k,
        evicted_v=torch.randn_like(k),
        token_indices=torch.arange(6),
        q_current=torch.randn(2, 2, 4),
        chunk_id=1,
        frame_ids=[2, 3],
        frame_positions=frame_positions,
        spatial_positions=spatial_positions,
        is_pre_rope=True,
    )

    assert stored is not None
    assert torch.equal(stored.frame_positions, frame_positions)
    assert torch.equal(stored.spatial_positions, spatial_positions)
    assert stored.rope_mode == "pre_rope"


def test_clean_only_rejects_denoising_eviction() -> None:
    rt = LifeCacheRuntime(LifeCacheRuntimeConfig(
        enabled=True,
        trace_only=False,
        compression="none",
        capture_clean_only=True,
    ))
    k = torch.randn(3, 2, 4)
    stored = rt.on_kv_evicted(
        layer_id=1,
        head_group="layout",
        evicted_k=k,
        evicted_v=torch.randn_like(k),
        token_indices=torch.arange(3),
        q_current=torch.randn(2, 2, 4),
        chunk_id=1,
        frame_ids=[0],
        capture_reason="denoising",
    )

    assert stored is None
    assert len(rt.bank) == 0


def test_oracle_mode_does_not_fall_back_to_sparse_recall() -> None:
    rt = LifeCacheRuntime(LifeCacheRuntimeConfig(
        enabled=True,
        trace_only=False,
        recall_enabled=True,
        oracle_mode="full_frame",
        oracle_layer=1,
        oracle_recall_frames=[9],
        oracle_allow_sparse_fallback=False,
    ))
    rt.bank.add(_token_set("historical"))
    recent_k = torch.randn(5, 2, 4)
    active_k, _, view = rt.compose_active_cache(
        layer_id=1,
        q=torch.randn(3, 2, 4),
        native_recent_k=recent_k,
        native_recent_v=torch.randn_like(recent_k),
        token_indices=torch.arange(5),
        head_group="layout",
        role=HeadRole.LAYOUT,
        current_frame=6,
    )

    assert active_k.shape[0] == recent_k.shape[0]
    assert view is not None
    assert CacheRegion.RECALL not in view.regions


def test_strict_oracle_mode_skips_unused_eviction_capture() -> None:
    rt = LifeCacheRuntime(LifeCacheRuntimeConfig(
        enabled=True,
        oracle_mode="full_frame",
        oracle_allow_sparse_fallback=False,
    ))
    assert not rt.should_capture_evictions()

    rt.config.oracle_allow_sparse_fallback = True
    assert rt.should_capture_evictions()


def test_runtime_reset_prevents_cross_video_memory_leak() -> None:
    rt = LifeCacheRuntime(LifeCacheRuntimeConfig(
        enabled=True,
        oracle_mode="full_frame",
        oracle_layer=1,
    ))
    rt.bank.add(_token_set("old-prompt"))
    rt.store_oracle_frame(
        layer_id=1,
        frame_idx=0,
        k_pre_rope=torch.randn(4, 2, 4),
        v=torch.randn(4, 2, 4),
    )
    rt.step = 12

    rt.reset()

    assert len(rt.bank) == 0
    assert rt._oracle_frames == {}
    assert rt.step == 0


def test_clean_block_capture_uses_requested_frame_not_block_tail() -> None:
    k = torch.arange(1 * 12 * 1 * 1).reshape(1, 12, 1, 1).float()
    v = k + 100
    slices = slice_clean_block_frames(
        k_pre_rope=k,
        v=v,
        local_end=12,
        block_start_frame=6,
        block_num_frames=3,
        target_frames=[6, 8],
        frame_seq_length=4,
    )

    assert [frame for frame, _, _ in slices] == [6, 8]
    assert torch.equal(slices[0][1].flatten(), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(slices[1][1].flatten(), torch.tensor([8.0, 9.0, 10.0, 11.0]))


def test_parallel_gate_zero_is_exact_and_head_mask_is_broadcast_safe() -> None:
    recent = torch.randn(1, 5, 3, 4)
    memory = torch.randn_like(recent)
    zero = fuse_parallel_attention(recent, memory, gate=0.0)
    assert zero is recent

    head_mask = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, 3, 1)
    fused = fuse_parallel_attention(
        recent,
        memory,
        gate=0.25,
        head_mask=head_mask,
        rms_match=False,
    )
    assert fused.shape == recent.shape
    assert torch.equal(fused[:, :, 1], recent[:, :, 1])


def test_trace_writer_replaces_previous_process_trace(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text("stale run\n", encoding="utf-8")

    writer = CacheTraceWriter(path)
    writer.write(CacheTraceEvent(step=0, layer_id=1, head_id=None, event="start"))

    content = path.read_text(encoding="utf-8")
    assert "stale run" not in content
    assert '"event": "start"' in content


def test_memory_head_policy_does_not_confuse_pf_labels_with_semantics() -> None:
    all_heads = LifeCacheRuntime(LifeCacheRuntimeConfig(memory_head_policy="all"))
    assert all_heads.resolve_memory_head_indices(4, pf_oscillating_heads=[0, 2]) == [0, 1, 2, 3]

    pf_ablation = LifeCacheRuntime(LifeCacheRuntimeConfig(memory_head_policy="pf_stable"))
    assert pf_ablation.resolve_memory_head_indices(4, pf_oscillating_heads=[0, 2]) == [1, 3]

    explicit = LifeCacheRuntime(LifeCacheRuntimeConfig(
        memory_head_policy="explicit",
        memory_head_indices=(3, 1, 3),
    ))
    assert explicit.resolve_memory_head_indices(4) == [1, 3]


def test_continuous_oracle_schedule_starts_after_native_window() -> None:
    rt = LifeCacheRuntime(LifeCacheRuntimeConfig(
        enabled=True,
        oracle_mode="full_frame",
        oracle_recall_start_frame=24,
        oracle_recall_stride=3,
    ))
    assert not rt._is_oracle_recall_frame(21)
    assert rt._is_oracle_recall_frame(24)
    assert rt._is_oracle_recall_frame(27)
    assert not rt._is_oracle_recall_frame(28)

    rt.config.oracle_recall_frames = [30, 60]
    assert rt._is_oracle_recall_frame(30)
    assert not rt._is_oracle_recall_frame(33)


def test_alignment_gate_rejects_opposing_stale_memory() -> None:
    recent = torch.ones(1, 2, 3, 4)
    opposing = -torch.ones_like(recent)
    fused = fuse_parallel_attention(
        recent,
        opposing,
        gate=0.5,
        rms_match=False,
        alignment_gate=True,
        alignment_threshold=0.0,
    )
    assert torch.equal(fused, recent)

    aligned = fuse_parallel_attention(
        recent,
        recent.clone(),
        gate=0.5,
        rms_match=False,
        alignment_gate=True,
        alignment_threshold=0.0,
    )
    assert torch.allclose(aligned, recent * 1.5)


def test_latent_trace_statistics_preserve_frame_and_channel_axes() -> None:
    latent = torch.arange(1 * 2 * 3 * 2 * 2).reshape(1, 2, 3, 2, 2).float()
    stats = tensor_statistics(latent, channel_dim=2)
    frames = frame_statistics(latent, frame_dim=1)

    assert stats["shape"] == [1, 2, 3, 2, 2]
    assert len(stats["channel_mean"]) == 3
    assert len(frames["frame_mean"]) == 2

    video = torch.zeros(1, 2, 3, 2, 2)
    video[:, :, 0] = 1.0
    video_frames = frame_statistics(video, frame_dim=1)
    assert torch.allclose(
        torch.tensor(video_frames["frame_luma"]),
        torch.full((2,), 0.2126),
    )
