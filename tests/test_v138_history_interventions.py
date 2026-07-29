import json
from pathlib import Path

import torch

from lifecycle_kv.head_profile import HeadProfileConfig, HeadProfileSession
from lifecycle_kv.history_interventions import (
    apply_history_rope,
    build_history_interventions,
    reposition_history_key,
)


def _attention(query, key, value):
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key)
    probs = scores.softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, value)


def _frequency_table(rows, complex_dim):
    row = torch.arange(rows, dtype=torch.float64).unsqueeze(1)
    column = torch.arange(complex_dim, dtype=torch.float64).unsqueeze(0)
    angles = 0.013 * row * (column + 1)
    return torch.polar(torch.ones_like(angles), angles)


def _apply_history_rope(raw, positions, height, width, freqs):
    _, sequence, heads, feature = raw.shape
    frames = len(positions)
    complex_dim = feature // 2
    split = freqs.split(
        [
            complex_dim - 2 * (complex_dim // 3),
            complex_dim // 3,
            complex_dim // 3,
        ],
        dim=1,
    )
    temporal = (
        split[0][positions]
        .view(frames, 1, 1, -1)
        .expand(frames, height, width, -1)
    )
    height_freq = (
        split[1][:height]
        .view(1, height, 1, -1)
        .expand(frames, height, width, -1)
    )
    width_freq = (
        split[2][:width]
        .view(1, 1, width, -1)
        .expand(frames, height, width, -1)
    )
    multiplier = torch.cat(
        (temporal, height_freq, width_freq), dim=-1
    ).reshape(sequence, 1, -1)
    value = torch.view_as_complex(
        raw.double().reshape(sequence, heads, -1, 2)
    )
    return torch.view_as_real(value * multiplier).flatten(2).unsqueeze(0)


def test_rope_reposition_has_identity_parity_and_reverses_content():
    generator = torch.Generator().manual_seed(5)
    frames, height, width = 5, 2, 2
    heads, feature = 2, 12
    frame_tokens = height * width
    raw = torch.randn(
        (1, frames * frame_tokens, heads, feature),
        generator=generator,
        dtype=torch.float64,
    )
    positions = torch.arange(7, 7 + frames)
    freqs = _frequency_table(32, feature // 2)
    roped = _apply_history_rope(raw, positions, height, width, freqs)
    grid = torch.tensor([[3, height, width]])
    normal = torch.arange(frames)
    identity = reposition_history_key(
        raw,
        grid_sizes=grid,
        freqs=freqs,
        target_t_pos=positions,
        frame_order=normal,
    )
    torch.testing.assert_close(identity, roped, atol=1e-9, rtol=1e-9)

    reverse = torch.arange(frames - 1, -1, -1)
    reversed_key = reposition_history_key(
        raw,
        grid_sizes=grid,
        freqs=freqs,
        target_t_pos=positions,
        frame_order=reverse,
    )
    raw_reversed = raw.reshape(
        1, frames, frame_tokens, heads, feature
    ).index_select(1, reverse).reshape_as(raw)
    expected = _apply_history_rope(
        raw_reversed, positions, height, width, freqs
    )
    torch.testing.assert_close(
        reversed_key, expected, atol=1e-9, rtol=1e-9
    )

    repeated = torch.tensor([3, 3, 3, 3, 4])
    frozen_key = reposition_history_key(
        raw,
        grid_sizes=grid,
        freqs=freqs,
        target_t_pos=positions,
        frame_order=repeated,
    )
    raw_frozen = raw.reshape(
        1, frames, frame_tokens, heads, feature
    ).index_select(1, repeated).reshape_as(raw)
    expected_frozen = _apply_history_rope(
        raw_frozen, positions, height, width, freqs
    )
    torch.testing.assert_close(
        frozen_key, expected_frozen, atol=1e-9, rtol=1e-9
    )


def test_pre_rope_reconstruction_detects_wrong_source_positions():
    generator = torch.Generator().manual_seed(17)
    frames, height, width = 5, 2, 2
    heads, feature = 2, 12
    raw = torch.randn(
        (1, frames * height * width, heads, feature),
        generator=generator,
        dtype=torch.float64,
    )
    positions = torch.arange(7, 7 + frames)
    freqs = _frequency_table(32, feature // 2)
    expected = _apply_history_rope(raw, positions, height, width, freqs)
    wrong = apply_history_rope(
        raw,
        grid_sizes=torch.tensor([[frames, height, width]]),
        freqs=freqs,
        temporal_positions=positions + 1,
    )
    assert float((wrong - expected).abs().max()) > 1e-3


def test_history_interventions_return_all_declared_outputs():
    generator = torch.Generator().manual_seed(9)
    frames, height, width = 6, 2, 2
    heads, feature = 2, 12
    frame_tokens = height * width
    raw_key = torch.randn(
        (1, frames * frame_tokens, heads, feature),
        generator=generator,
        dtype=torch.float64,
    )
    positions = torch.arange(10, 10 + frames)
    freqs = _frequency_table(32, feature // 2)
    history_key = _apply_history_rope(
        raw_key, positions, height, width, freqs
    )
    history_value = torch.randn(
        history_key.shape, generator=generator, dtype=torch.float64
    )
    query = torch.randn(
        (1, 3 * frame_tokens, heads, feature),
        generator=generator,
        dtype=torch.float64,
    )
    outputs, metadata = build_history_interventions(
        query=query,
        raw_history_key=raw_key,
        history_key=history_key,
        history_value=history_value,
        frame_seq_length=frame_tokens,
        current_frame=16,
        recent_frames=4,
        grid_sizes=torch.tensor([[3, height, width]]),
        freqs=freqs,
        attention_fn=_attention,
    )
    assert set(outputs) == {
        "reverse",
        "phase_shift",
        "freeze_latest",
        "value_mismatch",
    }
    assert all(output.shape == query.shape for output in outputs.values())
    assert metadata["pre_rope_sidecar"] == 1.0
    assert metadata["rope_reconstruction_relative_max"] < 1e-8
    assert metadata["rope_reconstruction_relative_rms"] < 1e-8
    assert metadata["recent_value_preservation_max"] == 0.0


def test_v4_profile_records_projected_descriptors(tmp_path):
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset_index": 0,
                "job_id": "intervention",
                "kind": "history_intervention",
                "base_prompt": "base",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = HeadProfileConfig(
        enabled=True,
        output_dir=tmp_path / "profiles",
        manifest_path=manifest,
        noisy_frames=(21,),
        noisy_timesteps=(1000,),
        clean_frames=(),
        recent_frames=4,
        spatial_samples=2,
        strict=True,
        history_interventions=True,
        projection_dim=8,
    )
    session = HeadProfileSession(config)
    session.begin_video(
        dataset_index=0,
        text_prompts=["base"],
        num_frames=120,
        frame_seq_length=4,
        num_frame_per_block=3,
        local_attn_size=21,
    )
    assert session.set_call_context(
        branch="base",
        mode="noisy",
        current_frame=21,
        nominal_timestep=1000,
        actual_timestep=999.0,
    )
    generator = torch.Generator().manual_seed(12)
    query = torch.randn((1, 12, 2, 8), generator=generator)
    current_key = torch.randn((1, 12, 2, 8), generator=generator)
    history_key = torch.randn((1, 24, 2, 8), generator=generator)
    history_value = torch.randn((1, 24, 2, 8), generator=generator)
    native = _attention(
        query,
        torch.cat((history_key, current_key), dim=1),
        torch.cat((history_value, history_value[:, :12]), dim=1),
    )
    intervention_outputs = {
        name: _attention(query, history_key, history_value)
        for name in ("reverse", "phase_shift", "freeze_latest", "value_mismatch")
    }
    session.record_attention(
        layer=0,
        query=query,
        current_key=current_key,
        history_key=history_key,
        history_value=history_value,
        native_output=native,
        frame_seq_length=4,
        attention_fn=_attention,
        history_intervention_outputs=intervention_outputs,
        history_intervention_metadata={
            "pre_rope_sidecar": 1.0,
            "rope_reconstruction_relative_max": 0.0,
            "rope_reconstruction_relative_rms": 0.0,
            "recent_value_preservation_max": 0.0,
        },
    )
    output = session.end_video(expected_layers=1)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["version"] == 4
    record = payload["records"][0]
    assert record["query_projection"].shape == (2, 2, 8)
    assert record["history_key_projection"].shape == (2, 6, 2, 8)
    assert "history_reverse_signature" in record
    assert "history_value_mismatch_signature" in record


def test_model_routes_interventions_only_through_explicit_profile_gate():
    root = Path(__file__).parents[1]
    model = (
        root
        / "third_party"
        / "Self-Forcing"
        / "wan"
        / "modules"
        / "causal_model.py"
    ).read_text(encoding="utf-8")
    assert "profile_session.wants_history_interventions()" in model
    assert "profile_session.config.history_interventions" in model
    assert 'kv_cache.get("k_pre_rope")' in model
    assert "raw_history_key=profile_raw_history_key" in model
    assert "build_history_interventions(" in model
    assert "history intervention profiling requires native SF" in model
