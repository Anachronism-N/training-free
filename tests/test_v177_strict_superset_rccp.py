from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v173_cache_compatibility as base  # noqa: E402
import analyze_v176_superset_rccp as analysis  # noqa: E402


def test_v177_contract_is_isolated_and_strict() -> None:
    contract = base.PROFILE_CONTRACTS["v177"]
    assert contract == {
        "version": 3,
        "method": "strict_superset_residual_cache_compatibility",
        "expected_budget": {
            "recent": 9,
            "coverage": 9,
            "episode": 9,
            "union": 17,
        },
        "calls": [0, 1, 2, 3],
        "expected_records_per_prompt_layer": 48,
        "trace_layers": {0, 10, 20, 29},
        "record_contract": "v177",
        "reference_superset": True,
    }
    assert base.PROFILE_CONTRACTS["v176"]["version"] == 2


def test_v177_preserves_the_frozen_holdout_split() -> None:
    old = analysis.frozen_prompt_split(range(128))
    corrected = analysis.frozen_prompt_split(
        range(128), discovery_seed=1762026
    )
    assert corrected == old
    discovery, validation, generation = corrected
    assert (len(discovery), len(validation), len(generation)) == (64, 32, 32)
    assert not (set(discovery) & set(validation))
    assert not (set(discovery) & set(generation))
    assert not (set(validation) & set(generation))


def test_v177_runtime_has_hard_representation_superset_audit() -> None:
    cache = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "adaptive_cache.py"
    ).read_text(encoding="utf-8")
    core = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "wan"
        / "modules"
        / "attention"
        / "core.py"
    ).read_text(encoding="utf-8")
    runner = (SCRIPTS / "run_v177_strict_superset_rccp_32gpu.sh").read_text(
        encoding="utf-8"
    )
    assert 'policy == "union" and contract == "v177"' in cache
    assert "middle_recent_min_t" in cache
    assert 'if profile_contract in {"v176", "v177"}:' in core
    assert "raise RuntimeError(message)" in core
    assert "representation_sets" in core
    assert 'source.startswith("anchor_")' in core
    assert 'source.startswith("anchor_")' in (
        SCRIPTS / "analyze_v173_cache_compatibility.py"
    ).read_text(encoding="utf-8")
    assert "candidate_representation_subset_verified" in core
    assert "duplicate physical anchor has non-equivalent" in cache
    assert "torch.equal(anchor.k, previous.k)" in cache
    assert 'if contract in {"v176", "v177"}' in cache
    assert "PROFILE_CONTRACT=v177" in runner
    assert "V177_OUT_ROOT" in runner


def _traced_budget(frames: list[tuple[int, str]]) -> dict:
    codebook = sorted({source for _, source in frames})
    source_to_code = {source: index for index, source in enumerate(codebook)}
    return {
        "max_frame_equivalents": len(frames),
        "per_sequence_frame_equivalents": [len(frames)] * base.HEADS,
        "selected_physical_frames_per_sequence": [
            [[frame, source_to_code[source]] for frame, source in frames]
            for _ in range(base.HEADS)
        ],
        "selected_source_codebook": codebook,
    }


def test_v177_loader_rejects_missing_representation(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    recent = [(0, "static_dynamic_rope")] + [
        (frame, "dynamic_saved_rope") for frame in range(7, 15)
    ]
    coverage = (
        [(0, "static_dynamic_rope")]
        + [(frame, "dynamic_saved_rope") for frame in range(11, 15)]
        + [
            (frame, "anchor_dynamic_rope:temporal_reservoir")
            for frame in (2, 3, 5, 6)
        ]
    )
    episode = (
        [(0, "static_dynamic_rope")]
        + [(frame, "dynamic_saved_rope") for frame in range(11, 15)]
        + [
            (4, "anchor_dynamic_rope:episode_reservoir"),
            (6, "anchor_dynamic_rope:episode_reservoir"),
        ]
        + [
            (7, "anchor_dynamic_rope:coherent_motion"),
            (8, "anchor_dynamic_rope:coherent_motion"),
        ]
    )
    # Missing (6, anchor) reproduces the v176 failure. Frame 6 existing as a
    # dynamic token would not satisfy this representation-aware audit.
    union = (
        [(0, "static_dynamic_rope")]
        + [(frame, "dynamic_saved_rope") for frame in range(7, 15)]
        + [
            (frame, "anchor_dynamic_rope:temporal_reservoir")
            for frame in (2, 3, 5)
        ]
        + [(4, "anchor_dynamic_rope:episode_reservoir")]
    )
    record = {
        "prompt_id": 0,
        "layer": 0,
        "heads": base.HEADS,
        "current_frame": 12,
        "call_index": 0,
        "cache_update_mode": "noisy",
        "cfg_branch": "cond",
        "profile_contract": "v177",
        "policies": {
            policy: {
                metric: ([0.1] * base.HEADS)
                for metric in (
                    "residual_relative_mse",
                    "residual_cosine",
                    "raw_relative_mse",
                    "raw_cosine",
                    "output_rms",
                )
            }
            for policy in base.POLICIES
        },
        "budgets": {
            "recent": _traced_budget(recent),
            "coverage": _traced_budget(coverage),
            "episode": _traced_budget(episode),
            "union": _traced_budget(union)
            | {
                "candidate_physical_superset_verified": True,
                "superset_verification_contract": "v177",
                "candidate_representation_subset_verified": True,
                "candidate_representation_subset_checks": 36,
                "candidate_representation_subset_failures": 0,
            },
        },
    }
    payload = {
        "version": 3,
        "contract": "v177",
        "method": "strict_superset_residual_cache_compatibility",
        "policies": list(base.POLICIES),
        "records": [record],
    }
    monkeypatch.setattr(base, "profile_paths", lambda _root: [Path("fake.pt")])
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(base, "sha256", lambda _path: "0" * 64)
    with pytest.raises(ValueError, match="not a teacher subset"):
        base.load_records(Path("unused"), contract="v177")


def test_v177_runner_requires_zero_warning_smoke() -> None:
    shared = (SCRIPTS / "run_v176_superset_rccp_32gpu.sh").read_text(
        encoding="utf-8"
    )
    assert '"teacher is not a cache-representation superset" not in log' in shared
    assert "assert len(records) == 1440" in shared
    assert "candidate_representation_subset_checks" in shared
    assert "candidate_representation_subset_verified" in shared
    assert "[profile-log-audit] PASS" in shared


def test_v177_loader_requires_explicit_representation_codebook() -> None:
    source = (SCRIPTS / "analyze_v173_cache_compatibility.py").read_text(
        encoding="utf-8"
    )
    assert "legacy or unknown representation family" in source
    assert "anchor_dynamic_rope:" in source
    assert "dynamic_time_mapped" in source


def test_v176_artifact_is_explicitly_invalidated() -> None:
    invalid = (ROOT / "runs" / "v176_superset_rccp" / "INVALID.md").read_text(
        encoding="utf-8"
    )
    assert "4,668" in invalid
    assert "must not be used" in invalid


def test_v177_analysis_declares_representation_teacher_contract() -> None:
    source = (SCRIPTS / "analyze_v176_superset_rccp.py").read_text(
        encoding="utf-8"
    )
    runner = (SCRIPTS / "run_v176_superset_rccp_32gpu.sh").read_text(
        encoding="utf-8"
    )
    assert "candidate_representation_superset_required" in source
    assert "physical_frame_and_representation_family" in source
    assert "teacher_requires_representation_candidate_superset" in runner
    assert "v177 analysis requires frozen input provenance" in source
    assert '--input-manifest "$INPUT_ROOT/manifest.json"' in runner
