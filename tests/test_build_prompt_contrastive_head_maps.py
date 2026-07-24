import json
from pathlib import Path

import pytest

from scripts.build_prompt_contrastive_head_maps import (
    PROMPT_RESPONSIVE,
    PROMPT_STABLE,
    _load_report,
    build_maps,
)


def _entries(prompt_rows, remote_rows):
    output = {}
    flat_prompt = [value for row in prompt_rows for value in row]
    flat_remote = [value for row in remote_rows for value in row]
    threshold = sum(
        remote - prompt
        for prompt, remote in zip(flat_prompt, flat_remote)
    ) / len(flat_prompt)
    width = len(prompt_rows[0])
    for layer, (prompt_row, remote_row) in enumerate(
        zip(prompt_rows, remote_rows)
    ):
        for head, (prompt, remote) in enumerate(zip(prompt_row, remote_row)):
            role_score = remote - prompt
            output[(layer, head)] = {
                "layer": layer,
                "head": head,
                "prompt_z": prompt,
                "remote_z": remote,
                "role_score": role_score,
                "label": (
                    PROMPT_STABLE
                    if role_score >= threshold
                    else PROMPT_RESPONSIVE
                ),
            }
    return output


def test_prompt_map_matches_pf_per_layer_budget_and_uses_low_sensitivity():
    pf = [
        [1, -1, 2, 1],
        [-1, 1, -1, 2],
    ]
    primary = _entries(
        [[0.0, 3.0, 1.0, 2.0], [4.0, 1.0, 3.0, 2.0]],
        [[3.0, 0.0, 2.0, 1.0], [0.0, 4.0, 1.0, 2.0]],
    )
    replica = _entries(
        [[0.1, 2.9, 1.1, 2.1], [3.9, 1.1, 3.1, 2.1]],
        [[3.0, 0.0, 2.0, 1.0], [0.0, 4.0, 1.0, 2.0]],
    )

    maps, metadata = build_maps(
        pf,
        primary,
        replica_entries=replica,
        random_seed=7,
    )

    assert metadata["stable_budgets_per_layer"] == [2, 1]
    assert maps["pf_binary"] == [
        [1, -1, -1, 1],
        [-1, 1, -1, -1],
    ]
    assert maps["prompt_pfcount"] == [
        [1, -1, 1, -1],
        [-1, 1, -1, -1],
    ]
    assert maps["prompt_inverse_pfcount"] == [
        [-1, 1, -1, 1],
        [1, -1, -1, -1],
    ]
    assert maps["prompt_replica_pfcount"] == maps["prompt_pfcount"]
    assert maps["prompt_consensus_pfcount"] == maps["prompt_pfcount"]
    assert all(
        row.count(1) == budget
        for row, budget in zip(
            maps["prompt_random_pfcount"],
            metadata["stable_budgets_per_layer"],
        )
    )


def test_natural_prompt_partition_is_independent_of_pf_count():
    pf = [[1, -1, 2, -1]]
    primary = _entries(
        [[-3.0, -2.0, 2.0, 3.0]],
        [[0.0, 0.0, 0.0, 0.0]],
    )

    maps, metadata = build_maps(
        pf,
        primary,
        replica_entries=None,
        random_seed=9,
    )

    assert maps["prompt_kmeans"] == [[1, 1, -1, -1]]
    assert metadata["prompt_kmeans"]["stable_center"] < 0
    assert metadata["prompt_kmeans"]["responsive_center"] > 0


def test_report_loader_rejects_incomplete_coordinates(tmp_path: Path):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "layer": 0,
                        "head": 0,
                        "prompt_z": 0,
                        "remote_z": 0,
                        "role_score": 0,
                        "label": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="coverage mismatch"):
        _load_report(report, rows=1, columns=2)
