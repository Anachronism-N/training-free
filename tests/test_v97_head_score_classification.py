from __future__ import annotations

import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

classifier = importlib.import_module("classify_v97_qk_head_scores")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_matrix(path: Path) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            [int(value) for value in row]
            for row in csv.reader(handle)
            if row
        ]


def test_offline_classification_preserves_scores_and_builds_controls(
    tmp_path,
    monkeypatch,
):
    scores = tmp_path / "scores.csv"
    fieldnames = (
        "layer",
        "head",
        "cfg_raw",
        "semantic_raw",
        "cfg_score",
        "semantic_score",
        "consensus_score",
        "positive_rate",
        "mean_logit",
        "mean_abs_logit",
        "signed_logit_mass",
        "sign_switch_rate",
        "dominant_period",
        "spectral_peak_ratio",
    )
    layer_scores = (
        (-2.0, -0.5, 0.5, 2.0),
        (-3.0, 0.0, 1.0, 3.0),
    )
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer, values in enumerate(layer_scores):
            for head, value in enumerate(values):
                writer.writerow(
                    {
                        "layer": layer,
                        "head": head,
                        "cfg_raw": abs(value),
                        "semantic_raw": abs(value),
                        "cfg_score": value,
                        "semantic_score": value,
                        "consensus_score": value,
                        "positive_rate": 0.75 if head % 2 == 0 else 0.25,
                        "mean_logit": value,
                        "mean_abs_logit": abs(value),
                        "signed_logit_mass": value / 4.0,
                        "sign_switch_rate": 0.1,
                        "dominant_period": 6.0,
                        "spectral_peak_ratio": 0.5,
                    }
                )
    original_hash = _sha256(scores)
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "head_count": 8,
                "score_definition": {"classification": None},
                "files": {"score_csv_sha256": original_hash},
            }
        ),
        encoding="utf-8",
    )
    pf = tmp_path / "pf.csv"
    with pf.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(
            [[1, -1, 2, 2], [-1, 1, 2, 1]]
        )
    output = tmp_path / "maps"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify",
            "--scores",
            str(scores),
            "--score-artifact",
            str(artifact),
            "--pf-labels",
            str(pf),
            "--output-dir",
            str(output),
            "--num-layers",
            "2",
            "--num-heads",
            "4",
            "--manual-thresholds",
            "0,1,2",
            "--main-threshold",
            "1",
            "--sign-thresholds",
            "0.5",
        ],
    )

    classifier.main()

    assert _sha256(scores) == original_hash
    assert _read_matrix(output / "prompt_tau_1.csv") == [
        [1, 1, 1, -1],
        [1, 1, 1, -1],
    ]
    assert _read_matrix(output / "prompt_tau_1_reversed.csv") == [
        [-1, 1, 1, 1],
        [-1, 1, 1, 1],
    ]
    random_map = _read_matrix(output / "prompt_tau_1_random.csv")
    assert all(row.count(-1) == 1 for row in random_map)
    assert _read_matrix(output / "sign_rpos_0p5.csv") == [
        [1, -1, 1, -1],
        [1, -1, 1, -1],
    ]
    assert _read_matrix(output / "pf_anchor_wave_vs_veil.csv") == [
        [1, 1, -1, -1],
        [1, 1, -1, 1],
    ]
    assert _read_matrix(output / "pf_native.csv") == [
        [1, -1, 2, 2],
        [-1, 1, 2, 1],
    ]
    report = json.loads(
        (output / "head_map_classification_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["classification_is_posthoc"]
    assert not report["pf_labels_used_to_construct_prompt_maps"]
    assert report["score_csv_sha256"] == original_hash


def test_classifier_rejects_modified_score_csv(tmp_path, monkeypatch):
    scores = tmp_path / "scores.csv"
    scores.write_text("layer,head\n0,0\n", encoding="utf-8")
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "head_count": 1,
                "score_definition": {"classification": None},
                "files": {"score_csv_sha256": "not-the-real-hash"},
            }
        ),
        encoding="utf-8",
    )
    pf = tmp_path / "pf.csv"
    pf.write_text("1\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify",
            "--scores",
            str(scores),
            "--score-artifact",
            str(artifact),
            "--pf-labels",
            str(pf),
            "--output-dir",
            str(tmp_path / "maps"),
            "--num-layers",
            "1",
            "--num-heads",
            "1",
        ],
    )

    try:
        classifier.main()
    except ValueError as error:
        assert "does not match immutable score artifact" in str(error)
    else:
        raise AssertionError("modified score CSV was accepted")
