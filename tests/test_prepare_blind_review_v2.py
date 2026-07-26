import csv
import json
import sys
from pathlib import Path

import pytest

from scripts.prepare_blind_review import (
    COMPLETE_NAME,
    FROZEN_NAME,
    SCORE_FIELDS,
    create_package,
    freeze_package,
    main,
    verify_frozen_package,
    verify_package,
)


def _source_run(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    run_root = tmp_path / "run"
    methods = ["base", "candidate"]
    for method in methods:
        directory = run_root / method
        directory.mkdir(parents=True)
        for index in range(2):
            (directory / f"{index}-0_ema.mp4").write_bytes(
                f"{method}-{index}".encode()
            )
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first prompt\nsecond prompt\n", encoding="utf-8")
    return run_root, prompts, methods


def _fill_scorecard(path: Path, method_count: int) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in (
            "identity_1_to_5",
            "background_1_to_5",
            "motion_1_to_5",
            "camera_1_to_5",
            "artifact_1_to_5",
            "prompt_alignment_1_to_5",
            "long_range_drift_1_to_5",
            "repetition_looping_1_to_5",
        ):
            row[field] = "4"
        for field in (
            "startup_flashback_0_or_1",
            "abrupt_jump_0_or_1",
            "polygon_noise_0_or_1",
        ):
            row[field] = "0"
        row["overall_rank"] = str(ord(row["label"]) - ord("A") + 1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_blind_package_hashes_sources_and_requires_stable_freeze(tmp_path: Path):
    run_root, prompts, methods = _source_run(tmp_path)
    output = tmp_path / "blind"
    private_output = tmp_path / "blind_private"
    create_package(
        run_root=run_root,
        methods=methods,
        prompts=prompts,
        output=output,
        private_output=private_output,
        prompt_count=2,
        seed=7,
        force=False,
    )

    completion = json.loads(
        (private_output / COMPLETE_NAME).read_text(encoding="utf-8")
    )
    assert completion["candidate_count"] == 4
    assert all(
        len(item["sha256"]) == 64
        for item in completion["candidate_inventory"]
    )
    verification = verify_package(
        run_root=run_root,
        methods=methods,
        prompts=prompts,
        output=output,
        private_output=private_output,
        prompt_count=2,
        seed=7,
    )

    _fill_scorecard(output / "scorecard.csv", len(methods))
    freeze_package(
        output=output,
        private_output=private_output,
        prompt_count=2,
        method_count=len(methods),
        verification=verification,
        force=False,
    )
    frozen = verify_frozen_package(
        output=output,
        private_output=private_output,
        prompt_count=2,
        method_count=len(methods),
        verification=verification,
    )
    assert (private_output / FROZEN_NAME).is_file()
    assert not (output / "key_private.json").exists()
    assert not (output / COMPLETE_NAME).exists()
    assert len(frozen["freeze_marker_sha256"]) == 64

    with (output / "scorecard.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="scorecard"):
        verify_frozen_package(
            output=output,
            private_output=private_output,
            prompt_count=2,
            method_count=len(methods),
            verification=verification,
        )


def test_blind_package_detects_manifest_tampering(tmp_path: Path):
    run_root, prompts, methods = _source_run(tmp_path)
    output = tmp_path / "blind"
    private_output = tmp_path / "blind_private"
    create_package(
        run_root=run_root,
        methods=methods,
        prompts=prompts,
        output=output,
        private_output=private_output,
        prompt_count=2,
        seed=7,
        force=False,
    )
    with (output / "manifest_public.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")

    with pytest.raises(ValueError, match="manifest_public"):
        verify_package(
            run_root=run_root,
            methods=methods,
            prompts=prompts,
            output=output,
            private_output=private_output,
            prompt_count=2,
            seed=7,
        )


def test_cli_legacy_default_keeps_private_ledger_in_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_root, prompts, methods = _source_run(tmp_path)
    output = tmp_path / "blind"
    private_output = tmp_path / "blind_private"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_blind_review.py",
            "--run-root",
            str(run_root),
            "--methods",
            *methods,
            "--prompts",
            str(prompts),
            "--output",
            str(output),
            "--prompt-count",
            "2",
            "--seed",
            "7",
        ],
    )

    assert main() == 0
    assert (private_output / "key_private.json").is_file()
    assert not (output / "key_private.json").exists()
    public = json.loads(
        (output / "manifest_public.json").read_text(encoding="utf-8")
    )
    assert "seed" not in public
    assert "methods" not in public
    for item in public["items"]:
        for candidate in item["candidates"]:
            assert set(candidate) == {"label", "video"}
            candidate_path = output / candidate["video"]
            assert candidate_path.stat().st_mtime == pytest.approx(946684800)
