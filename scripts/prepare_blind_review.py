#!/usr/bin/env python3
"""Create, verify, and freeze a randomized human blind-review package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import secrets
import shutil
import string
import uuid
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_VERSION = 2
INDEX_PATTERN = re.compile(r"^(\d+)-(\d+)_([^.]+)\.mp4$")
COMPLETE_NAME = ".complete.json"
FROZEN_NAME = "FROZEN.json"
SCORE_FIELDS = [
    "prompt_index",
    "label",
    "identity_1_to_5",
    "background_1_to_5",
    "motion_1_to_5",
    "camera_1_to_5",
    "artifact_1_to_5",
    "prompt_alignment_1_to_5",
    "startup_flashback_0_or_1",
    "abrupt_jump_0_or_1",
    "polygon_noise_0_or_1",
    "long_range_drift_1_to_5",
    "repetition_looping_1_to_5",
    "overall_rank",
    "failure_time_seconds",
    "notes",
]
ONE_TO_FIVE_FIELDS = {
    "identity_1_to_5",
    "background_1_to_5",
    "motion_1_to_5",
    "camera_1_to_5",
    "artifact_1_to_5",
    "prompt_alignment_1_to_5",
    "long_range_drift_1_to_5",
    "repetition_looping_1_to_5",
}
BINARY_FIELDS = {
    "startup_flashback_0_or_1",
    "abrupt_jump_0_or_1",
    "polygon_noise_0_or_1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _videos(method_dir: Path, count: int) -> list[Path]:
    indexed: dict[int, Path] = {}
    errors: list[str] = []
    for path in sorted(method_dir.glob("*.mp4"), key=lambda item: item.name):
        match = INDEX_PATTERN.fullmatch(path.name)
        if match is None:
            errors.append(f"malformed={path.name}")
            continue
        prompt_index = int(match.group(1))
        sample_index = int(match.group(2))
        if sample_index != 0:
            errors.append(f"unexpected_sample={path.name}")
            continue
        if prompt_index in indexed:
            errors.append(
                f"duplicate_index={prompt_index}:{indexed[prompt_index].name},{path.name}"
            )
            continue
        indexed[prompt_index] = path
    expected = set(range(count))
    missing = sorted(expected - set(indexed))
    extra = sorted(set(indexed) - expected)
    empty = sorted(path.name for path in indexed.values() if path.stat().st_size <= 0)
    if errors or missing or extra or empty:
        raise ValueError(
            f"{method_dir} video mismatch: errors={errors[:10]} "
            f"missing={missing[:10]} extra={extra[:10]} empty={empty[:10]}"
        )
    return [indexed[index] for index in range(count)]


def _prompt_lines(prompts: Path, count: int) -> list[str]:
    lines = [
        line.strip()
        for line in prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != count:
        raise ValueError(f"expected {count} prompts, found {len(lines)} in {prompts}")
    return lines


def _source_inventory(
    run_root: Path,
    methods: list[str],
    prompt_count: int,
) -> tuple[dict[str, list[Path]], list[dict[str, object]], str]:
    method_videos = {
        method: _videos(run_root / method, prompt_count) for method in methods
    }
    inventory: list[dict[str, object]] = []
    for method in methods:
        for prompt_index, path in enumerate(method_videos[method]):
            inventory.append(
                {
                    "method": method,
                    "prompt_index": prompt_index,
                    "file": str(path.resolve()),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return method_videos, inventory, _canonical_sha256(inventory)


def _copy_candidate(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source.resolve(), destination)
    # Do not expose source mtimes (or hard-link inode identity) through the
    # public review bundle. Candidate size is intrinsic to the encoded video,
    # but filesystem metadata must not become an assignment side channel.
    os.utime(destination, (946684800, 946684800))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _scorecard_identity(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {"prompt_index": int(row["prompt_index"]), "label": row["label"]}
        for row in rows
    ]


def _read_scorecard(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    return fields, rows


def _safe_remove_tree(path: Path) -> None:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor).resolve() or resolved == Path.home().resolve():
        raise ValueError(f"refusing to remove unsafe blind-review path: {resolved}")
    shutil.rmtree(resolved)


def _validate_package_targets(
    run_root: Path,
    output: Path,
    private_output: Path,
) -> None:
    source_root = run_root.resolve()
    for name, target in (
        ("public", output.resolve()),
        ("private", private_output.resolve()),
    ):
        if (
            target == Path(target.anchor).resolve()
            or target == Path.home().resolve()
            or target == source_root
            or target in source_root.parents
        ):
            raise ValueError(
                f"refusing unsafe {name} blind-review target: {target}"
            )


def _install_directory(temporary: Path, output: Path, *, force: bool) -> None:
    output = output.resolve()
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")
    backup = output.parent / f".{output.name}.backup.{uuid.uuid4().hex}"
    moved_old = False
    if output.exists():
        os.replace(output, backup)
        moved_old = True
    try:
        os.replace(temporary, output)
    except Exception:
        if moved_old and backup.exists():
            os.replace(backup, output)
        raise
    if moved_old:
        _safe_remove_tree(backup)


def _install_directory_pair(
    pairs: list[tuple[Path, Path]],
    *,
    force: bool,
) -> None:
    resolved_pairs = [
        (temporary.resolve(), output.resolve())
        for temporary, output in pairs
    ]
    if len({output for _, output in resolved_pairs}) != len(resolved_pairs):
        raise ValueError("public and private blind-review outputs must differ")
    for _, output in resolved_pairs:
        if output.exists() and not force:
            raise FileExistsError(
                f"{output} already exists; pass --force to replace it"
            )

    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for _, output in resolved_pairs:
            if output.exists():
                backup = output.parent / (
                    f".{output.name}.backup.{uuid.uuid4().hex}"
                )
                os.replace(output, backup)
                backups[output] = backup
        for temporary, output in resolved_pairs:
            os.replace(temporary, output)
            installed.append(output)
    except Exception:
        for output in reversed(installed):
            if output.exists():
                _safe_remove_tree(output)
        for output, backup in backups.items():
            if backup.exists():
                os.replace(backup, output)
        raise
    for backup in backups.values():
        if backup.exists():
            _safe_remove_tree(backup)


def create_package(
    *,
    run_root: Path,
    methods: list[str],
    prompts: Path,
    output: Path,
    private_output: Path,
    prompt_count: int,
    seed: int | None,
    force: bool,
) -> dict[str, object]:
    if len(methods) > len(string.ascii_uppercase):
        raise ValueError("at most 26 methods are supported")
    if len(methods) != len(set(methods)):
        raise ValueError("method names must be unique")
    prompt_lines = _prompt_lines(prompts, prompt_count)
    method_videos, source_inventory, source_fingerprint = _source_inventory(
        run_root, methods, prompt_count
    )
    output = output.resolve()
    private_output = private_output.resolve()
    _validate_package_targets(run_root, output, private_output)
    if output == private_output:
        raise ValueError("public and private blind-review outputs must differ")
    if output in private_output.parents or private_output in output.parents:
        raise ValueError(
            "private blind-review ledger must not be nested inside the "
            "public review bundle"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    private_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_public = output.parent / (
        f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    temporary_private = private_output.parent / (
        f".{private_output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    temporary_public.mkdir()
    temporary_private.mkdir(mode=0o700)
    secret_seed = int(seed) if seed is not None else secrets.randbits(128)
    try:
        rng = random.Random(secret_seed)
        public_entries: list[dict[str, object]] = []
        private_entries: list[dict[str, object]] = []
        score_rows: list[dict[str, object]] = []
        candidate_inventory: list[dict[str, object]] = []
        source_lookup = {
            (str(item["method"]), int(item["prompt_index"])): item
            for item in source_inventory
        }

        for prompt_index, prompt in enumerate(prompt_lines):
            shuffled_methods = list(methods)
            rng.shuffle(shuffled_methods)
            labels = string.ascii_uppercase[: len(shuffled_methods)]
            public_candidates: list[dict[str, object]] = []
            private_candidates: list[dict[str, object]] = []
            for label, method in zip(labels, shuffled_methods):
                source = method_videos[method][prompt_index]
                source_item = source_lookup[(method, prompt_index)]
                relative_video = (
                    Path(f"prompt_{prompt_index:04d}") / f"{label}.mp4"
                )
                _copy_candidate(source, temporary_public / relative_video)
                candidate = {
                    "prompt_index": prompt_index,
                    "label": label,
                    "video": str(relative_video),
                    "size": source_item["size"],
                    "sha256": source_item["sha256"],
                }
                candidate_inventory.append(candidate)
                public_candidates.append(
                    {
                        "label": label,
                        "video": str(relative_video),
                    }
                )
                private_candidates.append(
                    {
                        "label": label,
                        "method": method,
                        "source": str(source.resolve()),
                        "source_sha256": source_item["sha256"],
                    }
                )
                score_rows.append(
                    {
                        field: (
                            prompt_index
                            if field == "prompt_index"
                            else label
                            if field == "label"
                            else ""
                        )
                        for field in SCORE_FIELDS
                    }
                )
            public_entries.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "candidates": public_candidates,
                }
            )
            private_entries.append(
                {"prompt_index": prompt_index, "candidates": private_candidates}
            )

        public_path = temporary_public / "manifest_public.json"
        private_path = temporary_private / "key_private.json"
        scorecard_path = temporary_public / "scorecard.csv"
        _write_json(
            public_path,
            {
                "version": PACKAGE_VERSION,
                "assignment_hidden": True,
                "prompt_sha256": _sha256(prompts),
                "items": public_entries,
            },
        )
        _write_json(
            private_path,
            {
                "version": PACKAGE_VERSION,
                "seed": secret_seed,
                "run_root": str(run_root.resolve()),
                "methods": methods,
                "source_fingerprint": source_fingerprint,
                "items": private_entries,
            },
        )
        try:
            private_path.chmod(0o600)
        except OSError:
            pass
        with scorecard_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
            writer.writeheader()
            writer.writerows(score_rows)

        completion = {
            "version": PACKAGE_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_root": str(run_root.resolve()),
            "methods": methods,
            "prompt_count": prompt_count,
            "candidate_count": prompt_count * len(methods),
            "seed": secret_seed,
            "prompts": str(prompts.resolve()),
            "prompt_sha256": _sha256(prompts),
            "public_output": str(output),
            "private_output": str(private_output),
            "source_fingerprint": source_fingerprint,
            "source_inventory": source_inventory,
            "candidate_inventory": candidate_inventory,
            "manifest_public_sha256": _sha256(public_path),
            "key_private_sha256": _sha256(private_path),
            "scorecard_fields": SCORE_FIELDS,
            "scorecard_identity_sha256": _canonical_sha256(
                _scorecard_identity(
                    [
                        {key: str(value) for key, value in row.items()}
                        for row in score_rows
                    ]
                )
            ),
        }
        completion_path = temporary_private / COMPLETE_NAME
        _write_json(completion_path, completion)
        try:
            completion_path.chmod(0o600)
        except OSError:
            pass
        _install_directory_pair(
            [
                (temporary_private, private_output),
                (temporary_public, output),
            ],
            force=force,
        )
        try:
            private_output.chmod(0o700)
        except OSError:
            pass
    finally:
        for temporary in (temporary_public, temporary_private):
            if temporary.exists():
                _safe_remove_tree(temporary)
    return completion


def verify_package(
    *,
    run_root: Path,
    methods: list[str],
    prompts: Path,
    output: Path,
    private_output: Path,
    prompt_count: int,
    seed: int | None,
) -> dict[str, object]:
    output = output.resolve()
    private_output = private_output.resolve()
    completion_path = private_output / COMPLETE_NAME
    if not completion_path.is_file():
        raise ValueError(f"missing blind-review completion marker: {completion_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected_scalars = {
        "version": PACKAGE_VERSION,
        "run_root": str(run_root.resolve()),
        "methods": methods,
        "prompt_count": prompt_count,
        "candidate_count": prompt_count * len(methods),
        "prompts": str(prompts.resolve()),
        "prompt_sha256": _sha256(prompts),
        "public_output": str(output),
        "private_output": str(private_output),
    }
    if seed is not None:
        expected_scalars["seed"] = int(seed)
    elif not isinstance(completion.get("seed"), int):
        raise ValueError("blind-review private seed is missing or malformed")
    for key, expected in expected_scalars.items():
        if completion.get(key) != expected:
            raise ValueError(
                f"blind-review contract mismatch for {key}: "
                f"expected={expected!r} actual={completion.get(key)!r}"
            )
    _prompt_lines(prompts, prompt_count)
    _, source_inventory, source_fingerprint = _source_inventory(
        run_root, methods, prompt_count
    )
    if completion.get("source_fingerprint") != source_fingerprint:
        raise ValueError("blind-review source video fingerprint is stale")
    if completion.get("source_inventory") != source_inventory:
        raise ValueError("blind-review source inventory is stale")

    public_path = output / "manifest_public.json"
    private_path = private_output / "key_private.json"
    scorecard_path = output / "scorecard.csv"
    for path in (public_path, private_path, scorecard_path):
        if not path.is_file():
            raise ValueError(f"missing blind-review file: {path}")
    if _sha256(public_path) != completion.get("manifest_public_sha256"):
        raise ValueError("manifest_public.json hash mismatch")
    if _sha256(private_path) != completion.get("key_private_sha256"):
        raise ValueError("key_private.json hash mismatch")

    candidate_inventory = completion.get("candidate_inventory")
    if not isinstance(candidate_inventory, list):
        raise ValueError("blind-review candidate inventory is malformed")
    expected_candidate_paths: set[Path] = set()
    for item in candidate_inventory:
        if not isinstance(item, dict):
            raise ValueError("blind-review candidate inventory is malformed")
        relative = Path(str(item["video"]))
        candidate = output / relative
        expected_candidate_paths.add(relative)
        if not candidate.is_file():
            raise ValueError(f"missing blinded video: {candidate}")
        if candidate.stat().st_size != item.get("size"):
            raise ValueError(f"blinded video size mismatch: {candidate}")
        if _sha256(candidate) != item.get("sha256"):
            raise ValueError(f"blinded video hash mismatch: {candidate}")
    actual_candidate_paths = {
        path.relative_to(output)
        for path in output.glob("prompt_*/*.mp4")
    }
    if actual_candidate_paths != expected_candidate_paths:
        raise ValueError(
            "blind-review candidate set mismatch: "
            f"missing={sorted(map(str, expected_candidate_paths - actual_candidate_paths))[:10]} "
            f"extra={sorted(map(str, actual_candidate_paths - expected_candidate_paths))[:10]}"
        )

    fields, rows = _read_scorecard(scorecard_path)
    if fields != completion.get("scorecard_fields"):
        raise ValueError("scorecard schema mismatch")
    identity_hash = _canonical_sha256(_scorecard_identity(rows))
    if identity_hash != completion.get("scorecard_identity_sha256"):
        raise ValueError("scorecard prompt/label rows were reordered or replaced")
    return {
        "completion_sha256": _sha256(completion_path),
        "source_fingerprint": source_fingerprint,
        "candidate_count": len(candidate_inventory),
        "scorecard_rows": len(rows),
        "manifest_public_sha256": _sha256(public_path),
        "key_private_sha256": _sha256(private_path),
    }


def _parse_int(row: dict[str, str], field: str, *, minimum: int, maximum: int) -> int:
    raw = row.get(field, "").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"scorecard row prompt={row.get('prompt_index')} label={row.get('label')} "
            f"requires integer {field}, got {raw!r}"
        ) from error
    if not minimum <= value <= maximum:
        raise ValueError(
            f"scorecard row prompt={row.get('prompt_index')} label={row.get('label')} "
            f"{field} must be in [{minimum},{maximum}], got {value}"
        )
    return value


def validate_completed_scorecard(
    scorecard: Path,
    *,
    prompt_count: int,
    method_count: int,
) -> dict[str, object]:
    fields, rows = _read_scorecard(scorecard)
    if fields != SCORE_FIELDS:
        raise ValueError("scorecard schema mismatch")
    if len(rows) != prompt_count * method_count:
        raise ValueError(
            f"scorecard has {len(rows)} rows; expected {prompt_count * method_count}"
        )
    ranks: dict[int, set[int]] = {index: set() for index in range(prompt_count)}
    seen: set[tuple[int, str]] = set()
    for row in rows:
        prompt_index = _parse_int(
            row, "prompt_index", minimum=0, maximum=prompt_count - 1
        )
        label = row.get("label", "")
        if len(label) != 1 or label not in string.ascii_uppercase[:method_count]:
            raise ValueError(f"invalid scorecard candidate label: {label!r}")
        identity = (prompt_index, label)
        if identity in seen:
            raise ValueError(f"duplicate scorecard row: {identity}")
        seen.add(identity)
        for field in ONE_TO_FIVE_FIELDS:
            _parse_int(row, field, minimum=1, maximum=5)
        for field in BINARY_FIELDS:
            _parse_int(row, field, minimum=0, maximum=1)
        rank = _parse_int(
            row, "overall_rank", minimum=1, maximum=method_count
        )
        ranks[prompt_index].add(rank)
        failure_time = row.get("failure_time_seconds", "").strip()
        if failure_time:
            try:
                if float(failure_time) < 0:
                    raise ValueError
            except ValueError as error:
                raise ValueError(
                    f"failure_time_seconds must be blank or non-negative, got "
                    f"{failure_time!r}"
                ) from error
    expected_ranks = set(range(1, method_count + 1))
    for prompt_index, prompt_ranks in ranks.items():
        if prompt_ranks != expected_ranks:
            raise ValueError(
                f"prompt {prompt_index} ranks must be a permutation of "
                f"1..{method_count}; got {sorted(prompt_ranks)}"
            )
    return {
        "rows": len(rows),
        "scorecard_sha256": _sha256(scorecard),
    }


def freeze_package(
    *,
    output: Path,
    private_output: Path,
    prompt_count: int,
    method_count: int,
    verification: dict[str, object],
    force: bool,
) -> dict[str, object]:
    scorecard = output / "scorecard.csv"
    score_summary = validate_completed_scorecard(
        scorecard,
        prompt_count=prompt_count,
        method_count=method_count,
    )
    marker = private_output / FROZEN_NAME
    frozen = {
        "version": 1,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "completion_sha256": verification["completion_sha256"],
        "source_fingerprint": verification["source_fingerprint"],
        "candidate_count": verification["candidate_count"],
        **score_summary,
    }
    if marker.exists() and not force:
        current = json.loads(marker.read_text(encoding="utf-8"))
        comparable = dict(current)
        comparable.pop("frozen_utc", None)
        expected = dict(frozen)
        expected.pop("frozen_utc", None)
        if comparable == expected:
            return current
        raise FileExistsError(
            f"{marker} exists but is stale; pass --force-freeze after review"
        )
    temporary = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
    _write_json(temporary, frozen)
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    os.replace(temporary, marker)
    return frozen


def verify_frozen_package(
    *,
    output: Path,
    private_output: Path,
    prompt_count: int,
    method_count: int,
    verification: dict[str, object],
) -> dict[str, object]:
    marker = private_output / FROZEN_NAME
    if not marker.is_file():
        raise ValueError(
            f"missing human freeze marker: {marker}; fill scorecard.csv and run --freeze"
        )
    frozen = json.loads(marker.read_text(encoding="utf-8"))
    summary = validate_completed_scorecard(
        output / "scorecard.csv",
        prompt_count=prompt_count,
        method_count=method_count,
    )
    expected = {
        "version": 1,
        "completion_sha256": verification["completion_sha256"],
        "source_fingerprint": verification["source_fingerprint"],
        "candidate_count": verification["candidate_count"],
        **summary,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(
                f"stale human freeze marker for {key}: "
                f"expected={value!r} actual={frozen.get(key)!r}"
            )
    return {
        **verification,
        "scorecard_sha256": summary["scorecard_sha256"],
        "freeze_marker_sha256": _sha256(marker),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--private-output",
        type=Path,
        help=(
            "Private ledger directory. For backward-compatible callers the "
            "default is a sibling named <output>_private; preregistered "
            "experiments should pass this explicitly."
        ),
    )
    parser.add_argument("--prompt-count", type=int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Optional deterministic seed for tests. Omit for a secret "
            "cryptographically generated assignment seed."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--verify-frozen", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-freeze", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.private_output is None:
        args.private_output = args.output.with_name(
            f"{args.output.name}_private"
        )
    if args.prompt_count <= 0:
        raise ValueError("prompt count must be positive")
    if args.force and (args.verify or args.freeze or args.verify_frozen):
        raise ValueError("--force is only valid while creating a package")
    if args.force_freeze and not args.freeze:
        raise ValueError("--force-freeze requires --freeze")

    if not (args.verify or args.freeze or args.verify_frozen):
        result = create_package(
            run_root=args.run_root,
            methods=args.methods,
            prompts=args.prompts,
            output=args.output,
            private_output=args.private_output,
            prompt_count=args.prompt_count,
            seed=args.seed,
            force=args.force,
        )
        action = "created"
    else:
        verification = verify_package(
            run_root=args.run_root,
            methods=args.methods,
            prompts=args.prompts,
            output=args.output,
            private_output=args.private_output,
            prompt_count=args.prompt_count,
            seed=args.seed,
        )
        if args.freeze:
            result = freeze_package(
                output=args.output,
                private_output=args.private_output,
                prompt_count=args.prompt_count,
                method_count=len(args.methods),
                verification=verification,
                force=args.force_freeze,
            )
            action = "frozen"
        elif args.verify_frozen:
            result = verify_frozen_package(
                output=args.output,
                private_output=args.private_output,
                prompt_count=args.prompt_count,
                method_count=len(args.methods),
                verification=verification,
            )
            action = "verified-frozen"
        else:
            result = verification
            action = "verified"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_name(
            f".{args.output_json.name}.tmp.{os.getpid()}"
        )
        _write_json(temporary, result)
        os.replace(temporary, args.output_json)
    print(
        f"[blind-review] {action}: public={args.output} "
        f"private={args.private_output}"
    )
    if action == "created":
        print("[blind-review] fill scorecard.csv, then rerun this command with --freeze")
        print("[blind-review] do not reveal key_private.json until FROZEN.json exists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
