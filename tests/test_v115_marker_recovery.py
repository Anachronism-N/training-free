from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load():
    path = SCRIPTS / "recover_v115_done_markers.py"
    spec = importlib.util.spec_from_file_location("v115_recovery_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


recovery = _load()


def test_cell_payload_is_strict_and_roundtrips():
    payload = {
        "name": "candidate",
        "stage": "support",
        "prompt_kind": "single",
        "support_policy": "landmark",
        "suppress_policy": "recent8_sink1",
    }
    cell = recovery.cell_from_payload(payload)
    assert cell.name == "candidate"
    assert cell.uses_role_event is True

    try:
        recovery.cell_from_payload({**payload, "unknown": True})
    except ValueError as error:
        assert "unknown Cell fields" in str(error)
    else:
        raise AssertionError("unknown frozen config fields must fail")


def test_matching_contract_requires_exactly_one_hash(tmp_path):
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    path = contracts / "all.json"
    path.write_text(json.dumps({"experiment": recovery.EXPERIMENT}), encoding="utf-8")
    digest = recovery.sha256(path)

    assert recovery.matching_contract(tmp_path, digest) == path
    try:
        recovery.matching_contract(tmp_path, "0" * 64)
    except ValueError as error:
        assert "expected one frozen contract" in str(error)
    else:
        raise AssertionError("missing frozen contract must fail")


def test_log_validation_rejects_incomplete_runtime_markers(tmp_path):
    cell = recovery.Cell(
        "candidate",
        "support",
        "single",
        support_policy="landmark",
        suppress_policy="recent8_sink1",
    )
    log = tmp_path / "candidate.log"
    log.write_text(
        "\n".join(
            (
                "[PyramidKVRuntimePolicy]",
                "[HistoryPolarityPolicy] legacy_pf_labels=false "
                "exclusive_owner=true",
                "[PyramidKVRoleEvent]",
            )
        ),
        encoding="utf-8",
    )
    recovery.validate_log(log, cell)

    log.write_text("Traceback (most recent call last)", encoding="utf-8")
    try:
        recovery.validate_log(log, cell)
    except ValueError as error:
        assert "failure signatures" in str(error)
    else:
        raise AssertionError("traceback-bearing logs must fail")
