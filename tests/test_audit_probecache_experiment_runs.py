import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "audit_probecache_experiment_runs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_probecache_experiment_runs",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_cell(root, *, videos=1, log="done", trace=True):
    (root / "configs").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "traces").mkdir()
    (root / "cell").mkdir()
    (root / "configs" / "cell.env").write_text(
        "\n".join(
            [
                "name=cell",
                "method=probecache_full",
                "expected_videos=1",
                "frames=120",
                "seed=0",
                "trace_required=1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for index in range(videos):
        (root / "cell" / f"{index}.mp4").write_bytes(b"video")
    (root / "logs" / "cell.log").write_text(log, encoding="utf-8")
    if trace:
        (root / "traces" / "cell.probecache.jsonl").write_text(
            "\n".join(
                [
                    '{"event":"archive_update","archive_size":1}',
                    (
                        '{"event":"middle_selection","role":"persistent",'
                        '"accepted":true,"sync_t":4,"selected_times":[0]}'
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def test_run_audit_accepts_complete_cell(tmp_path):
    _write_cell(tmp_path)
    report = MODULE.audit_run_root(tmp_path)
    assert report["complete_count"] == 1
    assert report["incomplete_count"] == 0


def test_run_audit_detects_traceback_and_missing_outputs(tmp_path):
    _write_cell(
        tmp_path,
        videos=0,
        log="Traceback (most recent call last)\nRuntimeError: failed",
        trace=False,
    )
    report = MODULE.audit_run_root(tmp_path)
    assert report["incomplete_count"] == 1
    assert report["cells"][0]["issues"] == [
        "videos=0/1",
        "log_error",
        "missing_trace",
    ]


def test_trace_audit_rejects_recent_or_future_recall(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                '{"event":"archive_update","archive_size":1}',
                (
                    '{"event":"middle_selection","role":"reactive",'
                    '"accepted":true,"sync_t":4,"selected_times":[2,5]}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = MODULE.audit_trace(
        trace,
        recent_exclude_frames=4,
        archive_max_frames=24,
    )
    assert report["issues"] == [
        "trace_future_reads=1",
        "trace_recent_overlap=2",
    ]
