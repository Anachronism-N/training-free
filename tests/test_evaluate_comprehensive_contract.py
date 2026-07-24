from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_comprehensive.py"
)


def test_skip_flags_are_connected_to_single_video_evaluation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "skip_m3: bool = False" in source
    assert "skip_m4: bool = False" in source
    assert "if skip_m3:" in source
    assert "if skip_m4:" in source
    assert "skip_m3=args.skip_m3" in source
    assert "skip_m4=args.skip_m4" in source
