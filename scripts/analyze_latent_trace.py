from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, len(values))
    return float(np.polyfit(x, np.asarray(values, dtype=np.float64), 1)[0])


def summarize(path: Path) -> dict:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    blocks = [event for event in events if event["event"] == "denoised_block"]
    decoded = next(event for event in events if event["event"] == "decoded_video")

    latent_mean = [event["mean"] for event in blocks]
    latent_std = [event["std"] for event in blocks]
    latent_rms = [event["rms"] for event in blocks]
    luma = decoded.get("frame_luma", decoded["frame_mean"])
    quarter = max(1, len(luma) // 4)
    channel_mean = np.asarray([event["channel_mean"] for event in blocks], dtype=np.float64)
    channel_std = np.asarray([event["channel_std"] for event in blocks], dtype=np.float64)
    block_luma = np.asarray(
        [np.mean(part) for part in np.array_split(np.asarray(luma), len(blocks))],
        dtype=np.float64,
    )

    def channel_summary(values: np.ndarray) -> list[dict]:
        result = []
        for channel in range(values.shape[1]):
            series = values[:, channel]
            correlation = float(np.corrcoef(series, block_luma)[0, 1])
            result.append({
                "channel": channel,
                "slope": slope(series.tolist()),
                "luma_correlation": correlation,
            })
        return sorted(result, key=lambda item: abs(item["luma_correlation"]), reverse=True)

    return {
        "num_blocks": len(blocks),
        "decoded_frames": len(luma),
        "latent_mean_slope": slope(latent_mean),
        "latent_std_slope": slope(latent_std),
        "latent_rms_slope": slope(latent_rms),
        "latent_mean_first_quarter": float(np.mean(latent_mean[: max(1, len(blocks) // 4)])),
        "latent_mean_last_quarter": float(np.mean(latent_mean[-max(1, len(blocks) // 4) :])),
        "latent_std_first_quarter": float(np.mean(latent_std[: max(1, len(blocks) // 4)])),
        "latent_std_last_quarter": float(np.mean(latent_std[-max(1, len(blocks) // 4) :])),
        "luma_slope": slope(luma),
        "luma_first_quarter": float(np.mean(luma[:quarter])),
        "luma_last_quarter": float(np.mean(luma[-quarter:])),
        "luma_relative_change": float(
            np.mean(luma[-quarter:]) / max(np.mean(luma[:quarter]), 1e-8) - 1
        ),
        "channel_mean_luma": channel_summary(channel_mean),
        "channel_std_luma": channel_summary(channel_std),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = {path.parent.name: summarize(path) for path in args.traces}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
