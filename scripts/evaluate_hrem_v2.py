#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_aba_return import evaluate_video  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=3)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "native_raw", "native_reset", "oracle_episode0", "dual_episode_only", "hrem_v2"
        ],
    )
    parser.add_argument("--baseline", default="native_reset")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result: dict[str, object] = {"per_video": {}, "aggregate": {}, "paired_delta": {}}
    per_method: dict[str, list[dict]] = {}
    for method in args.methods:
        metrics = []
        for index in range(args.prompt_count):
            path = args.run_root / method / f"{index}-0_ema.mp4"
            if not path.exists():
                raise FileNotFoundError(path)
            value = evaluate_video(path)
            result["per_video"][f"{method}/{index}"] = value
            metrics.append(value)
        per_method[method] = metrics
        aggregate = {
            view: {
                key: float(np.mean([item[view][key] for item in metrics]))
                for key in (
                    "a1_a2", "a1_b", "scene_separation", "b_a2", "return_margin"
                )
            }
            for view in ("full", "background")
        }
        result["aggregate"][method] = aggregate
        print(method, json.dumps(aggregate, sort_keys=True))

    native = per_method.get(args.baseline)
    if native is not None:
        for method, metrics in per_method.items():
            if method == args.baseline:
                continue
            method_delta = {}
            for view in ("full", "background"):
                values = [
                    metrics[index][view]["return_margin"]
                    - native[index][view]["return_margin"]
                    for index in range(args.prompt_count)
                ]
                method_delta[view] = {
                    "return_margin_values": values,
                    "return_margin_mean": float(np.mean(values)),
                    "positive_prompts": int(sum(value > 0.0 for value in values)),
                }
            result["paired_delta"][method] = method_delta
            print("paired_delta", method, json.dumps(method_delta, sort_keys=True))

    output = args.output or (args.run_root / "metrics_aba.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
