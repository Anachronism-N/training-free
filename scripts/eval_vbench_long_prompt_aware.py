#!/usr/bin/env python3
"""Run VBench-Long while mapping numeric video ids back to real prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


INDEXED_STEM = re.compile(r"^(\d+)-0$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomically(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_prompt_items(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("comparison manifest is not an object")
    count = int(payload.get("prompt_count", -1))
    rows = payload.get("prompt_items")
    if count <= 0 or not isinstance(rows, list) or len(rows) != count:
        raise ValueError("comparison manifest has invalid prompt items")
    prompts: list[str] = []
    for expected_index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or int(row.get("index", -1)) != expected_index
            or not str(row.get("text", "")).strip()
        ):
            raise ValueError(
                f"invalid prompt item at index {expected_index}"
            )
        prompts.append(str(row["text"]).strip())
    return prompts


def _folder_stem(video_path: str) -> str:
    path = Path(video_path)
    for parent in path.parents:
        if INDEXED_STEM.fullmatch(parent.name):
            return parent.name
    raise ValueError(f"cannot recover indexed split folder from {video_path}")


def rewrite_full_info_prompts(
    full_info_path: Path,
    prompts: list[str],
    vbench_full_info_path: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(full_info_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("VBench full-info artifact is not a list")
    # Load auxiliary_info templates from VBench_full_info.json for dims
    # that require it (object_class, multiple_objects, color, scene,
    # spatial_relationship, appearance_style).
    aux_templates: dict[str, list] = {}
    if vbench_full_info_path is not None and vbench_full_info_path.exists():
        full_info = json.loads(vbench_full_info_path.read_text(encoding="utf-8"))
        for item in full_info:
            if not isinstance(item, dict) or "auxiliary_info" not in item:
                continue
            dims = item.get("dimension")
            dims_list = dims if isinstance(dims, list) else [dims]
            for d in dims_list:
                if d and d in item["auxiliary_info"]:
                    aux_templates.setdefault(d, []).append(item["auxiliary_info"][d])
    observed: set[int] = set()
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("VBench full-info row is not an object")
        videos = row.get("video_list")
        if not isinstance(videos, list) or not videos:
            raise ValueError("VBench full-info row has no videos")
        stem = _folder_stem(str(videos[0]))
        match = INDEXED_STEM.fullmatch(stem)
        if match is None:
            raise ValueError(f"invalid split folder stem: {stem}")
        index = int(match.group(1))
        if not 0 <= index < len(prompts):
            raise ValueError(f"prompt index outside manifest: {index}")
        if index in observed:
            raise ValueError(f"duplicate VBench prompt group: {index}")
        row["prompt_en"] = prompts[index]
        row["v129_prompt_index"] = index
        observed.add(index)
        # VBench build_full_info_json does not carry auxiliary_info when
        # using custom prompts. Cycle through VBench's own auxiliary_info
        # templates so dimensions that need it (object_class, color, etc.)
        # can compute their metrics.
        dims = row.get("dimension")
        dims_list = dims if isinstance(dims, list) else [dims]
        if "auxiliary_info" not in row:
            for d in dims_list:
                if d and d in aux_templates and aux_templates[d]:
                    template = aux_templates[d][index % len(aux_templates[d])]
                    row.setdefault("auxiliary_info", {})[d] = template
    expected = set(range(len(prompts)))
    if observed != expected:
        raise ValueError(
            f"VBench prompt coverage mismatch: missing={sorted(expected-observed)} "
            f"extra={sorted(observed-expected)}"
        )
    full_info_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "prompt_count": len(prompts),
        "mapped_count": len(observed),
        "indices": sorted(observed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vbench-root", required=True, type=Path)
    parser.add_argument("--comparison-manifest", required=True, type=Path)
    parser.add_argument("--videos_path", required=True, type=Path)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--full_json_dir", required=True, type=Path)
    parser.add_argument("--num_of_samples_per_prompt", type=int, default=1)
    parser.add_argument("--dev_flag", action="store_true")
    parser.add_argument(
        "--local-models",
        action="store_true",
        help="Load VBench models from VBENCH_CACHE_DIR without downloading.",
    )
    parser.add_argument(
        "--torch-hub-dir",
        type=Path,
        help="Shared torch.hub cache containing DINO/DINOv2 repositories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.vbench_root = args.vbench_root.resolve()
    args.comparison_manifest = args.comparison_manifest.resolve()
    prompts = load_prompt_items(args.comparison_manifest)
    sys.path.insert(0, str(args.vbench_root))
    import torch
    if args.torch_hub_dir is not None:
        torch.hub.set_dir(str(args.torch_hub_dir.expanduser().resolve()))
    from vbench2_beta_long import VBenchLong

    class PromptAwareVBenchLong(VBenchLong):
        def build_full_info_json(
            self,
            videos_path,
            name,
            dimension_list,
            prompt_list=None,
            special_str="",
            verbose=False,
            mode="vbench_standard",
            **kwargs,
        ):
            prompt_list = [] if prompt_list is None else prompt_list
            path = super().build_full_info_json(
                videos_path,
                name,
                dimension_list,
                prompt_list,
                special_str,
                verbose,
                mode,
                **kwargs,
            )
            if mode == "long_custom_input":
                vbench_full_info = Path(self.full_info_dir)
                report = rewrite_full_info_prompts(Path(path), prompts, vbench_full_info)
                report.update(
                    {
                        "version": 1,
                        "comparison_manifest": str(
                            args.comparison_manifest
                        ),
                        "comparison_manifest_sha256": sha256(
                            args.comparison_manifest
                        ),
                        "full_info": str(Path(path).resolve()),
                        "full_info_sha256": sha256(Path(path)),
                        "prompt_mapping": "comparison_manifest_exact",
                    }
                )
                write_json_atomically(
                    args.output_path / "prompt_mapping.json",
                    report,
                )
                print(
                    "[V129PromptMap] "
                    f"mapped={report['mapped_count']} "
                    f"manifest={args.comparison_manifest}",
                    flush=True,
                )
            return path

    args.output_path.mkdir(parents=True, exist_ok=True)
    evaluator = PromptAwareVBenchLong(
        torch.device("cuda"),
        str(args.full_json_dir),
        str(args.output_path),
    )
    evaluator.evaluate(
        videos_path=str(args.videos_path),
        name=f"v129_{args.dimension}",
        prompt_list=[],
        dimension_list=[args.dimension],
        local=bool(args.local_models),
        read_frame=False,
        mode="long_custom_input",
        sb_clip2clip_feat_extractor="dinov2",
        bg_clip2clip_feat_extractor="dreamsim",
        imaging_quality_preprocessing_mode="longer",
        clip_length_config="clip_length_mix.yaml",
        w_inclip=1.0,
        w_clip2clip=0.0,
        use_semantic_splitting=False,
        slow_fast_eval_config=str(
            args.vbench_root
            / "vbench2_beta_long"
            / "configs"
            / "slow_fast_params.yaml"
        ),
        dev_flag=bool(args.dev_flag),
        sb_mapping_file_path=str(
            args.vbench_root
            / "vbench2_beta_long"
            / "configs"
            / "subject_mapping_table.yaml"
        ),
        bg_mapping_file_path=str(
            args.vbench_root
            / "vbench2_beta_long"
            / "configs"
            / "background_mapping_table.yaml"
        ),
        num_of_samples_per_prompt=args.num_of_samples_per_prompt,
        static_filter_flag=False,
    )


if __name__ == "__main__":
    main()
