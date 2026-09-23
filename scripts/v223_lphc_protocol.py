#!/usr/bin/env python3
"""One fixed-seed dose/phase ablation, reusing the original SF/Ours videos."""
import os
from pathlib import Path

import v212_lphc_protocol as base
import v216_lphc_protocol as parent
import v219_lphc_protocol as reference
from summarize_v213_evidence import checked_hash
from prepare_v210_vbench_comparison import DIMENSIONS

LABEL = "v223"
EXPERIMENT = "v223_frozen_dose_phase_ablation32"
# Four uniformly spaced original GPU rows, all eight nodes; no metric selection.
SOURCE_INDICES = tuple(s for i, s in enumerate(reference.SOURCE_INDICES) if (i // 8) % 2 == 0)
REUSED = ("sf_fifo21", "ours_correct")
FRESH = ("strong_e1", "phase_full")


class Protocol(parent.Protocol):
    LABEL, EXPERIMENT, SOURCE_INDICES = LABEL, EXPERIMENT, SOURCE_INDICES
    STAGE, SEED, FRAMES = "ablation32", 21600, 120
    METHODS = REUSED + FRESH
    GATE_PAIRS = ()
    MECHANISM = (("ours_correct", "strong_e1"), ("ours_correct", "phase_full"))
    SMOKE_REQUIRED = False

    def build_specs(self, candidate):
        if candidate != "headwise_correct":
            raise ValueError("v223 requires the frozen v219 headwise method")
        specs = super().build_specs(candidate)
        specs.update(strong_e1={**specs["ours_correct"], "alpha": .10},
                     phase_full={**specs["ours_correct"], "phase": "full"})
        return specs

    def check_scope(self):
        super().check_scope()
        import export_lphc_horizon_evidence as evidence
        self.reference_root = Path(self.scope["reference_root"]).resolve()
        if self.reference_root == self.out:
            raise ValueError("reference and new output roots must differ")
        self.reference = evidence.load(self.reference_root, "v219")
        old = self.reference
        if (old["report_sha256"] != self.scope["reference_report_sha256"]
                or old["comparison"]["input_manifest_sha256"] != self.scope["reference_input_sha256"]
                or base.sha256(self.reference_root / "evaluation/vbench_comparison/comparison_manifest.json")
                != self.scope["reference_comparison_sha256"]):
            raise ValueError("v219 reference evidence changed")
        for key in ("authorized_nodes", "evidence_sha256", "selected_method", "primary_metric", "primary_window"):
            if self.scope[key] != old["scope"][key]:
                raise ValueError(f"v223 cannot retune {key}")
        self.EXTRA_METRICS = tuple(DIMENSIONS)
        self.PRIMARY_HYPOTHESIS = "Frozen Ours vs SF; dose and phase contrasts are supplemental, not a method search"

    def _check_inputs(self, data):
        super()._check_inputs(data)
        old = self.reference["inputs"]
        by_source = {r["source_index"]: r for r in old["prompt_items"]}
        for item in data["prompt_items"]:
            for key in ("text", "sha256", "effective_seed"):
                if item[key] != by_source[item["source_index"]][key]:
                    raise ValueError("reference prompt/seed changed")
        for method in self.METHODS:
            original = "sf_fifo21" if method == "sf_fifo21" else "ours_correct"
            if data["configs"][method]["sha256"] != old["configs"][original]["sha256"]:
                raise ValueError("ablation changed inference configuration")
        if parent.inference_paths(data["runtime_paths"]) != parent.inference_paths(old["runtime_paths"]):
            raise ValueError("inference operator changed; v219 videos cannot be reused")

    def assignment(self, source, slots):
        if tuple(slots) != tuple(map(str, range(8))):
            raise ValueError("retain original GPU slots 0..7")
        # Keep each prompt on its original physical GPU for receipt pairing.
        return base.assignment(source, slots, sources=reference.SOURCE_INDICES, num_nodes=8)

    def method_order(self, source):
        if source not in self.SOURCE_INDICES:
            raise ValueError("source is outside the frozen ablation subset")
        return FRESH if self.SOURCE_INDICES.index(source) % 2 == 0 else FRESH[::-1]

    def placement(self, slots):
        result = super().placement(slots)
        result.update(new_video_count=64, reused_video_count=64, active_gpu_count=32,
                      reference_root=str(self.reference_root), generation_methods=list(FRESH))
        return result

    def completion_path(self, stage, method, source):
        from run_v212_lphc import job_path
        if method in REUSED:
            return job_path(self.reference_root, reference.Protocol.STAGE, method, source) / "done.json"
        return job_path(self.out, stage, method, source) / "done.json"

    def reused_done(self, stage, method, source):
        if method not in REUSED:
            return None
        if stage != self.STAGE or source not in self.SOURCE_INDICES:
            raise ValueError("invalid reused job identity")
        path = self.completion_path(stage, method, source)
        expected = next(j for j in self.reference["comparison"]["jobs"]
                        if j["method"] == method and j["source_index"] == source)
        checked_hash(path, expected["done_sha256"])
        row = parent.read(path)
        media = Path(row["media"]["path"])
        if base.sha256(media) != row["media"]["sha256"]:
            raise ValueError("reference video changed")
        return row

    def require_reference_gate(self, data):
        self._check_inputs(data)
        # Source gates remain source gates; no fabricated new-run gate0 receipt.
        return {"kind": "unchanged_operator_reference_gate_reuse", "pass": True,
                "reference_root": str(self.reference_root),
                "reference_input_sha256": self.scope["reference_input_sha256"],
                "new_input_sha256": base.sha256(self.out / "inputs/manifest.json"),
                "new_gate_video_count": 0}

    def validate_generation_target(self, source, gpu_uuid, hostname):
        row = self.reused_done(self.STAGE, "ours_correct", source)
        if (row["gpu_uuid"], row["hostname"]) != (gpu_uuid, hostname):
            raise ValueError("run v223 on the original v219 node/GPU; do not reassign prompt bundles")

    def prepare(self, *args, **kwargs):
        data = super().prepare(*args, **kwargs)
        for source in self.SOURCE_INDICES:
            for method in REUSED:
                self.reused_done(self.STAGE, method, source)
        self.frozen_json(self.out / "decisions/reference_reuse.json", self.require_reference_gate(data))
        return data


def load(out=None):
    path = out or os.environ.get("V223_OUT_ROOT")
    if not path:
        raise ValueError("set V223_OUT_ROOT to a new v223_ directory")
    return Protocol(path)
