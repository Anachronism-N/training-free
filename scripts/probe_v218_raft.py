#!/usr/bin/env python3
"""GPU preflight: strict weight loading, actual backend receipt and synthetic flow diagnostics."""
import argparse
import json
from pathlib import Path
import sys

from v212_lphc_protocol import frozen_json, sha256
from v210_vbench_fingerprint import vbench_checkout_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.runtime_receipt.read_text())
    root = Path(receipt["target_root"])
    if receipt["fingerprint"] != vbench_checkout_fingerprint(root):
        raise ValueError("corrected runtime drift")
    sys.path.insert(0, str(root))
    import torch
    from easydict import EasyDict
    from vbench.dynamic_degree import DynamicDegree
    if Path(sys.modules[DynamicDegree.__module__].__file__).resolve() != (root/"vbench/dynamic_degree.py").resolve():
        raise ValueError("imported the wrong VBench module")
    model = DynamicDegree(EasyDict(model=str(args.checkpoint.resolve()), small=False,
                                  mixed_precision=False, alternate_corr=False), torch.device("cuda"))
    generator = torch.Generator(device="cpu").manual_seed(218)
    a = torch.rand((1, 3, 128, 128), generator=generator, dtype=torch.float32).mul(255).cuda()
    b = torch.roll(a, shifts=4, dims=-1)
    stats = {}
    with torch.no_grad():
        for name, image in (("identical", a), ("right_shift4", b)):
            _, flow = model.model(a, image, iters=20, test_mode=True)
            if not torch.isfinite(flow).all():
                raise FloatingPointError("RAFT returned nonfinite synthetic flow")
            crop = flow[:, :, 24:-24, 24:-24]
            stats[name] = {"median_x": crop[:, 0].median().item(), "median_y": crop[:, 1].median().item(),
                           "mean_magnitude": crop.square().sum(1).sqrt().mean().item()}
    result = {"runtime_receipt_sha256": sha256(args.runtime_receipt), "checkpoint_sha256": sha256(args.checkpoint),
              "strict_loading_pass": True, "synthetic_flow": stats,
              "boundary": "Diagnostic only: expect near-zero identical flow and positive x for right-shift. No video quality or method efficacy claim."}
    frozen_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
