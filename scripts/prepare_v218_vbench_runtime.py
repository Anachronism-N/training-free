#!/usr/bin/env python3
"""Build a separate VBench runtime restoring pinned upstream RAFT, never edit a running checkout."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from v210_vbench_fingerprint import vbench_checkout_fingerprint
from v212_lphc_protocol import frozen_json, sha256

PIN = "45e79ec14e69a2187202c675d2dbce1a71843d53"
DD = "vbench/dynamic_degree.py"
RAFT = "vbench/third_party/RAFT/core/raft.py"
STRICT_LOADER = '''
def load_model(self):
    import hashlib
    import json
    from pathlib import Path
    checkpoint = Path(self.args.model).resolve()
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    self.model = RAFT(self.args)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or not ckpt:
        raise ValueError("RAFT checkpoint must be a nonempty upstream state dict")
    if not all(isinstance(k, str) for k in ckpt):
        raise ValueError("RAFT state dict keys must be strings")
    state = {k.removeprefix("module."): v for k, v in ckpt.items()}
    if len(state) != len(ckpt):
        raise ValueError("RAFT checkpoint has colliding module prefixes")
    try:
        self.model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError("Wrong RAFT checkpoint for VBench upstream backend; do not use torchvision weights or strict=False") from error
    self.model.to(self.device)
    self.model.eval()
    print("[v218-raft-loaded] " + json.dumps({
        "backend": "vbench_upstream_raft", "class_module": type(self.model).__module__,
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest.hexdigest(),
        "strict": True, "state_entries": len(state), "input": "RGB float 0..255",
        "normalization": "inside upstream RAFT.forward: 2*(x/255)-1", "iterations": 20
    }), flush=True)
'''


def strict_source(upstream):
    tree = ast.parse(upstream)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DynamicDegree"]
    if len(classes) != 1:
        raise ValueError("unexpected pinned DynamicDegree module")
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "load_model"]
    if len(methods) != 1:
        raise ValueError("unexpected pinned RAFT loader")
    classes[0].body[classes[0].body.index(methods[0])] = ast.parse(STRICT_LOADER).body[0]
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def prepare(source, target, comparison_path, receipt_path):
    source, target = source.resolve(), target.resolve()
    comparison = json.loads(comparison_path.read_text())
    before = vbench_checkout_fingerprint(source)
    if before != comparison["vbench_fingerprint"] or before["head"] != PIN:
        raise ValueError("source VBench must be the exact pinned v215 evaluator snapshot")
    if source == target or target.is_relative_to(source) or receipt_path.resolve().is_relative_to(target):
        raise ValueError("use separate runtime and receipt paths outside the source checkout")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt["source_comparison_sha256"] != sha256(comparison_path)
                or receipt["target_root"] != str(target)
                or receipt["fingerprint"] != vbench_checkout_fingerprint(target)):
            raise ValueError("existing repaired runtime changed")
        return receipt
    if target.exists():
        raise ValueError("target already exists without its receipt; choose a new empty path")
    upstream = git(source, "show", f"{PIN}:{DD}").decode()
    payload = strict_source(upstream)
    subprocess.run(["git", "clone", "--no-hardlinks", str(source), str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "switch", "--detach", PIN], check=True)
    # Preserve all local compatibility repairs, except the broken DD backend.
    for name, digest in before["runtime_path_sha256"].items():
        src, dst = (source/name).resolve(), (target/name).resolve()
        if not src.is_relative_to(source) or not dst.is_relative_to(target):
            raise ValueError("runtime path escapes checkout")
        if name == DD:
            continue
        if digest is None:
            if dst.is_file():
                dst.unlink()
        else:
            if sha256(src) != digest:
                raise ValueError("source changed during copy")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    (target/DD).write_bytes(payload)
    # Verify the whole backend, not just its entry point: normalization and
    # checkpoint compatibility also depend on the encoder/update/correlation code.
    raft_paths = git(source, "ls-tree", "-r", "--name-only", PIN, "--", "vbench/third_party/RAFT").decode().splitlines()
    raft_hashes = {}
    for name in raft_paths:
        if not name.endswith(".py"):
            continue
        expected = git(source, "show", f"{PIN}:{name}")
        if not (target/name).is_file() or (target/name).read_bytes() != expected:
            raise ValueError(f"RAFT implementation differs from pinned upstream: {name}")
        raft_hashes[name] = hashlib.sha256(expected).hexdigest()
    if RAFT not in raft_hashes:
        raise ValueError("pinned upstream RAFT is incomplete")
    if vbench_checkout_fingerprint(source) != before:
        raise ValueError("source changed during repair preparation")
    receipt = {"version": 1, "source_comparison_sha256": sha256(comparison_path),
               "source_root": str(source), "target_root": str(target), "source_fingerprint": before,
               "fingerprint": vbench_checkout_fingerprint(target), "upstream_commit": PIN,
               "upstream_dynamic_sha256": hashlib.sha256(upstream.encode()).hexdigest(),
               "strict_dynamic_sha256": hashlib.sha256(payload).hexdigest(),
               "raft_sha256": raft_hashes[RAFT], "raft_python_sha256": raft_hashes,
               "changed_metric": "dynamic_degree", "generation_changed": False}
    frozen_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-vbench", type=Path, required=True)
    parser.add_argument("--target-vbench", type=Path, required=True)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_vbench, args.target_vbench, args.comparison_manifest, args.receipt), indent=2))


if __name__ == "__main__":
    main()
