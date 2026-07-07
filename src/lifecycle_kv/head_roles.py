from __future__ import annotations

import csv
import json
from pathlib import Path

from .cache_types import HeadRole


ROLE_ALIASES = {
    "anchor": HeadRole.ANCHOR,
    "layout": HeadRole.LAYOUT,
    "stable": HeadRole.LAYOUT,
    "static": HeadRole.LAYOUT,
    "spatial": HeadRole.LAYOUT,
    "recall": HeadRole.RECALL,
    "semantic": HeadRole.RECALL,
    "motion": HeadRole.MOTION,
    "dynamic": HeadRole.MOTION,
    "temporal": HeadRole.MOTION,
    "wave": HeadRole.WAVE,
    "osc": HeadRole.WAVE,
    "generic": HeadRole.GENERIC,
    "unknown": HeadRole.UNKNOWN,
}


def parse_head_role(value: str | int | float) -> HeadRole:
    text = str(value).strip().lower()
    if text in {"1", "sta+", "sta", "stable+"}:
        return HeadRole.LAYOUT
    if text in {"2", "sta-", "stable-"}:
        return HeadRole.GENERIC
    if text in {"-1", "osc", "oscillating"}:
        return HeadRole.WAVE
    return ROLE_ALIASES.get(text, HeadRole.UNKNOWN)


def load_head_roles(path: str | Path) -> dict[tuple[int, int], HeadRole]:
    """Load head-role priors from Pyramid CSV or Forcing-KV style JSON."""

    path = Path(path)
    if path.suffix.lower() == ".json":
        return _load_json_roles(path)
    if path.suffix.lower() == ".csv":
        return _load_csv_roles(path)
    raise ValueError(f"unsupported head-role file: {path}")


def _load_json_roles(path: Path) -> dict[tuple[int, int], HeadRole]:
    data = json.loads(path.read_text(encoding="utf-8"))
    roles: dict[tuple[int, int], HeadRole] = {}

    def add_group(items: object, role: HeadRole) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, dict):
                layer = item.get("layer") or item.get("layer_id")
                head = item.get("head") or item.get("head_id")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                layer, head = item[0], item[1]
            else:
                continue
            roles[(int(layer), int(head))] = role

    if isinstance(data, dict):
        for key, value in data.items():
            role = parse_head_role(key)
            if role != HeadRole.UNKNOWN:
                add_group(value, role)
            elif isinstance(value, dict):
                for sub_key, items in value.items():
                    add_group(items, parse_head_role(sub_key))
    return roles


def _load_csv_roles(path: Path) -> dict[tuple[int, int], HeadRole]:
    roles: dict[tuple[int, int], HeadRole] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lower = {k.lower(): v for k, v in row.items() if k is not None}
            layer = lower.get("layer") or lower.get("layer_id") or lower.get("block")
            head = lower.get("head") or lower.get("head_id")
            label = lower.get("role") or lower.get("label") or lower.get("class")
            if layer is None or head is None or label is None:
                continue
            roles[(int(layer), int(head))] = parse_head_role(label)
    return roles
