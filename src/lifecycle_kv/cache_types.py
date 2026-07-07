from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SlotState(str, Enum):
    CURRENT = "current"
    RECENT = "recent"
    ANCHOR = "anchor"
    COMPRESSED = "compressed"
    MOTION = "motion"
    STALE = "stale"
    INVALID = "invalid"
    DROPPED = "dropped"


class HeadRole(str, Enum):
    UNKNOWN = "unknown"
    ANCHOR = "anchor"
    LAYOUT = "layout"
    ENTITY = "entity"
    MOTION = "motion"
    WAVE = "wave"
    VEIL = "veil"


@dataclass
class CacheEntry:
    entry_id: str
    slot_state: SlotState
    layer_id: int
    head_id: int
    chunk_id: int
    token_start: int
    token_end: int

    # Pointer/index into tensor storage. This should point to actual K/V payload.
    kv_ptr: Optional[str] = None

    # Metadata used by the cache index/control plane.
    head_role: HeadRole = HeadRole.UNKNOWN
    scene_id: Optional[str] = None
    entity_ids: list[str] = field(default_factory=list)
    state_version: dict[str, str] = field(default_factory=dict)

    # Scores.
    trust_score: float = 1.0
    motion_score: float = 0.0
    stale_score: float = 0.0
    conflict_score: float = 0.0
    access_count: int = 0
    last_accessed_chunk: int = -1

    # Positional/RoPE metadata.
    rope_mode: str = "post_rope"  # or "pre_rope"
    rope_range: Optional[tuple[int, int]] = None
    position_map_ptr: Optional[str] = None

    extra: dict[str, Any] = field(default_factory=dict)
