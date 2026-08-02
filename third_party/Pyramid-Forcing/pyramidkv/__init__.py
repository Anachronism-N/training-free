"""PyramidKV frame selection strategies.

This package implements the [sink ... middle ... recent] architecture
for per-head KV cache management. Three middle strategies are available
and can be combined (union):

- CyclicStrategy: t mod T phase-bucket anchors
- LagStrategy: fixed-offset t-k anchors
- StrideStrategy: every k-th frame anchors
- MergeStrategy: spatiotemporal patch-block merging
- MotionEventStrategy: clean-value-change event anchors

HeadComposition ties sink_frames + middle strategies + recent_frames together
for each attention head. The factory module builds compositions from YAML config.
"""

from .base import FrameAnchor, HeadComposition, MiddleStrategy
from .cyclic import CyclicStrategy
from .lag import LagStrategy
from .stride import StrideStrategy
from .temporal_reservoir import (
    TemporalProfileAnchorStrategy,
    TemporalReservoirStrategy,
)
from .merge import MergeStrategy
from .motion_event import MotionEventStrategy
from .role_event import CoherentMotionStrategy, SemanticLandmarkStrategy
from .role_memory import (
    SemanticRetrievalStrategy,
    SparseSnapshotStrategy,
    TemporalPrototypeStrategy,
    UniqueSnapshotStrategy,
)
from .recent import RecentStrategy
from .factory import (
    HEAD_LABEL_MAP,
    build_compositions,
    load_head_labels,
)
from .config import PyramidKVConfig
from .cache import PyramidKVCache
from .adaptive_cache import AdaptiveKVCache
from .transition import (
    CacheTransitionConfig,
    CacheTransitionController,
    CacheTransitionDecision,
)
from .probecache import (
    ProbeCacheConfig,
    ProbeCacheController,
    ProbeCacheSelection,
)
from .prompt_warmup import PromptWarmupShield, PromptWarmupShieldConfig

__all__ = [
    "FrameAnchor",
    "HeadComposition",
    "MiddleStrategy",
    "CyclicStrategy",
    "LagStrategy",
    "StrideStrategy",
    "TemporalReservoirStrategy",
    "TemporalProfileAnchorStrategy",
    "MergeStrategy",
    "MotionEventStrategy",
    "SemanticLandmarkStrategy",
    "CoherentMotionStrategy",
    "SemanticRetrievalStrategy",
    "TemporalPrototypeStrategy",
    "UniqueSnapshotStrategy",
    "SparseSnapshotStrategy",
    "RecentStrategy",
    "HEAD_LABEL_MAP",
    "build_compositions",
    "load_head_labels",
    "PyramidKVConfig",
    "PyramidKVCache",
    "AdaptiveKVCache",
    "CacheTransitionConfig",
    "CacheTransitionController",
    "CacheTransitionDecision",
    "ProbeCacheConfig",
    "ProbeCacheController",
    "ProbeCacheSelection",
    "PromptWarmupShield",
    "PromptWarmupShieldConfig",
]
