"""Prompt-role-guided history exposure during autoregressive warmup."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Sequence


_TRACE_LOCK = threading.Lock()


@dataclass(frozen=True)
class PromptWarmupShieldConfig:
    enabled: bool = False
    blocks: int = 0
    release_span: int = 0
    mode: str = "middle"
    shield_labels: tuple[int, ...] = (-1,)
    layer_start: int = 0
    layer_end: int = -1
    trace_path: str | None = None
    debug: bool = False

    def validate(self) -> None:
        if self.blocks < 0:
            raise ValueError("prompt warmup blocks must be non-negative")
        if self.release_span < 0:
            raise ValueError("prompt warmup release_span must be non-negative")
        if self.mode not in {"middle", "history"}:
            raise ValueError("prompt warmup mode must be middle or history")
        if self.enabled and self.blocks <= 0:
            raise ValueError("enabled prompt warmup requires blocks > 0")
        if self.enabled and not self.shield_labels:
            raise ValueError("enabled prompt warmup requires shield labels")
        if self.layer_start < 0:
            raise ValueError("prompt warmup layer_start must be non-negative")
        if self.layer_end >= 0 and self.layer_end <= self.layer_start:
            raise ValueError("prompt warmup layer_end must exceed layer_start")


class PromptWarmupShield:
    """Deterministically shields history for selected prompt-role heads.

    The underlying PF strategies continue to update while reads are shielded.
    Once a head's release threshold is reached, it immediately uses the
    already-warm PF history instead of starting a second cache from scratch.
    """

    def __init__(
        self,
        config: PromptWarmupShieldConfig,
        *,
        layer_idx: int,
        head_labels: Sequence[int],
    ) -> None:
        config.validate()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.head_labels = tuple(int(value) for value in head_labels)
        self._last_trace_key: tuple[str, int, tuple[bool, ...]] | None = None
        self._layer_enabled = (
            config.enabled
            and config.layer_start <= self.layer_idx
            and (config.layer_end < 0 or self.layer_idx < config.layer_end)
        )
        labels = frozenset(config.shield_labels)
        self._eligible = tuple(
            self._layer_enabled and label in labels for label in self.head_labels
        )
        self._release_blocks = tuple(
            self._release_block(head_idx) if eligible else 0
            for head_idx, eligible in enumerate(self._eligible)
        )
        if config.debug and self._layer_enabled:
            print(
                "[PromptWarmupShield] "
                f"layer={self.layer_idx} mode={config.mode} "
                f"blocks={config.blocks} release_span={config.release_span} "
                f"eligible={sum(self._eligible)}/{len(self._eligible)} "
                f"labels={list(config.shield_labels)}",
                flush=True,
            )

    @property
    def enabled(self) -> bool:
        return self._layer_enabled and any(self._eligible)

    def _release_block(self, head_idx: int) -> int:
        if self.config.release_span <= 0:
            return self.config.blocks
        # Stable integer mixing avoids Python's process-randomized hash().
        phase = (
            self.layer_idx * 131 + int(head_idx) * 17
        ) % (self.config.release_span + 1)
        return self.config.blocks + phase

    def active_mask(self, block_id: int) -> tuple[bool, ...]:
        block_id = int(block_id)
        return tuple(
            eligible and block_id < release
            for eligible, release in zip(self._eligible, self._release_blocks)
        )

    def shields_middle(self, head_idx: int, block_id: int) -> bool:
        return self.active_mask(block_id)[int(head_idx)]

    def shields_sink(self, head_idx: int, block_id: int) -> bool:
        return (
            self.config.mode == "history"
            and self.shields_middle(head_idx, block_id)
        )

    def record(
        self,
        *,
        block_id: int,
        branch: str,
        active_mask: tuple[bool, ...] | None = None,
    ) -> None:
        if not self._layer_enabled:
            return
        mask = self.active_mask(block_id) if active_mask is None else active_mask
        key = (str(branch), int(block_id), tuple(mask))
        if key == self._last_trace_key:
            return
        self._last_trace_key = key
        payload = {
            "event": "prompt_warmup_shield",
            "layer": self.layer_idx,
            "block": int(block_id),
            "branch": str(branch),
            "mode": self.config.mode,
            "base_blocks": self.config.blocks,
            "release_span": self.config.release_span,
            "eligible_heads": sum(self._eligible),
            "active_heads": sum(mask),
            "released_heads": sum(self._eligible) - sum(mask),
            "shield_labels": list(self.config.shield_labels),
            "release_blocks": list(self._release_blocks),
        }
        if self.config.debug:
            print(
                "[PromptWarmupShield] "
                f"layer={self.layer_idx} block={block_id} branch={branch} "
                f"active={payload['active_heads']}/{payload['eligible_heads']} "
                f"mode={self.config.mode}",
                flush=True,
            )
        if self.config.trace_path:
            path = Path(self.config.trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(payload, sort_keys=True) + "\n"
            with _TRACE_LOCK:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line)

    def reset(self) -> None:
        self._last_trace_key = None
