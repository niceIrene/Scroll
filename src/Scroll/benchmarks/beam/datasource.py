"""BEAM datasource. Passive, no external channels.

The batch content (turns) is delivered to ``BeamAgent`` via the
unified ``log`` handle (auto-ingested into memoryspace tables before
the probe turn). The email + search channels inherit the
BaseDataSource defaults (no-op).
"""

from __future__ import annotations

from Scroll.core import BaseDataSource


class BeamDataSource(BaseDataSource):
    def __init__(self, seed: int, data_cfg: dict | None = None) -> None:
        self.seed = seed
        self.cfg = data_cfg or {}

    def begin_turn(self, turn_idx: int, env) -> list[str]:
        return env.begin_turn(turn_idx)

    def to_checkpoint(self) -> dict:
        return {"seed": self.seed}

    def from_checkpoint(self, data: dict) -> None:
        self.seed = data.get("seed", self.seed)
