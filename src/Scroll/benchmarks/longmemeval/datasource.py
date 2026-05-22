"""LongMemEval datasource — wraps :meth:`LongMemEvalEnv.begin_session`."""

from __future__ import annotations

from Scroll.core import BaseDataSource


class LongMemEvalDataSource(BaseDataSource):
    def __init__(self, seed: int, data_cfg: dict | None = None) -> None:
        self.seed = seed
        self.cfg = data_cfg or {}

    def begin_session(self, session_idx: int, env) -> list[str]:
        # Drives env.begin_session so the env can stage the current session
        # before agent.run_session spins up. Notes returned here are
        # prepended to the agent's session prompt under "Today's briefing".
        return env.begin_session(session_idx)

    def to_checkpoint(self) -> dict:
        return {"seed": self.seed}

    def from_checkpoint(self, data: dict) -> None:
        self.seed = data.get("seed", self.seed)
