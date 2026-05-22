"""LongMemEvalEnv — non-economic, session-streaming env.

A run answers a single QA item. ``begin_session(session_idx)`` exposes
the ``session_idx``-th haystack session (sorted by date) as a briefing
note + caches the structured turns on the env so agents can ingest the
session into their data store. ``step_session`` advances to the next
session and runs no scoring of its own — the only scored event is
the end-of-history probe wired up in :mod:`tasks.probes`.

Probe wiring lives in :mod:`tasks.probes`: at construction time we
register the loaded item as the *active probe target*, so
``get_probes_for_session(N)`` (where ``N == total_sessions``) returns
one ProbeSpec for this run. This is set per-run because the question
and its haystack length are item-specific.
"""

from __future__ import annotations

from typing import Any

from Scroll.core import BaseEnvironment, SessionResult, EnvSnapshot
from Scroll.benchmarks.longmemeval.catalog import LongMemEvalEnvConfig
from Scroll.benchmarks.longmemeval.dataset import (
    LongMemEvalItem,
    load_items,
    select_item,
)


_LME_SUBSTRATE_ENDGAME = """\
RUN STRUCTURE — LONGMEMEVAL:
- Past chat sessions are ingested by the harness directly into your
  log + memoryspace; you are not called per haystack session and have
  no per-session bookkeeping to do.
- ``today`` (1-indexed int) is bound as the current session ordinal.
- At the END of the run a single probe question fires asking about
  content from one or more past sessions. Reply formatting rules
  will be in the probe-mode system prompt that swaps in then, not
  here.
"""


class LongMemEvalEnv(BaseEnvironment):
    def __init__(self, cfg: LongMemEvalEnvConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.session_idx = 0

        # Memory optimization: when running per-QA via ``question_id``,
        # ``load_items`` auto-detects per-QA shards (~500KB each) under
        # ``<dataset>_shards/`` and loads only the matching shard
        # instead of the full ~264MB dataset. Reduces parallel-sweep
        # peak memory ~500× per subprocess. Run
        # ``scripts/shard_lme_dataset.py <dataset>`` once to enable.
        items = load_items(cfg.dataset_path, question_id=cfg.question_id)
        self.item: LongMemEvalItem = select_item(
            items, cfg.question_id, cfg.question_index
        )

        # Mutate cfg.num_sessions so the session-loop runs ``total_sessions``
        # ingestion iterations PLUS one extra "probe session" with no
        # haystack content. The probe fires at end-of-iteration on the
        # probe session, in a clean(er) history that doesn't share
        # context with any single haystack session's ingest activity.
        # ``is_terminal`` is updated to allow the +1 iteration through.
        cfg.num_sessions = self.item.total_sessions + 1

        self._current_session: list[dict] | None = None
        self._current_session_id: str | None = None
        self._current_session_date: str | None = None
        self._today_logs: list[str] = []

        # Register the active probe for this run. Imported lazily so
        # the tasks subpackage can also import env types without a cycle.
        from Scroll.benchmarks.longmemeval.tasks import probes as _probes
        _probes.set_active_probe(self.item, cfg)

    # ------------------------------------------------------------------
    # Per-session API
    # ------------------------------------------------------------------

    def begin_session(self, session_idx: int) -> list[str]:
        """Stage the next haystack session.

        Called by the datasource (and through it the session-loop) at the
        start of each session, *before* ``agent.run_session``. Returns a
        list of briefing lines that the framework prepends to the agent's
        session prompt as ``Today's briefing``.

        ``session_idx`` is the env's current session counter
        (``env.session_idx``), 0 at the first call.
        """
        idx = session_idx
        if idx >= self.item.total_sessions:
            self._current_session = None
            self._current_session_id = None
            self._current_session_date = None
            self._today_logs = ["no_session_remaining"]
            return list(self._today_logs)

        self._current_session = self.item.haystack_sessions[idx]
        self._current_session_id = self.item.haystack_session_ids[idx]
        self._current_session_date = self.item.haystack_dates[idx]
        notes = [
            f"session_id={self._current_session_id}",
            f"session_date={self._current_session_date}",
            f"turn_count={len(self._current_session)}",
        ]
        self._today_logs = notes
        return list(notes)

    def step_session(self) -> SessionResult:
        """Advance to the next session.

        LME has no economic state — ``SessionResult`` carries zeros and
        the only meaningful side-effect is incrementing ``self.session_idx``.
        """
        self.session_idx += 1
        return SessionResult(
            session_idx=self.session_idx,
            sold_units=0,
            revenue=0.0,
            machine_cash=0.0,
            cash=0.0,
        )

    # ``today_logs`` / ``net_worth`` inherit the BaseEnvironment defaults
    # (``[]`` and ``0.0``) — LME is a passive observation env with no
    # outcomes or economic state. ``_today_logs`` is still maintained
    # for ``build_snapshot`` / session-log consumers.

    def visible_state(self) -> dict:
        return {
            "session_idx": self.session_idx,
            "total_sessions": self.item.total_sessions,
            "current_session_id": self._current_session_id,
            "current_session_date": self._current_session_date,
            "current_session_turn_count": (
                len(self._current_session) if self._current_session else 0
            ),
            "question_date": self.item.question_date,
        }

    def is_terminal(self) -> bool:
        # Allow exactly one iteration past the last session for the
        # probe-only session (see ``__init__``: cfg.num_sessions is
        # total_sessions+1). Becomes terminal AFTER the probe session
        # has run.
        return self.session_idx > self.item.total_sessions

    def substrate_endgame_prompt(self) -> str:
        return _LME_SUBSTRATE_ENDGAME

    def probe_substrate_prompt(self) -> str:
        from Scroll.benchmarks.longmemeval.tasks.probes import LME_PROBE_FORMAT
        return LME_PROBE_FORMAT

    def probe_user_postscript(self) -> str:
        from Scroll.benchmarks.longmemeval.tasks.probes import compose_user_postscript
        return compose_user_postscript()

    def build_snapshot(self) -> EnvSnapshot:
        return EnvSnapshot(
            session_idx=self.session_idx,
            logs=list(self._today_logs),
            extra={
                "session_id": self._current_session_id,
                "session_date": self._current_session_date,
                "session_turn_count": (
                    len(self._current_session) if self._current_session else 0
                ),
                "total_sessions": self.item.total_sessions,
                "question_id": self.item.question_id,
                "question_type": self.item.question_type,
            },
        )

    # ------------------------------------------------------------------
    # Helpers consumed by agent ingest paths
    # ------------------------------------------------------------------

    @property
    def current_session(self) -> list[dict] | None:
        """Structured turns for today's session ([{role, content}, ...])."""
        return self._current_session

    @property
    def current_session_meta(self) -> dict[str, Any]:
        return {
            "session_id": self._current_session_id,
            "session_date": self._current_session_date,
            "turn_count": (
                len(self._current_session) if self._current_session else 0
            ),
        }

    @property
    def question_data(self) -> LongMemEvalItem:
        return self.item

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def to_checkpoint(self) -> dict:
        return {
            "session_idx": self.session_idx,
            "seed": self.seed,
            "question_id": self.item.question_id,
            "today_logs": list(self._today_logs),
        }

    @classmethod
    def from_checkpoint(cls, data: dict, cfg: LongMemEvalEnvConfig) -> "LongMemEvalEnv":
        # Force the cfg to point at the same item so the dataset reload
        # picks the same haystack. ``dataset_path`` is also assumed
        # unchanged across resume — config_hash will reject mismatched
        # configs at the checkpoint loader level.
        cfg.question_id = data["question_id"]
        env = cls(cfg, seed=data.get("seed", 0))
        env.session_idx = data["session_idx"]
        env._today_logs = list(data.get("today_logs") or [])
        return env
