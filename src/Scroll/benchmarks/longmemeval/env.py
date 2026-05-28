"""LongMemEvalEnv — non-economic, session-streaming env.

A run answers a single QA item. ``begin_turn(turn_idx)`` exposes
the ``turn_idx``-th haystack chat session (sorted by date) as a briefing
note + caches the structured turns on the env so agents can ingest the
chat session into their data store. ``step_turn`` advances to the next
chat session and runs no scoring of its own — the only scored event
is the end-of-history probe wired up in :mod:`tasks.probes`.

Probe wiring lives in :mod:`tasks.probes`: at construction time we
register the loaded item as the *active probe target*, so
``get_probes_for_turn(N)`` (where ``N == total_sessions + 1``)
returns one ProbeSpec for this run. This is set per-run because the
question and its haystack length are item-specific. PR #4 retires
the ``+1 probe-only turn`` trick in favor of an end-of-task probe.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from Scroll.core import BaseEnvironment, TurnResult, EnvSnapshot
from Scroll.benchmarks.longmemeval.datasource import (
    LongMemEvalItem,
    load_items,
    select_item,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LongMemEvalEnvConfig:
    """LongMemEval env config.

    A single LongMemEval *run* answers one QA item — its haystack of chat
    sessions is streamed session-by-session to the agent, and a final
    probe with the QA's question fires after the last session.

    ``num_turns`` is set automatically by the env to
    ``len(haystack_sessions) + 1`` (the +1 is the probe-only turn)
    once the QA item is loaded; the value in the JSON config is treated
    as a safety upper bound (the turn-loop also exits via
    :meth:`LongMemEvalEnv.is_terminal`).
    """

    # Upper bound on the turn count; the env overwrites this with the
    # loaded item's actual chat-session count + 1.
    num_turns: int = 100

    # Path to the LongMemEval JSON file. Three official variants:
    #   - longmemeval_oracle.json     (only evidence sessions; smoke-test)
    #   - longmemeval_s_cleaned.json  (~40 sessions / ~115K tokens; default)
    #   - longmemeval_m_cleaned.json  (~500 sessions; full benchmark)
    dataset_path: str = "external/longmemeval/data/longmemeval_s_cleaned.json"

    # Pick exactly one of these to select the QA item:
    #   - question_id: match by ``question_id`` field (most readable)
    #   - question_index: zero-based index into the loaded list (fallback)
    # If neither is set, the env errors at construction time.
    question_id: str | None = None
    question_index: int | None = None

    # GPT-4o-style judge config. Uses the OpenAI Python client (not the
    # Dashscope-compatible endpoint), so CN_DASHSCOPE_API_KEY is
    # required for the default judge_model. Set ``judge_api_base`` to
    # point at a local vLLM server hosting llama-3.1-70b-instruct
    # (LongMemEval's other supported judge) or any OpenAI-compatible
    # endpoint.
    judge_model: str = "gpt-4o-2024-08-06"
    judge_api_key_env: str = "CN_DASHSCOPE_API_KEY"
    judge_api_base: str | None = None

    # PR #4 loop-redesign switch. When ``False`` (default, the new
    # SCROLL-pure path): the env bulk-loads the haystack into ``E``
    # at task start via :meth:`ingest_all`, ``cfg.num_turns`` is
    # forced to ``0`` (no per-turn loop), and the probe fires once via
    # :meth:`get_end_of_task_probes`. When ``True`` (legacy fallback):
    # ``cfg.num_turns = total_sessions + 1``, the agent's per-turn
    # ``run_turn`` mirrors one chat session into ``E`` per iteration,
    # and the probe fires on the ``+1`` turn via the per-turn registry.
    # Flag exists to back out PR #4 cleanly if the new path regresses
    # judge scores in the parity sweep; PR #6 drops it once we've
    # validated parity (or kept the legacy path on purpose).
    agent_during_ingestion: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "LongMemEvalEnvConfig":
        """Construct from a raw config dict."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


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
        self.turn_idx = 0

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

        # PR #4: under the new SCROLL-pure path
        # (``cfg.agent_during_ingestion=False``, the default), the
        # haystack is bulk-loaded into ``E`` by :meth:`ingest_all` at
        # task start and the probe fires at end-of-task — so
        # ``num_turns = 0`` and the per-turn loop never runs.
        #
        # Under the legacy path (``cfg.agent_during_ingestion=True``),
        # we keep today's behavior: ``num_turns = total_sessions + 1``,
        # one ingestion iteration per chat session, and a virtual
        # ``+1`` probe-only turn. ``is_terminal`` allows the +1
        # iteration through.
        if cfg.agent_during_ingestion:
            cfg.num_turns = self.item.total_sessions + 1
        else:
            cfg.num_turns = 0

        self._current_session: list[dict] | None = None
        self._current_session_id: str | None = None
        self._current_session_date: str | None = None
        self._today_logs: list[str] = []

        # Register the active probe for this run. Imported lazily so
        # the tasks subpackage can also import env types without a cycle.
        from Scroll.benchmarks.longmemeval.tasks import probes as _probes
        _probes.set_active_probe(self.item, cfg)

    # ------------------------------------------------------------------
    # Per-turn API
    # ------------------------------------------------------------------

    def begin_turn(self, turn_idx: int) -> list[str]:
        """Stage the next haystack session.

        Called by the datasource (and through it the turn-loop) at the
        start of each turn, *before* ``agent.run_turn``. Returns a
        list of briefing lines that the framework prepends to the agent's
        turn prompt as ``Today's briefing``.

        ``turn_idx`` is the env's current turn counter
        (``env.turn_idx``), 0 at the first call. Today, ``turn_idx``
        also indexes directly into ``haystack_sessions``; PR #4
        decouples them.
        """
        idx = turn_idx
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

    def step_turn(self) -> TurnResult:
        """Advance to the next turn.

        LME has no economic state — ``TurnResult`` carries zeros and
        the only meaningful side-effect is incrementing ``self.turn_idx``.
        """
        self.turn_idx += 1
        return TurnResult(
            turn_idx=self.turn_idx,
            sold_units=0,
            revenue=0.0,
            machine_cash=0.0,
            cash=0.0,
        )

    # ``today_logs`` / ``net_worth`` inherit the BaseEnvironment defaults
    # (``[]`` and ``0.0``) — LME is a passive observation env with no
    # outcomes or economic state. ``_today_logs`` is still maintained
    # for ``build_snapshot`` / turn-log consumers.

    def visible_state(self) -> dict:
        return {
            "turn_idx": self.turn_idx,
            "total_sessions": self.item.total_sessions,
            "current_session_id": self._current_session_id,
            "current_session_date": self._current_session_date,
            "current_session_turn_count": (
                len(self._current_session) if self._current_session else 0
            ),
            "question_date": self.item.question_date,
        }

    def is_terminal(self) -> bool:
        # Legacy path (``agent_during_ingestion=True``): allow exactly
        # one iteration past the last chat session for the probe-only
        # turn (see ``__init__``: cfg.num_turns is total_sessions+1).
        # Becomes terminal AFTER the probe turn has run.
        # New path (``agent_during_ingestion=False``): num_turns is 0
        # so the per-turn loop never enters; this method's return
        # value is irrelevant in practice.
        return self.turn_idx > self.item.total_sessions

    # ------------------------------------------------------------------
    # Task lifecycle (PR #7: env exposes data only; LMEIngestor.bootstrap
    # owns the env → E bulk-load step)
    # ------------------------------------------------------------------

    def get_end_of_task_probes(self):
        """Return the QA item's single probe, fired at end-of-task.

        Empty under the legacy ``agent_during_ingestion=True`` path —
        the probe still rides on the ``+1`` per-turn registry there.
        """
        if self.cfg.agent_during_ingestion:
            return []
        from Scroll.benchmarks.longmemeval.tasks import probes as _probes
        if _probes._active_probe is None:
            return []
        return [_probes._active_probe]

    def substrate_endgame_prompt(self) -> str:
        return _LME_SUBSTRATE_ENDGAME

    # NOTE: no ``probe_substrate_prompt`` override — LME's format reminders
    # were folded into _LME_PROBE_HINT (most useful bits) + redundant
    # bullets removed; nothing left to inject at the env level.

    def probe_user_postscript(self) -> str:
        from Scroll.benchmarks.longmemeval.tasks.probes import compose_user_postscript
        return compose_user_postscript()

    def build_snapshot(self) -> EnvSnapshot:
        return EnvSnapshot(
            turn_idx=self.turn_idx,
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
            "turn_idx": self.turn_idx,
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
        env.turn_idx = data.get("turn_idx", data.get("session_idx", 0))
        env._today_logs = list(data.get("today_logs") or [])
        return env
