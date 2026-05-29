"""BeamEnv — non-economic, batch-streaming env (BEAM benchmark).

A run answers all probing questions for one BEAM chat. ``begin_day(day)``
exposes the ``day``-th batch (in chronological order) as a briefing
note + caches the structured turns on the env so the agent can ingest
them into the memoryspace. ``step_day`` advances to the next batch and
runs no scoring of its own — scoring fires on the probe day via the
probes registered in :mod:`tasks.probes`.
"""

from __future__ import annotations

from typing import Any

from Scroll.core import BaseEnvironment, TurnResult, EnvSnapshot
from Scroll.benchmarks.beam.catalog import BeamEnvConfig
from Scroll.benchmarks.beam.dataset import BeamItem, load_chat, normalize_beam_date


_BEAM_SUBSTRATE_ENDGAME = """\
RUN STRUCTURE — BEAM (Beyond a Million Tokens):
- Each iteration is one past chat batch (a time-anchored chunk of
  the long conversation). The transcript is inlined in the briefing —
  there are NO action tools to call inside the batch.
- ``today`` (1-indexed int) is the batch ordinal — which batch you
  are observing — NOT a real-time day. There WILL be probing
  questions at the end of the run covering 10 memory-ability
  categories (extraction, multi-session reasoning, knowledge update,
  temporal reasoning, abstention, contradiction resolution, event
  ordering, instruction following, preference following,
  summarization), so save anything you might want to recall.
- After your bookkeeping (or choosing not to), emit a response with
  NO ``python`` code block. The CodeAct loop treats that as the
  end-of-batch signal and advances the batch pointer.
- At the END of the run multiple probing questions fire in
  sequence; reply formatting rules will be in the probe-mode system
  prompt that swaps in then.
"""


class BeamEnv(BaseEnvironment):
    def __init__(self, cfg: BeamEnvConfig, seed: int) -> None:
        if not cfg.chat_id:
            raise ValueError(
                "BeamEnvConfig.chat_id is required — pick a chat to run "
                "(e.g. '1', '2', ...). Use Scroll.benchmarks.beam.dataset.list_chats() "
                "to enumerate available chats."
            )
        self.cfg = cfg
        self.seed = seed
        self.turn_idx = 0

        self.item: BeamItem = load_chat(
            cfg.dataset_root, cfg.scale, cfg.chat_id,
        )

        # Under the SCROLL-pure path
        # (``cfg.agent_during_ingestion=False``, the default), every
        # batch is bulk-loaded into ``E`` by :meth:`ingest_all` at
        # task start and probes fire end-of-task — so ``num_turns =
        # 0`` and the per-turn loop never runs.
        #
        # Under the legacy path (``cfg.agent_during_ingestion=True``):
        # ``num_turns = num_batches + 1`` — one batch per iteration,
        # plus a virtual ``+1`` probe-only turn. ``is_terminal``
        # allows the +1 iteration through.
        if cfg.agent_during_ingestion:
            cfg.num_turns = self.item.num_batches + 1
        else:
            cfg.num_turns = 0

        self._current_batch_turns: list[dict] | None = None
        self._current_batch_idx: int | None = None
        self._current_time_anchor: str | None = None
        self._today_logs: list[str] = []

        # Register all probing questions for this chat as ProbeSpecs.
        from Scroll.benchmarks.beam.tasks import probes as _probes
        _probes.set_active_item(self.item, cfg)

    # ------------------------------------------------------------------
    # Per-turn API
    # ------------------------------------------------------------------

    def begin_turn(self, turn_idx: int) -> list[str]:
        idx = turn_idx  # 0-based into self.item.batches
        if idx >= self.item.num_batches:
            self._current_batch_turns = None
            self._current_batch_idx = None
            self._current_time_anchor = None
            self._today_logs = ["no_batch_remaining"]
            return list(self._today_logs)

        batch = self.item.batches[idx]
        self._current_batch_turns = batch.turns
        self._current_batch_idx = batch.batch_idx
        self._current_time_anchor = batch.time_anchor
        notes = [
            f"batch_idx={batch.batch_idx}",
            f"time_anchor={batch.time_anchor}",
            f"turn_count={len(batch.turns)}",
        ]
        self._today_logs = notes
        return list(notes)

    def step_turn(self) -> TurnResult:
        self.turn_idx += 1
        return TurnResult(
            turn_idx=self.turn_idx,
            sold_units=0,
            revenue=0.0,
            machine_cash=0.0,
            cash=0.0,
        )

    # ``today_logs`` / ``net_worth`` inherit the BaseEnvironment defaults
    # (``[]`` and ``0.0``) — BEAM batches are passive observations with
    # no outcomes or economic state.

    def visible_state(self) -> dict:
        return {
            "turn_idx": self.turn_idx,
            "num_batches": self.item.num_batches,
            "current_batch_idx": self._current_batch_idx,
            "current_time_anchor": self._current_time_anchor,
            "current_batch_turn_count": (
                len(self._current_batch_turns) if self._current_batch_turns else 0
            ),
            "scale": self.item.scale,
            "chat_id": self.item.chat_id,
            "title": self.item.title,
        }

    def is_terminal(self) -> bool:
        # Legacy path (``agent_during_ingestion=True``): allow exactly
        # one iteration past the last batch for the probe-only turn.
        # SCROLL-pure path: num_turns is 0, so the per-turn loop
        # never enters and this return is irrelevant.
        return self.turn_idx > self.item.num_batches

    # ------------------------------------------------------------------
    # Task lifecycle — env exposes data only; BeamIngestor.bootstrap
    # owns the env → E bulk-load step.
    # ------------------------------------------------------------------

    def get_end_of_task_probes(self):
        """Return the chat's M probing questions, all fired at end-of-task.

        Empty under the legacy ``agent_during_ingestion=True`` path —
        probes still ride on the ``+1`` per-turn registry there.
        """
        if self.cfg.agent_during_ingestion:
            return []
        from Scroll.benchmarks.beam.tasks import probes as _probes
        return list(_probes._active_probes)

    def probe_isolation(self) -> str:
        """Forwarded from :attr:`BeamEnvConfig.probe_isolation`."""
        return str(getattr(self.cfg, "probe_isolation", "shared"))

    def substrate_endgame_prompt(self) -> str:
        return _BEAM_SUBSTRATE_ENDGAME

    def probe_substrate_prompt(self) -> str:
        from Scroll.benchmarks.beam.tasks.probes import BEAM_PROBE_FORMAT
        return BEAM_PROBE_FORMAT

    def probe_user_postscript(self) -> str:
        from Scroll.benchmarks.beam.tasks.probes import compose_user_postscript
        return compose_user_postscript()

    def build_snapshot(self) -> EnvSnapshot:
        return EnvSnapshot(
            turn_idx=self.turn_idx,
            logs=list(self._today_logs),
            extra={
                "scale": self.item.scale,
                "chat_id": self.item.chat_id,
                "batch_idx": self._current_batch_idx,
                "time_anchor": self._current_time_anchor,
                "batch_turn_count": (
                    len(self._current_batch_turns) if self._current_batch_turns else 0
                ),
                "num_batches": self.item.num_batches,
                "num_probes": self.item.num_probes,
            },
        )

    # ------------------------------------------------------------------
    # Helpers consumed by the agent's ingest path. Names mirror the
    # LongMemEval env so the agent can reuse the same ingest code shape
    # (current_session / current_session_meta).
    # ------------------------------------------------------------------

    @property
    def current_session(self) -> list[dict] | None:
        return self._current_batch_turns

    @property
    def current_session_meta(self) -> dict[str, Any]:
        # Convert BEAM's "Month-Day-Year" anchor to ISO YYYY-MM-DD so
        # the agent's ``session_date_to_iso`` (LME-shaped) parses it
        # cleanly into the memoryspace's ``session_date_iso`` column.
        iso = normalize_beam_date(self._current_time_anchor)
        return {
            "session_id": (
                f"{self.item.chat_id}_b{self._current_batch_idx}"
                if self._current_batch_idx is not None else None
            ),
            "session_date": iso or self._current_time_anchor,
            "session_date_raw": self._current_time_anchor,
            "turn_count": (
                len(self._current_batch_turns) if self._current_batch_turns else 0
            ),
        }

    @property
    def question_data(self) -> BeamItem:
        return self.item

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def to_checkpoint(self) -> dict:
        return {"turn_idx": self.turn_idx,
            "seed": self.seed,
            "scale": self.item.scale,
            "chat_id": self.item.chat_id,
            "today_logs": list(self._today_logs),
        }

    @classmethod
    def from_checkpoint(cls, data: dict, cfg: BeamEnvConfig) -> "BeamEnv":
        cfg.scale = data["scale"]
        cfg.chat_id = data["chat_id"]
        env = cls(cfg, seed=data.get("seed", 0))
        # ``session_idx`` is the legacy on-disk key for ``turn_idx``.
        env.turn_idx = data.get("turn_idx", data.get("session_idx", 0))
        env._today_logs = list(data.get("today_logs") or [])
        return env
