"""Vending SCROLL agent.

Applies the :class:`Scroll.core.ScrollAgent` harness to the vending
machine simulation. The harness owns ingestion of every iteration's
events; the agent reads from ``ms`` / ``log`` and reaches for the
recursive ``rlm`` sub-agent only when the structured surfaces miss.
(Module-level prompts still reference ``sub_llm`` — pending rewrite.)

Layout:
  - 7 base tables (``sales``, ``deliveries``, ``orders``,
    ``daily_finances``, ``daily_inventory``, ``supplier_prices``,
    ``notes``)
  - 5 analytic views (``weekly_revenue_by_sku``, ``rolling_7d_units_by_sku``,
    ``cogs_by_supplier``, ``inventory_turnover``, ``daily_pnl``)
  - Vector store over inbox emails, datasource notes, and env outcome
    lines.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from Scroll.core import ScrollAgent
from Scroll.core._codeact_agent import _wrap_closure_for_repl
from Scroll.core._models import LogEntry
from Scroll.core._tool_state import ToolState
from Scroll.benchmarks.vending.ingestor import (
    VendingIngestor,
    ensure_schema as _vending_ensure_schema,
    serialize_env_snapshot,
    serialize_inbox_mail,
)
from Scroll.benchmarks.vending.tools import _make_env_tool_closures
from Scroll.benchmarks.vending.agents.prompts import (
    SYSTEM_PROMPT,
    vending_day_prompt,
)


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# REPL namespace builder
# ---------------------------------------------------------------------------


def _make_env_namespace(state: ToolState) -> dict[str, Callable]:
    """Build the env-specific REPL globals dict for the vending agent.

    The env action tools are unwrapped from their AgentScope
    ``ToolResponse`` envelopes so REPL code sees plain strings. The
    session-loop ends when the model replies without a Python code
    block — there is no explicit advance-day primitive in the REPL.
    """
    closures = _make_env_tool_closures(state)
    ns: dict[str, Callable] = {}
    for fn in closures:
        name = fn.__name__
        if name.endswith("_tool"):
            name = name[: -len("_tool")]
        ns[name] = _wrap_closure_for_repl(fn)
    return ns


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class VendingAgent(ScrollAgent):
    """SCROLL agent for the vending environment.

    The harness auto-ingests every day's events into ``ms`` and the
    agent's REPL gets a query-only view (SQL + vector). Env-side
    actions (send_email, restock via run_sub_agent, etc.) remain
    available — they go through the env, not through ``ms`` writes.
    The recursive ``rlm`` sub-agent is exposed for free-text
    synthesis over recalled rows.
    """

    # SCROLL feature flag — RLM sub-agent.
    expose_rlm = True

    sys_prompt = SYSTEM_PROMPT

    def __init__(self, cfg, storage, data) -> None:
        # Cursor over the env's inbox: how many mails we've already
        # mirrored into E as ``kind="inbox_mail"`` entries. Each
        # ``receive_context`` advances it past the new tail. Set before
        # super().__init__ because receive_context can fire during
        # checkpoint resume.
        self._emitted_inbox_count = 0
        super().__init__(cfg, storage, data)

    # ----- SCROLL hooks -----

    ingestor_cls = VendingIngestor

    def _ensure_schema(self, memoryspace) -> None:
        _vending_ensure_schema(memoryspace)

    def _base_namespace(self) -> dict:
        # Env-specific tool closures (send_email, read_email,
        # get_money_balance, run_sub_agent, ...)
        return _make_env_namespace(self._tool_state)

    # NOTE: REPL globals docs are inlined into SYSTEM_PROMPT
    # (via _NAMESPACE_DOCS), so we don't override ``namespace_docs()``
    # — the base class returns "" and the framework just concatenates
    # the empty string. Matches LongMemEvalAgent's setup.

    def session_prompt(self, session_idx: int) -> str:
        # Vending: session_idx maps 1:1 to the calendar day, so we pass
        # it as the `day` arg to the domain helper.
        return vending_day_prompt(session_idx, self._tool_state.env)

    def _emit_context_entries(self, session_idx: int, notes: list[str]) -> None:
        """Land per-day briefing notes AND any new inbox mails into E.

        Vending: framework session == calendar day (1:1). The ingestor
        will pick these up on the next ``ms`` read and write them into
        the seven canonical tables + vector store.
        """
        # Briefing notes — one entry per note.
        for note in notes:
            self.log.append(LogEntry.make(
                session_idx=session_idx, role="system",
                content=str(note),
                metadata={"kind": "briefing_note"},
            ))
        # Inbox tail — emit one entry per newly-arrived mail.
        inbox = getattr(self._tool_state.data, "inbox", None)
        if inbox is not None:
            inbox_list = list(inbox)
            for idx in range(self._emitted_inbox_count, len(inbox_list)):
                mail = inbox_list[idx]
                self.log.append(LogEntry.make(
                    session_idx=session_idx, role="system",
                    content=f"from={getattr(mail, 'source', None)} "
                            f"subject={getattr(mail, 'subject', '')}",
                    metadata={
                        "kind": "inbox_mail",
                        **serialize_inbox_mail(mail, idx),
                    },
                ))
            self._emitted_inbox_count = len(inbox_list)

    def _emit_outcome_entries(self, session_idx: int, logs: list[str]) -> None:
        """Land per-day env outcome lines AND a single env snapshot into E."""
        for line in logs:
            self.log.append(LogEntry.make(
                session_idx=session_idx, role="system",
                content=str(line),
                metadata={"kind": "env_log"},
            ))
        env = self._tool_state.env
        if env is not None:
            self.log.append(LogEntry.make(
                session_idx=session_idx, role="system",
                content=f"env_snapshot day={session_idx}",
                metadata={
                    "kind": "env_snapshot",
                    **serialize_env_snapshot(env),
                },
            ))

    # ----- Probe answering -----
    # Vending treats a probe as just one more user message in the same
    # conversation: no system-prompt swap, no history rollback, no
    # separate probe agent. The probe Q+A enters the main history so
    # subsequent days' reasoning sees it (sliding memory compresses
    # older turns once the conversation grows).

    def answer_probe(self, question: str) -> str:
        old_day_ended = self._tool_state._session_ended
        self._tool_state._session_ended = False

        # Bind ``probe_question`` so REPL code can embed it directly in
        # rlm prompts. Restored in the finally block.
        old_pq = self._runtime.globals.get("probe_question")
        pq_was_present = "probe_question" in self._runtime.globals
        self._runtime.globals["probe_question"] = question

        self._history.append({"role": "user", "content": question})

        try:
            answer = asyncio.run(
                self._answer_probe_inner(max_iters=self.probe_max_iters)
            )
        except Exception as e:  # noqa: BLE001
            answer = f"[probe error: {e}]"
        finally:
            self._tool_state._session_ended = old_day_ended
            if pq_was_present:
                self._runtime.globals["probe_question"] = old_pq
            else:
                self._runtime.globals.pop("probe_question", None)
        return answer

    # ----- Checkpoint (inbox cursor; memoryspace handled by base) -----

    def to_checkpoint(self) -> dict:
        data = super().to_checkpoint()
        data["emitted_inbox_count"] = self._emitted_inbox_count
        return data

    def from_checkpoint(self, data: dict) -> None:
        super().from_checkpoint(data)
        # Back-compat: old checkpoints stored ``embedded_inbox_count``.
        self._emitted_inbox_count = data.get(
            "emitted_inbox_count",
            data.get("embedded_inbox_count", 0),
        )
