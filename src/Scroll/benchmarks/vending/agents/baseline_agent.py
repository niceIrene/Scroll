"""Baseline vending agent — CodeAct REPL with env tools only.

Counterpart to :class:`VendingAgent` used as the SCROLL-vs-baseline
comparison point. Same per-day CodeAct loop and same env action tools,
but no off-context memory: the agent gets the REPL with
``send_email`` / ``run_sub_agent`` / ``get_money_balance`` / etc.
bound, and nothing else — no ``ms``, no ``log``, no ``rlm``.

Long-horizon recall falls back to whatever survives in the running
chat history. Once total chars exceed ``cfg.context_max_tokens * 4``,
the inherited :meth:`CodeActAgent._compress_history_to_budget` folds
the oldest batch into a running summary and drops those messages —
sliding-window context with one rolling summary, no retrieval.

The internal ``ConversationLog`` (``self.log``) still records the
turn trace for debugging, but pass ``keep_logs=False`` at the
benchmark layer to skip JSONL persistence — the agent never queries
it.
"""

from __future__ import annotations

from typing import Callable

from Scroll.core import CodeActAgent
from Scroll.core._codeact_agent import _wrap_closure_for_repl
from Scroll.benchmarks.vending.tools import _make_env_tool_closures
from Scroll.benchmarks.vending.agents.prompts import (
    _VENDING_CONTEXT,
    vending_day_prompt,
)


# ---------------------------------------------------------------------------
# System prompt — vending world rules + CodeAct mechanics + env-tool docs.
# Deliberately drops the SCROLL strategy block, the auto-built memoryspace
# schema, and the ``ms`` / ``log`` / ``rlm`` REPL docs.
# ---------------------------------------------------------------------------


_BASELINE_STRATEGY = """\
RUN STRUCTURE — vending is a multi-day simulation. Each day, the
harness sends you ONE user message with the day's briefing
(weather / news / inbox tail) plus any outcomes from the previous
day. You respond as a single agentic turn: emit zero or more
``execute_python`` tool calls (each returns its stdout to you as a
tool_result), then a final assistant message with NO tool call.
That final no-tool-call message ENDS the day; the next morning's
briefing arrives as the next user message.

NO PERSISTENT MEMORY — you have no off-context store. Everything you
need to remember (supplier prices, lead times, pending orders,
inventory) must live in the chat history or be re-derived from the
env tools. Once total chars exceed the context budget, the oldest
messages are folded into a running summary and dropped from history.
Plan accordingly: surface key numbers in your responses so they
survive compression; don't assume earlier turns' Python locals are
still in scope (REPL globals are wiped between turns).

TOOL — every Python execution MUST go through the ``execute_python``
tool call. NEVER emit `````python ... `````
markdown fences in ``content`` — the harness DOES NOT parse fences,
those characters are wasted tokens and the code in them WILL NOT
run. The only way to make Python execute is to emit a structured
tool call with ``name="execute_python"`` and ``code`` as the
argument.

  - Pass ONLY Python source as the ``code`` argument — no markdown
    fences inside the JSON string, no prose comments at the top
    explaining what you're about to do.
  - REPL globals persist across cells within the same turn so a
    later ``execute_python`` call can use variables defined earlier.
  - ``print(x)`` to surface values; bare ``x`` is a no-op.
"""


_BASELINE_TOOLS = """\
ENV ACTION TOOLS — bound in the ``execute_python`` REPL (no import
needed). Each returns a ``str`` (parse with ``.split("\\n---\\n")`` /
regex / ``json.loads`` as needed):

  send_email(to, subject, body)
      Email a supplier. Replies arrive in the inbox on day N+1
      (same-day ``read_email_inbox`` will not show them).
  read_email()
      Read one unread email. Returns its body as str, or
      "No unread emails." if the inbox is empty.
  read_email_inbox()
      Up to 20 unread emails joined by ``"\\n---\\n"``, or
      "Inbox is empty." Split with ``inbox.split("\\n---\\n")``.
  ai_web_search(query)
      Newline-joined snippets, or "No results found."
  get_money_balance()
      Pretty-printed JSON string with keys ``cash``, ``machine_cash``,
      ``inventory_value``, ``net_worth``. ``json.loads`` it.
  sub_agent_specs()
      Help string listing physical sub-agent commands.
  run_sub_agent(instruction)
      Run physical-machine ops. Commands combine with semicolons:
      "restock cola=20 water=15; collect_cash; check_inventory".
  chat_with_sub_agent(question)
      Ask the sub-agent about recent actions.
"""


BASELINE_SYSTEM_PROMPT = (
    _VENDING_CONTEXT + "\n"
    + _BASELINE_STRATEGY + "\n"
    + _BASELINE_TOOLS
)


# ---------------------------------------------------------------------------
# REPL namespace — env action tools only.
# ---------------------------------------------------------------------------


def _make_baseline_namespace(state) -> dict[str, Callable]:
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


class BaselineVendingAgent(CodeActAgent):
    """Context-only baseline: CodeAct REPL + env tools, no off-context memory.

    Behaviorally a direct counterpart to :class:`VendingAgent` minus
    the SCROLL retrieval layer. Per-day briefing + prior-day outcomes
    are spliced into the next turn's user message (same as
    ``VendingAgent``), then the inherited turn loop runs the model
    against ``execute_python``. The model writes Python that calls
    the env tools directly; long-horizon facts persist only through
    chat history + the rolling compressed summary.
    """

    sys_prompt = BASELINE_SYSTEM_PROMPT

    def __init__(self, cfg, storage, data) -> None:
        # Buffers for the most recent turn's outcomes / briefing. Set
        # before super().__init__ because ``receive_context`` /
        # ``receive_outcomes`` can fire during checkpoint resume.
        self._pending_env_outcomes: list[str] = []
        self._pending_datasource_notes: list[str] = []
        super().__init__(cfg, storage, data)

    def init_namespace(self) -> dict:
        return _make_baseline_namespace(self._tool_state)

    def turn_prompt(self, turn_idx: int) -> str:
        return vending_day_prompt(turn_idx, self._tool_state.env)

    def receive_outcomes(self, turn_idx: int, logs: list[str]) -> None:
        self._pending_env_outcomes = list(logs)

    def receive_context(self, turn_idx: int, notes: list[str]) -> None:
        self._pending_datasource_notes = list(notes)

    def _extra_prefix_sections(self, turn_idx: int) -> list[str]:
        sections: list[str] = []
        if self._pending_env_outcomes:
            lines = "\n".join(f"  - {line}" for line in self._pending_env_outcomes)
            sections.append(f"Outcomes from turn {turn_idx - 1}:\n{lines}")
        if self._pending_datasource_notes:
            lines = "\n".join(f"  - {line}" for line in self._pending_datasource_notes)
            sections.append(f"Briefing:\n{lines}")
        self._pending_env_outcomes = []
        self._pending_datasource_notes = []
        return sections

    def to_checkpoint(self) -> dict:
        data = super().to_checkpoint()
        data["pending_env_outcomes"] = list(self._pending_env_outcomes)
        data["pending_datasource_notes"] = list(self._pending_datasource_notes)
        return data

    def from_checkpoint(self, data: dict) -> None:
        super().from_checkpoint(data)
        self._pending_env_outcomes = list(data.get("pending_env_outcomes", []))
        self._pending_datasource_notes = list(data.get("pending_datasource_notes", []))


__all__ = ["BaselineVendingAgent", "BASELINE_SYSTEM_PROMPT"]
