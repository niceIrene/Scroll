"""Vending-specific probe postscript + efficiency metrics.

Scoring itself lives in :mod:`Scroll.benchmarks.vending.tasks.probes`
(an LLM judge with ``judge.probe.{qid}`` trace spans). This module
keeps the per-probe user-message postscript (scorer-shaped reminders
that ride next to the question text) and the env-action efficiency
aggregator.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Probe user-message postscript
# ---------------------------------------------------------------------------

# Appended to the user-turn message inside ``inject_probe``.
# The agent may call ``execute_python`` to query memoryspace / log
# first, then commits its final answer as a no-tool-call message.
# The LLM judge (see :mod:`tasks.probes`) reads the final assistant
# message — typically the last ``Answer: ...`` line — against the
# deterministic ground truth.
VENDING_PROBE_USER_POSTSCRIPT = (
    "[PROBE — answer this question, then continue with the next "
    "business day.] "
    "You may call ``execute_python`` to query the memoryspace / log "
    "first. When ready, reply with a final assistant message that "
    "contains NO tool call. Format: reasoning first, then a single "
    "line starting with ``Answer: `` containing every value the "
    "question asks for. An LLM judge will compare your Answer line "
    "to the deterministic ground truth, allowing reasonable tolerance "
    "for numeric values and phrasing — ±5% on dollar amounts, "
    "±5 percentage points on percentages, ±10% on bare counts. Use "
    "the same unit symbol the question uses ($, %, or bare). For "
    "extremum questions (highest, most, cheapest, …) with a tie, "
    "list ALL tied items. Examples: ``Answer: 6 deliveries``, "
    "``Answer: $282.00``, ``Answer: 42% margin``, "
    "``Answer: machine=0 storage=27``. ``print(\"Answer: ...\")`` "
    "inside a code cell is NOT scored — only the plain-text reply is."
)


# action_log entries are env-action tool names (substrate agents do
# their data ops *inside* code cells, which don't reach action_log;
# only env tools tick state._action_log). So efficiency metrics here
# can only reason about decision frequency and overall env-tool usage.
_DECISION_TOOLS = {"send_email", "run_sub_agent"}


def compute_efficiency_metrics(
    daily_action_logs: list[list[str]],
) -> dict[str, Any]:
    """Aggregate env-action efficiency signals derived from action_log.

    Under the RLM substrate the agent's data ops (``db.query``,
    ``ms.sql_exec``, ``log.search`` …) happen inside code cells and
    are NOT mirrored into the ``ToolState._action_log`` — only env
    actions (send_email, run_sub_agent, etc.) are. So this reports
    only env-tool-level signals; for cell-level metrics use the
    ``conversation_log.jsonl`` or trace data.
    """
    total_days = len(daily_action_logs)
    if total_days == 0:
        return {"total_days": 0}

    total_decisions = 0
    total_actions = 0
    for actions in daily_action_logs:
        for action in actions:
            tool_name = action.split()[0].rstrip(":") if action else ""
            if not tool_name:
                continue
            total_actions += 1
            if tool_name in _DECISION_TOOLS:
                total_decisions += 1

    return {
        "total_days": total_days,
        "avg_decisions_per_day": round(total_decisions / total_days, 2),
        "avg_actions_per_day": round(total_actions / total_days, 2),
    }
