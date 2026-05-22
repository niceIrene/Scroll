"""Aggregate efficiency metrics for a LongMemEval run.

Per-session actions are minimal in LME — each "action_log" entry comes
from the agent's REPL bookkeeping (rlm calls, db queries, sub-
agent invocations). We just summarize.
"""

from __future__ import annotations

from typing import Any


def compute_efficiency_metrics(daily_action_logs: list[list[str]]) -> dict[str, Any]:
    total_days = len(daily_action_logs)
    if total_days == 0:
        return {"total_days": 0, "avg_actions_per_day": 0}

    total_actions = sum(len(actions) for actions in daily_action_logs)
    return {
        "total_days": total_days,
        "avg_actions_per_day": round(total_actions / total_days, 2),
    }
