"""Per-session briefing helpers + time-range extractor for LongMemEval.

Trimmed to just what ``LongMemEvalAgent`` consumes.
"""

from __future__ import annotations

import json
import re
from typing import Awaitable, Callable


def make_time_range_extractor(
    oneshot: Callable[[str], Awaitable[str]],
    *,
    today_iso_provider: Callable[[], str | None] | None = None,
) -> Callable[[str], Awaitable[dict | None]]:
    """Build an async ``extract_time_range(question) -> {start, end} | None``.

    Implements the LongMemEval paper §5.4 time-aware query expansion:
    one LLM call extracts a date range from the question, and the
    caller filters retrieval to that range BEFORE pulling content.

    Returns ``None`` (no time scope) for questions without temporal
    cues, or ``{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}`` when a
    range is detectable. Robust to malformed JSON: parse failures
    return ``None`` rather than raise.
    """
    async def extract_time_range(question: str) -> dict | None:
        today_iso = today_iso_provider() if today_iso_provider else None
        today_clause = f"Today is {today_iso}." if today_iso else ""
        prompt = (
            "Extract the date range relevant to answering this "
            "question, expressed in ISO YYYY-MM-DD form.\n"
            f"{today_clause}\n"
            "Reply with STRICT JSON, no prose, exactly one of:\n"
            '  {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}\n'
            '  {"start": null, "end": null}\n'
            "Use null/null when the question has no temporal scope. "
            "For 'current X' or 'now', set both to today's date. "
            "For 'last week / last month / last year', infer the "
            "concrete range from today's date.\n\n"
            f"Question: {question}\n\nJSON:"
        )
        try:
            raw = await oneshot(prompt)
        except Exception:  # noqa: BLE001
            return None
        m = re.search(r"\{[^{}]*\}", raw or "")
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except (ValueError, json.JSONDecodeError):
            return None
        if not isinstance(obj, dict):
            return None
        start = obj.get("start")
        end = obj.get("end")
        if start is None and end is None:
            return None
        iso = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if start is not None and not iso.match(str(start)):
            return None
        if end is not None and not iso.match(str(end)):
            return None
        return {"start": start, "end": end}

    return extract_time_range


def probe_session_body(session_idx: int) -> str:
    """Universal probe-session body (used on session N+1).

    The probe session has no haystack content — the agent's only
    legitimate cell is ``wait_for_next_day()`` to trigger the probe.
    """
    return (
        f"Session {session_idx}: PROBE SESSION. Reply with ONLY this single cell:\n"
        "```python\nwait_for_next_day()\n```\n"
        "The probe will fire immediately after. Any other cell here "
        "(data fetches, inspections, prep) is wasted compute — the "
        "probe-time prompt will give you everything you need for "
        "retrieval and answering. Just advance."
    )


def handle_only_body() -> str:
    """Session-loop body pointing the agent at the unified log for the
    current session. ``LongMemEvalAgent`` auto-advances and rarely sees
    this body, but it remains the canonical haystack-session prompt.
    """
    return (
        "The current chat session is in the unified log: "
        "``log.slice(session_idx=today, kind=\"chat_turn\")`` (returns "
        "``LogEntry`` objects) or ``today_session()`` (native "
        "dicts). Past sessions: ``log.slice(session_idx=N, "
        "kind=\"chat_turn\")``.\n\n"
        "Apply your session-loop contract (system prompt) and call "
        "``wait_for_next_day()`` when done."
    )
