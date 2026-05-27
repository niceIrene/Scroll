"""Shared chat-memory primitives used by both LongMemEval and BEAM agents.

Both benchmarks have the same shape (chat-haystack + end-of-history
probes); their agents share:

  - ``make_time_range_extractor`` — builds the agent-facing async
    ``extract_time_range(question)`` helper (one one-shot LLM call to
    pull a YYYY-MM-DD window out of a probe question).
  - ``probe_session_body`` / ``handle_only_body`` — universal session-
    loop bodies (LME auto-advances and never sees them; BEAM does).
  - ``write_chat_turn_entries`` — mirrors a session's chat turns into
    the unified log E as ``kind="chat_turn"`` entries (one per turn).
    The X = f(E) contract: the transcript IS in E; agents only differ
    by their retrieval function ``f``.
  - ``make_chat_memory_namespace`` — REPL namespace shared by every
    chat-memory agent (``wait_for_next_day`` + stdlib pre-binds +
    optional ``today_session`` reader on the log).

Previously these lived under ``Scroll.benchmarks.longmemeval.agents``
and BEAM cross-imported from the LME package. Moved here so neither
benchmark reaches into the other.
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

from Scroll.core import LogEntry
from Scroll.core._codeact_agent import make_wait_for_next_day


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
    this body; BEAM does see it on haystack-session turns.
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


# ---------------------------------------------------------------------------
# Chat-turn entries: source of truth for the session's transcript.
#
# Each haystack-session turn is written to the unified log E as one
# ``LogEntry(role=..., metadata={"kind": "chat_turn", "turn_idx": K})``.
# This keeps the X = f(E) contract: the chat transcript IS in E (every
# agent reads from the same log), the only difference between agents
# is the retrieval ``f``. Compare to dropping it in via prompt text
# (anti-pattern, encourages re-typing) or via a separate env handle
# (bypasses E, breaks the experimental contract).
# ---------------------------------------------------------------------------


def write_chat_turn_entries(log, env, session_idx: int) -> int:
    """Mirror the current chat session into the unified log as one
    ``kind="chat_turn"`` entry per turn. Idempotent — if the session's
    chat_turn entries already exist (e.g. on checkpoint resume), no
    new entries are written.

    Each entry's metadata carries ``session_date`` (raw string) and
    ``session_date_iso`` (parsed ISO date, or ``None`` if unparseable)
    so date-difference probes ("how many days between visit X and
    visit Y") can resolve calendar dates from the log alone, without
    requiring the agent to have stored a date column in its memoryspace.

    Returns the number of entries written this call.
    """
    session = getattr(env, "current_session", None)
    if not session:
        return 0

    # Idempotency guard: skip if any chat_turn entry for this session
    # already exists in the log (resume case, or accidental double-call).
    for e in log.entries:
        if e.session_idx == session_idx and (e.metadata or {}).get("kind") == "chat_turn":
            return 0

    meta = getattr(env, "current_session_meta", {}) or {}
    session_date_raw = meta.get("session_date")
    session_id = meta.get("session_id")
    session_date_iso: str | None = None
    if session_date_raw:
        # Best-effort ISO parse; failures (free-text dates) leave
        # session_date_iso as None — the raw string is still queryable.
        try:
            from Scroll.benchmarks.longmemeval._time_utils import session_date_to_iso
            session_date_iso = session_date_to_iso(str(session_date_raw))
        except Exception:  # noqa: BLE001
            session_date_iso = None

    written = 0
    for turn_idx, turn in enumerate(session):
        md: dict[str, Any] = {"kind": "chat_turn", "turn_idx": turn_idx}
        if session_date_raw is not None:
            md["session_date"] = str(session_date_raw)
        if session_date_iso is not None:
            md["session_date_iso"] = session_date_iso
        if session_id is not None:
            md["session_id"] = str(session_id)
        log.append(LogEntry.make(
            session_idx=session_idx,
            role=str(turn.get("role", "user")),
            content=str(turn.get("content", "")),
            metadata=md,
        ))
        written += 1
    return written


def _make_today_session(agent) -> Callable[[], list[dict[str, Any]]]:
    """Return a function that yields the current chat session as a list
    of turn dicts. Reads from the unified log
    (``log.slice(session_idx=today, kind="chat_turn")``) — the
    chat_turn entries are the source of truth; this wrapper just
    returns native dicts instead of ``LogEntry`` objects.
    """

    def today_session() -> list[dict[str, Any]]:
        log = getattr(agent, "log", None)
        today = getattr(agent, "_current_session", None)
        if log is None or today is None:
            return []
        out: list[dict[str, Any]] = []
        for e in log.entries:
            md = e.metadata or {}
            if e.session_idx == today and md.get("kind") == "chat_turn":
                out.append({
                    "role": e.role,
                    "content": e.content,
                    "turn_idx": md.get("turn_idx"),
                    # Surface session_date so the agent can ingest it
                    # without a separate env lookup. session_date_iso
                    # is the parsed YYYY-MM-DD form (None if the raw
                    # date didn't parse).
                    "session_date": md.get("session_date"),
                    "session_date_iso": md.get("session_date_iso"),
                })
        return out

    return today_session


def make_chat_memory_namespace(state, agent=None) -> dict[str, Callable]:
    """Return the REPL namespace common to every chat-memory agent
    (LongMemEval, BEAM, …).

    ``state`` is the framework :class:`ToolState` (used by
    ``wait_for_next_day``). ``agent`` is the agent instance (used by
    ``today_session`` to read chat_turn entries from the agent's log).

    Also pre-binds a small stdlib surface (``re``, ``json``,
    ``Counter``, ``defaultdict``, ``date``, ``timedelta``,
    ``itertools``) so probe-time cells don't waste a turn on
    ``import`` boilerplate. Anything else still needs an explicit
    ``import`` — full stdlib access is intentional.
    """
    import re as _re
    import json as _json
    import itertools as _itertools
    from collections import Counter as _Counter, defaultdict as _defaultdict
    from datetime import date as _date, timedelta as _timedelta

    ns: dict[str, Callable] = {
        "wait_for_next_day": make_wait_for_next_day(state),
        "re": _re,
        "json": _json,
        "itertools": _itertools,
        "Counter": _Counter,
        "defaultdict": _defaultdict,
        "date": _date,
        "timedelta": _timedelta,
    }
    if agent is not None:
        ns["today_session"] = _make_today_session(agent)
    return ns
