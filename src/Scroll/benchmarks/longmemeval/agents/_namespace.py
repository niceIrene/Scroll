"""REPL namespace builder for LongMemEval agents.

LME has no env actions — the only callables each agent gets are
``wait_for_next_day`` and ``today_session`` (a convenience handle
that reads the current session's chat turns out of the unified log).
Strategy-specific objects (``log``, ``ms``, ``rlm``) are added by the
agent class itself.
"""

from __future__ import annotations

from typing import Any, Callable

from Scroll.core import LogEntry
from Scroll.core._codeact_agent import make_wait_for_next_day


# ---------------------------------------------------------------------------
# Chat-turn entries: source of truth for the session's transcript.
#
# Each LME ``haystack_session`` turn is written to the unified log as one
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


def make_today_session(agent) -> Callable[[], list[dict[str, Any]]]:
    """Return a function that yields the current chat session as a list
    of turn dicts. Reads from the unified log
    (``log.slice(session_idx=today, kind="chat_turn")``) — the chat_turn
    entries are the source of truth, this function is just an ergonomic
    wrapper that returns native dicts instead of ``LogEntry`` objects.

    Equivalent to (and the agent can call this directly instead):
        [{"role": e.role, "content": e.content,
          "turn_idx": e.metadata.get("turn_idx")}
         for e in log.slice(session_idx=today, kind="chat_turn")]

    Returns:
        ``list[dict]`` — one per turn in the current session, in order.
        Empty on the probe session (no haystack content) or before
        chat_turn entries are written.
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


def make_lme_namespace(state, agent=None) -> dict[str, Callable]:
    """Return the REPL namespace common to every LME agent.

    ``state`` is the framework :class:`ToolState` (used by
    ``wait_for_next_day``). ``agent`` is the LME agent instance
    (used by ``today_session`` to read chat_turn entries from the
    agent's log).

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

    ns = {
        "wait_for_next_day": make_wait_for_next_day(state),
        # stdlib helpers pre-bound (paper-style; matches the prompt's
        # CODE-WRITING AFFORDANCES section).
        "re": _re,
        "json": _json,
        "itertools": _itertools,
        "Counter": _Counter,
        "defaultdict": _defaultdict,
        "date": _date,
        "timedelta": _timedelta,
    }
    if agent is not None:
        ns["today_session"] = make_today_session(agent)
    return ns
