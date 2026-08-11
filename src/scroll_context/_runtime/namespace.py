from __future__ import annotations

import inspect
import re
from datetime import date
from typing import Any

from scroll_context._runtime.memoryspace import MemorySpace, or_terms


# Names ``populate()`` injects — excluded from the working-memory digest so it
# shows only the variables the *model* created, not the fixtures it was given.
_INJECTED = ("ms", "days_between", "or_terms")


def populate(
    ns: dict[str, Any],
    *,
    memoryspace: MemorySpace,
) -> None:
    """Inject the model-visible names into the runtime's REPL namespace.

    The model writes Python that resolves these names: `ms` for the read-only
    query window onto its durable cross-session history (ATTACHed as
    `hist.conversation_history`, with `ms.search(...)` / `ms.sql_query(...)` and
    `ms.session_id` / `ms.task_id` to scope to the current run or to the task
    across all runs); `days_between(d1, d2, inclusive=False)` for reliable
    calendar math; and `or_terms([...])` to build a safe FTS5 OR match
    expression (auto-quotes phrases/hyphenated terms). The model keeps its own
    working data in plain variables it assigns here — they persist across
    `execute_python` calls (see ``describe``).
    """
    ns["ms"] = memoryspace
    ns["days_between"] = _days_between
    ns["or_terms"] = or_terms


def _describe_value(val: Any) -> str:
    """One-line ``type, size`` descriptor for a persisted namespace variable."""
    if isinstance(val, str):
        return f"str, {len(val)} chars"
    if isinstance(val, (list, tuple, set, frozenset)):
        return f"{type(val).__name__}, {len(val)} items"
    if isinstance(val, dict):
        return f"dict, {len(val)} keys"
    if isinstance(val, (bool, int, float)):
        return f"{type(val).__name__} = {val!r}"
    if inspect.isfunction(val):
        return "function"
    return type(val).__name__


def describe(ns: dict[str, Any]) -> str:
    """Deterministic snapshot of the model-created variables in the namespace.

    The ``execute_python`` namespace persists across turns, so a variable the
    model stored survives even after the turn that printed it scrolls out of the
    window. Listing those variables each turn (name + type + size) keeps the
    model aware of its working set without re-querying. Injected fixtures
    (``ms``, helpers), imported modules, and dunder names are hidden so only the
    model's own data shows. Empty when nothing has been stored yet.
    """
    rows = []
    for name, val in ns.items():
        if name.startswith("__") or name in _INJECTED or inspect.ismodule(val):
            continue
        rows.append(f"  - {name} ({_describe_value(val)})")
    if not rows:
        return "vars: (empty)"
    return "vars:\n" + "\n".join(sorted(rows))


_DATE_RE = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")


def _parse_date(value: Any) -> date:
    """Pull a calendar date out of the messy formats dates actually arrive in.

    Accepts ISO (``2023-06-01``, ``2023-06-01T09:00``) *and* the LongMemEval
    style (``2023/06/01 (Thu) 09:00``) — anything with a leading
    ``YYYY[-/]M[-/]D``; the weekday/time tail is ignored.
    """
    m = _DATE_RE.search(str(value))
    if not m:
        raise ValueError(f"days_between: no YYYY-MM-DD / YYYY/MM/DD date in {value!r}")
    y, mo, d = (int(g) for g in m.groups())
    return date(y, mo, d)


def _days_between(d1: str, d2: str, inclusive: bool = False) -> int:
    """Absolute number of days between two dates.

    Parses both ISO (``2023-06-01``) and LongMemEval (``2023/06/01 (Thu) 09:00``)
    formats — pass ``question_date`` / ``session_date_iso`` straight in. Always
    non-negative; ``inclusive=True`` counts both endpoints (adds 1). LLM calendar
    math is unreliable past ~2-week deltas — compute it here instead.
    """
    n = abs((_parse_date(d2) - _parse_date(d1)).days)
    return n + 1 if inclusive else n
