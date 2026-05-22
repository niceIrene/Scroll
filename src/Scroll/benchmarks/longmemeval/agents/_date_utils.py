"""Shared date-resolution helpers for LongMemEval agents.

Used by ``code_auto.py`` (auto-ingest of ``event_dates``) and
``code_agent.py`` (parse_observations + REPL-bound ``resolve_date``).
Lives in its own module to prevent the ~180 lines of regex + resolver
logic from drifting between the two agents.

Three-date model:
  observation_date — when it was said (session_date_iso)
  referenced_date  — what date the phrase points to (ISO, may be None)
  relative_offset  — signed days from observation to referenced

Usage:
  from Scroll.benchmarks.longmemeval.agents._date_utils import resolve_date
  iso, offset = resolve_date("last week", anchor_iso="2023-05-15")
  # → ("2023-05-08", -7)
"""

from __future__ import annotations

import re
from datetime import date, timedelta


# ---------------------------------------------------------------------------
# Lexical building blocks (used by both this module and code_auto's ``_DATE_RE``)
# ---------------------------------------------------------------------------

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DOW = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"

_MONTH_TO_NUM = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_DAY_OF_WEEK_NUM = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_DAYS_PER_UNIT = {"day": 1, "week": 7, "month": 30, "year": 365}


# ---------------------------------------------------------------------------
# Compiled regexes — absolute date forms
# ---------------------------------------------------------------------------

_MD_YYYY_RE = re.compile(
    rf"\b(?P<m>{_MONTHS})\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(?P<y>\d{{4}}))?\b",
    re.IGNORECASE,
)
_D_M_YYYY_RE = re.compile(
    rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m>{_MONTHS})(?:,?\s+(?P<y>\d{{4}}))?\b",
    re.IGNORECASE,
)
_ISO_RE = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b")
_SLASH_RE = re.compile(r"\b(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{2,4})\b")


# ---------------------------------------------------------------------------
# Compiled regexes — relative date forms
# ---------------------------------------------------------------------------

# "X days/weeks/... ago" or "X days/weeks/... from now" or "X days later"
_RELATIVE_NUM_RE = re.compile(
    r"\b(?P<n>\d{1,3})\s+(?P<unit>day|week|month|year)s?\s+(?P<dir>ago|from\s+now|later)\b",
    re.IGNORECASE,
)
# "in X days/weeks/months/years"
_RELATIVE_IN_RE = re.compile(
    r"\bin\s+(?P<n>\d{1,3})\s+(?P<unit>day|week|month|year)s?\b",
    re.IGNORECASE,
)
# "last/next Monday-Sunday"
_RELATIVE_DOW_RE = re.compile(
    rf"\b(?P<dir>last|next)\s+(?P<dow>{_DOW})\b",
    re.IGNORECASE,
)
# "last/next/this week/month/year"
_RELATIVE_UNIT_RE = re.compile(
    r"\b(?P<dir>last|next|this)\s+(?P<unit>week|month|year)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------

def _date_text_to_iso(date_text: str, fallback_year: int | None = None) -> str | None:
    """Best-effort ISO conversion for ABSOLUTE dates only.

    Returns None for relative dates ("last week", "tomorrow") — those go
    through ``_resolve_relative_date`` against a session-date anchor.
    """
    s = date_text.strip()
    m = _ISO_RE.match(s)
    if m:
        return f"{int(m['y']):04d}-{int(m['m']):02d}-{int(m['d']):02d}"
    m = _SLASH_RE.match(s)
    if m:
        y = int(m["y"])
        if y < 100:
            y += 2000
        return f"{y:04d}-{int(m['m']):02d}-{int(m['d']):02d}"
    m = _MD_YYYY_RE.match(s)
    if not m:
        m = _D_M_YYYY_RE.match(s)
    if m:
        mon = _MONTH_TO_NUM.get(m["m"].lower())
        if not mon:
            return None
        y = int(m["y"]) if m["y"] else fallback_year
        if not y:
            return None
        return f"{y:04d}-{mon:02d}-{int(m['d']):02d}"
    return None


def _add_days(anchor: date, offset: int) -> tuple[str, int]:
    """Anchor + N days → (ISO, offset). Trivial helper, factored out for
    readability of the resolver."""
    return (anchor + timedelta(days=offset)).isoformat(), offset


def _resolve_relative_date(
    date_text: str,
    anchor_iso: str | None,
) -> tuple[str | None, int | None]:
    """Resolve a relative-date phrase against a session anchor.

    Returns ``(iso_str_or_None, offset_days_or_None)``. Both fields are
    None when the phrase can't be resolved or no anchor is provided.
    For "month"/"year" units we use 30/365 day approximations — fine
    for ranking and BETWEEN filters; not exact calendar arithmetic.
    """
    if not anchor_iso:
        return None, None
    try:
        anchor = date.fromisoformat(anchor_iso[:10])
    except ValueError:
        return None, None

    s = date_text.strip().lower()

    # yesterday / today / tomorrow
    if s == "yesterday":
        return _add_days(anchor, -1)
    if s == "today":
        return _add_days(anchor, 0)
    if s == "tomorrow":
        return _add_days(anchor, 1)

    # last/next <weekday>
    m = _RELATIVE_DOW_RE.match(s)
    if m:
        direction = m.group("dir").lower()
        target_dow = _DAY_OF_WEEK_NUM[m.group("dow").lower()]
        anchor_dow = anchor.weekday()
        if direction == "last":
            # Most recent target_dow strictly before anchor; if anchor IS
            # target_dow, "last X" means 7 days back.
            diff = (anchor_dow - target_dow) % 7
            offset = -(diff if diff != 0 else 7)
        else:  # next
            # Nearest target_dow strictly after anchor; if same dow, +7.
            diff = (target_dow - anchor_dow) % 7
            offset = diff if diff != 0 else 7
        return _add_days(anchor, offset)

    # last/next/this <week|month|year>
    m = _RELATIVE_UNIT_RE.match(s)
    if m:
        direction = m.group("dir").lower()
        unit = m.group("unit").lower()
        days = _DAYS_PER_UNIT[unit]
        if direction == "last":
            offset = -days
        elif direction == "next":
            offset = +days
        else:  # this
            offset = 0
        return _add_days(anchor, offset)

    # "in X <unit>"
    m = _RELATIVE_IN_RE.match(s)
    if m:
        n = int(m.group("n"))
        unit = m.group("unit").lower()
        return _add_days(anchor, +n * _DAYS_PER_UNIT[unit])

    # "X <unit> ago" / "X <unit> from now" / "X <unit> later"
    m = _RELATIVE_NUM_RE.match(s)
    if m:
        n = int(m.group("n"))
        unit = m.group("unit").lower()
        direction = m.group("dir").lower()
        if direction == "ago":
            return _add_days(anchor, -n * _DAYS_PER_UNIT[unit])
        else:  # "from now" or "later"
            return _add_days(anchor, +n * _DAYS_PER_UNIT[unit])

    return None, None


def resolve_date(
    text: str,
    anchor_iso: str | None = None,
    *,
    fallback_year: int | None = None,
) -> tuple[str | None, int | None]:
    """Top-level date resolver.

    Returns ``(iso_or_None, offset_days_or_None)``. Tries absolute parse
    first ("January 31, 2024", "2024-01-31", "1/31/2024"); on miss falls
    back to relative resolution against ``anchor_iso`` ("last week",
    "tomorrow", "in 2 weeks", "next Monday"). When both an absolute ISO
    and an anchor are available, also computes the offset.

    Examples:
        resolve_date("2024-03-15")                  -> ("2024-03-15", None)
        resolve_date("last Monday", "2024-03-15")   -> ("2024-03-11", -4)
        resolve_date("in 2 weeks",  "2024-03-15")   -> ("2024-03-29", 14)
        resolve_date("yesterday",   "2024-03-15")   -> ("2024-03-14", -1)
    """
    iso = _date_text_to_iso(text, fallback_year=fallback_year)
    if iso is None:
        return _resolve_relative_date(text, anchor_iso)
    if anchor_iso:
        try:
            anchor = date.fromisoformat(anchor_iso[:10])
            target = date.fromisoformat(iso)
            return iso, (target - anchor).days
        except ValueError:
            pass
    return iso, None


# Backward-compat alias used by code_auto's auto-ingest path.
_resolve_date_text = resolve_date
