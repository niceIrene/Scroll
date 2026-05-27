"""Date/time helpers for the LongMemEval ingest path.

Two separate parsers live here (both consumed only by ``ingestor.py``):

1. ``session_date_to_iso`` — fixed-format session metadata. The dataset
   stores each session's timestamp as ``"2023/05/20 (Sat) 02:21"``
   (YYYY/MM/DD + optional weekday + optional HH:MM). One regex, no
   anchor, one-shot ISO conversion. Used to populate
   ``chat_turns.session_ts_iso``.

2. ``resolve_date`` — free-text date phrases found inside chat content
   ("last week", "March 15", "yesterday", "3 days ago"). Combines
   absolute-form parsers (ISO / slash / "March 15, 2024" / "15 of
   March") with relative-form parsers anchored against a session
   date. Used by the typed-extraction layer (``extract_typed=True``)
   to populate the ``event_dates`` table. LME's default path skips
   typed extraction; BEAM still uses it.

The two parsers don't share code — they handle different input shapes
— but they live together because they're both LME ingest concerns
and both are dead weight outside the ingestor.
"""

from __future__ import annotations

import re
from datetime import date, timedelta


# =============================================================================
# 1. Session-metadata timestamp → ISO  (used unconditionally on every ingest)
# =============================================================================

# Examples seen in the dataset:
#   "2023/05/20 (Sat) 02:21"
#   "2023/05/20 02:21"
#   "2023/5/9 (Tue) 9:05"
# The day-of-week parenthetical and any zero-padding are optional.
_SESSION_TS_RE = re.compile(
    r"""
    ^\s*
    (?P<y>\d{4}) [/-] (?P<m>\d{1,2}) [/-] (?P<d>\d{1,2})
    (?: \s+ \([^)]*\) )?           # optional "(Sat)" weekday
    (?:
        \s+
        (?P<H>\d{1,2}) : (?P<M>\d{2})
        (?: : (?P<S>\d{2}) )?
    )?
    \s*$
    """,
    re.VERBOSE,
)


def session_date_to_iso(raw: str | None) -> str | None:
    """Parse a free-text ``session_date`` into ISO 8601 (no timezone).

    Returns ``None`` for unparseable input — callers should store NULL
    in that case rather than crashing the ingest.
    """
    if not raw:
        return None
    m = _SESSION_TS_RE.match(raw)
    if not m:
        return None
    y = int(m.group("y"))
    mm = int(m.group("m"))
    dd = int(m.group("d"))
    H = int(m.group("H") or 0)
    M = int(m.group("M") or 0)
    S = int(m.group("S") or 0)
    if not (1 <= mm <= 12 and 1 <= dd <= 31 and 0 <= H <= 23 and 0 <= M <= 59 and 0 <= S <= 59):
        return None
    return f"{y:04d}-{mm:02d}-{dd:02d}T{H:02d}:{M:02d}:{S:02d}"


# =============================================================================
# 2. Free-text date phrases inside chat content → ISO + offset
#    (used only by typed extraction; BEAM today, LME if extract_typed=True)
# =============================================================================
#
# Three-date model:
#   observation_date — when it was said (session_date_iso)
#   referenced_date  — what date the phrase points to (ISO, may be None)
#   relative_offset  — signed days from observation to referenced
#
# Usage:
#   iso, offset = resolve_date("last week", anchor_iso="2023-05-15")
#   # → ("2023-05-08", -7)

# ---------------------------------------------------------------------------
# Lexical building blocks (also referenced by the ingestor's ``_DATE_RE``)
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


# Alias kept for the ingestor's typed-extraction path, which imports
# this under the leading-underscore name.
_resolve_date_text = resolve_date
