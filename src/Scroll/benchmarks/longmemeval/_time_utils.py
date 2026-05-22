"""Time helpers for LongMemEval ingest paths.

The dataset stores ``session_date`` as a free-text string — e.g.
``"2023/05/20 (Sat) 02:21"``. Sorting that string lexically *almost*
works (the YYYY/MM/DD prefix is monotonic), but to enable proper
date-range filtering during temporal-reasoning probes we also store
an ISO-8601 form alongside it. Used by ``LongMemEvalAgent``'s auto-ingest.
"""

from __future__ import annotations

import re

# Examples seen in the dataset:
#   "2023/05/20 (Sat) 02:21"
#   "2023/05/20 02:21"
#   "2023/5/9 (Tue) 9:05"
# The day-of-week parenthetical and any zero-padding are optional.
_DATE_RE = re.compile(
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
    m = _DATE_RE.match(raw)
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
