"""Ingest a LongMemEval haystack (``sessions.json``) into a seed history DB.

A LongMemEval instance's haystack is a list of prior chat **sessions**, each a
list of ``{role, content}`` turns with a session date. We write every turn as a
``conversation_history`` row (``kind="conversation"``) so a later scroll-agent
session sharing the same ``task_id`` recalls it via ``ms.search(scope="task")`` —
the prior dialogue becomes the agent's long-term memory rather than a file it reads.

Design mirrors ``evals/beam/ingest.py`` so the two memory evals present an
identical seed schema to the agent (and the same ``prompts/system.md`` wording):

- **One seed session per haystack session** (``session_id = f"seed:{task_id}:s{N}"``),
  all under the shared ``task_id``; ``step_index = N`` is the 1-based session number.
- Each turn's ``content`` is tagged ``[Session N | <ISO date>] role: <text>`` (what
  FTS indexes and the agent reads back); ``metadata.date`` is the sortable ISO date
  for range/ordering queries.
- ``msg_index`` is monotonic across the whole conversation → order by it for chronology.
- The WAL is checkpointed at the end so the single ``.db`` file is complete and can
  be copied once into the chat's shared history DB (see the runner). Every probe then
  reads/writes that one DB, isolated by ``run_id`` (see ``SEED_RUN_ID``).
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from scroll_context import HistoryStore
from scroll_context import LogEntry
from scroll_context.manager import _clip_sentences

# Sentinel ``run_id`` every seeded prior-conversation row carries. MUST be
# "seed": ScrollContextManager.seed_index_map() keys the in-context memory map
# off ``run_id='seed'``, and the runner passes ``shared_run_ids=(SEED_RUN_ID,)``
# so a probe retrieves the seeded conversation plus its own turns, never a
# sibling probe's. (Same value/role as beam's SEED_RUN_ID.)
SEED_RUN_ID = "seed"

# LongMemEval session dates look like ``"2023/05/20 (Sat) 02:21"``; question_date
# similarly. Pull out the leading Y/M/D and render sortable ISO ``YYYY-MM-DD``
# (the raw form is not lexically sortable, so date-range/ordering probes need this).
_DATE_RE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})")

_SESSION_HEADLINE_MAX = 120  # chars — a session's index line, not a paragraph
_MILESTONES_PER_SESSION = 10  # headlines sampled per seed session

# Assistant replies often open with a throwaway interjection ("Certainly!",
# "Sure,") before the substance — strip it so the headline gist starts on real
# content. Require punctuation right after the word so content uses ("Great job")
# are left intact.
_FILLER_RE = re.compile(
    r"^(?:certainly|sure|absolutely|of course|no problem|ok(?:ay)?)[,!.:;]+\s*",
    re.IGNORECASE,
)


def to_iso_date(raw: str | None) -> str | None:
    """Return the sortable ISO date ``YYYY-MM-DD`` from a LongMemEval date string."""
    m = _DATE_RE.search(raw or "")
    if not m:
        return None
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"


def _strip_filler(gist: str) -> str:
    """Drop a leading conversational interjection from an assistant reply."""
    stripped = _FILLER_RE.sub("", gist, count=1).lstrip()
    return stripped or gist


def _format_content(session_num: int, iso_date: str | None, role: str, text: str) -> str:
    tag = (
        f"[Session {session_num} | {iso_date}]"
        if iso_date
        else f"[Session {session_num}]"
    )
    return f"{tag} {role or 'user'}: {(text or '').strip()}"


def _evenly_spaced(positions: list[int], k: int) -> set[int]:
    """``k`` indices evenly spaced across ``positions`` (always incl. ends)."""
    if len(positions) <= k:
        return set(positions)
    step = (len(positions) - 1) / (k - 1)
    return {positions[round(j * step)] for j in range(k)}


def _session_headline(session_num: int, iso_date: str | None, text: str) -> str:
    """A deterministic one-line gist for a seed session milestone turn.

    ``Session N | <date> — <message, trimmed>``. Milestones are sampled from a
    session's ASSISTANT turns — mirroring the live eviction index, where only the
    model's own turns carry a headline — so the seeded map reads as a date-ordered
    table of contents of each session's arc, not a per-turn transcript.
    """
    tag = f"Session {session_num} | {iso_date}" if iso_date else f"Session {session_num}"
    gist = _strip_filler(" ".join((text or "").split()))
    gist = _clip_sentences(gist, _SESSION_HEADLINE_MAX)
    return f"{tag} — {gist}" if gist else tag


def build_seed_db(
    sessions: list[dict],
    task_id: str,
    db_path: str | Path,
    *,
    seed_index: bool = True,
) -> int:
    """Write every turn of ``sessions`` into a fresh seed DB. Returns row count.

    ``sessions`` is the ``sessions`` list of a task's ``sessions.json``: each item
    is ``{session_id?, date, turns: [{role, content}]}``, in chronological order
    (the task generator sorts them). Rows are appended under
    ``session_id=f"seed:{task_id}:s{N}"`` with the shared ``task_id`` and
    ``run_id=SEED_RUN_ID`` so retrieval at ``scope="task"`` spans all seed sessions.

    ``seed_index=False`` is the index-OFF ablation: every row's ``headline`` is left
    NULL so the seed index data is physically absent from the DB (an agent can
    otherwise reach it via ``SELECT headline …`` even with the in-context map off).
    Without precomputed headlines we sample a few milestone turns per session, evenly
    spaced across its ASSISTANT turns and capped at ``_MILESTONES_PER_SESSION``.
    """
    store = HistoryStore(db_path)
    # The seed DB is a rebuildable cache; skip per-turn fsync for a fast bulk build.
    # The final wal_checkpoint(TRUNCATE) still flushes durably.
    store._conn.execute("PRAGMA synchronous=NORMAL")

    # Flatten to (session_num, iso_date, role, content) so headline sampling can index
    # by global turn position, mirroring beam's per-session assistant sampling.
    turns: list[dict[str, Any]] = []
    for idx, session in enumerate(sessions):
        session_num = idx + 1
        iso_date = to_iso_date(session.get("date"))
        raw_sid = session.get("session_id")
        for msg in session.get("turns", []):
            turns.append(
                {
                    "session_num": session_num,
                    "iso_date": iso_date,
                    "role": msg.get("role"),
                    "content": msg.get("content", ""),
                    "session_id": raw_sid,
                }
            )

    headline_pos: set[int] = set()
    if seed_index:
        assistant_pos: dict[int, list[int]] = {}
        first_pos: dict[int, int] = {}
        for i, t in enumerate(turns):
            n = t["session_num"]
            first_pos.setdefault(n, i)
            if (t.get("role") or "") == "assistant":
                assistant_pos.setdefault(n, []).append(i)
        for n, fp in first_pos.items():
            positions = assistant_pos.get(n) or [fp]
            headline_pos |= _evenly_spaced(positions, _MILESTONES_PER_SESSION)

    msg_index = 0
    try:
        for i, t in enumerate(turns):
            n = t["session_num"]
            headline = (
                _session_headline(n, t["iso_date"], t["content"])
                if (seed_index and i in headline_pos)
                else None
            )
            store.append(
                session_id=f"seed:{task_id}:s{n}",
                run_id=SEED_RUN_ID,
                task_id=task_id,
                entry=LogEntry(
                    kind="conversation",
                    role=t["role"],
                    content=_format_content(n, t["iso_date"], t["role"], t["content"]),
                    step_index=n,
                    msg_index=msg_index,
                    headline=headline,
                    metadata={
                        "session_num": n,
                        "session_id": t.get("session_id"),
                        "date": t["iso_date"],
                    },
                ),
            )
            msg_index += 1
    finally:
        store.close()

    # Collapse the WAL into the main file so a plain file copy is complete
    # (the runner copies this DB per chat for isolation).
    conn = sqlite3.connect(str(Path(db_path).expanduser()))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    return msg_index


def build_seed_db_for_task(
    task_dir: str | Path,
    task_id: str,
    db_path: str | Path,
    *,
    seed_index: bool = True,
) -> int:
    """Convenience: read ``<task_dir>/sessions.json`` and build its seed DB."""
    task_dir = Path(task_dir)
    payload = json.loads((task_dir / "sessions.json").read_text(encoding="utf-8"))
    sessions = payload.get("sessions", []) if isinstance(payload, dict) else payload
    return build_seed_db(sessions, task_id, db_path, seed_index=seed_index)
