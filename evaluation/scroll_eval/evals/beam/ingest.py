"""Ingest a BEAM ``chat.json`` into a durable conversation_history seed DB.

A BEAM conversation is a list of *batches* (sessions), each a list of turn
groups of message dicts. We write every message as a ``conversation_history``
row (``kind="conversation"``) so a later scroll-agent session sharing the same
``task_id`` can recall it via ``ms.search(scope="task")`` — turning the prior
dialogue into the agent's long-term memory rather than a file it greps.

Design points:
- **One seed session per batch** (``session_id = f"seed:{task_id}:s{batch}"``),
  all under the shared ``task_id`` — faithfully representing "N prior sessions".
- Each turn is tagged ``[Session N | <ISO date>] role: <text>`` in ``content``
  (what FTS indexes and the agent reads back), so cross-session / temporal /
  ordering questions are answerable. ``batch``/``time_anchor`` (raw) /``date``
  (sortable ISO)/ids are also kept in ``metadata``.
- BEAM's in-dialogue ``->-> i,j`` index markers are stripped from the text.
- The WAL is checkpointed at the end so the single ``.db`` file is complete and
  can be copied once into the chat's shared history DB (see the runner); every
  probe then reads/writes that one DB, isolated by ``run_id`` (see
  ``SEED_RUN_ID``) rather than by a private file copy.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from scroll_context import HistoryStore
from scroll_context import LogEntry
from scroll_context.manager import _clip_sentences

# Sentinel ``run_id`` every seeded prior-conversation row carries. It marks a
# row as the chat's shared "long-term memory" tier — visible to every probe of
# the chat — rather than a probe's own session turn. Both the seed-index map and
# the per-probe retrieval isolation (``ms.search(scope='task')``) key off it, so
# define it once here, in the layer that writes the seed rows.
SEED_RUN_ID = "seed"

# BEAM tags some user turns with a trailing "->-> 1,1" routing marker. Strip
# every occurrence (defensively, anywhere it appears), then tidy whitespace.
_MARKER = re.compile(r"\s*->->\s*[\d,]+")


def clean_content(text: str) -> str:
    """Remove BEAM ``->-> i,j`` markers and trim surrounding whitespace."""
    return _MARKER.sub("", text or "").strip()


def to_iso_date(anchor: str | None) -> str | None:
    """Convert a BEAM ``time_anchor`` to a sortable ISO date.

    BEAM stamps dates as ``Month-DD-YYYY`` (e.g. ``"July-01-2024"``), which does
    not sort or range-filter lexically. Return the ISO form (``"2024-07-01"``)
    so date-range / ordering queries can compare it directly; ``None`` if the
    anchor is missing or not in the expected format.
    """
    if not anchor:
        return None
    try:
        return datetime.strptime(anchor, "%B-%d-%Y").date().isoformat()
    except ValueError:
        return None


def _first_dated(batch: dict) -> str | None:
    """The first non-empty ``time_anchor`` among a batch's messages, if any."""
    for group in batch.get("turns", []):
        for msg in group:
            anchor = msg.get("time_anchor")
            if anchor:
                return anchor
    return None


def iter_turns(chat: list[dict]) -> Iterator[dict]:
    """Yield messages in conversation order across batches and turn groups.

    Each yielded dict: ``{batch, time_anchor, role, content, per_batch_id,
    question_type}``. ``chat`` is the parsed ``chat.json`` (list of batches).
    """
    for batch in chat:
        batch_number = batch.get("batch_number")
        # BEAM stamps a date only on the FIRST message of each batch (the rest
        # are null). Derive the batch's session date — from the batch field if
        # present (500K/1M), else the first dated message (10M) — and propagate
        # it to every turn, so each row is date-filterable via
        # metadata.time_anchor and its content tag shows the date. Date-range
        # probes (when did X happen / what happened between dates) depend on
        # every turn knowing its date, not just the session opener.
        batch_time_anchor = batch.get("time_anchor") or _first_dated(batch)
        for group in batch.get("turns", []):
            for msg in group:
                yield {
                    "batch": batch_number,
                    "time_anchor": msg.get("time_anchor") or batch_time_anchor,
                    "role": msg.get("role"),
                    "content": msg.get("content", ""),
                    "per_batch_id": msg.get("id"),
                    "question_type": msg.get("question_type"),
                }


def _format_content(turn: dict) -> str:
    anchor = turn.get("time_anchor")
    session = turn.get("batch")
    # Prefer the sortable ISO date in the visible tag so chronology survives
    # into snippet excerpts and is directly comparable to the question's dates;
    # fall back to the raw anchor if it doesn't parse.
    date_str = to_iso_date(anchor) or anchor
    tag = f"[Session {session} | {date_str}]" if date_str else f"[Session {session}]"
    role = turn.get("role") or "user"
    return f"{tag} {role}: {clean_content(turn.get('content', ''))}"


_SESSION_HEADLINE_MAX = 120  # chars — a session's index line, not a paragraph
_MILESTONES_PER_SESSION = 10  # headlines sampled per seed session

# Assistant replies open with a throwaway interjection ("Certainly!", "Sure,")
# before the substance — strip it so the headline gist starts on real content.
# Require punctuation right after the word so content uses ("Great job", "Sure
# thing") are left intact.
_FILLER_RE = re.compile(
    r"^(?:certainly|sure|absolutely|of course|no problem|ok(?:ay)?)[,!.:;]+\s*",
    re.IGNORECASE,
)


def _strip_filler(gist: str) -> str:
    """Drop a leading conversational interjection from an assistant reply."""
    stripped = _FILLER_RE.sub("", gist, count=1).lstrip()
    return stripped or gist


def _evenly_spaced(positions: list[int], k: int) -> set[int]:
    """``k`` indices evenly spaced across ``positions`` (always incl. ends)."""
    if len(positions) <= k:
        return set(positions)
    step = (len(positions) - 1) / (k - 1)
    return {positions[round(j * step)] for j in range(k)}


def _session_headline(turn: dict, *, anchor: str | None = None) -> str:
    """A deterministic one-line gist for a seed session milestone turn.

    ``Session N | <date> — <message, trimmed>``. Milestones are sampled from a
    session's ASSISTANT turns — mirroring the live eviction index, where only
    the model's own turns carry a headline — so the seeded map reads as a
    date-ordered table of contents of each session's arc, not a per-turn
    transcript. ``anchor`` overrides the turn's own date — only a session's
    opening turn carries a ``time_anchor``, so the caller passes the session
    date to keep every milestone dated.
    """
    anchor = anchor if anchor is not None else turn.get("time_anchor")
    session = turn.get("batch")
    tag = f"Session {session} | {anchor}" if anchor else f"Session {session}"
    gist = _strip_filler(" ".join(clean_content(turn.get("content", "")).split()))
    gist = _clip_sentences(gist, _SESSION_HEADLINE_MAX)
    return f"{tag} — {gist}" if gist else tag


def build_seed_db(
    chat: list[dict],
    task_id: str,
    db_path: str | Path,
    headlines: dict[str, str] | None = None,
    *,
    seed_index: bool = True,
) -> int:
    """Write every turn of ``chat`` into a fresh seed DB. Returns row count.

    Rows are appended under ``session_id=f"seed:{task_id}:s{batch}"`` (one seed
    session per batch) with the shared ``task_id`` and ``run_id=SEED_RUN_ID`` so
    retrieval at ``scope="task"`` spans all seed sessions (and, in the shared
    history DB, stays scoped to this seed tier plus the calling probe's own
    session). ``msg_index`` is monotonic across the whole conversation;
    ``step_index`` carries the batch number.

    ``headlines`` (from ``headlines.py`` / ``<task_dir>/headlines.json``) maps an
    assistant message ``id`` -> a model-written one-line headline. When given,
    every assistant turn present in the map carries that headline; an assistant
    turn absent from the map (e.g. a middle turn under ``--endpoints``) gets no
    headline, so a sparse map ingests cleanly. When omitted we fall back to
    extractive milestone headlines — a few evenly-spaced assistant turns per
    session, trimmed from their own content.

    ``seed_index=False`` is the index-OFF ablation: every row's ``headline`` is
    left NULL so the seed index data is *physically absent* from the DB, not just
    hidden from the prompt. Without this, the agent can still reach the index by
    selecting the ``headline`` column via ``ms.sql_query`` even with the in-context
    map disabled — a leak that makes ``--no-index`` runs still "use the index".
    """
    store = HistoryStore(db_path)
    # The seed DB is a rebuildable cache (regenerated from chat.json), so it needs
    # no per-commit durability during the bulk build. HistoryStore.append commits
    # (and, at the default synchronous=FULL, fsyncs) once per turn — tens of
    # thousands of fsyncs stall ingestion for minutes on a slow/contended disk.
    # NORMAL fsyncs only at checkpoints; the final wal_checkpoint(TRUNCATE) still
    # flushes durably, so the on-disk result is identical, just far faster.
    store._conn.execute("PRAGMA synchronous=NORMAL")
    msg_index = 0
    turns = list(iter_turns(chat))
    # Without precomputed headlines we sample a few milestone turns per session,
    # evenly-spaced across its ASSISTANT turns (mirroring the live eviction
    # index, where only the model's own turns carry a headline), capped at
    # _MILESTONES_PER_SESSION. Dense enough that the agent can outline a session
    # from a near-free `WHERE headline IS NOT NULL` query instead of reading full
    # content. A session with no assistant turn falls back to its opener.
    batch_anchor: dict = {}
    headline_pos: set = set()
    if headlines is None and seed_index:
        first_pos: dict = {}
        assistant_pos: dict = {}
        for i, t in enumerate(turns):
            b = t.get("batch")
            if b not in first_pos:
                first_pos[b] = i
                batch_anchor[b] = t.get("time_anchor")  # the session's date
            if (t.get("role") or "") == "assistant":
                assistant_pos.setdefault(b, []).append(i)
        for b, fp in first_pos.items():
            positions = assistant_pos.get(b) or [fp]
            headline_pos |= _evenly_spaced(positions, _MILESTONES_PER_SESSION)
    try:
        for i, turn in enumerate(turns):
            batch = turn.get("batch")
            if not seed_index:
                # Index-OFF ablation: never persist headlines, so the seed index
                # data isn't reachable via `SELECT headline FROM …`.
                headline = None
            elif headlines is not None:
                # Model-written headline per assistant turn, keyed by its id.
                headline = (
                    headlines.get(str(turn.get("per_batch_id")))
                    if (turn.get("role") or "") == "assistant"
                    else None
                )
            else:
                # Milestone turns get a dated headline (assistant turns carry no
                # time_anchor of their own, so pass the session's).
                headline = (
                    _session_headline(turn, anchor=batch_anchor.get(batch))
                    if i in headline_pos
                    else None
                )
            store.append(
                session_id=f"seed:{task_id}:s{batch}",
                run_id=SEED_RUN_ID,
                task_id=task_id,
                entry=LogEntry(
                    kind="conversation",
                    role=turn.get("role"),
                    content=_format_content(turn),
                    step_index=batch if isinstance(batch, int) else None,
                    msg_index=msg_index,
                    headline=headline,
                    metadata={
                        "batch": batch,
                        "per_batch_id": turn.get("per_batch_id"),
                        "time_anchor": turn.get("time_anchor"),
                        # Sortable ISO form of time_anchor for date-range /
                        # ordering queries (the raw anchor is not sortable).
                        "date": to_iso_date(turn.get("time_anchor")),
                        "question_type": turn.get("question_type"),
                    },
                ),
            )
            msg_index += 1
    finally:
        store.close()

    # Collapse the WAL into the main file so a plain file copy is complete
    # (the runner copies this DB per probe for isolation).
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
    dense_headlines: bool = False,
) -> int:
    """Convenience: read ``<task_dir>/chat.json`` and build its seed DB.

    If ``<task_dir>/headlines.json`` exists (written by ``headlines.py``), its
    model-written per-turn headlines are used; otherwise the extractive
    milestone fallback applies. ``dense_headlines=True`` (the ``--dense``
    toggle) reads ``headlines-dense.json`` instead — the dense per-exchange
    variant kept alongside the default file — and fails loudly if that file is
    missing, since the caller explicitly asked for it. ``seed_index=False``
    (the ``--no-index`` ablation) leaves the ``headline`` column NULL
    regardless — see ``build_seed_db``.
    """
    import json

    task_dir = Path(task_dir)
    chat = json.loads((task_dir / "chat.json").read_text(encoding="utf-8"))
    if dense_headlines:
        headlines_path = task_dir / "headlines-dense.json"
        if not headlines_path.exists():
            raise FileNotFoundError(
                f"--dense requested but {headlines_path} does not exist"
            )
    else:
        headlines_path = task_dir / "headlines.json"
    headlines = (
        json.loads(headlines_path.read_text(encoding="utf-8"))
        if headlines_path.exists()
        else None
    )
    return build_seed_db(chat, task_id, db_path, headlines=headlines, seed_index=seed_index)
