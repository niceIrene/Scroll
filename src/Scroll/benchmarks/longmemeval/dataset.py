"""LongMemEval dataset loader.

Each item in the JSON file (oracle / s_cleaned / m_cleaned) carries:

  - ``question_id``        : unique id (suffix ``_abs`` marks abstention items)
  - ``question_type``      : skill bucket — drives the judge prompt template
  - ``question`` / ``answer`` / ``question_date``
  - ``haystack_session_ids`` / ``haystack_dates`` / ``haystack_sessions``
  - ``answer_session_ids``  : evidence session ids (used by the upstream
                              retrieval-recall metric; we don't score
                              against this — passive memory only)

The oracle file's ``haystack_session_ids`` are NOT pre-sorted; we sort
by ``haystack_dates`` on load so the session loop replays sessions in
chronological order. ``s_cleaned`` and ``m_cleaned`` are already sorted
upstream (we still sort defensively — cheap and idempotent).

``has_answer`` flags inside session turns mark evidence the agent must
not be allowed to filter by — including them in the auto-ingest schema
would be GT leakage. They're dropped at load time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)


@dataclass
class LongMemEvalItem:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_session_ids: list[str]
    haystack_dates: list[str]
    haystack_sessions: list[list[dict]]
    answer_session_ids: list[str]
    is_abstention: bool

    @property
    def total_sessions(self) -> int:
        return len(self.haystack_sessions)


def load_items(
    path: str | Path,
    question_id: str | None = None,
) -> list[LongMemEvalItem]:
    """Load LME items from ``path``.

    Memory optimization: if ``question_id`` is given AND a shard at
    ``<path-without-ext>_shards/<qid>.json`` exists, load ONLY that
    shard (~500KB) instead of the full dataset (~264MB for s_cleaned).
    Subprocesses doing per-QA work get 500× memory reduction this way.

    Falls back to loading the full file if the shard is absent — so
    sharding is purely opt-in (run ``scripts/shard_lme_dataset.py``
    once to enable).
    """
    p = Path(path)
    # Try shard fast-path
    if question_id is not None:
        shard_dir = p.with_suffix("").as_posix() + "_shards"
        shard = Path(shard_dir) / f"{question_id}.json"
        if shard.exists():
            p = shard

    raw = json.loads(p.read_text(encoding="utf-8"))
    out: list[LongMemEvalItem] = []
    for entry in raw:
        sids = list(entry.get("haystack_session_ids") or [])
        dates = list(entry.get("haystack_dates") or [])
        sessions = list(entry.get("haystack_sessions") or [])
        if not (len(sids) == len(dates) == len(sessions)):
            _log.warning(
                "qid=%s session-list lengths disagree (sids=%d, dates=%d, sessions=%d) — skipping",
                entry.get("question_id"),
                len(sids), len(dates), len(sessions),
            )
            continue
        triples = sorted(zip(dates, sids, sessions), key=lambda t: t[0])
        sorted_dates = [t[0] for t in triples]
        sorted_sids = [t[1] for t in triples]
        sorted_sessions = [_clean_session(t[2]) for t in triples]
        qid = str(entry["question_id"])
        out.append(LongMemEvalItem(
            question_id=qid,
            question_type=str(entry["question_type"]),
            question=str(entry["question"]),
            answer=str(entry["answer"]),
            question_date=str(entry.get("question_date", "")),
            haystack_session_ids=sorted_sids,
            haystack_dates=sorted_dates,
            haystack_sessions=sorted_sessions,
            answer_session_ids=list(entry.get("answer_session_ids") or []),
            is_abstention="_abs" in qid,
        ))
    return out


def _clean_session(turns: list[dict]) -> list[dict]:
    """Strip ``has_answer`` from every turn — including it would leak GT."""
    out: list[dict] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        out.append({
            "role": str(t.get("role", "user")),
            "content": str(t.get("content", "")),
        })
    return out


def select_item(
    items: list[LongMemEvalItem],
    question_id: str | None = None,
    question_index: int | None = None,
) -> LongMemEvalItem:
    if question_id is not None:
        for it in items:
            if it.question_id == question_id:
                return it
        raise KeyError(f"question_id={question_id!r} not in dataset")
    if question_index is not None:
        if question_index < 0 or question_index >= len(items):
            raise IndexError(f"question_index={question_index} out of range [0, {len(items)})")
        return items[question_index]
    raise ValueError(
        "select_item requires either question_id or question_index"
    )
