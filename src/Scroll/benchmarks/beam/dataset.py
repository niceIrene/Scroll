"""BEAM dataset loader.

Each chat lives at ``<dataset_root>/<scale>/<chat_id>/`` with files:

  - ``chat.json``              — list of batches, each a list of turn
                                  lists; flattened here into
                                  per-batch ``list[{role, content}]``
                                  to fit the SCROLL session shape.
  - ``probing_questions/probing_questions.json``
                               — dict[category, list[question dict]]
  - ``topic.json``             — chat-level title / theme / subtopics.
"""

from __future__ import annotations

import calendar
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)


# BEAM time anchors are formatted ``Month-Day-Year`` (e.g.
# ``"March-15-2024"``). Normalize to ISO ``YYYY-MM-DD`` so the auto-ingest
# pipeline can write valid ``session_date_iso`` / ``session_ts_iso``
# columns. Returns ``None`` on unparseable input.
_MONTHS_LC: dict[str, int] = {
    m.lower(): i for i, m in enumerate(calendar.month_name) if m
}
_MONTHS_LC.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

_BEAM_DATE_RE = re.compile(
    r"^\s*(?P<month>[A-Za-z]+)[-_ /](?P<day>\d{1,2})[-_ /](?P<year>\d{4})\s*$"
)


def normalize_beam_date(raw: str | None) -> str | None:
    """``"March-15-2024"`` -> ``"2024-03-15"`` (ISO ``YYYY-MM-DD``).

    Returns ``None`` for empty / unparseable input. Falls back to
    accepting already-ISO ``YYYY-MM-DD`` strings unchanged.
    """
    if not raw:
        return None
    s = str(raw).strip()
    iso_m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if iso_m:
        y, mo, d = iso_m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    m = _BEAM_DATE_RE.match(s)
    if not m:
        return None
    month_idx = _MONTHS_LC.get(m.group("month").lower())
    if month_idx is None:
        return None
    day = int(m.group("day"))
    year = int(m.group("year"))
    if not (1 <= day <= 31):
        return None
    return f"{year:04d}-{month_idx:02d}-{day:02d}"


@dataclass
class BeamBatch:
    """One time-anchored batch within a BEAM chat — analogous to a
    LongMemEval session."""

    batch_idx: int                  # 1-based
    time_anchor: str | None         # e.g. "March-15-2024"
    turns: list[dict]               # [{role, content}, ...]


@dataclass
class BeamProbingQuestion:
    """One probing question with its rubric."""

    category: str                   # e.g. "abstention", "knowledge_update"
    index: int                      # position within the category
    question: str
    ideal_response: str
    rubric: list[str]
    difficulty: str | None = None
    extras: dict = field(default_factory=dict)


@dataclass
class BeamItem:
    """One BEAM chat — the unit of one Scroll run."""

    scale: str
    chat_id: str
    title: str | None
    theme: str | None
    batches: list[BeamBatch]
    probing_questions: list[BeamProbingQuestion]

    @property
    def num_batches(self) -> int:
        return len(self.batches)

    @property
    def num_probes(self) -> int:
        return len(self.probing_questions)


def _flatten_batch_turns(raw_turns) -> list[dict]:
    """BEAM batches have ``turns`` as a list-of-lists (each inner list
    is a few back-and-forth messages); flatten to a flat list of
    ``{role, content}`` dicts.
    """
    flat: list[dict] = []
    for turn_list in raw_turns or []:
        if isinstance(turn_list, dict):
            # Defensive: some batches may already be flat.
            flat.append({
                "role": str(turn_list.get("role", "user")),
                "content": str(turn_list.get("content", "")),
            })
            continue
        for msg in turn_list or []:
            if not isinstance(msg, dict):
                continue
            flat.append({
                "role": str(msg.get("role", "user")),
                "content": str(msg.get("content", "")),
            })
    return flat


def load_chat(dataset_root: str | Path, scale: str, chat_id: str) -> BeamItem:
    """Load one BEAM chat by (scale, chat_id)."""
    root = Path(dataset_root) / scale / str(chat_id)
    if not root.is_dir():
        raise FileNotFoundError(
            f"BEAM chat not found at {root}. Expected layout: "
            f"<dataset_root>/<scale>/<chat_id>/{{chat.json,topic.json,"
            f"probing_questions/probing_questions.json}}"
        )

    chat_raw = json.loads((root / "chat.json").read_text(encoding="utf-8"))
    if not isinstance(chat_raw, list):
        raise ValueError(f"{root/'chat.json'}: expected top-level list of batches")

    batches: list[BeamBatch] = []
    for entry in chat_raw:
        if not isinstance(entry, dict):
            continue
        # Time anchor: batch-level field is often null in BEAM's data;
        # in practice each message inside the batch carries the same
        # ``time_anchor``. Take the first non-empty one as the batch's
        # anchor so chronological filtering still works.
        anchor = entry.get("time_anchor")
        if not anchor:
            for turn_list in entry.get("turns") or []:
                if not isinstance(turn_list, list):
                    continue
                for msg in turn_list:
                    if isinstance(msg, dict) and msg.get("time_anchor"):
                        anchor = msg["time_anchor"]
                        break
                if anchor:
                    break
        batches.append(BeamBatch(
            batch_idx=int(entry.get("batch_number") or len(batches) + 1),
            time_anchor=anchor,
            turns=_flatten_batch_turns(entry.get("turns")),
        ))

    topic_path = root / "topic.json"
    topic: dict = {}
    if topic_path.exists():
        try:
            topic = json.loads(topic_path.read_text(encoding="utf-8")) or {}
        except (ValueError, json.JSONDecodeError):
            _log.warning("%s: failed to parse; treating as empty topic", topic_path)

    pq_path = root / "probing_questions" / "probing_questions.json"
    pq_raw = json.loads(pq_path.read_text(encoding="utf-8"))
    probing: list[BeamProbingQuestion] = []
    for category, items in (pq_raw or {}).items():
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            rubric = item.get("rubric") or []
            if not isinstance(rubric, list):
                rubric = [str(rubric)]
            extras = {
                k: v for k, v in item.items()
                if k not in ("question", "ideal_response", "rubric", "difficulty")
            }
            probing.append(BeamProbingQuestion(
                category=str(category),
                index=idx,
                question=str(item.get("question") or ""),
                ideal_response=str(item.get("ideal_response") or ""),
                rubric=[str(r) for r in rubric],
                difficulty=item.get("difficulty"),
                extras=extras,
            ))

    return BeamItem(
        scale=scale,
        chat_id=str(chat_id),
        title=topic.get("title"),
        theme=topic.get("theme"),
        batches=batches,
        probing_questions=probing,
    )


def list_chats(dataset_root: str | Path, scale: str) -> list[str]:
    """Return the chat_ids present under ``<dataset_root>/<scale>/`` (sorted numerically)."""
    root = Path(dataset_root) / scale
    if not root.is_dir():
        return []
    ids = [p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    ids.sort(key=int)
    return ids
