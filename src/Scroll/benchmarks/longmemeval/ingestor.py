"""LongMemEval ``E → W`` ingestor.

Consumes ``kind="chat_turn"`` entries from the conversation log (E) and
populates the LongMemEval memoryspace (W): five tables
(``chat_turns``, ``user_preferences``, ``event_dates``, ``facts``,
``sessions``) plus a ``rounds`` view, plus per-turn / per-session
vector store entries.

The ingestor is the **only** writer of W — both at runtime (via
``Memoryspace._maybe_catch_up``) and offline (via the
``Scroll rebuild-w`` CLI). Replaying the same jsonl from scratch
through this module must produce a byte-for-byte equivalent W.

Regex-only — no LLM at ingest time. The regexes themselves are
chosen to be conservative; LongMemEval's evaluation does the
semantic heavy-lifting at probe time via the agent's CodeAct calls
to ``rlm`` / ``ms.sql_exec`` / ``ms.vector_query``.
"""

from __future__ import annotations

import re
from typing import Iterable

from Scroll.core._ingestor import Ingestor
from Scroll.core._models import LogEntry
from Scroll.tools.memoryspace import Memoryspace

from Scroll.benchmarks.longmemeval._time_utils import (
    _DOW,
    _MONTHS,
    _resolve_date_text,
    session_date_to_iso,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Slimmed LME schema. Previously held 5 tables; the v4 500-QA tool-usage
# audit showed user_preferences / event_dates / facts received <2% of
# agent queries (1-2 hits out of 250+), and sessions.session_text was a
# duplicate of chat_turns content with even rarer access. All four were
# pre-extracted "convenience" surfaces the agent never reached for —
# qwen3.7-max overwhelmingly prefers WHERE content LIKE on the raw
# rounds view. BEAM still ships those tables in its own schema and uses
# the typed-extraction layer of LMEIngestor (see ``extract_typed``).
_LME_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS chat_turns (
    session_idx      INTEGER NOT NULL,
    turn_idx         INTEGER NOT NULL,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    session_id       TEXT,
    session_date_iso TEXT,                    -- "YYYY-MM-DD"
    session_ts_iso   TEXT,                    -- full ISO ts, sortable
    PRIMARY KEY (session_idx, turn_idx)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_idx      INTEGER PRIMARY KEY,
    session_id       TEXT,
    session_date_iso TEXT,
    session_ts_iso   TEXT
);

CREATE INDEX IF NOT EXISTS idx_chat_turns_role     ON chat_turns(role);
CREATE INDEX IF NOT EXISTS idx_chat_turns_date     ON chat_turns(session_date_iso);
CREATE INDEX IF NOT EXISTS idx_sessions_date_iso   ON sessions(session_date_iso);

CREATE VIEW IF NOT EXISTS rounds AS
SELECT
    u.session_idx      AS session_idx,
    u.turn_idx         AS round_idx,
    u.session_date_iso AS session_date_iso,
    u.content          AS user_msg,
    a.content          AS assistant_msg
FROM chat_turns u
JOIN chat_turns a
  ON a.session_idx = u.session_idx AND a.turn_idx = u.turn_idx + 1
WHERE u.role = 'user' AND a.role = 'assistant';

-- FTS5 keyword index over chat_turns.content. `external content` mode:
-- the FTS table stores its own tokenized index but reads content via
-- the rowid back-pointer to chat_turns, so no row duplication.
-- Porter stemmer collapses inflections ("running" matches "run"); the
-- unicode61 tokenizer handles punctuation + casefolding. This replaces
-- `WHERE content LIKE '%X%'` full-table scans (O(rows)) with FTS index
-- lookups (O(matching docs)) — critical on M-split / BEAM where the
-- haystack grows 10-100x larger than S-split.
CREATE VIRTUAL TABLE IF NOT EXISTS chat_turns_fts
USING fts5(
    content,
    content='chat_turns',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Keep the FTS index in lockstep with chat_turns. chat_turns is
-- append-only at runtime (ingestor never UPDATEs / DELETEs rows), so
-- an AFTER INSERT trigger is sufficient. Delete/update triggers are
-- included for offline ``rebuild-w`` correctness.
CREATE TRIGGER IF NOT EXISTS chat_turns_fts_ai AFTER INSERT ON chat_turns
BEGIN
    INSERT INTO chat_turns_fts(rowid, content)
    VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chat_turns_fts_ad AFTER DELETE ON chat_turns
BEGIN
    INSERT INTO chat_turns_fts(chat_turns_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chat_turns_fts_au AFTER UPDATE ON chat_turns
BEGIN
    INSERT INTO chat_turns_fts(chat_turns_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
    INSERT INTO chat_turns_fts(rowid, content)
    VALUES (new.rowid, new.content);
END;
"""


def ensure_schema(ms: Memoryspace) -> None:
    """Apply the LongMemEval schema to a fresh memoryspace."""
    ms.sqlite.executescript(_LME_SCHEMA_SQL)
    ms.sqlite.commit()


# ---------------------------------------------------------------------------
# Regex extractors (relocated from agents/agent.py — they belong with the
# ingest module, not the agent class)
# ---------------------------------------------------------------------------

_PREF_POS_RE = re.compile(
    r"\bI\s+(?:like|love|prefer|enjoy|want|need|am\s+into|am\s+a\s+fan\s+of)\s+"
    r"(?P<target>[^\.\!\?\n]{3,100})",
    re.IGNORECASE,
)
_PREF_NEG_RE = re.compile(
    r"\bI\s+(?:dislike|hate|don'?t\s+(?:like|want|enjoy|prefer|care\s+for)|"
    r"can'?t\s+stand|am\s+not\s+(?:into|a\s+fan\s+of))\s+"
    r"(?P<target>[^\.\!\?\n]{3,100})",
    re.IGNORECASE,
)

_PREF_ACQUIRED_RE = re.compile(
    r"\bI\s+(?:just|recently|finally|already|now)?\s*"
    r"(?:bought|got|acquired|purchased|received|picked\s+up|"
    r"signed\s+up\s+for|subscribed\s+to|installed|downloaded|"
    r"started\s+using|switched\s+to|adopted)\s+"
    r"(?P<target>[^\.\!\?\n]{3,100})",
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    rf"\b("
    rf"\d{{4}}-\d{{2}}-\d{{2}}"
    rf"|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}"
    rf"|(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})(?:,?\s+\d{{4}})?"
    rf"|yesterday|today|tomorrow"
    rf"|last\s+(?:week|month|year|{_DOW})"
    rf"|next\s+(?:week|month|year|{_DOW})"
    rf"|this\s+(?:week|month|year)"
    rf"|\d{{1,3}}\s+(?:days?|weeks?|months?|years?)\s+ago"
    rf"|in\s+\d{{1,3}}\s+(?:days?|weeks?|months?|years?)"
    rf"|\d{{1,3}}\s+(?:days?|weeks?|months?|years?)\s+(?:from\s+now|later)"
    rf")\b",
    re.IGNORECASE,
)

_ADV = r"(?:just|recently|finally|already|also|now)?\s*"

_FACT_PATTERNS: tuple[tuple[str, "re.Pattern"], ...] = (
    ("possession", re.compile(
        rf"\bI\s+{_ADV}(?:have|own|bought|got|purchased|acquired|received|picked\s+up)\s+"
        r"(?P<target>[^\.\!\?\n]{3,100})",
        re.IGNORECASE,
    )),
    ("plan", re.compile(
        r"\bI(?:'?m|\s+am)?\s+(?:planning\s+to|going\s+to|gonna|will|plan\s+to|"
        r"intend\s+to|aim\s+to|hope\s+to|about\s+to)\s+"
        r"(?P<target>[^\.\!\?\n]{3,100})",
        re.IGNORECASE,
    )),
    ("attribute", re.compile(
        r"\bI"
        r"(?:"
        r"(?:'?m|\s+am)\s+(?:a|an)\s+(?P<target_a>[^\.\!\?\n]{3,100})"
        r"|"
        r"(?:'?m|\s+am)?\s+"
        r"(?:currently|now)?\s*"
        r"(?:work(?:ing)?\s+as|employed\s+as|live(?:s|d)?\s+in|living\s+in|"
        r"from|based\s+in|originally\s+from)\s+"
        r"(?P<target_b>[^\.\!\?\n]{3,100})"
        r")",
        re.IGNORECASE,
    )),
    ("change", re.compile(
        r"\bI\s+(?:used\s+to|no\s+longer|just\s+started|recently\s+started|"
        r"recently\s+stopped|stopped|switched\s+to)\s+"
        r"(?P<target>[^\.\!\?\n]{3,100})",
        re.IGNORECASE,
    )),
    ("current_state", re.compile(
        r"\bI(?:'?m|\s+am)\s+(?:currently|now|right\s+now|these\s+days)\s+"
        r"(?P<target>[^\.\!\?\n]{3,100})",
        re.IGNORECASE,
    )),
)


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z\"'(])", text)
    return [p.strip() for p in parts if p.strip()]


def _extract_preferences(text: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for sent in _split_sentences(text):
        for m in _PREF_POS_RE.finditer(sent):
            target = m.group("target").strip().rstrip(",;:")
            if target:
                out.append(("like", target[:100], sent[:300]))
        for m in _PREF_NEG_RE.finditer(sent):
            target = m.group("target").strip().rstrip(",;:")
            if target:
                out.append(("dislike", target[:100], sent[:300]))
    return out


def _extract_acquired_preferences(text: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for sent in _split_sentences(text):
        for m in _PREF_ACQUIRED_RE.finditer(sent):
            target = m.group("target").strip().rstrip(",;:")
            if target:
                out.append(("acquired", target[:100], sent[:300]))
    return out


def _extract_dates(
    text: str,
    *,
    anchor_iso: str | None,
    fallback_year: int | None,
) -> list[tuple[str, str | None, int | None, str]]:
    out: list[tuple[str, str | None, int | None, str]] = []
    for sent in _split_sentences(text):
        for m in _DATE_RE.finditer(sent):
            date_text = m.group(1)
            iso, offset = _resolve_date_text(
                date_text, anchor_iso=anchor_iso, fallback_year=fallback_year,
            )
            out.append((date_text[:80], iso, offset, sent[:300]))
    return out


def _extract_facts(text: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for sent in _split_sentences(text):
        for fact_type, pat in _FACT_PATTERNS:
            for m in pat.finditer(sent):
                gd = m.groupdict()
                target = gd.get("target") or gd.get("target_a") or gd.get("target_b")
                if not target:
                    continue
                target = target.strip().rstrip(",;:")
                if not target:
                    continue
                out.append((fact_type, target[:120], sent[:300]))
    return out


# ---------------------------------------------------------------------------
# LMEIngestor — groups consecutive chat_turn entries per session, then
# writes per-session aggregates (sessions row, per-session vector entry).
# ---------------------------------------------------------------------------


class LMEIngestor(Ingestor):
    """E → W for LongMemEval (and BEAM via :class:`BeamIngestor`).

    Consumes ``kind="chat_turn"`` entries. Each entry carries
    ``metadata = {session_id, session_date, session_date_iso, turn_idx}``;
    the ingestor groups them by ``session_idx`` and emits:

      - one ``chat_turns`` row per turn,
      - one ``sessions`` row per session,
      - per-turn + per-session vector entries,
      - and (only when ``extract_typed=True``) per-turn
        ``user_preferences`` / ``facts`` / ``event_dates`` rows plus
        a ``sessions.session_text`` full-transcript column.

    Other entry kinds (``code_exec``, ``rlm_call``, ``briefing_note``,
    etc.) are silently ignored — they don't carry chat data.

    ``extract_typed`` controls the typed-value extraction layer
    (preferences / facts / event_dates / session_text). Default is
    ``False`` — the v4 500-QA audit showed the LME agent issued <2%
    of queries against those tables, so they're not worth populating
    on the LME path. BEAM's :class:`BeamIngestor` flips it back to
    ``True`` because its prompt still references those tables and
    its schema retains them.
    """

    extract_typed: bool = False

    def __init__(self, ms: Memoryspace) -> None:
        self.ms = ms

    def consume(self, entries: Iterable[LogEntry]) -> None:
        # Group chat_turn entries by turn_idx (= chat-session-idx, since
        # the LME chat-history loader emits one entry per chat-session
        # message with that chat session's index as the LogEntry's
        # turn_idx). Preserve original order within each session.
        by_session: dict[int, list[LogEntry]] = {}
        for entry in entries:
            kind = (entry.metadata or {}).get("kind")
            if kind != "chat_turn":
                continue
            by_session.setdefault(entry.turn_idx, []).append(entry)

        if not by_session:
            return

        for session_idx, turns in by_session.items():
            self._ingest_session_turns(session_idx, turns)
        self.ms.sqlite.commit()

    def _ingest_session_turns(
        self, session_idx: int, turns: list[LogEntry],
    ) -> None:
        cur = self.ms.sqlite.cursor()
        meta0 = turns[0].metadata or {}
        raw_date = str(meta0.get("session_date") or "")
        ts_iso = (
            str(meta0.get("session_date_iso"))
            if meta0.get("session_date_iso")
            else (session_date_to_iso(raw_date) if raw_date else None)
        )
        fallback_year: int | None = None
        if ts_iso:
            try:
                fallback_year = int(ts_iso[:4])
            except (ValueError, IndexError):
                fallback_year = None
        session_id_str = str(meta0.get("session_id") or "")
        session_date_iso_str = ts_iso[:10] if ts_iso else None

        all_session_facts: list[str] = []
        for entry in turns:
            meta = entry.metadata or {}
            turn_idx = int(meta.get("turn_idx", 0))
            role = str(entry.role)
            content = str(entry.content)
            cur.execute(
                "INSERT OR IGNORE INTO chat_turns "
                "(session_idx, turn_idx, role, content, session_id, "
                " session_date_iso, session_ts_iso) VALUES (?,?,?,?,?,?,?)",
                (session_idx, turn_idx, role, content,
                 session_id_str, session_date_iso_str, ts_iso),
            )
            if not content:
                continue
            turn_facts: list[str] = []
            if self.extract_typed:
                if role == "user":
                    pref_rows = list(_extract_preferences(content))
                    pref_rows.extend(_extract_acquired_preferences(content))
                    for polarity, target, sentence in pref_rows:
                        try:
                            cur.execute(
                                "INSERT OR IGNORE INTO user_preferences "
                                "(session_idx, turn_idx, polarity, target, sentence) "
                                "VALUES (?,?,?,?,?)",
                                (session_idx, turn_idx, polarity, target, sentence),
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        turn_facts.append(f"{polarity}={target}")
                    for fact_type, fact_text, sentence in _extract_facts(content):
                        try:
                            cur.execute(
                                "INSERT OR IGNORE INTO facts "
                                "(session_idx, turn_idx, fact_type, fact_text, sentence) "
                                "VALUES (?,?,?,?,?)",
                                (session_idx, turn_idx, fact_type, fact_text, sentence),
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        turn_facts.append(f"{fact_type}={fact_text}")
                for date_text, date_iso, offset, sentence in _extract_dates(
                    content, anchor_iso=session_date_iso_str,
                    fallback_year=fallback_year,
                ):
                    try:
                        cur.execute(
                            "INSERT INTO event_dates "
                            "(session_idx, turn_idx, date_text, date_iso, "
                            " relative_offset_days, sentence) "
                            "VALUES (?,?,?,?,?,?)",
                            (session_idx, turn_idx, date_text, date_iso, offset, sentence),
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    turn_facts.append(f"date={date_iso or date_text}")
                all_session_facts.extend(turn_facts)
            facts_blob = ""
            if turn_facts:
                facts_blob = "\n[facts] " + "; ".join(turn_facts)
            try:
                self.ms.vectors.add(
                    f"sess{session_idx}_turn{turn_idx}_{role}", content + facts_blob,
                )
            except Exception:  # noqa: BLE001
                pass

        session_text = " ".join(
            f"{str(e.role)}: {str(e.content)}" for e in turns
        )

        if session_text:
            try:
                if self.extract_typed:
                    # BEAM path: schema has the session_text column.
                    cur.execute(
                        "INSERT OR REPLACE INTO sessions "
                        "(session_idx, session_id, session_date_iso, session_ts_iso, "
                        " session_text) VALUES (?,?,?,?,?)",
                        (session_idx, session_id_str, session_date_iso_str, ts_iso,
                         session_text[:20000]),
                    )
                else:
                    # LME path: trimmed schema, no session_text column.
                    cur.execute(
                        "INSERT OR REPLACE INTO sessions "
                        "(session_idx, session_id, session_date_iso, session_ts_iso) "
                        "VALUES (?,?,?,?)",
                        (session_idx, session_id_str, session_date_iso_str, ts_iso),
                    )
            except Exception:  # noqa: BLE001
                pass

            session_facts_blob = ""
            if all_session_facts:
                seen: set[str] = set()
                deduped = [
                    f for f in all_session_facts
                    if not (f in seen or seen.add(f))
                ]
                session_facts_blob = "\n[session facts] " + "; ".join(deduped)
            date_tag = (
                f"\n[session_date] {ts_iso or raw_date}"
                if (ts_iso or raw_date) else ""
            )
            try:
                self.ms.vectors.add(
                    f"sess{session_idx}_session",
                    session_text[:4000] + date_tag + session_facts_blob,
                )
            except Exception:  # noqa: BLE001
                pass


__all__ = ["LMEIngestor", "ensure_schema"]
