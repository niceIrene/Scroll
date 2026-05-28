"""BEAM ``E → W`` ingestor.

BEAM and LongMemEval are both chat-memory benchmarks with identical
ingest semantics: chat turns land in ``E`` as ``kind="chat_turn"``
entries with ``metadata = {session_id, session_date, session_date_iso,
turn_idx}``; the ingestor consume/group/extract pipeline is shared.

NOTE: this is the last remaining BEAM→LME cross-package import.
The pipeline (``LMEIngestor``) is shared, but each env keeps its own
schema (BEAM ships the original 5-table layout + typed extraction;
LME dropped 3 tables for the v4 simplification). Extracting the
pipeline into ``Scroll.tools.chat_memory`` and giving each env a
thin subclass with only its schema would close the loop — deferred
on the bet that the schemas may diverge further (FTS5 variants,
typed-value tables) and a base-class refactor would be premature.
"""

from __future__ import annotations

from Scroll.tools.memoryspace import Memoryspace

# Pipeline shared across chat-memory benchmarks; schema below is
# BEAM-specific. See module docstring for the deferred split.
from Scroll.benchmarks.longmemeval.ingestor import LMEIngestor


# Schema is byte-for-byte identical to LME — same tables, same indices,
# same ``rounds`` view. Kept as its own copy here (rather than imported
# from LME) so per-env divergence stays local.
_BEAM_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS chat_turns (
    session_idx      INTEGER NOT NULL,
    turn_idx         INTEGER NOT NULL,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    session_id       TEXT,
    session_date_iso TEXT,                    -- ISO "YYYY-MM-DD"
    session_ts_iso   TEXT,                    -- full ISO ts, sortable
    PRIMARY KEY (session_idx, turn_idx)
);

CREATE TABLE IF NOT EXISTS user_preferences (
    session_idx INTEGER NOT NULL,
    turn_idx    INTEGER NOT NULL,
    polarity    TEXT NOT NULL,                -- 'like' | 'dislike' | 'acquired'
    target      TEXT NOT NULL,
    sentence    TEXT NOT NULL,
    PRIMARY KEY (session_idx, turn_idx, polarity, target)
);

CREATE TABLE IF NOT EXISTS event_dates (
    session_idx          INTEGER NOT NULL,
    turn_idx             INTEGER NOT NULL,
    date_text            TEXT NOT NULL,
    date_iso             TEXT,
    relative_offset_days INTEGER,
    sentence             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    session_idx INTEGER NOT NULL,
    turn_idx    INTEGER NOT NULL,
    fact_type   TEXT NOT NULL,
    fact_text   TEXT NOT NULL,
    sentence    TEXT NOT NULL,
    PRIMARY KEY (session_idx, turn_idx, fact_type, fact_text)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_idx      INTEGER PRIMARY KEY,
    session_id       TEXT,
    session_date_iso TEXT,
    session_ts_iso   TEXT,
    session_text     TEXT
);

CREATE INDEX IF NOT EXISTS idx_chat_turns_role          ON chat_turns(role);
CREATE INDEX IF NOT EXISTS idx_chat_turns_date          ON chat_turns(session_date_iso);
CREATE INDEX IF NOT EXISTS idx_user_preferences_target  ON user_preferences(target);
CREATE INDEX IF NOT EXISTS idx_user_preferences_polarity ON user_preferences(polarity);
CREATE INDEX IF NOT EXISTS idx_event_dates_iso          ON event_dates(date_iso);
CREATE INDEX IF NOT EXISTS idx_event_dates_offset       ON event_dates(relative_offset_days);
CREATE INDEX IF NOT EXISTS idx_facts_type               ON facts(fact_type);
CREATE INDEX IF NOT EXISTS idx_sessions_date_iso        ON sessions(session_date_iso);

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
"""


def ensure_schema(ms: Memoryspace) -> None:
    ms.sqlite.executescript(_BEAM_SCHEMA_SQL)
    ms.sqlite.commit()


class BeamIngestor(LMEIngestor):
    """env → E → W for BEAM.

    Inherits from :class:`LMEIngestor` — the consume/group/extract
    pipeline is identical, but BEAM keeps the typed-extraction layer
    on (preferences / facts / event_dates / session_text) because
    BEAM's prompt + schema still reference those tables. LME's
    default is ``extract_typed=False``.

    Overrides :meth:`bootstrap` to walk ``env.item.batches`` instead
    of LME's ``env.item.haystack_sessions``.
    """

    extract_typed = True

    def bootstrap(self, env, log) -> None:
        """Bulk-stage every batch's chat turns into ``E``.

        Iterates ``env.item.batches`` in chronological order; for each
        batch, calls :meth:`BeamEnv.begin_turn` to stage it on the
        env and :func:`write_chat_turn_entries` to mirror its turns
        into ``log`` as ``kind="chat_turn"`` entries. No-op under
        the legacy ``env.cfg.agent_during_ingestion=True`` path.
        """
        if getattr(env.cfg, "agent_during_ingestion", False):
            return
        from Scroll.tools.chat_memory import write_chat_turn_entries
        for idx in range(env.item.num_batches):
            env.begin_turn(idx)
            write_chat_turn_entries(log, env, idx + 1)
        env._current_batch_turns = None
        env._current_batch_idx = None
        env._current_time_anchor = None
        env._today_logs = []


__all__ = ["BeamIngestor", "ensure_schema"]
