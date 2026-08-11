from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from scroll_context._runtime.types import LogEntry


# v2 added the `prose` / `code` columns and a two-column FTS5 index (code split
# out of the BM25-ranked prose). Pre-v2 DBs are migrated in place on open.
_SCHEMA_VERSION = "2"
_BUSY_TIMEOUT_MS = 5000


# Columns of conversation_history, in INSERT order (minus the autoincrement seq).
# `prose`/`code` are the code-split derivatives of `content` (see _split_code);
# they back the FTS5 index, while `content` keeps the full raw turn for recall.
_INSERT_COLUMNS = (
    "session_id", "run_id", "task_id", "step_index", "msg_index",
    "kind", "role", "name", "content", "prose", "code",
    "tool_call_id", "tool_input", "tool_state", "headline", "blocks", "metadata",
    "created_at",
)


# A fenced code block: ``` or ~~~ opener (with optional info string), body, and a
# matching closer. DOTALL so a block spans newlines; non-greedy so adjacent
# blocks don't merge. Indented/HTML code is left in prose (rare in chat turns).
_FENCE_RE = re.compile(r"(?P<f>```|~~~)(?P<lang>[^\n`]*)\n(?P<body>.*?)(?P=f)", re.DOTALL)


def _split_code(content: str | None) -> tuple[str, str | None]:
    """Split a turn into (prose, code): prose with code fences elided, code joined.

    BM25 over a column that mixes prose and code lets code tokens (identifiers,
    keywords, ``[:6274]``-style slices) win rank without carrying answer-bearing
    meaning, and a leading code dump eats the whole search preview before any
    reasoning text. So each fenced block is pulled into ``code`` and replaced in
    ``prose`` by a ``‹code:lang›`` marker — the two are indexed as separate FTS
    columns so prose ranks cleanly and code is retrievable on its own track. The
    full raw turn is still kept verbatim in ``content`` for faithful recall.

    Returns ``("", None)`` for empty input; ``code`` is ``None`` when the turn
    has no fenced blocks.
    """
    if not content:
        return "", None
    codes: list[str] = []

    def _repl(m: re.Match) -> str:
        codes.append(m.group("body"))
        lang = (m.group("lang") or "").strip()
        return f"‹code:{lang}›" if lang else "‹code›"

    prose = _FENCE_RE.sub(_repl, content)
    return prose, ("\n\n".join(codes) if codes else None)


class HistoryStore:
    """Durable, file-backed conversation history shared across sessions.

    Owns the *read-write* connection to a single SQLite file holding the
    predefined ``conversation_history`` table. Every event the agent appends
    is write-through-persisted here with full structure (`blocks`, tool args,
    state) so a later session can retrieve it. The model reaches the same file
    *read-only* through its ``MemorySpace`` (ATTACHed ``hist`` schema), so this
    writer and those readers coexist under WAL.

    Lifetime spans many sessions: the file is never dropped or deleted —
    ``close()`` only closes this connection.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.quarantined_to: Path | None = None
        try:
            self._open_and_init()
        except sqlite3.DatabaseError as exc:
            # A corrupt / unreadable DB (truncated file, stale WAL trio, bad
            # page) would otherwise raise on open and crash EVERY task at
            # startup with no trajectory. Quarantine the bad file (and its
            # -wal/-shm sidecars) and recreate fresh, degrading "broken memory"
            # to "lost history" instead of a dead run.
            self._quarantine(exc)
            self._open_and_init()

    def _open_and_init(self) -> None:
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        # Probe for corruption that only surfaces on read (the exact failure
        # mode that produced 'disk I/O error' on a stale WAL trio).
        row = self._conn.execute("PRAGMA quick_check").fetchone()
        if not row or row[0] != "ok":
            raise sqlite3.DatabaseError(f"quick_check failed: {row[0] if row else None}")
        self._init_schema()

    def _quarantine(self, exc: Exception) -> None:
        """Move the unreadable DB + its -wal/-shm aside with a timestamp."""
        try:
            self._conn.close()
        except (AttributeError, sqlite3.Error):
            pass
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(self._path) + suffix)
            if not src.exists():
                continue
            dest = Path(f"{self._path}.corrupt-{ts}{suffix}")
            try:
                src.rename(dest)
                if suffix == "":
                    self.quarantined_to = dest
            except OSError:
                try:
                    src.unlink()  # last resort so a fresh DB can be created
                except OSError:
                    pass
        print(
            f"[HistoryStore] {self._path} was unreadable ({exc}); quarantined to "
            f"{self.quarantined_to} and recreated a fresh store.",
            file=sys.stderr,
        )

    @property
    def path(self) -> Path:
        return self._path

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_history (
                    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT NOT NULL,
                    run_id       TEXT,
                    task_id      TEXT,
                    step_index   INTEGER,
                    msg_index    INTEGER,
                    kind         TEXT NOT NULL,
                    role         TEXT,
                    name         TEXT,
                    content      TEXT,
                    prose        TEXT,
                    code         TEXT,
                    tool_call_id TEXT,
                    tool_input   TEXT,
                    tool_state   TEXT,
                    headline     TEXT,
                    blocks       TEXT,
                    metadata     TEXT,
                    created_at   TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS ch_session ON conversation_history(session_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS ch_task ON conversation_history(task_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS ch_kind ON conversation_history(kind)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS _meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            # Bring a pre-v2 table up to schema (add prose/code, backfill them)
            # BEFORE building the FTS index — 'rebuild' reads those columns.
            self._ensure_columns()
            self._backfill_split()
            self._conn.execute(
                "INSERT INTO _meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_SCHEMA_VERSION,),
            )
            self._init_fts()

    def _ensure_columns(self) -> None:
        """Add later-version columns to an older table (no-op if present).

        ``prose``/``code`` arrived in v2; ``headline`` predates the FTS v3 index
        but a truly legacy DB may lack the column, and the external-content FTS
        references it — so ensure all three before ``_init_fts``.
        """
        cols = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(conversation_history)")
        }
        for col in ("prose", "code", "headline"):
            if col not in cols:
                self._conn.execute(
                    f"ALTER TABLE conversation_history ADD COLUMN {col} TEXT"
                )

    def _backfill_split(self) -> None:
        """Derive prose/code for rows that predate the split (``prose IS NULL``).

        A fresh insert always sets ``prose`` (to ``""`` at minimum), so only
        legacy rows match — this is a one-time pass on first open after upgrade.
        """
        rows = self._conn.execute(
            "SELECT seq, content FROM conversation_history WHERE prose IS NULL"
        ).fetchall()
        for r in rows:
            prose, code = _split_code(r["content"])
            self._conn.execute(
                "UPDATE conversation_history SET prose = ?, code = ? WHERE seq = ?",
                (prose, code, r["seq"]),
            )

    def _init_fts(self) -> None:
        """Create the three-column FTS5 index over `prose`/`code`/`headline`.

        External-content FTS5 (``content='conversation_history'``) indexes
        without duplicating the text; it's kept in sync by ``append``. Splitting
        ``prose`` and ``code`` into separate columns lets the reader rank BM25 on
        prose alone (code tokens no longer inflate rank) and retrieve code on its
        own track. ``headline`` (v3) indexes the model-written turn/session
        summaries — a second, summary-register vocabulary layer that can match
        question-altitude phrasing that never occurs verbatim in any turn
        (``MemorySpace.search`` fuses headline hits into its results). The
        `porter` tokenizer stems ("index" matches "indexing"). A pre-v3 index
        (single-column v1 or prose/code v2) is dropped and rebuilt from the
        content table on first open; if this build of SQLite lacks FTS5, we
        degrade silently — ``MemorySpace.search`` falls back to LIKE.
        """
        try:
            existed = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='conversation_history_fts'"
            ).fetchone()
            if existed:
                cols = {
                    r["name"]
                    for r in self._conn.execute(
                        "PRAGMA table_info(conversation_history_fts)"
                    )
                }
                if not {"prose", "code", "headline"} <= cols:
                    # Stale v1 (single `content`) or v2 (prose/code) index —
                    # drop and rebuild with the v3 schema.
                    self._conn.execute("DROP TABLE conversation_history_fts")
                    existed = None
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS conversation_history_fts "
                "USING fts5(prose, code, headline, content='conversation_history', "
                "content_rowid='seq', tokenize='porter')"
            )
            if not existed:
                # Back-fill any rows that predate the index.
                self._conn.execute(
                    "INSERT INTO conversation_history_fts(conversation_history_fts) "
                    "VALUES('rebuild')"
                )
            self._fts = True
        except sqlite3.OperationalError:
            self._fts = False

    # --- write path ----------------------------------------------------

    def append(
        self,
        *,
        session_id: str,
        run_id: str | None,
        task_id: str | None,
        entry: LogEntry,
    ) -> int:
        """Write-through one event. Returns the assigned ``seq`` (watermark)."""
        prose, code = _split_code(entry.content)
        row = (
            session_id,
            run_id,
            task_id,
            entry.step_index,
            entry.msg_index,
            entry.kind,
            entry.role,
            entry.name,
            entry.content,
            prose,
            code,
            entry.tool_call_id,
            _to_json(entry.tool_input),
            entry.tool_state,
            entry.headline,
            _to_json(entry.blocks),
            _to_json(entry.metadata or None),
            entry.created_at or datetime.now(timezone.utc).isoformat(),
        )
        placeholders = ", ".join("?" for _ in _INSERT_COLUMNS)
        with self._conn:
            cur = self._conn.execute(
                f"INSERT INTO conversation_history ({', '.join(_INSERT_COLUMNS)}) "
                f"VALUES ({placeholders})",
                row,
            )
            seq = int(cur.lastrowid)
            if self._fts:
                # Keep the external-content FTS5 index in sync (prose + code +
                # headline).
                self._conn.execute(
                    "INSERT INTO conversation_history_fts(rowid, prose, code, headline) "
                    "VALUES (?, ?, ?, ?)",
                    (seq, prose or "", code or "", entry.headline or ""),
                )
            return seq

    # --- read path (backs the `log` handle, scoped to one session) -----

    def query_log(
        self,
        session_id: str,
        *,
        kind: str | None = None,
        name: str | None = None,
        role: str | None = None,
        tail: int | None = None,
    ) -> list[LogEntry]:
        clauses = ["session_id = ?"]
        params: list = [session_id]
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if name is not None:
            clauses.append("name = ?")
            params.append(name)
        if role is not None:
            clauses.append("role = ?")
            params.append(role)
        where = " AND ".join(clauses)
        if tail is not None:
            if tail <= 0:
                return []
            sql = (
                f"SELECT * FROM conversation_history WHERE {where} "
                f"ORDER BY seq DESC LIMIT ?"
            )
            params.append(tail)
            rows = list(self._conn.execute(sql, params))
            rows.reverse()  # back to chronological order
        else:
            sql = f"SELECT * FROM conversation_history WHERE {where} ORDER BY seq"
            rows = list(self._conn.execute(sql, params))
        return [_row_to_entry(r) for r in rows]

    def count(self, session_id: str) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) AS n FROM conversation_history WHERE session_id = ?",
            (session_id,),
        )
        return int(cur.fetchone()["n"])

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __repr__(self) -> str:
        return f"<HistoryStore path={self._path}>"


def _to_json(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _from_json(text):
    if text is None:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _row_to_entry(row: sqlite3.Row) -> LogEntry:
    return LogEntry(
        kind=row["kind"],
        role=row["role"],
        name=row["name"],
        content=row["content"],
        metadata=_from_json(row["metadata"]) or {},
        step_index=row["step_index"],
        msg_index=row["msg_index"],
        tool_call_id=row["tool_call_id"],
        tool_input=_from_json(row["tool_input"]),
        tool_state=row["tool_state"],
        headline=row["headline"],
        blocks=_from_json(row["blocks"]),
        created_at=row["created_at"],
    )
