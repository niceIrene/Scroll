"""Multi-format data memoryspace that an LLM agent can manage autonomously.

Backends: in-memory SQLite, JSON key-value store, sparse vector store, file store.
The agent decides what to create and how to organise its data.

API style: every read/write method returns a **native Python value**
(``list[dict]``, ``int``, ``None``, etc.) and raises on missing keys
or SQL errors — the REPL surfaces tracebacks back to the agent so
``try/except sqlite3.Error`` works as expected. Error returns as
strings (``"SQL_ERROR: ..."``, ``"NOT_FOUND: ..."``) are not used.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Scroll.tools._embed import _embed, _cos


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

@dataclass
class VectorEntry:
    key: str
    text: str
    vec: dict[str, float]


class VectorStore:
    def __init__(self) -> None:
        self.entries: list[VectorEntry] = []

    def add(self, key: str, text: str) -> None:
        self.entries.append(VectorEntry(key=key, text=text, vec=_embed(text)))

    def query(self, query_text: str, top_k: int = 5) -> list[tuple[str, str, float]]:
        if not self.entries:
            return []
        q = _embed(query_text)
        scored = [(e.key, e.text, _cos(q, e.vec)) for e in self.entries]
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(k, t, round(s, 4)) for k, t, s in scored[:top_k]]

    def count(self) -> int:
        return len(self.entries)

    def delete(self, key: str) -> int:
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.key != key]
        return before - len(self.entries)


# ---------------------------------------------------------------------------
# Memoryspace
# ---------------------------------------------------------------------------

class Memoryspace:
    """Multi-format data memoryspace backing `f = query(·, W)`.

    Design note: this is `W` in design/2026-04-24.md §2 — the
    intermediate structure that generalizes `D` (fixed SQL schema)
    beyond a single relational store. Tools dispatched here are the
    agent-facing `f`; rows are populated either by the agent itself
    (Adaptive manual, row 3-2) or by an env-specific auto-builder
    (Adaptive auto, row 3-1 — see e.g. ``vending/agents/adaptive_auto.py``).
    """

    def __init__(self) -> None:
        self.sqlite = sqlite3.connect(":memory:")
        self.sqlite.row_factory = sqlite3.Row
        self.json_store: dict[str, Any] = {}
        self.vectors = VectorStore()
        self.file_store: dict[str, str] = {}

        # E → W ingest plumbing. ``attach`` wires both at agent init;
        # the catch-up watermark advances when ``_maybe_catch_up`` consumes
        # new entries. Decoupled from this class so memoryspace doesn't
        # depend on ConversationLog / Ingestor types (any duck-type with
        # ``.entries`` (list) and ``.consume(iterable)`` works).
        self._log = None  # type: ignore[assignment]
        self._ingestor = None  # type: ignore[assignment]
        self._watermark: int = 0
        self._catching_up: bool = False

    # ----- E → W catch-up -----

    def attach(self, log, ingestor) -> None:
        """Bind an event log + ingestor so reads auto-catch-up.

        After ``attach``, every public read/write method on this
        memoryspace first checks whether ``len(log.entries) > watermark``
        and, if so, calls ``ingestor.consume(log.entries[watermark:])``
        before serving the call. This enforces the invariant that any
        query against ``W`` reflects all of ``E`` up to the moment of
        the query.
        """
        self._log = log
        self._ingestor = ingestor

    def _maybe_catch_up(self) -> None:
        """Consume any unprocessed tail of E into W.

        Cheap when there's nothing new (one ``len()`` + one comparison).
        Re-entrancy is guarded so ingestor implementations may call into
        the memoryspace without recursing.
        """
        if self._catching_up or self._ingestor is None or self._log is None:
            return
        n = len(self._log.entries)
        if n <= self._watermark:
            return
        self._catching_up = True
        try:
            tail = list(self._log.entries[self._watermark:n])
            self._ingestor.consume(tail)
            self._watermark = n
        finally:
            self._catching_up = False

    def bootstrap(self, env, log) -> None:
        """Run the attached ingestor's bootstrap step.

        Called once per task by the harness (``_run_task``) via
        ``agent.bootstrap(env)`` → ``ScrollAgent.bootstrap`` → here.
        Delegates to the attached ingestor's
        :meth:`Ingestor.bootstrap` (default no-op), which is the
        single place env-specific task-wide data ingestion lives
        (LME haystack chat sessions, BEAM batches; vending is a
        no-op).

        Does NOT call ``consume`` — the appended entries are
        materialized into ``W`` on the next ``ms`` read via lazy
        ``_maybe_catch_up()``.

        No-op when no ingestor is attached.
        """
        if self._ingestor is None:
            return
        self._ingestor.bootstrap(env, log)

    # ----- SQL -----

    def sql_exec(
        self,
        statement: str,
        params: list | tuple | dict | None = None,
    ) -> list[dict] | int | None:
        """Execute a single SQL statement.

        ``params`` is an optional binding for ``?`` (sequence) or
        ``:name`` (mapping) placeholders — DB-API style. Without it,
        agents typically reach for ``db.query(sql, [v])`` habit and
        get ``TypeError`` for the extra positional arg.

        Returns:
            ``list[dict]`` for SELECT (and other row-returning queries),
            ``int`` (rowcount) for INSERT / UPDATE / DELETE,
            ``None`` for CREATE / DROP / PRAGMA / etc.
        Raises:
            ``sqlite3.Error`` on syntax / runtime failures — let it
            propagate; the REPL surfaces the traceback.
        """
        self._maybe_catch_up()
        if params is None:
            cur = self.sqlite.execute(statement)
        elif isinstance(params, dict):
            cur = self.sqlite.execute(statement, params)
        elif isinstance(params, (list, tuple)):
            cur = self.sqlite.execute(statement, tuple(params))
        else:
            raise TypeError(
                f"params must be a list/tuple (for `?` placeholders) "
                f"or a dict (for `:name` placeholders); got "
                f"{type(params).__name__}"
            )
        self.sqlite.commit()
        if cur.description is not None:
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        return cur.rowcount if cur.rowcount >= 0 else None

    # ----- JSON -----

    def json_write(self, key: str, data: Any) -> None:
        """Store ``data`` under ``key``. ``data`` must be JSON-serializable.

        Raises ``TypeError`` for non-serializable values (datetime, sets,
        custom objects, etc.) — required so checkpoint save/restore
        round-trips identically.
        """
        self._maybe_catch_up()
        # Force JSON round-trip at write time so the in-memory shape is
        # already canonical (dict/list/str/int/float/bool/None). This
        # guarantees ``to_checkpoint → from_checkpoint`` is identity.
        self.json_store[key] = json.loads(json.dumps(data))

    def json_read(self, key: str) -> Any:
        """Return the stored Python value. Raises ``KeyError`` if ``key`` is unknown."""
        self._maybe_catch_up()
        if key not in self.json_store:
            raise KeyError(f"json key not found: {key!r}")
        return self.json_store[key]

    def json_list(self) -> list[str]:
        """Return all stored JSON keys (sorted)."""
        self._maybe_catch_up()
        return sorted(self.json_store.keys())

    # ----- Vector -----

    def vector_store_add(self, key: str, text: str) -> int:
        """Add a vector entry. Returns the new total entry count."""
        self._maybe_catch_up()
        self.vectors.add(key, text)
        return self.vectors.count()

    def vector_query(self, query: str, top_k: int = 5) -> list[tuple[str, str, float]]:
        """Return ``[(key, text, similarity), ...]`` ranked by cosine similarity (newest highest first).

        Empty list if the vector store is empty. ``similarity`` is in
        ``[-1, 1]`` and is rounded to 4 decimals.
        """
        self._maybe_catch_up()
        return self.vectors.query(query, top_k=int(top_k))

    def vector_delete(self, key: str) -> int:
        """Delete entries matching ``key``. Returns the number deleted."""
        self._maybe_catch_up()
        return self.vectors.delete(key)

    # ----- File -----

    def file_write(self, name: str, content: str) -> None:
        """Store ``content`` (a string) under ``name``."""
        self._maybe_catch_up()
        self.file_store[name] = content

    def file_read(self, name: str) -> str:
        """Return the stored content. Raises ``KeyError`` if ``name`` is unknown."""
        self._maybe_catch_up()
        if name not in self.file_store:
            raise KeyError(f"file not found: {name!r}")
        return self.file_store[name]

    def file_list(self) -> dict[str, int]:
        """Return ``{name: size_in_chars, ...}`` for every stored file."""
        self._maybe_catch_up()
        return {name: len(content) for name, content in self.file_store.items()}

    # ----- Introspection -----

    def schema_inspect(self) -> dict[str, Any]:
        """Return a structured snapshot of W: tables, views, json keys, vectors, files.

        Shape::

            {
                "tables": [
                    {"name": str, "ddl": str, "rows": int, "last_row": dict | None},
                    ...
                ],
                "views":  [
                    {"name": str, "ddl": str, "rows": int},
                    ...
                ],
                "json_keys":    [str, ...],
                "vector_count": int,
                "files":        [str, ...],
            }

        Iterate over ``schema["tables"]`` / ``schema["views"]`` to walk
        the SQL layer; ``last_row`` previews the most recently inserted
        row for non-empty tables (``None`` otherwise).
        """
        self._maybe_catch_up()
        out: dict[str, Any] = {"tables": [], "views": [], "json_keys": [], "vector_count": 0, "files": []}

        cur = self.sqlite.execute(
            "SELECT name, sql, type FROM sqlite_master "
            "WHERE type IN ('table', 'view') ORDER BY type, name"
        )
        for row in cur.fetchall():
            name, ddl, kind = row["name"], row["sql"], row["type"]
            if name == "sqlite_sequence":
                continue
            count = self.sqlite.execute(
                f"SELECT COUNT(*) FROM [{name}]"
            ).fetchone()[0]
            entry: dict[str, Any] = {"name": name, "ddl": ddl, "rows": count}
            if kind == "table":
                last_row: dict | None = None
                if count > 0:
                    try:
                        last_cur = self.sqlite.execute(
                            f"SELECT * FROM [{name}] ORDER BY rowid DESC LIMIT 1"
                        )
                        cols = [d[0] for d in last_cur.description]
                        last_row = dict(zip(cols, last_cur.fetchone()))
                    except sqlite3.Error:
                        last_row = None
                entry["last_row"] = last_row
                out["tables"].append(entry)
            else:
                out["views"].append(entry)

        out["json_keys"] = sorted(self.json_store.keys())
        out["vector_count"] = self.vectors.count()
        out["files"] = sorted(self.file_store.keys())
        return out

    # ----- Checkpoint -----

    def to_checkpoint(self) -> dict:
        sql_dump = "\n".join(self.sqlite.iterdump())
        return {
            "sql_dump": sql_dump,
            "json_store": self.json_store,
            "file_store": dict(self.file_store),
            "vectors": [{"key": e.key, "text": e.text} for e in self.vectors.entries],
            "watermark": self._watermark,
        }

    def from_checkpoint(self, data: dict) -> None:
        # Restore SQLite
        self.sqlite.close()
        self.sqlite = sqlite3.connect(":memory:")
        self.sqlite.row_factory = sqlite3.Row
        self.sqlite.executescript(data["sql_dump"])

        self.json_store = data.get("json_store", {})
        self.file_store = data.get("file_store", {})

        # Restore vectors (re-embed text)
        self.vectors = VectorStore()
        for v in data.get("vectors", []):
            self.vectors.add(v["key"], v["text"])

        # Restore the E → W watermark. Any tail of E past this offset
        # (from a crashed run) will be consumed on the next ms read.
        self._watermark = int(data.get("watermark", 0))

    # ----- Helpers for heuristic router -----

    def get_sql_tables(self) -> list[str]:
        cur = self.sqlite.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row["name"] for row in cur.fetchall()]

    def get_table_columns(self, table: str) -> list[str]:
        try:
            cur = self.sqlite.execute(f"PRAGMA table_info([{table}])")
            return [row["name"] for row in cur.fetchall()]
        except sqlite3.Error:
            return []

    # ----- Dump memoryspace to disk -----

    def dump_memoryspace(self, output_dir: str | Path) -> None:
        """Save the entire memoryspace state to *output_dir*/memoryspace/."""
        ms = Path(output_dir) / "memoryspace"
        ms.mkdir(parents=True, exist_ok=True)

        # 1. SQL → dump each table as a JSON file + save schema
        sql_dir = ms / "sql"
        sql_dir.mkdir(exist_ok=True)
        schema_lines: list[str] = []
        cur = self.sqlite.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        for row in cur.fetchall():
            table = row["name"]
            ddl = row["sql"]
            schema_lines.append(f"-- {table}\n{ddl};\n")
            data_cur = self.sqlite.execute(f"SELECT * FROM [{table}]")
            cols = [d[0] for d in data_cur.description] if data_cur.description else []
            rows = [dict(zip(cols, r)) for r in data_cur.fetchall()]
            (sql_dir / f"{table}.json").write_text(
                json.dumps(rows, indent=2, default=str), encoding="utf-8"
            )
        (sql_dir / "_schema.sql").write_text("\n".join(schema_lines), encoding="utf-8")

        # 2. JSON store
        if self.json_store:
            json_dir = ms / "json"
            json_dir.mkdir(exist_ok=True)
            for key, value in self.json_store.items():
                safe_name = key.replace("/", "_").replace("\\", "_")
                (json_dir / f"{safe_name}.json").write_text(
                    json.dumps(value, indent=2, default=str), encoding="utf-8"
                )

        # 3. Vector store
        if self.vectors.entries:
            vec_dir = ms / "vectors"
            vec_dir.mkdir(exist_ok=True)
            entries = [{"key": e.key, "text": e.text} for e in self.vectors.entries]
            (vec_dir / "entries.json").write_text(
                json.dumps(entries, indent=2, default=str), encoding="utf-8"
            )

        # 4. File store
        if self.file_store:
            file_dir = ms / "files"
            file_dir.mkdir(exist_ok=True)
            for name, content in self.file_store.items():
                safe_name = name.replace("/", "_").replace("\\", "_")
                (file_dir / safe_name).write_text(content, encoding="utf-8")
