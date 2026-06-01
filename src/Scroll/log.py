"""The append-only event log E.

:class:`ConversationLog` is the writer + persistence backend for E: an
in-memory list of :class:`Scroll.core.LogEntry` records mirrored to
``conversation_log.jsonl`` (flush + fsync per append, so a crash
loses at most one in-flight LLM call).

The agent reads from this through :class:`Scroll.tools._log_handle.LogHandle`
— never writes to it directly. Harness code (CodeActAgent, run_turn,
make_dspy_rlm) holds the raw log for writes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from Scroll.core import LogEntry


class ConversationLog:
    """Append-only log of LogEntry records — `E` in design/2026-04-24.md §1.

    The log is the agent's memory made durable: every chat turn that
    lands in the agent's InMemoryMemory also lands here, including
    both action tool calls and retrieval (e.g. ``rlm``) tool
    calls plus their results. ``rlm`` reads from this object;
    ``W`` is the derived memoryspace (see ``tools/memoryspace.py``).

    When `jsonl_path` is set, `append` writes the entry to disk
    immediately (with a flush + fsync) so a crash loses at most the
    in-flight LLM call.
    """

    def __init__(self, jsonl_path: str | Path | None = None) -> None:
        self.entries: list[LogEntry] = []
        self._jsonl_path: Path | None = Path(jsonl_path) if jsonl_path else None
        self._fh = None
        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._jsonl_path, "a", encoding="utf-8")

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, entry: LogEntry) -> None:
        self.entries.append(entry)
        self._flush_entry(entry)

    def extend(self, entries: Iterable[LogEntry]) -> None:
        for entry in entries:
            self.append(entry)

    def _flush_entry(self, entry: LogEntry) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(entry.to_dict(), default=str) + "\n")
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def truncate_after(self, count: int) -> None:
        """Keep only the first `count` entries in memory and on disk.

        Called on resume: the checkpoint stores `entry_count`, and any
        jsonl lines past that point were written by the crashed run
        mid-session.
        """
        if count < 0:
            count = 0
        self.entries = self.entries[:count]
        if self._jsonl_path is None:
            return
        self.close()
        if count == 0:
            self._jsonl_path.write_text("", encoding="utf-8")
        else:
            # Rewrite with only the kept entries (simpler than byte-accurate truncation).
            lines = [json.dumps(e.to_dict(), default=str) for e in self.entries]
            self._jsonl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self._fh = open(self._jsonl_path, "a", encoding="utf-8")

    def to_checkpoint(self) -> dict:
        """Return a minimal pointer — the jsonl file is the source of truth."""
        return {"entry_count": len(self.entries)}

    def load_from_jsonl(self, entry_count: int | None = None) -> None:
        """Re-populate `entries` from the jsonl file (up to `entry_count` lines).

        Called on resume before any new appends.
        """
        if self._jsonl_path is None or not self._jsonl_path.exists():
            return
        loaded: list[LogEntry] = []
        for line in self._jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                loaded.append(LogEntry.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
            if entry_count is not None and len(loaded) >= entry_count:
                break
        self.entries = loaded
        # If the file had more lines than entry_count, truncate back to the pointer.
        if entry_count is not None:
            self.truncate_after(entry_count)
