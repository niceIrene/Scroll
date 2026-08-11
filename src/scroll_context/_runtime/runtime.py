from __future__ import annotations

from pathlib import Path
from typing import Any

from scroll_context._runtime import namespace as _namespace
from scroll_context._runtime.exec import Executor, stdout_cap_for
from scroll_context._runtime.history import HistoryStore
from scroll_context._runtime.memoryspace import MemorySpace
from scroll_context._runtime.types import ExecutionResult, LogEntry


_DEFAULT_EXECUTE_TIMEOUT_S = 30.0
_DEFAULT_MEMORY_DB = "~/.scroll/history.db"


class ScrollRuntime:
    """Per-session long-context substrate.

    Owns the REPL namespace — where the model keeps its working data in plain
    variables that persist across ``execute_python`` calls — and the model's
    ``MemorySpace`` (a read-only query window onto the durable history, ATTACHed
    read-only). The durable, cross-session ``conversation_history`` lives in a
    file-backed ``HistoryStore`` that the runtime writes through on every
    ``append_log`` — so a later session can retrieve a past session's record
    via SQL against ``hist.conversation_history``.

    A *session* is one ``agent.run()`` execution, identified by ``session_id``
    (``f"{run_id}:{task_id}"``). Lifetime is that one session: instantiate at
    the top of ``run()``, call ``close()`` in a ``finally`` block — the
    connections are released, but the history *file* persists.
    """

    def __init__(
        self,
        *,
        history_db_path: str | Path | None = None,
        session_id: str = "local",
        run_id: str | None = None,
        task_id: str | None = None,
        shared_run_ids: tuple[str, ...] = (),
        execute_timeout_s: float = _DEFAULT_EXECUTE_TIMEOUT_S,
        history_max_tokens: int | None = None,
    ) -> None:
        self._session_id = session_id
        self._run_id = run_id
        self._task_id = task_id

        db_path = history_db_path if history_db_path is not None else _DEFAULT_MEMORY_DB
        self._history = HistoryStore(db_path)
        self._ms = MemorySpace(
            history_db_path=self._history.path,
            session_id=session_id,
            task_id=task_id,
            # Run ids that are a shared tier under ms.search(scope='task') — e.g.
            # an eval's seeded prior sessions — so sibling sessions sharing this
            # history DB stay isolated to "shared tier + own session".
            shared_run_ids=shared_run_ids,
        )
        self._persisted_seq = 0

        self._ns: dict[str, Any] = {}
        _namespace.populate(self._ns, memoryspace=self._ms)
        # Cap a single execute_python's stdout to a fraction of the in-context
        # budget so one print can't flood the window (see stdout_cap_for).
        self._executor = Executor(
            self._ns,
            timeout_s=execute_timeout_s,
            max_stdout_chars=stdout_cap_for(history_max_tokens),
        )

    # --- public API ---------------------------------------------------

    async def execute(self, source: str) -> ExecutionResult:
        return await self._executor.execute(source)

    def append_log(self, entry: LogEntry) -> None:
        """Write-through one event into the durable conversation history.

        Returns nothing, but advances the persist watermark (``persisted_seq``)
        so the agent can assert a Msg is durable before evicting it from the
        in-context window.
        """
        seq = self._history.append(
            session_id=self._session_id,
            run_id=self._run_id,
            task_id=self._task_id,
            entry=entry,
        )
        self._persisted_seq = max(self._persisted_seq, seq)

    @property
    def persisted_seq(self) -> int:
        return self._persisted_seq

    def log_entries(self) -> list[LogEntry]:
        return self._history.query_log(self._session_id)

    def digest(self) -> str:
        """Deterministic working-notes snapshot of the model's persisted vars."""
        return _namespace.describe(self._ns)

    @property
    def namespace(self) -> dict[str, Any]:
        return self._ns

    @property
    def memoryspace(self) -> MemorySpace:
        return self._ms

    @property
    def history(self) -> HistoryStore:
        return self._history

    @property
    def session_id(self) -> str:
        return self._session_id

    def close(self) -> None:
        self._ms.close()
        self._history.close()
