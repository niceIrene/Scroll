"""ScrollRuntime — long-context management substrate for agents.

The runtime gives an agent a persistent Python REPL whose namespace exposes
the model's in-memory SQLite scratch space (`ms`), with the durable,
file-backed cross-session ``conversation_history`` ATTACHed read-only as
``hist`` for retrieval (and ``ms.session_id`` to scope to the current run).
Task-environment tools like ``bash`` live at the agent layer (dispatched to
``ctx.environment`` directly), not in this namespace.

Every event the agent appends is write-through-persisted into the durable
``conversation_history`` table, so a later session can retrieve a past
session's record. Agents instantiate `ScrollRuntime` at the top of `run()`,
route `execute_python` calls into `runtime.execute(source)`, and `close()` in
a `finally` (connections are released; the history file persists).
"""
from scroll_context._runtime.history import HistoryStore
from scroll_context._runtime.runtime import ScrollRuntime
from scroll_context._runtime.types import (
    ExecutionResult,
    LogEntry,
)

__all__ = [
    "ScrollRuntime",
    "HistoryStore",
    "ExecutionResult",
    "LogEntry",
]
