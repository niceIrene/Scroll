from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LogEntry:
    """A single entry the agent appends to the runtime's event log.

    The log mirrors the agent's outer-loop perspective: model turns, tool
    calls it issued, and tool results it observed. The model can introspect
    it from inside execute_python via the namespace `log` handle.

    Beyond the flattened text (`content`), an entry can carry the *structured*
    payload of its turn so the durable backing store (``conversation_history``)
    can reconstruct the exact wire-format Msg later: ``blocks`` holds the
    serialized block dicts, ``tool_input`` the tool arguments, ``tool_call_id``
    links a call to its result, ``tool_state`` the result state. These default
    to empty so legacy construction (kind/role/name/content/metadata) is intact.
    """
    kind: str                              # e.g. "model_turn", "tool_call", "tool_result"
    role: str | None = None                # "assistant" / "tool" / ...
    name: str | None = None                # tool name (for tool_call/tool_result)
    content: str | None = None             # text body
    metadata: dict[str, Any] = field(default_factory=dict)
    # --- structured payload (for faithful persistence/recall) -------------
    step_index: int | None = None          # loop step that produced this entry
    msg_index: int | None = None            # index of the owning Msg in history
    tool_call_id: str | None = None         # links tool_call <-> tool_result
    tool_input: Any = None                  # tool args (dict or JSON-able)
    tool_state: str | None = None           # ToolResultState value for results
    headline: str | None = None             # one-line self-authored (or extractive)
                                            # digest of the turn; the durable index
                                            # scanned when deciding what to recall
    blocks: list[dict[str, Any]] | None = None  # exact serialized block dicts
    created_at: str | None = None           # event's own timestamp; append() falls
                                            # back to write-time now() when unset


@dataclass(frozen=True)
class ExecutionResult:
    """What runtime.execute() returns for a single execute_python call."""
    stdout: str
    stderr: str
    error: str | None = None
