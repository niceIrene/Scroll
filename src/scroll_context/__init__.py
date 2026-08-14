"""OpenAI-chat-format frontend to scroll's context-management mechanism.

`scroll_context._runtime` is the substrate (durable history, retrieval, REPL,
eviction index) and is deliberately message-format-agnostic. This package is
the *orchestration* of that substrate for harnesses whose conversation is a
plain OpenAI chat-completions message list (``[{"role": ..., "content": ...,
"tool_calls": [...]}, ...]``) — the same algorithms `scroll_agent_A` runs over
AgentScope ``Msg`` objects, rewritten against dict messages:

- write-through persistence of every turn (`record_*`),
- token-budget eviction folded into the in-context ``EvictionIndex``,
- observation aging (stubbing old tool outputs),
- the per-turn ephemeral ``[working memory]`` digest,
- the ``scroll_repl`` tool (persistent Python REPL with ``ms`` recall).

It must not import from any specific harness (only ``scroll_context._runtime`` and
the stdlib) so any OpenAI-format agent loop can reuse it; the first consumer is
LOCA-bench's ReAct runner.
"""

from scroll_context._runtime.history import HistoryStore
from scroll_context._runtime.types import LogEntry
from scroll_context.manager import ScrollContextManager, shared_session_spans
from scroll_context.prompts import (
    SCROLL_PROMPT_PROTOCOL,
    SCROLL_VAR_PROMPT_PROTOCOL,
    core_prompt,
    index_prompt,
    protocol_prompt,
    strip_headline_schema,
    var_prompt,
)
from scroll_context.tool import SCROLL_REPL_TOOL_NAME, SCROLL_REPL_TOOL_SCHEMA

__all__ = [
    "HistoryStore",
    "LogEntry",
    "ScrollContextManager",
    "shared_session_spans",
    "SCROLL_PROMPT_PROTOCOL",
    "SCROLL_VAR_PROMPT_PROTOCOL",
    "core_prompt",
    "index_prompt",
    "protocol_prompt",
    "strip_headline_schema",
    "var_prompt",
    "SCROLL_REPL_TOOL_NAME",
    "SCROLL_REPL_TOOL_SCHEMA",
]
