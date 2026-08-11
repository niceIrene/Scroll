"""scroll-context — scroll's context management for LLM agent loops.

The public API is :class:`ScrollContextManager`: per-session context
management over a plain OpenAI chat-completions message list
(``[{"role": ..., "content": ..., "tool_calls": [...]}, ...]``):

- write-through persistence of every turn (`record_*`),
- token-budget eviction folded into the in-context eviction-index map,
- observation aging (stubbing old tool outputs),
- the seeded prior-sessions ``[memory]`` map (`seed_index_map`),
- the per-step ephemeral ``[working memory]`` digest,
- the ``scroll_repl`` tool (persistent Python REPL with ``ms`` recall),
- the model-facing prompt protocol (`core.md` / `index.md`), assembled per
  configuration by `protocol_prompt` — the single source of truth for how the
  model is taught to manage its context.

``HistoryStore`` and ``LogEntry`` are additionally public for *ingestion*:
writing prior conversation tiers (e.g. a benchmark's seeded haystack) into a
history DB that a manager then reads via ``shared_run_ids`` /
``seed_index_map``. Everything else under ``scroll_context._runtime`` is a
private implementation detail — reach it only through the manager.

The package imports nothing from any agent harness (only the stdlib and
``opentelemetry-api``), so any OpenAI-format agent loop can embed it.
"""

from scroll_context._runtime.history import HistoryStore
from scroll_context._runtime.types import LogEntry
from scroll_context.manager import ScrollContextManager
from scroll_context.prompts import (
    HEADLINE_SCHEMA_FRAGMENTS,
    SCROLL_PROMPT_PROTOCOL,
    core_prompt,
    index_prompt,
    protocol_prompt,
    strip_headline_schema,
)
from scroll_context.tool import SCROLL_REPL_TOOL_NAME, SCROLL_REPL_TOOL_SCHEMA

__version__ = "0.1.0"

__all__ = [
    "ScrollContextManager",
    "HistoryStore",
    "LogEntry",
    "HEADLINE_SCHEMA_FRAGMENTS",
    "SCROLL_PROMPT_PROTOCOL",
    "core_prompt",
    "index_prompt",
    "protocol_prompt",
    "strip_headline_schema",
    "SCROLL_REPL_TOOL_NAME",
    "SCROLL_REPL_TOOL_SCHEMA",
]
