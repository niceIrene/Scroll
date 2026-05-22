"""Agent-facing REPL primitives for the SCROLL substrate.

Three primitives, bound by ``ScrollAgent`` into the agent's REPL globals:

  - :class:`LogHandle` — read-only view of E (the event log).
  - :class:`Memoryspace` — read+write view of W (the derived memoryspace:
    SQL + JSON + vectors + files).
  - :func:`make_dspy_rlm` — factory for ``rlm(query, context=...)``, a
    recursive sub-agent backed by :class:`dspy.RLM`. The bound callable
    is what the agent's REPL sees as ``rlm``.
"""

from __future__ import annotations

from Scroll.tools._log_handle import LogHandle
from Scroll.tools._rlm import make_dspy_rlm
from Scroll.tools.memoryspace import Memoryspace

__all__ = [
    "LogHandle",
    "Memoryspace",
    "make_dspy_rlm",
]
