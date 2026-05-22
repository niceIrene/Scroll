"""REPL namespace builder for the vending agent.

Returns ``{name: callable}`` bound into the agent's REPL globals:
the env action tools (unwrapped from their AgentScope ToolResponse
envelopes so REPL code sees plain strings), plus
``wait_for_next_day`` that raises ``EndOfSession`` to end the day cleanly.
"""

from __future__ import annotations

import logging
from typing import Callable

from Scroll.core._codeact_agent import (
    _wrap_closure_for_repl,
    make_wait_for_next_day,
)
from Scroll.core._tool_state import ToolState
from Scroll.benchmarks.vending.tools import _make_env_tool_closures

_log = logging.getLogger(__name__)


def make_env_namespace(state: ToolState) -> dict[str, Callable]:
    closures = _make_env_tool_closures(state)
    ns: dict[str, Callable] = {}
    for fn in closures:
        name = fn.__name__
        if name.endswith("_tool"):
            name = name[: -len("_tool")]
        if name == "wait_for_next_day":
            continue
        ns[name] = _wrap_closure_for_repl(fn)
    ns["wait_for_next_day"] = make_wait_for_next_day(state)
    return ns


__all__ = ["make_env_namespace"]
