"""LongMemEval has no env action tools.

The agent's only "action" each session is observing the served chat;
data-management primitives (``log``, ``ms``, ``rlm``, ``rlm``)
come from the agent's namespace builder, not from this module.
"""

from __future__ import annotations

from typing import Any


def register_env_tools(toolkit: Any, state: Any) -> None:
    return None
