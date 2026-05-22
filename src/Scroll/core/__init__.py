"""Core abstractions and runtime for the SCROLL benchmark framework.

SCROLL — **S**ession-as-**C**ontext: **R**ecoverable **O**ff-context **L**LM **L**og.
The harness pattern: long-term memory lives outside the LLM context as an
event log ``E`` plus a derived memoryspace ``W``; the agent uses
**CodeAct** (writing Python in a sandboxed REPL with ``log`` and ``ms``
bound) to retrieve into context on demand.

Design symbol ↔ code map:

| Symbol                   | Code                                                          |
|--------------------------|---------------------------------------------------------------|
| ``E`` (event log)        | :class:`Scroll.log.ConversationLog`                       |
| ``log`` (REPL handle)    | :class:`Scroll.tools._log_handle.LogHandle`                   |
| ``W`` (memoryspace)      | :class:`Scroll.tools.memoryspace.Memoryspace`                 |
| ``W = build(E)``         | env-specific subclass of :class:`ScrollAgent`                  |
| ``rlm(query, context)``  | :func:`Scroll.tools._rlm.make_dspy_rlm`                       |

The formal symbols stay in docs/design; Python identifiers are idiomatic.
"""

from Scroll.core._models import AgentConfig, BaseEnvConfig, SessionResult, Event, LogEntry, RunStats

# Back-compat alias — older code may still import ``DayResult``.
DayResult = SessionResult
from Scroll.core._evaluation import EnvSnapshot, ProbeResult, ProbeSpec
from Scroll.core._environment import BaseEnvironment, BaseDataSource
from Scroll.core._agent import BaseAgent
from Scroll.core._codeact_agent import CodeActAgent
from Scroll.core._scroll_agent import ScrollAgent
from Scroll.core._ingestor import Ingestor
from Scroll.core._checkpoint import serialize_rng, restore_rng
from Scroll.core._runner import load_config, parse_env_config, parse_agent_config, apply_overrides, expand_ablations
from Scroll.core._registry import get_env, register_env, EnvEntry

__all__ = [
    # agents
    "BaseAgent",
    "CodeActAgent",
    "ScrollAgent",
    "Ingestor",
    # environment
    "BaseEnvironment",
    "BaseDataSource",
    # models
    "AgentConfig",
    "BaseEnvConfig",
    "SessionResult",
    "DayResult",  # back-compat alias
    "Event",
    "LogEntry",
    "RunStats",
    # evaluation
    "EnvSnapshot",
    "ProbeResult",
    "ProbeSpec",
    # checkpoint
    "serialize_rng",
    "restore_rng",
    # runner
    "load_config",
    "parse_env_config",
    "parse_agent_config",
    "apply_overrides",
    "expand_ablations",
    # registry
    "get_env",
    "register_env",
    "EnvEntry",
]
