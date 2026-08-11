"""Canonical tool definitions shared across agent families.

Historically, all three families (wild, base, agentscope) presented the same
two-tool surface (bash + submit_answer) to the model so cross-family
benchmark comparisons isolated loop-effect from tool-spec-effect. That
constant is still exported here and existing agents still use it unchanged.

In addition, this module now exposes a small registry (`TOOLS`) and a
`select_tools(names)` helper. New agents (and future per-benchmark configs)
can pull a tool subset by name instead of importing the fixed list (e.g. an
agent exposing execute_python alongside bash + submit_answer).

base_agents passes a tool schema list to a ChatModelBase and dispatches
tool calls inline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BASH_DESCRIPTION = (
    "Execute a shell command inside the agent's Linux container. "
    "Returns combined stdout/stderr and the exit code."
)

SUBMIT_ANSWER_DESCRIPTION = (
    "Submit your final answer and end the task. Call this when you've "
    "verified the task is complete."
)

EXECUTE_PYTHON_DESCRIPTION = (
    "Run Python source in the agent's persistent runtime namespace. "
    "Use this to filter, structure, and query past observations via handles "
    "in the namespace (log, ms, fs, bash, submit_answer). Variables, imports "
    "and function defs persist across calls. Returns the captured stdout/stderr."
)


_BASH_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": BASH_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run (passed to bash -lc).",
                }
            },
            "required": ["command"],
        },
    },
}

_SUBMIT_ANSWER_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": SUBMIT_ANSWER_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Brief summary of what you did or the requested output.",
                }
            },
            "required": ["answer"],
        },
    },
}

_EXECUTE_PYTHON_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": EXECUTE_PYTHON_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": (
                        "Python source to execute in the persistent runtime "
                        "namespace. Use print(...) to surface values."
                    ),
                }
            },
            "required": ["source"],
        },
    },
}


# Canonical legacy surface (base_agent_A). MUST remain byte-identical to the
# pre-registry export — order and contents preserved.
OPENAI_TOOLS_SCHEMA: list[dict] = [_BASH_SCHEMA, _SUBMIT_ANSWER_SCHEMA]


# Registry of all known tools by name. New tools live here and are pulled by
# `select_tools(names)`. Agents and per-benchmark configs use the registry;
# the legacy list above is reserved for backwards-compat imports.
TOOLS: dict[str, dict] = {
    "bash": _BASH_SCHEMA,
    "submit_answer": _SUBMIT_ANSWER_SCHEMA,
    "execute_python": _EXECUTE_PYTHON_SCHEMA,
}


def select_tools(names: list[str]) -> list[dict]:
    """Return OpenAI-schema entries for the given tool names, in input order.

    Raises ValueError on any unknown name so misconfigured YAML fails fast
    instead of silently dropping a tool from the model's surface.
    """
    unknown = [name for name in names if name not in TOOLS]
    if unknown:
        known = sorted(TOOLS.keys())
        raise ValueError(
            f"unknown tool name(s): {unknown!r}; registered tools are {known!r}"
        )
    return [TOOLS[name] for name in names]


@dataclass(frozen=True)
class BashResult:
    stdout: str
    stderr: str
    exit_code: int


def format_bash_observation(r: BashResult) -> str:
    return f"exit={r.exit_code}\n{r.stdout}\n{r.stderr}".strip()


def budget_notice(
    *,
    elapsed_s: float | None = None,
    wall_time_s: float | None = None,
    tokens_used: int | None = None,
    max_tokens: int | None = None,
    steps_used: int | None = None,
    max_steps: int | None = None,
    threshold: float = 0.25,
) -> str | None:
    """One-line budget warning when an axis is near exhaustion, else ``None``.

    Each axis (wall-time, tokens, turns) is reduced to a *fraction remaining*;
    the message names the tightest one and roughly how much is left — but **only**
    once some axis has dropped below ``threshold`` of its budget (default 25%).
    The rest of the run it returns ``None``. The point is to surface budget
    pressure to the model **only when it's close to the limit** (not every turn),
    so it consolidates its remaining work into the current turn and submits —
    rather than being nagged about budget on every step.

    Axes with missing data (``None`` budget or ``None`` usage) are skipped, so a
    caller can pass only the axes it tracks.
    """
    fracs: list[tuple[float, str]] = []
    if wall_time_s and elapsed_s is not None and wall_time_s > 0:
        left = max(0.0, wall_time_s - elapsed_s)
        fracs.append((left / wall_time_s, f"~{int(left)}s of wall-time"))
    if max_tokens and tokens_used is not None and max_tokens > 0:
        left = max(0, max_tokens - tokens_used)
        fracs.append((left / max_tokens, f"~{left} tokens"))
    if max_steps and steps_used is not None and max_steps > 0:
        left = max(0, max_steps - steps_used)
        fracs.append((left / max_steps, f"{left} turn(s)"))
    if not fracs:
        return None
    frac, label = min(fracs, key=lambda pair: pair[0])
    if frac >= threshold:
        return None
    return (
        f"[budget] Only {label} left. Stop working in small increments: do the "
        "most valuable remaining work now — batch it into this one tool call — "
        "then submit_answer with your best result before the budget runs out."
    )


async def run_bash(environment: Any, command: str, *, timeout_sec: int = 60) -> BashResult:
    """Canonical bash implementation shared by all three agent families.

    Used by all three families to ensure the OBSERVATION format is identical
    across them (another isolation invariant for fair comparison).
    """
    if environment is None:
        return BashResult(
            stdout="",
            stderr="bash requires a container environment",
            exit_code=2,
        )
    try:
        result = await environment.exec(command, timeout_sec=timeout_sec)
    except RuntimeError as exc:
        # Harbor raises a bare RuntimeError("Command timed out after N seconds")
        # when a command exceeds timeout_sec. Surface it to the agent as a
        # recoverable observation (exit 124, like coreutils `timeout`) instead
        # of letting it propagate and kill the whole trial.
        if "timed out" in str(exc).lower():
            return BashResult(
                stdout="",
                stderr=f"command timed out after {timeout_sec} seconds",
                exit_code=124,
            )
        raise
    return BashResult(
        stdout=getattr(result, "stdout", "") or "",
        stderr=getattr(result, "stderr", "") or "",
        exit_code=int(getattr(result, "return_code", 0)),
    )
