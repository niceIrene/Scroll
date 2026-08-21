from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    """Snapshot of what Harbor hands to an agent for a single task."""
    task_id: str
    instruction: str                       # the raw instruction.md contents
    metadata: dict[str, Any] = field(default_factory=dict)


class TerminationReason(str, Enum):
    SUCCESS = "success"
    BUDGET = "budget"
    ERROR = "error"
    GAVE_UP = "gave_up"


@dataclass
class Step:
    index: int
    thought: str | None  # the model's visible *text* output (not chain-of-thought)
    action: dict[str, Any] | None  # {"tool": name, "args": {...}} or None
    observation: str | None
    tokens_in: int = 0
    tokens_out: int = 0
    # Reasoning/thinking-mode chain-of-thought for this turn, surfaced here for
    # readable trajectories. None when thinking is off (or the turn emitted none);
    # it is *not* re-sent to the model (see agent loop) — this is for inspection.
    reasoning: str | None = None


@dataclass
class Trajectory:
    task_id: str
    steps: list[Step]
    final_answer: str | None
    terminated: TerminationReason
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopContext:
    """What an agent's run() receives.

    Per spec §6.2, agents own their tools and prompts — load from your own
    subfolder. The runner does not provide either.

    llm_openai      raw openai.OpenAI client (unused by the base_agents family;
                    kept for API compatibility).
    llm_agentscope  agentscope.model.ChatModelBase; base_agents use this.
    model_name      bare model name (e.g. 'qwen3.7-max') — for tracing and
                    so each agent can pass the right name to its client.
    tracer          shared OTel tracer.
    budget          token + wall-time budget (placeholder; unused in v0).
    environment     Harbor BaseEnvironment.
    tools           Optional list of OpenAI-format tool schemas. When set
                    (via the YAML `tools:` field), agents that opt in read
                    this instead of importing the canonical surface. None
                    means "use whatever default the agent picks for itself."
    run_id          Concrete, run-stable id (None outside the harness).
                    Combined with task_id into a per-session id by agents that
                    use the durable conversation store.
    history_db_path Path to the file-backed cross-session conversation_history
                    store. None lets the agent/runtime pick its default.
    history_max_tokens  Token bound for the in-context history window an agent
                    sends to the LLM. None means "don't bound".
    summary_chunk_tokens  Per-call input size (tokens) for the summary_baseline
                    arm's summarization calls. None = the agent's default.
                    Other agents ignore it.
    logs_dir        Directory where the harness writes this task's artifacts
                    (trajectory.json, etc.). Agents may drop extra debug
                    artifacts here. None outside the harness.
    system_prompt   Optional task-specific system-prompt addendum. When set, an
                    agent appends it to its own bundled system.md (capability
                    prompt) rather than replacing it — lets an eval supply
                    task-family-specific framing (e.g. BEAM's memory/grounding
                    rules) on top of the shared agent's capability description.
                    None = the agent uses its bundled system.md alone.
    shared_run_ids  Run ids whose rows form a *shared background tier* in a
                    history DB shared across sibling sessions: under
                    ms.search(scope='task') they stay visible to every session,
                    while all other task rows are limited to the current session
                    (so siblings don't leak into each other). An eval that seeds
                    prior turns under a sentinel run id (BEAM seeds under
                    'seed') passes that id here. Empty (default) keeps the plain
                    "all runs of this task" scan — no isolation.
    """
    llm_openai: Any
    llm_agentscope: Any
    model_name: str
    tracer: Any
    budget: Any
    environment: Any = None
    tools: list[dict] | None = None
    run_id: str | None = None
    history_db_path: str | None = None
    history_max_tokens: int | None = None
    summary_chunk_tokens: int | None = None
    logs_dir: str | None = None
    system_prompt: str | None = None
    shared_run_ids: tuple[str, ...] = ()
