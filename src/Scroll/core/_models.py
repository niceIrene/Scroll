"""Domain-free data classes used across the benchmark framework."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class LogEntry:
    """A single entry in the unified conversation log.

    The log mirrors the agent's chat turns (system prompt, user prompts,
    assistant responses, tool calls, tool results, compression summaries).
    Entries are append-only and persisted to conversation_log.jsonl.
    """

    session_idx: int
    ts: float
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_call: dict | None = None   # {id, name, arguments}  when role="assistant"
    tool_result: dict | None = None  # {id, name, output}     when role="tool"
    metadata: dict = field(default_factory=dict)

    @classmethod
    def make(
        cls,
        session_idx: int,
        role: str,
        content: str = "",
        tool_call: dict | None = None,
        tool_result: dict | None = None,
        metadata: dict | None = None,
    ) -> "LogEntry":
        return cls(
            session_idx=session_idx,
            ts=time.time(),
            role=role,
            content=content,
            tool_call=tool_call,
            tool_result=tool_result,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_idx": self.session_idx,
            "ts": self.ts,
            "role": self.role,
            "content": self.content,
            "tool_call": self.tool_call,
            "tool_result": self.tool_result,
            "metadata": self.metadata,
        }

    # Dict-style ergonomics: LMs habitually call ``entry.get("content")``
    # / ``entry["role"]`` from the REPL (especially when iterating over
    # ``log.entries``), since the conversation_log.jsonl on disk is dicts
    # and the docs read like a record. Without these the agent gets
    # ``AttributeError: 'LogEntry' object has no attribute 'get'`` and
    # has to retry. Mapping access mirrors :meth:`to_dict` keys.
    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        try:
            return self.to_dict()[key]
        except KeyError as e:
            raise KeyError(
                f"LogEntry has no field {key!r}; valid keys: "
                f"session_idx, ts, role, content, tool_call, tool_result, metadata"
            ) from e

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    @classmethod
    def from_dict(cls, d: dict) -> "LogEntry":
        return cls(
            session_idx=int(d["session_idx"]),
            ts=float(d.get("ts", 0.0)),
            role=str(d.get("role", "system")),
            content=str(d.get("content", "")),
            tool_call=d.get("tool_call"),
            tool_result=d.get("tool_result"),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass
class BaseEnvConfig:
    """Cross-environment simulation config.

    Every env's own ``EnvConfig`` should expose at least these fields.
    The benchmark loop only reads ``num_sessions``; subclass / wrapper
    dataclasses add env-specific knobs (vending: machine_slots,
    lead_time_days; future envs: max_turns, domain, etc.).
    """

    num_sessions: int = 30

    @classmethod
    def from_dict(cls, d: dict) -> "BaseEnvConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class AgentConfig:
    """Agent configuration shared across all environments."""

    policy: str = "heuristic"
    qwen_model_name: str = "deepseek-v4-flash"
    qwen_api_key_env: str = "US_DASHSCOPE_API_KEY"
    qwen_api_base: str | None = None      # hardcoded base URL override
    qwen_api_base_env: str | None = None  # env var name to read base URL from
    # Anthropic-only knobs (ignored by the OpenAI-compatible paths). Set
    # ``qwen_auth_mode`` to ``"bearer"`` when hitting an OAuth-style
    # proxy that wants ``Authorization: Bearer`` instead of the SDK's
    # default ``x-api-key`` header (e.g. the idealab Anthropic proxy).
    # ``qwen_default_headers`` is merged into every request as extra
    # HTTP headers (``x-app``, ``anthropic-beta``, ``User-Agent``, etc.).
    qwen_auth_mode: str = "api_key"
    qwen_default_headers: dict[str, str] | None = None

    # Override Dashscope's qwen ``extra_body.enable_thinking``. None =
    # use the default (which Scroll sets to False to suppress
    # thinking tokens on qwen-family models). Set True to enable
    # thinking for models that benefit (e.g. qwen3.6-max-preview's
    # chain-of-thought).
    enable_thinking: bool | None = None

    # qwen-native reasoning intensity (token-budget on the thinking
    # channel). Only takes effect alongside ``enable_thinking=True``;
    # forwarded as ``extra_body.thinking_budget`` to Dashscope. Typical
    # range 1024-32768. ``None`` omits the field (Dashscope uses model
    # default). Coverage varies by model — qwen3-*-thinking variants
    # honor it reliably; mainline qwen3.7-max may silently ignore
    # both enable_thinking and thinking_budget via compat-mode.
    thinking_budget: int | None = None

    message_budget: int = 2000
    max_iters_per_turn: int = 50
    context_max_tokens: int = 30000
    # Per-call output token cap passed to the chat completion. Forces the
    # agent to stay concise (and to commit to a probe `Answer:` line) within
    # a known budget rather than relying on after-the-fact truncation.
    max_output_tokens: int = 4096
    memory_strategy: str = "summarizing"

    # Agent substrate. Currently only "codeact_rlm" is supported (the
    # RLM-style Python-REPL agent; see core/_codeact_agent.py). The
    # field is kept so future substrates can be added without re-
    # migration.
    substrate: str = "codeact_rlm"

    # X = f(E) memory mode. When None, uses the agent class's `memory_mode`
    # default. JSON configs only set this to override the class default.
    # Recognized values: "step" | "sliding". "step" wipes the LM history
    # per session and relies on REPL globals for within-session
    # continuity. "sliding" keeps history across sessions and trims
    # oldest turns when over the context budget — the passive
    # `f = mem(E)` baseline.
    memory_mode: str | None = None

    # Cross-task memoryspace persistence. Path to a JSON file shared across
    # multiple independent task runs (e.g. LongMemEval's per-QA sub-runs
    # under one orchestrator invocation). When set, ``CodeActAgent``
    # subclasses that opt in via ``shared_memoryspace_keys`` will load
    # selected memoryspace JSON keys at agent init and write them back at
    # session end, with file locking for parallel-safe access.
    shared_memoryspace_path: str | None = None

    # When False, ``LongMemEvalAgent._on_probe_complete`` early-returns
    # instead of distilling new procedural hints. Set to False when
    # running a weaker model with a pre-baked playbook (or hints from
    # a stronger model) — we want the model to READ the curated
    # guidance but NOT pollute the shared store with its own
    # (lower-quality) distillations.
    enable_distillation: bool = True

    # When False, ``LongMemEvalAgent.probe_user_hint`` skips appending
    # the ``DISTILLED PLAYBOOK`` static block to the probe prompt. Set
    # to False for A/B variants that consume raw procedural_hints from
    # the shared memoryspace instead of (or in addition to) the curated
    # playbook.
    enable_playbook: bool = True

    # Per-run override of ``LongMemEvalAgent.procedural_hints_in_prompt``
    # (default 5 from the class attr). When None, the class default
    # is used. Set higher for runs where the shared memoryspace is
    # pre-seeded from a strong-model run with rich hints per qtype.
    procedural_hints_in_prompt: int | None = None


    # Vending-specific defaults (will move to env config later)
    target_margin: float = 0.65
    reorder_days_cover: int = 10
    order_interval_days: int = 2
    collect_interval_days: int = 7
    restock_units_per_sku: int = 20
    llm_max_order_per_sku: int = 60

    @classmethod
    def from_dict(cls, d: dict) -> AgentConfig:
        """Construct from a raw config dict, using dataclass defaults for missing keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Event:
    """A timestamped event in the simulation."""

    session_idx: int
    kind: str
    text: str


@dataclass
class SessionResult:
    """Results from stepping the environment forward one session."""

    session_idx: int
    sold_units: int
    revenue: float
    machine_cash: float
    cash: float


@dataclass
class RunStats:
    """Aggregate results from a single benchmark run.

    Only ``strategy`` / ``seed`` / ``active_sessions`` / ``probe_avg_score``
    / ``efficiency`` are env-agnostic. Env-specific headline metrics
    (vending's ``net_worth`` / ``units_sold`` / ``bankrupt``) live in
    ``env_metrics``, populated by :meth:`BaseEnvironment.report_run_metrics`.
    """

    strategy: str
    seed: int
    active_sessions: int = 0
    probe_avg_score: float = 0.0
    efficiency: dict = field(default_factory=dict)
    env_metrics: dict = field(default_factory=dict)
