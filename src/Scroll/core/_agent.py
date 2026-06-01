"""Base agent abstractions and shared model helpers.

The LLM agents now live in :mod:`Scroll.core._codeact_agent` (the
RLM-style substrate). This module retains:

  - :class:`BaseAgent` — the lifecycle interface every agent
    (heuristic or LLM-based) implements.
  - :func:`_init_model` — builds an AgentScope ``OpenAIChatModel`` +
    returns the ``Msg`` class. Used by both the top-level model and
    sub-LM helpers.
  - :func:`_extract_text_from_response` — pulls the concatenated text
    from a ``ChatResponse`` / ``Msg``-like object.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Scroll.core._environment import BaseEnvironment

CHARS_PER_TOKEN = 4
MODEL_CALL_MAX_RETRIES = 3
MODEL_CALL_RETRY_DELAY = 2  # seconds

_log = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Interface that all benchmark agents must implement.

    Task lifecycle: ``bootstrap`` → ``start_session`` → per-turn loop
    (``receive_context → run_turn → receive_outcomes``, zero or more
    iterations) → end-of-task probes → ``end_session``. Vending runs
    the loop every turn; LME / BEAM in their default mode
    (``agent_during_ingestion=false``) set ``num_turns=0`` and answer
    purely from ``W`` via end-of-task probes.

    A *session* = agent-instance lifetime. Today every env runs one
    session per task; ``start_session`` / ``end_session`` exist so
    future multi-session envs (or BEAM ``probe_isolation="fresh"``) can
    spawn a fresh agent context per session. Both default to no-ops.
    """

    last_context: list[str]

    def bootstrap(self, env: BaseEnvironment) -> None:
        """Hook fired once per task, BEFORE ``start_session``.

        Owns the task-wide env→E ingestion step. Default no-op.
        :class:`ScrollAgent` overrides to delegate to its memoryspace's
        attached ingestor — so the actual env-specific bulk-load
        code (LME haystack, BEAM batches) lives in the ingestor, not
        the agent.

        Distinct from :meth:`start_session` so that ``probe_isolation
        ="fresh"`` (BEAM) can spawn fresh agent sessions per probe
        without re-running ingest each time. Ingest happens once;
        start_session may happen multiple times.
        """
        return None

    def start_session(self) -> None:
        """Hook fired when a new agent session begins.

        Called once by the harness (``_run_task``) before the first
        ``run_turn``, and again per probe under
        ``probe_isolation="fresh"``. Default no-op. Subclasses
        override to wipe / rebuild per-session in-context state
        (e.g. ``LongMemEvalAgent`` reloading cross-task distilled
        hints at session start). Must be safe to call multiple
        times in a task.
        """
        return None

    def end_session(self) -> None:
        """Hook fired when an agent session ends.

        Called once by the harness (``_run_task``) after the last
        ``run_turn`` and any end-of-task probes. Default no-op.
        Subclasses override for per-instance teardown (e.g. flushing
        distilled hints out to the cross-task store, releasing
        connections held by a sub-LM client).
        """
        return None

    @abstractmethod
    def run_turn(self, env: BaseEnvironment) -> list[str]:
        """Execute one simulation turn. Returns a list of action log strings."""

    @abstractmethod
    def receive_outcomes(self, turn_idx: int, logs: list[str]) -> None:
        """Receive environment outcome logs after step_turn."""

    @abstractmethod
    def receive_context(self, turn_idx: int, notes: list[str]) -> None:
        """Receive external data source notes at the start of a turn."""

    @abstractmethod
    def answer_probe(self, question: str) -> str:
        """Answer a data-quality probe question. Returns the answer string."""

    @property
    @abstractmethod
    def message_count(self) -> int:
        """Total LLM messages consumed so far (for budget tracking)."""

    @property
    @abstractmethod
    def turn_ended(self) -> bool:
        """Whether the agent has signaled end-of-turn."""

    def probe_user_hint(self, probe=None) -> str:
        """Strategy-specific reminder appended to the probe user message.

        Complements :meth:`BaseEnvironment.probe_user_postscript` (which
        carries env-level scorer-format rules — Answer-line shape, units,
        tolerances). This hook carries STRATEGY-level retrieval guidance
        — what tools the agent has and how to use them at probe time
        (e.g. ``LongMemEvalAgent`` reminds the model to use ``ms`` / ``log``
        / ``rlm`` rather than answering from chat history alone).

        ``probe`` (optional) is the active ``ProbeSpec``. Subclasses that
        want per-probe-type routing (e.g. ``LongMemEvalAgent`` dispatching
        prompts by ``probe.question_type``) override and read it; older
        subclasses ignoring the arg stay source-compatible because the
        default value is ``None``.

        Putting this text NEXT TO the probe question (instead of in the
        agent's ``sys_prompt``) keeps it out of the turn-loop prompt
        where it's irrelevant and brings it adjacent to the actual
        probe Q where the model is most likely to act on it.

        Default: empty (no hint). Override in subclasses.
        """
        return ""

    def probe_user_question_prefix(self, probe=None) -> str:
        """Text prepended IMMEDIATELY BEFORE the probe question text.

        Sits inside the same user-turn message, between the framework
        ``[PROBE — qid]`` header and ``probe.question``. Use it for
        per-probe data the model needs RIGHT NEXT TO the question
        (e.g. LongMemEvalAgent wraps with ``Today's Date: YYYY-MM-DD
        (latest session_idx=N)\\nQuestion: `` so date arithmetic
        anchors and KU ``ORDER BY session_idx DESC LIMIT 1`` queries
        have the index in hand).

        Default: empty. Override in subclasses.
        """
        return ""

    async def _on_probe_complete(
        self,
        *,
        probe,
        agent_answer: str,
        score: float,
        ground_truth,
    ) -> None:
        """Hook fired after a probe is answered and scored.

        Default no-op. Subclasses (e.g. ``LongMemEvalAgent``) override to
        distill success/failure patterns from the trajectory + score
        and persist them to a cross-task store, so future probes of
        the same question type can be primed with what worked. This
        is the system-side analog of ``code_agent`` writing its own
        ``lessons`` — but driven by the harness from probe outcomes,
        not by the agent.

        Async so distillation can call sub-LM without blocking the
        single-threaded run loop. Fires in ``inject_probe`` after
        ``scoring_fn`` returns, so subclasses see the final score.
        """
        return None

    def augment_efficiency(self, efficiency: dict) -> None:
        """Hook: agent-side fields to merge into the per-run efficiency dict.

        Called by ``benchmark.py`` after the env's ``compute_efficiency_metrics``
        and after substrate-level token/LM-call accounting. Default no-op.
        Subclasses mutate ``efficiency`` in place to surface env-specific
        counters that aren't visible to the env (e.g. LME's per-task
        distilled-lesson count).
        """
        return None

    def to_checkpoint(self) -> dict:
        """Serialize agent state for checkpoint/resume."""
        raise NotImplementedError

    def from_checkpoint(self, data: dict) -> None:
        """Restore agent state from checkpoint data."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Model helpers (shared by CodeActAgent + sub-LM `rlm()` callable)
# ---------------------------------------------------------------------------


def _init_model(cfg):
    """Create a chat model from agent config. Returns ``(model, Msg)``.

    Routes to AgentScope's ``OpenAIChatModel`` (Chat Completions API) by
    default, or to ``OpenAIResponsesChatModel`` (Responses API) when the
    model name signals a Responses-only model — currently any GPT-5 ``-pro-``
    variant or anything prefixed ``openai.`` (Dashscope's compatibility router
    only exposes those via the Responses endpoint, and `client.chat.completions`
    on them returns 500 ``input is required``).
    """
    from agentscope.message import Msg
    from agentscope.model import OpenAIChatModel
    from Scroll.core._chat_model_adapters import (
        AnthropicMessagesChatModel,
        OpenAIResponsesChatModel,
    )

    api_key = os.getenv(cfg.qwen_api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"Missing API key env: {cfg.qwen_api_key_env}. "
            "Set it to your DashScope key before running."
        )

    _base_env = getattr(cfg, "qwen_api_base_env", None)
    _base_override = (
        getattr(cfg, "qwen_api_base", None)
        or (os.environ.get(_base_env) if _base_env else None)
    )
    name_lc = cfg.qwen_model_name.lower()
    is_qwen = "qwen" in name_lc
    is_responses_api = (
        name_lc.startswith("openai.")
        or "gpt-5" in name_lc
    )
    is_anthropic_native = name_lc.startswith("claude-")

    if is_anthropic_native:
        # Anthropic SDK has its own default base_url (api.anthropic.com);
        # only pass an override when caller explicitly sets one.
        model = AnthropicMessagesChatModel(
            model_name=cfg.qwen_model_name,
            api_key=api_key,
            base_url=_base_override or os.environ.get("ANTHROPIC_BASE_URL"),
            generate_kwargs={"max_tokens": cfg.max_output_tokens},
            auth_mode=getattr(cfg, "qwen_auth_mode", "api_key"),
            default_headers=getattr(cfg, "qwen_default_headers", None),
        )
        return model, Msg

    base_url = (
        _base_override
        or os.environ.get("US_DASHSCOPE_BASE_URL")
        or os.environ.get("CN_DASHSCOPE_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    if is_responses_api:
        # Responses API uses ``max_output_tokens``. The wrapper's ``__call__``
        # also tolerates the Chat-Completions key names and rewrites them.
        generate_kwargs: dict = {"max_output_tokens": cfg.max_output_tokens}
        model = OpenAIResponsesChatModel(
            model_name=cfg.qwen_model_name,
            api_key=api_key,
            stream=False,
            client_kwargs={"base_url": base_url},
            generate_kwargs=generate_kwargs,
        )
        return model, Msg

    tokens_key = "max_tokens" if is_qwen else "max_completion_tokens"
    generate_kwargs = {tokens_key: cfg.max_output_tokens}
    if is_qwen:
        # cfg.enable_thinking overrides; otherwise default False to
        # avoid spending tokens on thinking on weak qwen tiers.
        _thinking = getattr(cfg, "enable_thinking", None)
        if _thinking is None:
            _thinking = False
        extra_body: dict = {"enable_thinking": bool(_thinking)}
        # qwen-native reasoning intensity. Only meaningful when
        # enable_thinking is True; pass-through to Dashscope which
        # may honor it (qwen3-*-thinking variants) or silently
        # ignore (mainline qwen3.7-max via compat-mode — empirically
        # unconfirmed).
        _budget = getattr(cfg, "thinking_budget", None)
        if _budget is not None:
            extra_body["thinking_budget"] = int(_budget)
        generate_kwargs["extra_body"] = extra_body
    model = OpenAIChatModel(
        model_name=cfg.qwen_model_name,
        api_key=api_key,
        stream=False,
        client_kwargs={"base_url": base_url},
        generate_kwargs=generate_kwargs,
    )
    return model, Msg


def _extract_text_from_response(result) -> str:
    """Pull the concatenated text from a ``ChatResponse`` or ``Msg``-like object."""
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(p for p in parts if p).strip()
    return ""


__all__ = [
    "BaseAgent",
    "CHARS_PER_TOKEN",
    "MODEL_CALL_MAX_RETRIES",
    "MODEL_CALL_RETRY_DELAY",
    "_init_model",
    "_extract_text_from_response",
]
