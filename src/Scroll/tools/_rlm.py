"""Recursive Language Model (RLM) helper for SCROLL agents.

``make_dspy_rlm`` returns an async ``rlm(query, *, context="") -> str``
callable backed by :class:`dspy.RLM`. RLM is a sub-agent that runs its
own Python REPL, can call its own batched sub-LLMs, and iteratively
explores a context blob to answer a query. SCROLL agents bind it as
``rlm`` in the REPL when ``expose_rlm = True`` so a code cell can
delegate semantic operations (span extraction, paraphrase
classification, ranking) that SQL / vector search can't express.

Mirrors call / result into the agent's :class:`ConversationLog` with
``metadata.kind`` of ``"rlm_call"`` / ``"rlm_result"`` so prior RLM
answers are recoverable via ``log.semantic_search(...,
kind="rlm_result")``.

Implementation notes:
- ``dspy.LM`` is built lazily on first call so import-time errors
  (e.g. missing API key) don't crash environments where RLM isn't
  used.
- The same ``dspy.LM`` instance is pinned via ``dspy.context(lm=...)``
  so concurrent ``asyncio.gather`` over multiple RLM calls within one
  task stay isolated.
"""

from __future__ import annotations

import os

import dspy

from Scroll.core._models import LogEntry


_RLM_PROMPT_PERSIST_CAP = 50_000
_RLM_OUTPUT_PERSIST_CAP = 50_000


def _cap_for_log(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... ({label} truncated for log: {len(text)} chars total)"


def _build_dspy_lm(cfg) -> dspy.LM:
    """Build a ``dspy.LM`` from the agent config.

    Mirrors the endpoint selection in ``Scroll.core._agent._init_model``
    so the dspy backend (via litellm) talks to the same OpenAI-compatible
    server the agent's top-level model uses. Routed through litellm's
    ``openai/<model>`` prefix.

    Reasoning models (gpt-5 / o1 / o3 / o4 family — and ``openai.``-
    prefixed router names) require ``temperature=1.0`` and
    ``max_tokens>=16000`` per dspy's validator; non-reasoning models keep
    ``temperature=0.0`` for deterministic sub-LM calls.
    """
    api_key = os.getenv(cfg.qwen_api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"Missing API key env: {cfg.qwen_api_key_env}. Set it before running."
        )
    base_env = getattr(cfg, "qwen_api_base_env", None)
    base_url = (
        getattr(cfg, "qwen_api_base", None)
        or (os.environ.get(base_env) if base_env else None)
        or os.environ.get("US_DASHSCOPE_BASE_URL")
        or os.environ.get("CN_DASHSCOPE_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    name_lc = cfg.qwen_model_name.lower()
    is_reasoning = (
        "gpt-5" in name_lc
        or name_lc.startswith("o1")
        or name_lc.startswith("o3")
        or name_lc.startswith("o4")
        or "openai." in name_lc
    )
    if is_reasoning:
        return dspy.LM(
            model=f"openai/{cfg.qwen_model_name}",
            api_key=api_key,
            api_base=base_url,
            max_tokens=max(16000, cfg.max_output_tokens),
            temperature=1.0,
        )
    return dspy.LM(
        model=f"openai/{cfg.qwen_model_name}",
        api_key=api_key,
        api_base=base_url,
        max_tokens=cfg.max_output_tokens,
        temperature=0.0,
    )


def make_dspy_rlm(
    cfg,
    *,
    log,
    day_provider,
    max_iterations: int = 8,
    max_llm_calls: int = 20,
    max_output_chars: int = 10_000,
):
    """Build an async ``rlm(query, *, context="") -> str`` callable.

    Args:
        cfg: agent config (must expose ``qwen_api_key_env``,
            ``qwen_model_name``, ``max_output_tokens``).
        log: the agent's :class:`ConversationLog` (or ``None`` to skip
            mirroring).
        day_provider: zero-arg callable returning the current day index
            (used to tag mirrored log entries).
        max_iterations / max_llm_calls / max_output_chars: passed
            through to :class:`dspy.RLM`.

    Returns:
        An async callable. ``await rlm(query=..., context=...)`` runs
        one RLM sub-agent loop and returns the answer string. Each call
        appends one ``rlm_call`` entry and one ``rlm_result`` entry to
        ``log``.
    """
    state: dict = {"lm": None, "module": None, "counter": 0}

    async def rlm(query=None, *, context: str = "", **kwargs) -> str:
        # Accept ``rlm(prompt)`` positional and ``prompt=`` keyword shapes
        # so simple "one prompt in, one string out" callers work without
        # spelling out ``query=`` + ``context=``.
        if query is None:
            query = kwargs.pop("prompt", "")
        query_str = str(query)
        context_str = str(context)

        if state["lm"] is None:
            state["lm"] = _build_dspy_lm(cfg)
            state["module"] = dspy.RLM(
                "context, query -> answer",
                max_iterations=max_iterations,
                max_llm_calls=max_llm_calls,
                max_output_chars=max_output_chars,
                sub_lm=state["lm"],
            )

        # Pin the LM for this call (thread-local; safe under concurrent
        # asyncio.gather over rlm calls within one task).
        with dspy.context(lm=state["lm"]):
            prediction = await state["module"].aforward(
                context=context_str, query=query_str,
            )
        answer = getattr(prediction, "answer", None)
        if answer is None:
            answer = str(prediction)
        answer_str = str(answer)

        if log is not None:
            day = int(day_provider()) if day_provider is not None else 0
            state["counter"] += 1
            call_id = f"rlm-{day}-{state['counter']}"
            log.append(LogEntry.make(turn_idx=day, role="assistant",
                tool_call={
                    "id": call_id, "name": "rlm",
                    "arguments": {
                        "query": _cap_for_log(query_str, _RLM_PROMPT_PERSIST_CAP, "query"),
                        "context": _cap_for_log(context_str, _RLM_PROMPT_PERSIST_CAP, "context"),
                    },
                },
                metadata={
                    "kind": "rlm_call",
                    "query_chars": len(query_str),
                    "context_chars": len(context_str),
                },
            ))
            log.append(LogEntry.make(turn_idx=day, role="tool",
                tool_result={
                    "id": call_id, "name": "rlm",
                    "output": _cap_for_log(answer_str, _RLM_OUTPUT_PERSIST_CAP, "answer"),
                },
                metadata={
                    "kind": "rlm_result",
                    "output_chars": len(answer_str),
                },
            ))
        return answer_str

    rlm._label = "rlm"  # type: ignore[attr-defined]
    return rlm
