"""Harness bookkeeping: ``ToolState`` and the ``ToolResponse`` helper.

Tracks per-session counters, action log, and accumulated LM usage
across the agent's session-loop calls. Lives in ``core/`` because
it's harness state, not an agent-facing data surface. The legacy
env-side closures (in ``vending/tools.py``) still return AgentScope
``ToolResponse`` objects internally; the namespace builder unwraps
them to plain ``str`` before exposing to the agent's REPL (see
:func:`Scroll.core._codeact_agent._wrap_closure_for_repl`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Lazy imports for agentscope types — resolved at first call site.
_ToolResponse = None
_TextBlock = None


def _ensure_imports() -> None:
    global _ToolResponse, _TextBlock
    if _ToolResponse is None:
        from agentscope.message import TextBlock
        from agentscope.tool import ToolResponse

        _ToolResponse = ToolResponse
        _TextBlock = TextBlock


MAX_RESULT_CHARS = 2000


def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text)} chars total)"


def _resp(text: str, *, interrupted: bool = False):
    """Wrap ``text`` in an AgentScope ``ToolResponse``.

    Legacy tool closures (in ``vending/tools.py`` and
    ``tools/sql_tools.py``) still return ``ToolResponse`` for
    backward-compatibility with any non-substrate caller. The REPL
    namespace builder unwraps it back to ``str`` for the agent.
    """
    _ensure_imports()
    return _ToolResponse(
        content=[_TextBlock(type="text", text=text)],
        is_interrupted=interrupted,
    )


# ---------------------------------------------------------------------------
# ToolState — shared mutable state across all tool calls within a run
# ---------------------------------------------------------------------------


@dataclass
class ToolState:
    """Mutable state shared by all tool functions within a single run.

    Tool closures (in :mod:`Scroll.benchmarks.vending.tools`) close over a
    single instance, so this is the canonical place for cross-tool
    bookkeeping (lifetime message budget, today's action log, etc.).

    Data storage (``memoryspace``) lives on the agent itself, not here.
    """

    env: object | None = None
    data: object | None = None
    cfg: object | None = None

    # Session lifecycle
    _session_ended: bool = False
    _message_count: int = 0
    _action_log: list[str] = field(default_factory=list)
    _sub_agent_log: list[str] = field(default_factory=list)

    # Token + LM-call accounting (lifetime, persisted in checkpoint).
    # Populated via ``_record_lm_usage`` from the model wrapper —
    # captures BOTH session-loop / probe-time agent calls AND rlm
    # calls (they share the same wrapped model object).
    _lm_calls: int = 0
    _prompt_tokens: int = 0
    _completion_tokens: int = 0

    def reset_session(self) -> None:
        """Reset per-session transient state (called at start of each session).

        ``_message_count`` and the LM-token counters survive — they're
        lifetime budget / cost counters.
        """
        self._session_ended = False
        self._action_log = []
        self._sub_agent_log = []

    # Back-compat alias for callers that still use the old name.
    reset_day = reset_session

    def _tick(self) -> None:
        self._message_count += 1

    def _record_lm_usage(self, response: object) -> None:
        """Pull token usage from a model response; accumulate lifetime totals.

        Defensively probes common shapes: ``response.usage``,
        ``response.metadata.usage``, ``response._raw_response.usage``.
        Falls back to a no-op when no usage info is exposed (e.g.
        non-OpenAI backend, streaming response without final usage).
        """
        p, c = _extract_usage_tokens(response)
        if p == 0 and c == 0:
            return
        self._lm_calls += 1
        self._prompt_tokens += p
        self._completion_tokens += c

    def to_checkpoint(self) -> dict:
        return {
            "_message_count": self._message_count,
            "_lm_calls": self._lm_calls,
            "_prompt_tokens": self._prompt_tokens,
            "_completion_tokens": self._completion_tokens,
        }

    def from_checkpoint(self, data: dict) -> None:
        self._message_count = data.get("_message_count", 0)
        self._lm_calls = data.get("_lm_calls", 0)
        self._prompt_tokens = data.get("_prompt_tokens", 0)
        self._completion_tokens = data.get("_completion_tokens", 0)


def _extract_usage_tokens(response: object) -> tuple[int, int]:
    """Pull ``(prompt_tokens, completion_tokens)`` from a model response.

    Returns ``(0, 0)`` if no usage info is exposed. Tolerates both
    Pydantic-object usage (``.prompt_tokens`` attribute) and dict
    usage (``["prompt_tokens"]`` key); checks several common locations
    where wrappers stash the OpenAI-style usage object.
    """
    candidates: list[object] = []

    # ChatResponse / DictMixin objects override __getattr__ to dict-style
    # lookup, so a missing attribute raises KeyError instead of returning
    # the default. Wrap each probe in try/except so the function never
    # crashes — it just falls through to (0, 0) when nothing matches.
    def _safe_getattr(obj, name):
        try:
            return getattr(obj, name, None)
        except (KeyError, AttributeError):
            return None

    u = _safe_getattr(response, "usage")
    if u is not None:
        candidates.append(u)
    md = _safe_getattr(response, "metadata")
    if md is not None:
        u2 = _safe_getattr(md, "usage")
        if u2 is None and isinstance(md, dict):
            u2 = md.get("usage")
        if u2 is not None:
            candidates.append(u2)
    for raw_attr in ("_raw_response", "raw_response", "raw"):
        raw = _safe_getattr(response, raw_attr)
        if raw is not None:
            u3 = _safe_getattr(raw, "usage")
            if u3 is not None:
                candidates.append(u3)
                break

    # OpenAI-native uses ``prompt_tokens`` / ``completion_tokens``;
    # AgentScope's ``ChatUsage`` uses ``input_tokens`` / ``output_tokens``.
    # Try both, in either attribute or dict form. Use _safe_getattr
    # because DictMixin types raise KeyError (not AttributeError) on
    # missing attrs, which breaks hasattr()/getattr(default=...).
    for u in candidates:
        for prompt_key, completion_key in (
            ("prompt_tokens", "completion_tokens"),
            ("input_tokens", "output_tokens"),
        ):
            p = _safe_getattr(u, prompt_key)
            if p is not None:
                c = _safe_getattr(u, completion_key)
                return (int(p or 0), int((c or 0)))
            if isinstance(u, dict) and prompt_key in u:
                return (
                    int(u.get(prompt_key, 0) or 0),
                    int(u.get(completion_key, 0) or 0),
                )
    return 0, 0
