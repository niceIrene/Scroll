"""CodeAct agent

Subclasses customize via:

  - ``sys_prompt``           (role + strategy, ends up after the substrate prelude)
  - ``init_namespace()``     (objects to expose in the REPL globals)
  - ``namespace_docs()``     (documentation appended to the system prompt)
  - ``clear_namespace_each_turn`` (whether to wipe globals between turns; default True)

History is kept continuous across turns. When total chars exceed
``cfg.context_max_tokens * 4``, ``_compress_history_to_budget`` folds
the oldest batch into a running ``_compressed_summary`` and drops
those messages from ``self._history``. The summary is re-injected as
an extra system message at every ``_call_model``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import signal
import time
from abc import abstractmethod
from functools import wraps
from typing import TYPE_CHECKING, Any

from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

_tracer = otel_trace.get_tracer(__name__)

from Scroll.core._agent import (
    BaseAgent,
    MODEL_CALL_MAX_RETRIES,
    MODEL_CALL_RETRY_DELAY,
    _extract_text_from_response,
    _init_model,
)
from Scroll.core._codeact_runtime import (
    CellResult,
    CellRuntime,
)
from Scroll.core._models import LogEntry
from Scroll.tools._log_handle import LogHandle

if TYPE_CHECKING:
    from Scroll.core._environment import BaseEnvironment

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text / response normalization helpers
# ---------------------------------------------------------------------------

# Probe scorers extract the agent's commitment via this regex. Mirrored
# from ``vending/tasks/rewards.py`` so the substrate's stdout-fallback in
# ``_answer_probe_inner`` can detect when no plain-text ``Answer:`` line
# exists and append the cell's stdout so the scorer can find one there.
_ANSWER_LINE_RE = re.compile(
    r"(?im)^[\s>#\-*•]*\**\s*answer\s*\**\s*[:\-]\**\s*(.+?)\s*$"
)


def split_response_into_text_and_tools(
    response: Any,
) -> tuple[str, list[dict]]:
    """Normalize a model response into ``(text, tool_use_blocks)``.

    The model adapters (OpenAI / Anthropic / DashScope) return slightly
    different shapes. This helper folds them into one canonical pair:

    - ``text``: plain assistant prose (empty string if none)
    - ``tool_use_blocks``: list of ``{"type": "tool_use", "id": ..., "name": ..., "input": {"code": ...}}``

    Anthropic-shape responses with ``content`` as a list of blocks are
    handled natively; OpenAI-shape responses with a string ``content``
    are returned as text-only. Tool calls in either form are surfaced
    as the same Anthropic-style dict so callers can write one parse path.
    """
    text_parts: list[str] = []
    tool_use_blocks: list[dict] = []
    content = getattr(response, "content", None)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", "") or "")
                elif btype == "tool_use":
                    tool_use_blocks.append(block)
            elif isinstance(block, str):
                text_parts.append(block)
    elif isinstance(content, str):
        text_parts.append(content)
    text = "\n".join(p for p in text_parts if p).strip()
    return text, tool_use_blocks


# Refusal phrases the model emits when it "imagines" the search
# returned nothing. Used by abstention-detection (LME probe path).
_REFUSAL_RE = re.compile(
    r"(?i)\b("
    r"don'?t have (that|the|this) information"
    r"|no information( from| in| about| anywhere|.*conversation)"
    r"|chat history does not contain"
    r"|haven'?t discussed"
    r"|never (mentioned|discussed)"
    r"|cannot (answer|determine|find)"
    r"|insufficient information"
    r"|nothing (about|mentioned|discussed)"
    r"|i (have )?searched (thoroughly|across|all|and)"
    r")"
)


def _is_abstention_text(text: str) -> bool:
    """Detect the canonical 'I don't have that information' refusal.

    Used by ``_answer_probe_inner`` to force one retry when the agent
    abstains without running any memoryspace query. Matches the
    benchmark's exact required phrasing plus a couple of common
    near-equivalents the model occasionally drifts to.
    """
    if not text:
        return False
    t = text.lower()
    return (
        "don't have that information from our conversations" in t
        or "do not have that information from our conversations" in t
        or "no information from our conversations" in t
    )


# Per-cell stdout/stderr cap when persisting to ConversationLog. Much
# larger than what the LM ever sees (4 K via to_user_message) but still
# bounded — without this, an agent that prints log entries can grow the
# log geometrically since each LogEntry's persisted output gets re-read
# and re-printed by later cells.
_PERSIST_STDOUT_CAP = 50_000
_PERSIST_STDERR_CAP = 10_000


def _cap(text: str, limit: int, label: str) -> str:
    """Trim ``text`` to ``limit`` chars; append a marker if truncated."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... ({label} truncated for log: {len(text)} chars total)"


# ---------------------------------------------------------------------------
# Agent-facing tool schema
#
# ``execute_python`` is the single tool every CodeAct agent exposes:
# the model writes a Python snippet, the framework runs it in the
# persistent REPL (:class:`CellRuntime`), and the result comes back as
# a ``tool`` message. Subclasses (e.g. ``ScrollAgent`` and its env
# subclasses) inherit ``tools_schema`` via ``CodeActAgent`` — those
# don't add new tool schemas; env-specific action tools (send_email,
# run_sub_agent, …) get bound into the REPL namespace and are invoked
# AS Python from inside an ``execute_python`` cell.
# ---------------------------------------------------------------------------


EXECUTE_PYTHON_TOOL_NAME = "execute_python"

EXECUTE_PYTHON_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": EXECUTE_PYTHON_TOOL_NAME,
        "description": (
            "Execute a Python snippet in the persistent REPL. REPL "
            "globals persist across calls in the same turn, so a later "
            "call can use variables defined earlier. The snippet's "
            "stdout / stderr / any traceback comes back as a "
            "tool_result. Pass ONLY Python source as ``code`` — no "
            "markdown fences, no prose."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python source. Use ``print(x)`` to surface "
                        "values (the REPL does NOT auto-display the "
                        "last expression)."
                    ),
                },
            },
            "required": ["code"],
        },
    },
}


# ---------------------------------------------------------------------------
# LM wrapper + tool closure adapters + span-label op detection
# ---------------------------------------------------------------------------


class _UsageTrackingModel:
    """Thin wrapper around the LLM model that records token usage to
    ``ToolState`` on every call.

    Every CodeActAgent has exactly one wrapped model object, shared by
    its turn-loop calls (via ``_call_model``) AND by any closure that
    captures ``self._model`` (e.g. LME / BEAM's ``_oneshot``). So wrapping
    once at agent ``__init__`` time covers all paths — every LLM
    response that flows through the agent contributes to the
    accumulated counters, with no callers needing to change.
    """

    def __init__(self, inner: object, tool_state) -> None:
        self._inner = inner
        self._state = tool_state

    async def __call__(self, *args, **kwargs):
        response = await self._inner(*args, **kwargs)
        try:
            self._state._record_lm_usage(response)
        except Exception:  # noqa: BLE001
            # Never let usage tracking crash the run.
            pass
        return response

    def __getattr__(self, name):
        # Forward attribute access to the wrapped model so callers
        # that read configuration (model_name, etc.) still work.
        return getattr(self._inner, name)


def _strategy_to_user_text(value: Any) -> str:
    """Normalize a tool/closure return value to a plain string for stdout.

    Existing closures return AgentScope ``ToolResponse`` objects (a
    ``content`` list of ``TextBlock``); when called from the REPL we
    want a plain ``str`` so ``print(read_email())`` works.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    content = getattr(value, "content", None)
    if content is None:
        return str(value)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text", "")
            parts.append(text or "")
        return "\n".join(p for p in parts if p)
    return str(content)


def _wrap_closure_for_repl(fn):
    """Wrap a tool closure so it returns ``str`` instead of ``ToolResponse``.

    Preserves async-ness. The wrapped function keeps the closure's
    name, docstring, and signature (via ``functools.wraps``).
    """
    if inspect.iscoroutinefunction(fn):
        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            return _strategy_to_user_text(await fn(*args, **kwargs))
        return async_wrapper

    @wraps(fn)
    def sync_wrapper(*args, **kwargs):
        return _strategy_to_user_text(fn(*args, **kwargs))
    return sync_wrapper


# Object.method patterns + bare names worth surfacing in span labels.
# Order matters: longer names first so partial overlaps are matched
# correctly by the regex alternation.
_CELL_OP_PATTERN = re.compile(
    r"\b(?:memoryspace|ms|log)\.(\w+)"
    r"|\b(rlm|schema_inspect)\b"
)


def _detect_cell_ops(code: str, max_ops: int = 4) -> list[str]:
    """Return the distinct API ops touched by a code cell.

    Keeps insertion order (stable per-run) and caps at ``max_ops`` so
    span names stay readable. Returns e.g.
    ``['memoryspace.sql_exec', 'memoryspace.vector_query']`` — the
    fully-qualified form for object methods, bare for free functions.
    """
    ops: list[str] = []
    seen: set[str] = set()
    for m in _CELL_OP_PATTERN.finditer(code):
        # Group 0 is the full match (including object prefix); use it
        # directly so "memoryspace.sql_exec" stays distinct from
        # "ms.sql_exec" even though they alias the same object.
        op = m.group(0)
        if op not in seen:
            seen.add(op)
            ops.append(op)
        if len(ops) >= max_ops:
            break
    return ops


class CodeActAgent(BaseAgent):
    """RLM-style agent over a persistent Python REPL.

    Subclasses must override:
      - ``sys_prompt`` — class attribute: role + strategy text.
      - ``init_namespace(self) -> dict`` — objects to bind into REPL globals.
      - ``namespace_docs(self) -> str`` — appended to the system prompt to
        teach the LM what's in its namespace.
    """

    sys_prompt: str = ""
    clear_namespace_each_turn: bool = True
    # Probe-time iteration cap. Tools-heavy strategies (memoryspace + log
    # querying) may need more cells than mem-only baselines — bump this
    # in subclasses if the agent can't reliably finish recall in 5 cells.
    probe_max_iters: int = 5
    # Tool schemas passed to ``self._model(..., tools=tools_schema)``.
    # Default: just ``execute_python`` (the Python REPL primitive).
    # Subclasses with additional tools can extend this list.
    tools_schema: list[dict] = [EXECUTE_PYTHON_TOOL_SCHEMA]

    def __init__(self, cfg, storage, data) -> None:
        from Scroll.core._tool_state import ToolState

        self.cfg = cfg
        self.log = storage  # ConversationLog (E)
        self.data = data
        self.last_context: list[str] = []
        self._current_turn: int = 0
        self._tool_state = ToolState(data=data, cfg=cfg)
        self._iter_count: int = 0

        raw_model, self._msg_cls = _init_model(self.cfg)
        # Wrap so every model call (turn-loop + closures that capture
        # ``self._model``, e.g. LME / BEAM's ``_oneshot``) records token
        # usage onto ``self._tool_state`` for benchmark efficiency reporting.
        self._model = _UsageTrackingModel(raw_model, self._tool_state)
        self._runtime = CellRuntime()
        self._history: list[dict] = []  # OpenAI-style {role, content}
        # Sliding-mode summary of dropped messages. Empty for step
        # mode (the default). Auto-spliced into the prompt by
        # ``_call_model`` when populated.
        self._compressed_summary: str = ""

        # Subclasses may stash extra state on self before
        # ``super().__init__`` so it's available to ``init_namespace``.
        self._rebuild_namespace()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def init_namespace(self) -> dict:
        """Return a dict of objects to bind into the REPL globals.

        Subclass override. Should always include ``log`` (a
        ``LogHandle``) and any env-tool callables; may include domain
        objects like ``memoryspace``, ``rlm``.
        """
        return {"log": LogHandle(self.log)}

    def namespace_docs(self) -> str:
        """Return Markdown-ish docs describing the namespace.

        Appended to the system prompt so the LM knows what to call.
        """
        return ""

    def extra_system_prompt(self) -> str:
        """Return additional system-prompt text appended after namespace docs.

        Subclass hook for content that depends on env-side state at
        run-time (e.g. tau-bench appends the active domain's policy.md
        here so the agent always has the operating manual in context
        without having to print it from inside a cell).
        """
        return ""

    # ------------------------------------------------------------------
    # Per-turn prompts
    # ------------------------------------------------------------------

    @abstractmethod
    def turn_prompt(self, turn_idx: int) -> str:
        """Return the user-message body for turn ``turn_idx``.

        Env-specific — each benchmark phrases its turn boundary
        differently (vending = calendar day, LME/BEAM = probe vs.
        ingestion turn), so this has no env-agnostic default.
        """

    def _compose_turn_user_msg(self, turn_idx: int) -> str:
        """Build the per-turn user message: namespace reminder + body.

        Subclasses that need to splice env-side state in front of the
        body (e.g. VendingAgent's pending outcomes / briefing) override
        :meth:`_extra_prefix_sections`.
        """
        body = self.turn_prompt(turn_idx)
        prefix_sections: list[str] = []
        ns_reminder = self._namespace_reminder()
        if ns_reminder:
            prefix_sections.append(ns_reminder)
        prefix_sections.extend(self._extra_prefix_sections(turn_idx))
        if prefix_sections:
            body = "\n\n".join(prefix_sections) + "\n\n" + body
        return body

    def _extra_prefix_sections(self, turn_idx: int) -> list[str]:
        """Env-specific prefix sections to splice before ``turn_prompt``.

        Default: none. Override to render buffered env state (e.g.
        prior-turn outcomes, datasource briefings) into the user
        message. Called once per turn from :meth:`_compose_turn_user_msg`;
        implementations should also clear any consumed state.
        """
        return []

    def _namespace_reminder(self) -> str:
        """One-line listing of REPL globals bound inside ``execute_python``.

        Prepended to every turn's user message so the model is far less
        likely to invent ``view_storage()`` or ``from tools import ...``
        when writing the ``code`` argument of an ``execute_python`` tool
        call. Listing is short — names only — to keep token cost negligible.
        """
        keep_objects = {"log", "memoryspace", "ms", "rlm", "today"}
        names: list[str] = []
        for name, val in self._runtime.globals.items():
            if name.startswith("_") or name == "__builtins__":
                continue
            if callable(val) or name in keep_objects:
                names.append(name)
        if not names:
            return ""
        names.sort()
        return (
            f"Bound in the ``execute_python`` REPL (no ``import`` needed): "
            f"{', '.join(names)}."
        )

    def _env_endgame_prompt(self) -> str:
        """Per-env "RUN STRUCTURE" section — sourced from the live env.

        The substrate is environment-neutral; each env (vending,
        taubench, longmemeval) overrides
        :meth:`BaseEnvironment.substrate_endgame_prompt` to spell out
        what one iteration is, what ``today`` means, and how to end an
        iteration. Returns ``""`` if the env isn't wired up yet
        (constructor path) — the prompt is only consumed inside
        ``run_turn``, where env is set.
        """
        env = getattr(self, "_tool_state", None) and self._tool_state.env
        if env is None:
            return ""
        try:
            return (env.substrate_endgame_prompt() or "").strip()
        except Exception:  # noqa: BLE001
            return ""

    def _env_probe_prompt(self) -> str:
        """Per-env probe-mode format rules — sourced from the live env.

        Mirrors :meth:`_env_endgame_prompt` but for probe mode: each
        env supplies its own scorer-shaped format block via
        :meth:`BaseEnvironment.probe_substrate_prompt`. Tau-bench's
        probes are passive and never call ``answer_probe`` so this
        path is unused there; vending and longmemeval have meaningfully
        different format contracts (deterministic regex vs LLM judge)
        and override.
        """
        env = getattr(self, "_tool_state", None) and self._tool_state.env
        if env is None:
            return ""
        try:
            return (env.probe_substrate_prompt() or "").strip()
        except Exception:  # noqa: BLE001
            return ""

    def _full_sys_prompt(self) -> str:
        """Compose env endgame + agent role + namespace docs + extra."""
        parts: list[str] = []
        endgame = self._env_endgame_prompt()
        if endgame:
            parts.append(endgame)
        parts.append(self.sys_prompt.strip())
        ns_docs = self.namespace_docs().strip()
        if ns_docs:
            parts.append(ns_docs)
        extra = self.extra_system_prompt().strip()
        if extra:
            parts.append(extra)
        return "\n\n".join(p for p in parts if p)

    def _probe_sys_prompt(self) -> str:
        """System prompt used while answering a probe.

        Default: identical to :meth:`_full_sys_prompt` (no swap; probe
        runs in the same system-prompt context as the turn loop).
        Subclass override hook for envs that want different rules at
        probe time — LME / BEAM override this to drop the env's
        "RUN STRUCTURE" endgame (irrelevant during probe) and inject
        any per-env probe-format hints.
        """
        return self._full_sys_prompt()

    # ------------------------------------------------------------------
    # Namespace lifecycle
    # ------------------------------------------------------------------

    def _rebuild_namespace(self) -> None:
        """Reset the REPL globals to a freshly built namespace.

        Always injects ``today`` (the current 1-indexed turn number) so
        agent code can write ``log.range_by_day(today - 7, today - 1)``
        without parsing the turn index out of the briefing text.
        """
        ns = self.init_namespace()
        ns.setdefault("today", self._current_turn)
        self._runtime.reset(initial_globals=ns)

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------

    @property
    def message_count(self) -> int:
        return self._tool_state._message_count

    @property
    def turn_ended(self) -> bool:
        return self._tool_state._turn_ended

    def receive_outcomes(self, turn_idx: int, logs: list[str]) -> None:
        # No-op concrete impl to satisfy BaseAgent's abstract contract.
        # Subclasses that need to render outcomes into the next turn's
        # prompt (e.g. VendingAgent) override this to buffer ``logs``.
        pass

    def receive_context(self, turn_idx: int, notes: list[str]) -> None:
        # No-op concrete impl to satisfy BaseAgent's abstract contract.
        # Subclasses that need to render notes into the next turn's
        # prompt (e.g. VendingAgent) override this to buffer ``notes``.
        pass

    def run_turn(self, env) -> list[str]:
        turn_idx = env.turn_idx
        self._current_turn = turn_idx + 1
        self._tool_state.env = env
        self._tool_state.reset_turn()

        if self.clear_namespace_each_turn:
            self._rebuild_namespace()
        else:
            # Even when globals persist across days, refresh ``today``
            # so the agent's worked-example pattern (``log.range_by_day(
            # today - 7, today - 1)``) keeps pointing at the real turn.
            self._runtime.update_globals(today=self._current_turn)

        prompt = self._compose_turn_user_msg(turn_idx + 1)

        # Keep history across turns; old messages get summarized into
        # ``_compressed_summary`` and dropped by
        # ``_compress_history_to_budget`` once total chars exceed budget.
        # The summary is re-injected at every ``_call_model``.
        if not self._history or self._history[0].get("role") != "system":
            self._history = [
                {"role": "system", "content": self._full_sys_prompt()},
            ]
        self._history.append({"role": "user", "content": prompt})
        # Mirror the turn-start user prompt into E (matches the legacy
        # SlidingMemory.add path).
        self.log.append(LogEntry.make(
            turn_idx=self._current_turn,
            role="user",
            content=prompt,
        ))

        # Track Ctrl+C across asyncio.run boundaries (same pattern as LLMAgent).
        interrupted = False
        prev_handler = signal.getsignal(signal.SIGINT)

        def _flag_interrupt(sig, frame):
            nonlocal interrupted
            interrupted = True
            if callable(prev_handler) and prev_handler not in (
                signal.SIG_IGN,
                signal.SIG_DFL,
            ):
                prev_handler(sig, frame)
            else:
                raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _flag_interrupt)
        last_err: Exception | None = None
        try:
            for attempt in range(1, MODEL_CALL_MAX_RETRIES + 1):
                try:
                    asyncio.run(self._run_turn_inner())
                    last_err = None
                    break
                except KeyboardInterrupt:
                    interrupted = True
                    break
                except Exception as e:
                    last_err = e
                    if attempt < MODEL_CALL_MAX_RETRIES:
                        _log.warning(
                            "Model call failed (attempt %d/%d): %s — retrying in %ds",
                            attempt,
                            MODEL_CALL_MAX_RETRIES,
                            e,
                            MODEL_CALL_RETRY_DELAY,
                        )
                        time.sleep(MODEL_CALL_RETRY_DELAY)
            if last_err is not None:
                _log.error("agent_error on turn %d: %s", turn_idx, last_err)
        finally:
            signal.signal(signal.SIGINT, prev_handler)

        if interrupted:
            raise KeyboardInterrupt

        actions = list(self._tool_state._action_log)
        self.last_context = self._snapshot_history()

        if not self._tool_state._turn_ended:
            self._tool_state._turn_ended = True
        return actions

    async def _run_turn_inner(self) -> None:
        """Turn-loop inner: one user message → assistant tool-use loop.

        The on-history wire shape is OpenAI Chat Completions tool-call
        format (assistant messages carry ``tool_calls``; tool results
        are ``role="tool"`` keyed by ``tool_call_id``). Per-provider
        asymmetries (OpenAI Responses API, Anthropic Messages) are
        absorbed by the adapters in ``_chat_model_adapters``, so this
        loop sees one shape regardless of provider.

        - Model sees ``self.tools_schema`` (typically just
          ``execute_python``) and responds with either plain content
          (= end of turn) or one+ tool-use blocks (decoded from the
          provider via the adapter).
        - Each tool-use block is executed in the persistent REPL and
          the result is appended as a ``role="tool"`` message.
        - The loop iterates until the model emits NO tool calls (= done)
          or ``max_iters_per_turn`` is hit.
        """
        max_iters = self.cfg.max_iters_per_turn
        ended = False
        for it in range(max_iters):
            self._iter_count += 1
            await self._compress_history_to_budget()
            override = await self._on_pre_cell_async()

            # Force a tool call on the first iter so the model can't
            # short-circuit the turn by emitting Python-as-prose; from
            # the second iter on, "auto" lets the model commit a final
            # no-tool-call response when its work for the day is done.
            tool_choice = "required" if it == 0 else None
            response = await self._call_model(
                override, tool_choice_override=tool_choice,
            )
            text, tool_use_blocks = split_response_into_text_and_tools(response)
            if not text and not tool_use_blocks:
                # Empty response — treat as no-op end.
                break

            # Build assistant message (content + tool_calls) and mirror
            # into both LLM context (self._history) and event log E.
            assistant_msg: dict = {
                "role": "assistant",
                "content": text or "",
            }
            tool_calls_payload: list[dict] = []
            for idx, block in enumerate(tool_use_blocks):
                tc_id = block.get("id") or (
                    f"tc-d{self._current_turn}-{it}-{idx}"
                )
                raw_input = block.get("input")
                if isinstance(raw_input, dict):
                    code = raw_input.get("code", "") or ""
                else:
                    raw_str = block.get("raw_input") or ""
                    try:
                        parsed = (
                            json.loads(raw_str) if raw_str else {}
                        )
                        code = (
                            parsed.get("code", "")
                            if isinstance(parsed, dict) else ""
                        )
                    except Exception:  # noqa: BLE001
                        code = ""
                tool_calls_payload.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "execute_python",
                        "arguments": json.dumps({"code": code}),
                    },
                })
            if tool_calls_payload:
                assistant_msg["tool_calls"] = tool_calls_payload
            self._history.append(assistant_msg)

            self.log.append(LogEntry.make(
                turn_idx=self._current_turn,
                role="assistant",
                content=text,
                metadata={
                    "kind": "lm_turn",
                    "iter": it,
                    "has_tool_calls": bool(tool_calls_payload),
                },
            ))

            if not tool_use_blocks:
                # No tool call — agent has nothing more to do this
                # turn. End of turn.
                break

            # Execute each tool call, append a role=tool message per result.
            for idx, block in enumerate(tool_use_blocks):
                tc_id = tool_calls_payload[idx]["id"]
                tc_args = json.loads(
                    tool_calls_payload[idx]["function"]["arguments"]
                )
                code = tc_args.get("code", "") or ""
                if not code.strip():
                    self._history.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": (
                            "[error: execute_python called with empty "
                            "code. Pass actual Python source as the "
                            "``code`` argument.]"
                        ),
                    })
                    continue

                ops = _detect_cell_ops(code)
                ops_label = ", ".join(ops) if ops else "no-ops"
                line_count = code.count("\n") + 1
                span_name = (
                    f"code_exec d{self._current_turn}.i{it}.{idx} "
                    f"({line_count}L) [{ops_label}]"
                )
                code_preview = code if len(code) <= 3000 else (
                    code[:2900] + "\n\n# ... [truncated; see cell.code_full]"
                )
                with _tracer.start_as_current_span(span_name) as span:
                    span.set_attributes({
                        "tool.name": "execute_python",
                        "cell.turn": self._current_turn,
                        "cell.iter": it,
                        "cell.tool_idx": idx,
                        "cell.iter_lifetime": self._iter_count,
                        "cell.ops": ",".join(ops),
                        "cell.line_count": line_count,
                        "cell.code_chars": len(code),
                        "cell.code_full": code,
                        "input.value": code_preview,
                        "input.mime_type": "text/plain",
                    })
                    result = await self._runtime.execute_cell(code)
                    output_parts: list[str] = []
                    if result.stdout:
                        output_parts.append(result.stdout)
                    if result.stderr:
                        output_parts.append("[stderr]\n" + result.stderr)
                    if result.exception:
                        output_parts.append("[exception]\n" + result.exception)
                    summary_bits = [
                        f"stdout={len(result.stdout or '')}c",
                        f"ops=[{ops_label}]",
                    ]
                    if result.turn_ended:
                        summary_bits.append("turn_ended=True")
                    if result.exception:
                        summary_bits.append("EXCEPTION")
                    cell_summary = " | ".join(summary_bits)
                    output_preview = (
                        ("\n\n".join(output_parts))[:2000]
                        if output_parts
                        else (
                            f"[cell ran, no stdout]\n\n"
                            f"summary: {cell_summary}\n\n"
                            f"code (first 200c):\n{code[:200]}"
                        )
                    )
                    stdout_preview = (
                        (result.stdout or "")[:200].replace("\n", " ⏎ ")
                    )
                    stdout_lines = (
                        result.stdout.count("\n") if result.stdout else 0
                    ) + (1 if result.stdout else 0)
                    span.set_attributes({
                        "openinference.span.kind": "TOOL",
                        "output.value": output_preview,
                        "output.mime_type": "text/plain",
                        "cell.summary": cell_summary,
                        "cell.stdout_preview": stdout_preview or "[empty]",
                        "cell.stdout_lines": stdout_lines,
                        "cell.stdout_chars": len(result.stdout or ""),
                        "code_exec.turn_ended": result.turn_ended,
                        "code_exec.has_exception": (
                            result.exception is not None
                        ),
                    })
                    if result.exc is not None:
                        span.record_exception(result.exc)
                        span.set_status(Status(
                            StatusCode.ERROR,
                            f"{type(result.exc).__name__}: {result.exc}"[:200],
                        ))
                        span.set_attributes({
                            "exception.type": type(result.exc).__name__,
                            "exception.message": str(result.exc)[:500],
                        })
                self._record_cell(code, result, it)

                self._history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result.to_user_message(),
                })

                if result.turn_ended:
                    ended = True
                    break

            if ended:
                break

        await self._on_turn_end_async()
        if not ended:
            # Either fell off the iteration cap, or model emitted a
            # no-tool-call response (= "I'm done with this turn").
            self._tool_state._turn_ended = True

    async def _call_model(
        self,
        messages_override: list | None = None,
        *,
        tool_choice_override: str | None = None,
    ):
        """One model call against the current history (or override).

        For sliding mode, splices ``_compressed_summary`` in as an extra
        system message after index 0 so dropped-and-summarized history
        stays visible to the LM. No-op when summary is empty or
        ``messages_override`` is provided (callers managing their own
        message list handle injection themselves).

        Per-call quota-aware retry: Dashscope occasionally returns 429
        / insufficient_quota under load. Turn-level retry (in run_turn)
        is too coarse for these — restarting a whole turn on a single
        flaky call wastes work and re-fires the same throttle. So we
        retry HERE with longer backoff (15s, 30s) on quota-class errors
        only; non-quota exceptions bubble to the turn-level retry as
        before.
        """
        msgs = messages_override if messages_override is not None else self._history
        if (
            self._compressed_summary
            and messages_override is None
            and msgs
            and msgs[0].get("role") == "system"
        ):
            msgs = [
                msgs[0],
                {
                    "role": "system",
                    "content": (
                        "Earlier conversation summary (older turns "
                        "compacted to save context):\n"
                        + self._compressed_summary
                    ),
                },
                *msgs[1:],
            ]
        tools = self.tools_schema or None
        if tool_choice_override is not None:
            tool_choice = tool_choice_override
        else:
            tool_choice = "auto" if tools else None
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                return await self._model(msgs, tools=tools, tool_choice=tool_choice)
            except Exception as e:  # noqa: BLE001
                last_err = e
                err_str = str(e).lower()
                is_quota = any(
                    s in err_str for s in ("429", "quota", "rate", "insufficient")
                )
                if attempt < 3 and is_quota:
                    delay = 15 * attempt  # 15s, 30s
                    _log.warning(
                        "Model call quota error (attempt %d/3): %.150s — backing off %ds",
                        attempt, e, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
        # Unreachable in practice (loop either returns or raises), but
        # keep mypy happy.
        raise last_err  # type: ignore[misc]

    async def _on_pre_cell_async(self) -> list | None:
        """Hook fired before each model call.

        Returns either a message list to send for THIS turn (in
        OpenAI-dict form), or ``None`` to use ``self._history``
        unchanged. Default: ``None`` (no-op). Subclasses may override
        to inject extra context into the system prompt per call
        without permanently rewriting history.
        """
        return None

    async def _on_turn_end_async(self) -> None:
        """Hook fired after the per-turn loop finishes (before asyncio.run exits).

        Default no-op. Subclasses may override to flush per-turn
        state into external storage before the next turn's
        namespace wipe.
        """
        return None

    # ------------------------------------------------------------------
    # Logging / mirroring
    # ------------------------------------------------------------------

    def _record_cell(self, code: str, result: CellResult, iter_idx: int) -> None:
        """Mirror a cell run into ConversationLog (E).

        Layout matches the legacy ``tool_use`` / ``tool_result`` shape
        so visualizations and trajectory dumps don't have to special-
        case the substrate.

        Each cell's stdout is capped at ``_PERSIST_STDOUT_CAP`` chars
        before being written to the log. Without the cap, an RLM-style
        agent that does ``for e in log: print(e)`` re-serializes prior
        cells' (already-stored) outputs into its own stdout, and the
        persisted entries grow geometrically — runs eventually OOM or
        spend all their time in JSON encoding (see incident:
        ``output/rlm_1_351d966a/`` where line sizes hit 422 MB after
        9 days). The cap is much larger than what the LM ever sees
        (``CellResult.to_user_message`` truncates to 4 K) so the log
        remains useful for offline inspection without runaway growth.
        """
        cell_id = f"cell-{self._current_turn}-{iter_idx}"
        self.log.append(LogEntry.make(
            turn_idx=self._current_turn,
            role="assistant",
            tool_call={
                "id": cell_id,
                "name": "code_exec",
                "arguments": {"code": code},
            },
            metadata={"kind": "code"},
        ))
        stdout_for_log = _cap(result.stdout or "", _PERSIST_STDOUT_CAP, "stdout")
        stderr_for_log = _cap(result.stderr or "", _PERSIST_STDERR_CAP, "stderr")
        output_blob = stdout_for_log + (
            ("\n[stderr]\n" + stderr_for_log) if stderr_for_log else ""
        ) + (
            ("\n[exception]\n" + result.exception) if result.exception else ""
        ) + (
            "\n[turn ended]" if result.turn_ended else ""
        )
        self.log.append(LogEntry.make(
            turn_idx=self._current_turn,
            role="tool",
            tool_result={
                "id": cell_id,
                "name": "code_exec",
                "output": output_blob.strip(),
            },
            metadata={"kind": "stdout", "turn_ended": result.turn_ended},
        ))

    def _record_probe_cell(
        self, code: str, result: CellResult, iter_idx: int,
        thought: str = "",
    ) -> None:
        """Mirror a probe-time cell run into ConversationLog.

        Same shape as :meth:`_record_cell` but tagged with
        ``metadata.kind`` of ``"probe_code"`` / ``"probe_stdout"`` and
        a ``probe-cell-`` id prefix so probe lookups are distinguishable
        from turn-loop cells. ``extract_tool_trace`` reads from
        ``agent.log.entries[prev_log_count:]`` to populate
        ``ProbeResult.tool_trace``; without this mirror the trace is
        always empty.

        ``thought`` carries the agent's reasoning text emitted alongside
        the tool call (the assistant message's plain-text content
        block, written between cells). It's stored on the probe_code
        entry's ``content`` field so trajectory distillation captures
        THINK / DO / SEE triples, not just DO / SEE.
        """
        cell_id = f"probe-cell-{self._current_turn}-{iter_idx}"
        self.log.append(LogEntry.make(
            turn_idx=self._current_turn,
            role="assistant",
            content=thought or "",
            tool_call={
                "id": cell_id,
                "name": "code_exec",
                "arguments": {"code": code},
            },
            metadata={"kind": "probe_code"},
        ))
        stdout_for_log = _cap(result.stdout or "", _PERSIST_STDOUT_CAP, "stdout")
        stderr_for_log = _cap(result.stderr or "", _PERSIST_STDERR_CAP, "stderr")
        output_blob = stdout_for_log + (
            ("\n[stderr]\n" + stderr_for_log) if stderr_for_log else ""
        ) + (
            ("\n[exception]\n" + result.exception) if result.exception else ""
        )
        self.log.append(LogEntry.make(
            turn_idx=self._current_turn,
            role="tool",
            tool_result={
                "id": cell_id,
                "name": "code_exec",
                "output": output_blob.strip(),
            },
            metadata={"kind": "probe_stdout"},
        ))

    def _snapshot_history(self) -> list[str]:
        """Return a human-readable summary of this turn's LM messages."""
        lines: list[str] = []
        for msg in self._history:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            preview = content if isinstance(content, str) else str(content)
            preview = preview.strip().replace("\n", " ")[:200]
            lines.append(f"[{role}] {preview}")
        return lines

    @staticmethod
    def _estimate_msg_chars(msg: dict) -> int:
        """Rough char size of an OpenAI-style ``{role, content}`` dict."""
        size = len(str(msg.get("role", ""))) + 4
        content = msg.get("content", "")
        if isinstance(content, str):
            size += len(content)
        else:
            size += len(str(content))
        return size

    @staticmethod
    def _msg_to_text(msg: dict) -> str:
        """Flatten a history dict to a single line for the summary prompt."""
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        return f"[{role}] {content}"

    def _total_history_chars(self) -> int:
        return sum(self._estimate_msg_chars(m) for m in self._history)

    def _collect_oldest_batch(self, target_chars: int) -> list[int]:
        """Indices of oldest non-system messages totaling ~``target_chars``."""
        indices: list[int] = []
        chars = 0
        for i, m in enumerate(self._history):
            if i == 0 and m.get("role") == "system":
                continue
            indices.append(i)
            chars += self._estimate_msg_chars(m)
            if chars >= target_chars:
                break
        return indices

    async def _compress_history_to_budget(self) -> None:
        """Summarize oldest non-system messages, then drop them.

        Called as a pre-call hook before every model invocation; no-op
        when the total history is under the char budget. When over,
        picks the oldest batch (~budget / 4), asks the model to fold it
        into the running ``_compressed_summary``, then deletes those
        messages from ``_history``. The summary is re-injected as an
        extra system message at every ``_call_model``, so the LM still
        sees old context — just compacted.

        On LLM failure: messages are still dropped (better to truncate
        than to grow unbounded).
        """
        budget = max(self.cfg.context_max_tokens, 4096) * 4
        if self._total_history_chars() <= budget:
            return

        indices = self._collect_oldest_batch(target_chars=budget // 4)
        if not indices:
            return

        batch_msgs = [self._history[i] for i in indices]
        batch_text = "\n".join(self._msg_to_text(m) for m in batch_msgs)
        if len(batch_text) > 8000:
            batch_text = batch_text[:8000] + "\n... (truncated)"

        prompt_parts = [
            "Summarize the following agent conversation history into a "
            "compact paragraph. Preserve key facts, numbers, decisions, "
            "supplier prices, and action outcomes.\n",
        ]
        if self._compressed_summary:
            prompt_parts.append(f"\nPrevious summary:\n{self._compressed_summary}\n")
        prompt_parts.append(f"\nNew messages to incorporate:\n{batch_text}")
        prompt = "".join(prompt_parts)

        with _tracer.start_as_current_span("memory.compress") as span:
            span.set_attribute("memory.msgs_to_drop", len(indices))
            try:
                response = await self._model([{"role": "user", "content": prompt}])
                summary_text = _extract_text_from_response(response).strip()
                if summary_text:
                    self._compressed_summary = summary_text
                    span.set_attribute("memory.summary_chars", len(summary_text))
            except Exception:  # noqa: BLE001
                # Summarization failed; still drop the messages so the
                # next model call doesn't blow the context.
                _log.warning(
                    "Sliding memory compression failed — dropping batch without summary",
                    exc_info=True,
                )
                span.set_attribute("memory.summary_failed", True)

        # Delete in reverse so earlier indices stay valid.
        for i in sorted(indices, reverse=True):
            del self._history[i]

    # ------------------------------------------------------------------
    # Probes
    # ------------------------------------------------------------------

    def answer_probe(self, question: str) -> str:
        """Run the REPL loop on a probe question.

        Lets the LM optionally write code (e.g. ``log.search("…")`` or
        ``db.query("…")``) before committing to ``Answer: …``. Capped
        at a small number of cells to avoid runaway probes.

        Probe-mode adjustments:

        - The system prompt is swapped to ``_probe_sys_prompt()``,
          which replaces the turn-loop substrate rules (always reply
          with a code block; no-code response ends the day) with
          probe-specific rules (plain-text ``Answer:`` required; do not
          end the turn). Without this swap the model can drop into
          a no-code turn intending to end the day mid-probe, before it
          has committed an answer.
        - History handling is unified across memory modes: probes
          run *inside the same agent session* — we keep ``self._history``
          (current turn's messages: system + briefing + cell traces),
          replace the system prompt at index 0 with the probe-mode
          version, and append the question. This way step-mode agents
          see the in-flight reasoning context while answering (matching what
          sliding always did), and the prompt's "you must query for
          factual probes" rule pushes them to re-fetch from memoryspace
          / log instead of trusting mid-turn intermediate values
          they might recall. For sliding we additionally trim to
          context budget since its history
          spans days. ``self._history`` is restored after the probe
          so the Q+A doesn't leak into working memory.
        """
        old_day_ended = self._tool_state._turn_ended
        self._tool_state._turn_ended = False

        history_backup = list(self._history)
        probe_sys_msg = {
            "role": "system",
            "content": self._probe_sys_prompt(),
        }

        # Bind ``probe_question`` so REPL code can embed it directly in
        # rlm prompts (e.g. ``await rlm(query=probe_question,
        # context=body)``). Prompt examples reference this name; without
        # the binding weak models hit ``NameError`` on copy-pasted code.
        old_pq = self._runtime.globals.get("probe_question")
        pq_was_present = "probe_question" in self._runtime.globals
        self._runtime.globals["probe_question"] = question

        try:
            new_history = list(self._history)
            if new_history and new_history[0].get("role") == "system":
                new_history[0] = probe_sys_msg
            else:
                new_history = [probe_sys_msg] + new_history
            new_history.append({"role": "user", "content": question})
            self._history = new_history
            answer = asyncio.run(self._answer_probe_inner(max_iters=self.probe_max_iters))
        except Exception as e:  # noqa: BLE001
            answer = f"[probe error: {e}]"
        finally:
            self._history = history_backup
            self._tool_state._turn_ended = old_day_ended
            if pq_was_present:
                self._runtime.globals["probe_question"] = old_pq
            else:
                self._runtime.globals.pop("probe_question", None)
        return answer

    async def _answer_probe_inner(self, max_iters: int) -> str:
        """Probe answering loop using the OpenAI tool-call protocol.

        Differs from :meth:`_run_turn_inner` in two probe-specific
        ways:
          1. Force one retry on a zero-tool-call abstention so the
             agent has to actually query memoryspace before refusing.
          2. Stdout fallback: if the agent puts ``Answer:`` inside a
             ``print(...)`` instead of a plain-text reply, append the
             last cell's stdout so the regex scorer can find the line.

        Other probe-specific behaviors (LME's grace turn, evidence
        anchoring, etc.) live in env-specific overrides.
        """
        last_text = ""
        last_stdout = ""
        cells_executed = 0
        zero_cell_abstain_forced = False
        for it in range(max_iters):
            await self._compress_history_to_budget()
            override = await self._on_pre_cell_async()
            response = await self._call_model(override)
            text, tool_use_blocks = split_response_into_text_and_tools(response)
            if not text and not tool_use_blocks:
                break

            assistant_msg: dict = {
                "role": "assistant",
                "content": text or "",
            }
            tool_calls_payload: list[dict] = []
            for idx, block in enumerate(tool_use_blocks):
                tc_id = block.get("id") or (
                    f"probe-tc-d{self._current_turn}-{it}-{idx}"
                )
                raw_input = block.get("input")
                if isinstance(raw_input, dict):
                    code = raw_input.get("code", "") or ""
                else:
                    raw_str = block.get("raw_input") or ""
                    try:
                        parsed = (
                            json.loads(raw_str) if raw_str else {}
                        )
                        code = (
                            parsed.get("code", "")
                            if isinstance(parsed, dict) else ""
                        )
                    except Exception:  # noqa: BLE001
                        code = ""
                tool_calls_payload.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "execute_python",
                        "arguments": json.dumps({"code": code}),
                    },
                })
            if tool_calls_payload:
                assistant_msg["tool_calls"] = tool_calls_payload
            self._history.append(assistant_msg)
            last_text = text

            if not tool_use_blocks:
                # Zero-tool-call response. If the agent abstained without
                # ever querying memoryspace, reject the answer once and
                # require at least one probe call.
                if (
                    cells_executed == 0
                    and not zero_cell_abstain_forced
                    and _is_abstention_text(text)
                ):
                    zero_cell_abstain_forced = True
                    self._history.append({
                        "role": "user",
                        "content": (
                            "[ABSTENTION REJECTED — zero tool calls.] "
                            "You answered \"I don't have that information\" "
                            "without calling ``execute_python`` for any "
                            "memoryspace query. Per the SEARCH BEFORE YOU "
                            "REFUSE rule, a zero-call abstention is wrong by "
                            "default — the data may be in your store even "
                            "when nothing in your recent chat history "
                            "mentions it.\n\n"
                            "Make at least ONE ``execute_python`` call now: "
                            "``ms.sql_exec(...)`` or ``log.semantic_search(...)`` "
                            "or ``log.findall(...)`` using keywords from the "
                            "probe question. If the query genuinely returns "
                            "nothing relevant, you may abstain in your NEXT "
                            "response based on what you observed."
                        ),
                    })
                    last_text = ""  # discard the rejected abstention
                    continue
                break

            turn_ended = False
            for idx, block in enumerate(tool_use_blocks):
                tc_id = tool_calls_payload[idx]["id"]
                tc_args = json.loads(
                    tool_calls_payload[idx]["function"]["arguments"]
                )
                code = tc_args.get("code", "") or ""
                if not code.strip():
                    self._history.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": (
                            "[error: execute_python called with empty "
                            "code. Pass actual Python source as the "
                            "``code`` argument.]"
                        ),
                    })
                    continue

                cells_executed += 1
                ops = _detect_cell_ops(code)
                ops_label = ", ".join(ops) if ops else "no-ops"
                line_count = code.count("\n") + 1
                span_name = (
                    f"probe_cell d{self._current_turn}.i{it}.{idx} "
                    f"({line_count}L) [{ops_label}]"
                )
                code_preview = code if len(code) <= 3000 else (
                    code[:2900] + "\n\n# ... [truncated; see cell.code_full]"
                )
                with _tracer.start_as_current_span(span_name) as span:
                    span.set_attributes({
                        "tool.name": "execute_python",
                        "cell.turn": self._current_turn,
                        "cell.iter": it,
                        "cell.tool_idx": idx,
                        "cell.kind": "probe",
                        "cell.ops": ",".join(ops),
                        "cell.line_count": line_count,
                        "cell.code_chars": len(code),
                        "cell.code_full": code,
                        "input.value": code_preview,
                        "input.mime_type": "text/plain",
                    })
                    result = await self._runtime.execute_cell(code)
                    output_parts: list[str] = []
                    if result.stdout:
                        output_parts.append(result.stdout)
                    if result.stderr:
                        output_parts.append("[stderr]\n" + result.stderr)
                    if result.exception:
                        output_parts.append("[exception]\n" + result.exception)
                    summary_bits = [
                        f"stdout={len(result.stdout or '')}c",
                        f"ops=[{ops_label}]",
                    ]
                    if result.exception:
                        summary_bits.append("EXCEPTION")
                    cell_summary = " | ".join(summary_bits)
                    output_preview = (
                        ("\n\n".join(output_parts))[:2000]
                        if output_parts
                        else (
                            f"[probe cell ran, no stdout]\n\n"
                            f"summary: {cell_summary}\n\n"
                            f"code (first 200c):\n{code[:200]}"
                        )
                    )
                    stdout_preview = (
                        (result.stdout or "")[:200].replace("\n", " ⏎ ")
                    )
                    stdout_lines = (
                        result.stdout.count("\n") if result.stdout else 0
                    ) + (1 if result.stdout else 0)
                    span.set_attributes({
                        "openinference.span.kind": "TOOL",
                        "output.value": output_preview,
                        "output.mime_type": "text/plain",
                        "cell.summary": cell_summary,
                        "cell.stdout_preview": stdout_preview or "[empty]",
                        "cell.stdout_lines": stdout_lines,
                        "cell.stdout_chars": len(result.stdout or ""),
                        "code_exec.has_exception": (
                            result.exception is not None
                        ),
                    })
                    if result.exc is not None:
                        span.record_exception(result.exc)
                        span.set_status(Status(
                            StatusCode.ERROR,
                            f"{type(result.exc).__name__}: {result.exc}"[:200],
                        ))

                last_stdout = result.stdout or ""
                self._record_probe_cell(code, result, it, thought=text)
                self._history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result.to_user_message(),
                })
                if result.turn_ended:
                    turn_ended = True
                    break

            if turn_ended:
                break

        # Fallback for agents that put ``print(f"Answer: ...")`` inside
        # their final code cell instead of a plain-text reply.
        if last_stdout and not _ANSWER_LINE_RE.search(last_text):
            return last_text + "\n\n[stdout]\n" + last_stdout
        return last_text

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def to_checkpoint(self) -> dict:
        return {
            "tool_state": self._tool_state.to_checkpoint(),
            "iter_count": self._iter_count,
            "current_turn": self._current_turn,
            "history": list(self._history),
            "compressed_summary": self._compressed_summary,
        }

    def from_checkpoint(self, data: dict) -> None:
        self._tool_state.from_checkpoint(data["tool_state"])
        self._iter_count = int(data.get("iter_count", 0))
        self._current_turn = int(data.get("current_turn", 0))
        self._history = list(data.get("history", []))
        self._compressed_summary = data.get("compressed_summary", "")


__all__ = [
    "CodeActAgent",
    "_strategy_to_user_text",
    "_wrap_closure_for_repl",
]
