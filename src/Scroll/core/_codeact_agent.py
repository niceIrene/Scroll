"""CodeAct agent

Subclasses customize via:

  - ``sys_prompt``           (role + strategy, ends up after the substrate prelude)
  - ``init_namespace()``     (objects to expose in the REPL globals)
  - ``namespace_docs()``     (documentation appended to the system prompt)
  - ``clear_namespace_each_session`` (whether to wipe globals between days; default True)
  - ``memory_mode``          ("step" | "sliding"; default "step" — the
                             REPL is the within-session memory)
"""

from __future__ import annotations

import asyncio
import logging
import re
import signal
import time
import traceback
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
    EndOfSession,
    extract_python_block,
)
from Scroll.core._models import LogEntry
from Scroll.tools._log_handle import LogHandle

if TYPE_CHECKING:
    from Scroll.core._environment import BaseEnvironment

_log = logging.getLogger(__name__)


# Probe scorers extract the agent's commitment via this regex. Mirrored
# from ``vending/tasks/rewards.py`` so the substrate's stdout-fallback in
# ``_answer_probe_inner`` can detect when no plain-text ``Answer:`` line
# exists and append the cell's stdout so the scorer can find one there.
_ANSWER_LINE_RE = re.compile(
    r"(?im)^[\s>#\-*•]*\**\s*answer\s*\**\s*[:\-]\**\s*(.+?)\s*$"
)


# Matches the closing fence of a python code block (``` on its own line,
# possibly with a trailing language tag we don't expect). Used to detect
# response shape "code + post-code prose" — see ``_split_after_code_block``.
_CODE_FENCE_END_RE = re.compile(r"^```\s*$", re.MULTILINE)


def _split_after_code_block(text: str) -> tuple[str, str]:
    """Split ``text`` at the END of its last fenced code block.

    Returns ``(prefix_inclusive_of_block, suffix_after_block)``. The
    suffix is what the model wrote AFTER its code block — if the
    suffix contains a final-answer line or a "no information" style
    conclusion, the model committed to a result before the cell had a
    chance to run. We use this to detect + strip premature conclusions
    so they don't pollute later iterations or get returned as the
    final answer when the loop exhausts its budget.

    If no closing fence is found, returns ``(text, "")``.
    """
    matches = list(_CODE_FENCE_END_RE.finditer(text))
    if not matches:
        return text, ""
    end = matches[-1].end()
    return text[:end], text[end:]


# Refusal phrases the model emits when it "imagines" the search
# returned nothing. Used together with ``_ANSWER_LINE_RE`` to detect
# premature conclusions in post-code prose.
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


def _is_premature_conclusion(suffix: str) -> bool:
    """Return True iff post-code suffix looks like a final-answer
    commitment that the model wrote BEFORE seeing the cell's stdout.

    Conservative: only triggers on suffixes that contain an explicit
    ``Answer:`` line or a refusal phrase. Plain "let me check the
    output below" intent statements don't trigger.
    """
    s = suffix.strip()
    if not s:
        return False
    if _ANSWER_LINE_RE.search(s):
        return True
    if _REFUSAL_RE.search(s) and len(s) > 30:
        return True
    return False


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


SUBSTRATE_PROMPT = """\
You operate by writing Python code in a persistent REPL.

EACH TURN:
1. Read the user message (or the stdout of your previous cell).
2. Reply with exactly one fenced Python block:
       ```python
       # your code here
       ```
3. The code runs. Its stdout / stderr / any exception comes back to you
   as the next user message. Variables you assign persist across cells
   within the same iteration.

PRINTING:
- The REPL does NOT auto-display the last expression. ``print(x)`` to
  see ``x``. Bare ``x`` on its own line is a no-op.
- Long outputs are truncated when shown back to you. Slice or
  summarize before printing if the data is large.

ERRORS:
- If your code raises, you'll see the traceback in the next turn. Fix
  and try again. Variables defined before the error still persist.

What "one iteration" means and how it ends is environment-specific —
read the section below for the rules of this run.
"""


# Replaces SUBSTRATE_PROMPT for the duration of a probe. Captures only
# what is universal across envs: the session-loop rules are suspended,
# lookup against durable data tools is encouraged, and REPL semantics
# are unchanged. Reply *formatting* (Answer line, units, tolerances,
# what counts as a correct response) varies by scorer and lives in
# each env's :meth:`BaseEnvironment.probe_substrate_prompt` — see
# ``vending/tasks/rewards.py`` (deterministic regex format) and
# ``longmemeval/tasks/probes.py`` (LLM-judge format). Tau-bench probes
# are passive and never exercise this path.
#
# The chat history at probe time IS the same-iteration session (the
# current iteration's REPL turns), not a wiped context — see
# ``answer_probe`` for the splice. Values visible in that history may
# be intermediate / partial (e.g. early-iteration estimates, sub-agent
# prints), so the prompt still pushes the model toward a fresh lookup
# against the durable namespace tools.
PROBE_SUBSTRATE_PROMPT = """\
PROBE-ANSWERING MODE.

You are being asked a one-off probe question. The session-loop substrate
rules that force every reply into a fenced Python block are
SUSPENDED for this turn — a final plain-text reply is acceptable.
Any iteration-end actions named in your env-specific section have
also been disabled (replaced with no-op shims) so a stray call
cannot affect timing.

HOW TO ANSWER — depends on what's in your namespace:
- If you HAVE data tools (log, memoryspace, rlm — whichever
  apply): you can EITHER emit one Python block to query the durable
  copy first, OR answer directly from the chat history / what you
  remember from earlier this run if you're already confident. Both
  are valid. Use the lookup path when the chat history doesn't show
  the value you need, or when what you see is partial /
  intermediate / cross-iteration-stale; use the memory path when
  the answer is already plainly in your context, since every cell
  costs message budget.
- If you have NO data tools (only env action tools, no memoryspace /
  log / rlm): you have nothing to query. Answer directly from
  the chat history — that's your only source.

Whether either path is reliable for THIS scorer (regex vs LLM
judge, units, tolerances) is environment-specific — see the
section below.

REPL BEHAVIOR (unchanged when you do emit a cell):
- ``print(x)`` to see ``x``; bare ``x`` is a no-op.
- Long outputs are truncated when shown back to you.
- If your code raises, the traceback comes back; fix and retry.

Reply formatting (Answer line, units, what the scorer accepts) is
environment-specific — see the section below for this run's rules.
"""


class _UsageTrackingModel:
    """Thin wrapper around the LLM model that records token usage to
    ``ToolState`` on every call.

    Every CodeActAgent has exactly one wrapped model object, shared by
    its session-loop calls (via ``_call_model``) AND by any closure that
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


def _probe_safe_wait_for_next_session():
    """Probe-mode no-op shim for ``wait_for_next_day``.

    Replaces the real ``wait_for_next_day`` in the REPL globals while
    a probe is being answered, so a stray call from the model can't
    end the loop before it commits to a plain-text answer. Prints a
    short reminder to stdout (which the model sees in the next turn).
    """
    print(
        "[probe] wait_for_next_day() is a NO-OP during probe "
        "answering — probes do not end the session. Reply in PLAIN TEXT "
        "starting with `Answer: ` instead."
    )


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
    if asyncio.iscoroutinefunction(fn):
        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            return _strategy_to_user_text(await fn(*args, **kwargs))
        return async_wrapper

    @wraps(fn)
    def sync_wrapper(*args, **kwargs):
        return _strategy_to_user_text(fn(*args, **kwargs))
    return sync_wrapper


def make_wait_for_next_session(state) -> callable:
    """Build a ``wait_for_next_session`` (a.k.a. ``wait_for_next_day``)
    that raises ``EndOfSession``.

    Mirrors the action-logging side-effects of the legacy
    ``vending.tools.wait_for_next_day`` (tick the message counter,
    set ``_session_ended``, append to the action log) before raising so
    the runtime catches it and ends the session.
    """
    def wait_for_next_session():
        """End the current session and advance to the next one.

        Call this when you have finished all actions for the current
        session. Any code in the same cell after this call will not run.
        """
        state._tick()
        state._session_ended = True
        state._action_log.append("wait_for_next_session")
        raise EndOfSession()
    return wait_for_next_session


# Back-compat alias — agent system prompts mention ``make_wait_for_next_day``
# / ``wait_for_next_day`` by name in many places, and per-env namespace
# builders still bind the old key. Keep the old function symbol working.
make_wait_for_next_day = make_wait_for_next_session


# Object.method patterns + bare names worth surfacing in span labels.
# Order matters: longer names first so partial overlaps are matched
# correctly by the regex alternation.
_CELL_OP_PATTERN = re.compile(
    r"\b(?:memoryspace|ms|log)\.(\w+)"
    r"|\b(rlm|wait_for_next_day|schema_inspect)\b"
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
    clear_namespace_each_session: bool = True
    memory_mode: str = "step"
    # Probe-time iteration cap. Tools-heavy strategies (memoryspace + log
    # querying) may need more cells than mem-only baselines — bump this
    # in subclasses if the agent can't reliably finish recall in 5 cells.
    probe_max_iters: int = 5

    # Cross-task memoryspace persistence. List of memoryspace JSON keys to
    # share across independent task runs through ``cfg.shared_memoryspace_path``.
    # Subclass opt-in: memoryspace-having strategies (LME ``code_agent``)
    # override this to e.g. ``["lessons"]`` so accumulated procedural
    # memory survives between QA items. Default empty (no sharing).
    shared_memoryspace_keys: list[str] = []

    def __init__(self, cfg, storage, data) -> None:
        from Scroll.core._tool_state import ToolState

        self.cfg = cfg
        self.log = storage  # ConversationLog (E)
        self.storage = storage  # backwards-compat alias during transition
        self.data = data
        self.last_context: list[str] = []
        self._current_session: int = 0
        self._pending_env_outcomes: list[str] = []
        self._pending_datasource_notes: list[str] = []
        self._tool_state = ToolState(data=data, cfg=cfg)
        self._iter_count: int = 0

        raw_model, self._msg_cls = _init_model(self.cfg)
        # Wrap so every model call (session-loop + closures that capture
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

        # Cross-task memoryspace seed (only fires when both ``shared_memoryspace_keys``
        # is non-empty AND ``cfg.shared_memoryspace_path`` points at a real
        # file). Runs after ``_rebuild_namespace`` so the memoryspace is
        # fully constructed before we seed it from the shared store.
        self._load_shared_memoryspace_state()

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
    # Per-session prompts
    # ------------------------------------------------------------------

    def session_prompt(self, session_idx: int) -> str:
        return (
            f"Day {session_idx} has started.\n\n"
            "Decide what to do today. Write Python to take actions. "
            "When you are done, call wait_for_next_day()."
        )

    def _compose_session_user_msg(self, session_idx: int) -> str:
        """Prepend yesterday's env outcomes + today's datasource notes."""
        body = self.session_prompt(session_idx)
        prefix_sections: list[str] = []
        ns_reminder = self._namespace_reminder()
        if ns_reminder:
            prefix_sections.append(ns_reminder)
        if self._pending_env_outcomes:
            lines = "\n".join(f"  - {line}" for line in self._pending_env_outcomes)
            prefix_sections.append(f"Overnight events from session_idx {session_idx - 1}:\n{lines}")
        if self._pending_datasource_notes:
            lines = "\n".join(f"  - {line}" for line in self._pending_datasource_notes)
            prefix_sections.append(f"Today's briefing:\n{lines}")
        self._pending_env_outcomes = []
        self._pending_datasource_notes = []
        if prefix_sections:
            body = "\n\n".join(prefix_sections) + "\n\n" + body
        return body

    def _namespace_reminder(self) -> str:
        """One-line listing of REPL globals to remind the model what's bound.

        Prepended to every session_idx's user message so the model is far less
        likely to invent ``view_storage()`` or ``from tools import ...``.
        Listing is short — names only — to keep token cost negligible.
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
            f"REPL globals (call directly, no `import` needed): "
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
        ``run_session``, where env is set.
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
        """Compose substrate + env endgame + agent role + namespace docs + extra."""
        parts = [SUBSTRATE_PROMPT.strip()]
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

        Swaps SUBSTRATE_PROMPT for PROBE_SUBSTRATE_PROMPT so the
        "always emit a fenced Python block" / "call wait_for_next_day
        when done" iteration-loop rules don't fight the required
        plain-text reply, then appends the env's
        ``probe_substrate_prompt`` (scorer-shaped format rules — e.g.
        vending's regex Answer line, LME's lenient judge format) so
        the LM knows exactly what the scorer wants. The endgame
        block is dropped here — its end-of-iteration rules are not
        relevant during a probe. Strategy text, namespace docs, and
        any ``extra_system_prompt`` (e.g. tau-bench's domain policy)
        are kept — the model still needs to know what tools it has
        and which rules to follow.
        """
        parts = [PROBE_SUBSTRATE_PROMPT.strip()]
        env_probe = self._env_probe_prompt()
        if env_probe:
            parts.append(env_probe)
        parts.append(self.sys_prompt.strip())
        ns_docs = self.namespace_docs().strip()
        if ns_docs:
            parts.append(ns_docs)
        extra = self.extra_system_prompt().strip()
        if extra:
            parts.append(extra)
        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Namespace lifecycle
    # ------------------------------------------------------------------

    def _rebuild_namespace(self) -> None:
        """Reset the REPL globals to a freshly built namespace.

        Always injects ``today`` (the current 1-indexed session_idx number) so
        agent code can write ``log.range_by_day(today - 7, today - 1)``
        without parsing the session_idx out of the briefing text.
        """
        ns = self.init_namespace()
        ns.setdefault("today", self._current_session)
        self._runtime.reset(initial_globals=ns)

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------

    @property
    def message_count(self) -> int:
        return self._tool_state._message_count

    @property
    def session_ended(self) -> bool:
        return self._tool_state._session_ended

    def receive_outcomes(self, session_idx: int, logs: list[str]) -> None:
        self._pending_env_outcomes = list(logs)

    def receive_context(self, session_idx: int, notes: list[str]) -> None:
        self._pending_datasource_notes = list(notes)

    def run_session(self, env) -> list[str]:
        session_idx = env.session_idx
        self._current_session = session_idx + 1
        self._tool_state.env = env
        self._tool_state.reset_session()

        if self.clear_namespace_each_session:
            self._rebuild_namespace()
        else:
            # Even when globals persist across days, refresh ``today``
            # so the agent's worked-example pattern (``log.range_by_day(
            # today - 7, today - 1)``) keeps pointing at the real session.
            self._runtime.update_globals(today=self._current_session)

        prompt = self._compose_session_user_msg(session_idx + 1)

        if self.memory_mode == "sliding":
            # Passive `mem(E)` baseline (Basic, design §0): keep
            # yesterday's LM history and append today's briefing. Old
            # messages are summarized into ``_compressed_summary`` and
            # dropped by ``_compress_history_to_budget`` inside the
            # async loop (called before each ``_call_model``). This is
            # the only `f` Basic has — without it, cross-session recall is
            # impossible.
            if not self._history or self._history[0].get("role") != "system":
                self._history = [
                    {"role": "system", "content": self._full_sys_prompt()},
                ]
            self._history.append({"role": "user", "content": prompt})
        else:
            # Default ("step"): fresh history per session. Within-session
            # continuity comes from REPL globals — stale LM turns from
            # the previous session don't need to live in the prompt.
            self._history = [
                {"role": "system", "content": self._full_sys_prompt()},
                {"role": "user", "content": prompt},
            ]
        # Mirror the session-start user prompt into E (matches the legacy
        # SlidingMemory.add path).
        self.log.append(LogEntry.make(
            session_idx=self._current_session,
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
                    asyncio.run(self._run_session_inner())
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
                _log.error("agent_error on session %d: %s", session_idx, last_err)
        finally:
            signal.signal(signal.SIGINT, prev_handler)

        if interrupted:
            raise KeyboardInterrupt

        actions = list(self._tool_state._action_log)
        self.last_context = self._snapshot_history()

        if not self._tool_state._session_ended:
            self._tool_state._session_ended = True
        return actions

    async def _run_session_inner(self) -> None:
        max_iters = self.cfg.max_iters_per_turn
        ended = False
        for it in range(max_iters):
            self._iter_count += 1
            if self.memory_mode == "sliding":
                await self._compress_history_to_budget()
            override = await self._on_pre_cell_async()

            response = await self._call_model(override)
            text = _extract_text_from_response(response)
            if not text:
                # Empty response — treat as no-op end.
                break

            code = extract_python_block(text)
            # Defensive: strip "premature conclusion" prose written
            # AFTER the code block in the same response. Models
            # sometimes commit to "Answer: no info" in the same turn
            # they write the search code — basing the conclusion on an
            # imagined empty stdout. Keeping it in history biases the
            # next iteration. Day-loop concern is purely about
            # contaminating later days' reasoning.
            stripped_premature = False
            if code is not None:
                _, suffix = _split_after_code_block(text)
                if _is_premature_conclusion(suffix):
                    text = text[:len(text) - len(suffix)] + (
                        "\n\n[note: post-code conclusion stripped — wait "
                        "for the cell's stdout before committing to a "
                        "result.]"
                    )
                    stripped_premature = True

            self._history.append({"role": "assistant", "content": text})

            self.log.append(LogEntry.make(
                session_idx=self._current_session,
                role="assistant",
                content=text,
                metadata={
                    "kind": "lm_turn",
                    "iter": it,
                    "has_code": bool(code),
                    "stripped_premature_conclusion": stripped_premature,
                },
            ))

            if code is None:
                # No code emitted — accept the assistant text as a
                # final answer (probe path) or as a "I'm done" signal.
                break

            ops = _detect_cell_ops(code)
            ops_label = ", ".join(ops) if ops else "no-ops"
            line_count = code.count("\n") + 1
            span_name = (
                f"code_exec d{self._current_session}.i{it} "
                f"({line_count}L) [{ops_label}]"
            )
            # Phoenix's SpanDetailsQuery resolver chokes on large /
            # non-standard-mime ``input.value``s (we saw it fail with
            # ``text/x-python`` and multi-KB code blobs). Cap the
            # rendered preview, use ``text/plain`` so Phoenix's UI
            # picks a known renderer, and stash the full code under a
            # separate attribute that's never rendered in the input
            # panel but is still searchable / inspectable.
            code_preview = code if len(code) <= 3000 else (
                code[:2900] + "\n\n# ... [truncated; see cell.code_full]"
            )
            with _tracer.start_as_current_span(span_name) as span:
                span.set_attributes({
                    "tool.name": "code_exec",
                    "cell.session": self._current_session,
                    "cell.iter": it,
                    "cell.iter_lifetime": self._iter_count,
                    "cell.ops": ",".join(ops),
                    "cell.line_count": line_count,
                    "cell.code_chars": len(code),
                    "cell.code_full": code,
                    "input.value": code_preview,
                    "input.mime_type": "text/plain",
                })
                result = await self._runtime.execute_cell(code)
                # Build the output panel preview the same way the agent
                # sees it (stdout + stderr + traceback) so the tool-
                # output panel doesn't silently hide the exception text.
                # ALWAYS non-empty: a cell that did INSERT + wait but
                # didn't print anything still ran successfully — give
                # Phoenix something to render so the user can tell
                # "no output" from "render failure".
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
                if result.session_ended:
                    summary_bits.append("session_ended=True")
                if result.exception:
                    summary_bits.append("EXCEPTION")
                cell_summary = " | ".join(summary_bits)
                output_preview = (
                    ("\n\n".join(output_parts))[:2000]
                    if output_parts
                    else f"[cell ran, no stdout]\n\nsummary: {cell_summary}\n\ncode (first 200c):\n{code[:200]}"
                )
                stdout_preview = (result.stdout or "")[:200].replace("\n", " ⏎ ")
                stdout_lines = (result.stdout.count("\n") if result.stdout else 0) + (1 if result.stdout else 0)
                span.set_attributes({
                    "openinference.span.kind": "TOOL",
                    "output.value": output_preview,
                    "output.mime_type": "text/plain",
                    "cell.summary": cell_summary,
                    "cell.stdout_preview": stdout_preview or "[empty]",
                    "cell.stdout_lines": stdout_lines,
                    "cell.stdout_chars": len(result.stdout or ""),
                    "code_exec.session_ended": result.session_ended,
                    "code_exec.has_exception": result.exception is not None,
                })
                if result.exc is not None:
                    # Attach as a proper OTel exception event so it shows
                    # up under Phoenix's "Events" tab on the span (not
                    # just buried in output text). Also flip span status
                    # to ERROR so failed cells are findable in the UI.
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

            user_text = result.to_user_message()
            self._history.append({"role": "user", "content": user_text})

            if result.session_ended:
                ended = True
                break

        await self._on_session_end_async()
        # Cross-task persistence. Fires AFTER the subclass hook so any
        # last-cell ``json_write`` is captured. No-op unless the agent
        # has opted in via ``shared_memoryspace_keys`` and the config
        # points at a shared path.
        self._persist_shared_memoryspace_state()
        if not ended:
            # Fell off the iteration cap — flag end-of-session so the
            # benchmark loop advances. Mirrors ReActAgent's behavior.
            self._tool_state._session_ended = True

    async def _call_model(self, messages_override: list | None = None):
        """One model call against the current history (or override).

        For sliding mode, splices ``_compressed_summary`` in as an extra
        system message after index 0 so dropped-and-summarized history
        stays visible to the LM. No-op when summary is empty or
        ``messages_override`` is provided (callers managing their own
        message list handle injection themselves).

        Per-call quota-aware retry: Dashscope occasionally returns 429
        / insufficient_quota under load. Session-level retry (in run_session)
        is too coarse for these — restarting a whole session on a single
        flaky call wastes work and re-fires the same throttle. So we
        retry HERE with longer backoff (15s, 30s) on quota-class errors
        only; non-quota exceptions bubble to the session-level retry as
        before.
        """
        msgs = messages_override if messages_override is not None else self._history
        if (
            self.memory_mode == "sliding"
            and self._compressed_summary
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
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                return await self._model(msgs)
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

    async def _on_session_end_async(self) -> None:
        """Hook fired after the per-session loop finishes (before asyncio.run exits).

        Default no-op. Subclasses may override to flush per-session
        state into external storage before the next session's
        namespace wipe.
        """
        return None

    # ------------------------------------------------------------------
    # Cross-task memoryspace persistence
    #
    # Mechanism only — agents opt in via ``shared_memoryspace_keys`` and
    # supply the storage primitive via ``get_memoryspace()``. Used by
    # LongMemEval's ``code_agent`` so ``lessons`` (procedural query-
    # rewrite memory) accumulates across independent QA sub-runs that
    # the orchestrator launches in separate processes.
    # ------------------------------------------------------------------

    def get_memoryspace(self) -> Any | None:
        """Return the memoryspace object when the subclass has one.

        Default ``None``. Memoryspace-having subclasses override to return
        ``self.memoryspace``. The cross-task persistence code uses this
        to locate the ``json_read`` / ``json_write`` surface; a ``None``
        return turns the whole mechanism into a no-op.
        """
        return None

    def _filter_for_shared_write(self, key: str, value: Any) -> Any:
        """Filter a memoryspace value before writing to the shared store.

        Default identity. Subclass override to drop entries that are
        task-specific and shouldn't leak across runs (e.g. user-
        specific vocab mappings in LME ``lessons``).
        """
        return value

    def _merge_shared_value(self, key: str, disk: Any, local: Any) -> Any:
        """Merge the on-disk shared value with the agent's local value.

        Called under the file lock during persist, AFTER
        ``_filter_for_shared_write`` was applied to ``local``. Needed
        because parallel task runs may have written to disk since this
        agent loaded its initial seed — naive overwrite would clobber
        their contributions.

        Default rules:
          - both list  → concat + dedup by JSON-serialized form,
            preserving first-seen order (append-only-log semantics)
          - both dict  → ``{**disk, **local}`` (local wins per-key)
          - disk None  → local
          - otherwise  → local (last-writer-wins for scalars)

        Subclass override when the value has a richer schema (e.g.
        scored entries where ``max(score)`` is the right merge).
        """
        if disk is None:
            return local
        if isinstance(disk, list) and isinstance(local, list):
            import json as _json
            seen: set[str] = set()
            out: list = []
            for entry in disk + local:
                try:
                    repr_ = _json.dumps(entry, sort_keys=True, default=str)
                except (TypeError, ValueError):
                    repr_ = repr(entry)
                if repr_ in seen:
                    continue
                seen.add(repr_)
                out.append(entry)
            return out
        if isinstance(disk, dict) and isinstance(local, dict):
            return {**disk, **local}
        return local

    def _load_shared_memoryspace_state(self) -> None:
        """Seed the memoryspace with previously-persisted shared keys.

        No-op when (a) the subclass hasn't opted in via
        ``shared_memoryspace_keys``, (b) ``cfg.shared_memoryspace_path`` is
        unset, (c) ``get_memoryspace()`` returns None, or (d) the file
        doesn't exist yet (first task run).

        Errors are swallowed (logged at DEBUG) — a corrupt shared file
        shouldn't poison subsequent task runs. The next persist will
        rewrite it cleanly.
        """
        if not self.shared_memoryspace_keys:
            return
        path = getattr(self.cfg, "shared_memoryspace_path", None)
        if not path:
            return
        ms = self.get_memoryspace()
        if ms is None:
            return
        import json as _json
        from pathlib import Path as _Path
        p = _Path(path)
        if not p.exists():
            return
        try:
            raw = p.read_text(encoding="utf-8")
            shared = _json.loads(raw) if raw.strip() else {}
        except (OSError, ValueError) as exc:
            _log.debug("shared memoryspace load failed at %s: %s", path, exc)
            return
        if not isinstance(shared, dict):
            return
        for key in self.shared_memoryspace_keys:
            if key in shared:
                try:
                    ms.json_write(key, shared[key])
                except Exception as exc:  # noqa: BLE001
                    _log.debug("shared memoryspace seed %r failed: %s", key, exc)

    def _persist_shared_memoryspace_state(self) -> None:
        """Merge filtered local memoryspace state into the shared store.

        Called after every ``_on_session_end_async`` so lessons learned
        through any session survive even if the task crashes mid-run.
        Uses ``fcntl.flock(LOCK_EX)`` for parallel-safe read-modify-
        write across the sub-run subprocesses spawned by the
        orchestrator.

        No-op (early return) under the same conditions as ``_load_*``.
        """
        if not self.shared_memoryspace_keys:
            return
        path = getattr(self.cfg, "shared_memoryspace_path", None)
        if not path:
            return
        ms = self.get_memoryspace()
        if ms is None:
            return
        import fcntl as _fcntl
        import json as _json
        from pathlib import Path as _Path
        p = _Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _log.debug("shared memoryspace mkdir failed at %s: %s", p.parent, exc)
            return
        # ``a+`` creates if missing, leaves cursor at EOF; we seek(0)
        # explicitly to read. Hold LOCK_EX for the full read-modify-
        # write so a concurrent persist in a sibling subprocess waits.
        try:
            with open(p, "a+", encoding="utf-8") as f:
                _fcntl.flock(f.fileno(), _fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    raw = f.read()
                    try:
                        shared = _json.loads(raw) if raw.strip() else {}
                    except ValueError:
                        shared = {}
                    if not isinstance(shared, dict):
                        shared = {}
                    for key in self.shared_memoryspace_keys:
                        try:
                            local = ms.json_read(key)
                        except KeyError:
                            continue
                        except Exception as exc:  # noqa: BLE001
                            _log.debug("shared memoryspace read %r failed: %s", key, exc)
                            continue
                        filtered = self._filter_for_shared_write(key, local)
                        shared[key] = self._merge_shared_value(
                            key, shared.get(key), filtered,
                        )
                    f.seek(0)
                    f.truncate()
                    f.write(_json.dumps(shared, indent=2, default=str))
                finally:
                    _fcntl.flock(f.fileno(), _fcntl.LOCK_UN)
        except OSError as exc:
            _log.debug("shared memoryspace persist failed at %s: %s", path, exc)

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
        cell_id = f"cell-{self._current_session}-{iter_idx}"
        self.log.append(LogEntry.make(
            session_idx=self._current_session,
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
            "\n[session ended]" if result.session_ended else ""
        )
        self.log.append(LogEntry.make(
            session_idx=self._current_session,
            role="tool",
            tool_result={
                "id": cell_id,
                "name": "code_exec",
                "output": output_blob.strip(),
            },
            metadata={"kind": "stdout", "session_ended": result.session_ended},
        ))

    def _record_probe_cell(
        self, code: str, result: CellResult, iter_idx: int,
        thought: str = "",
    ) -> None:
        """Mirror a probe-time cell run into ConversationLog.

        Same shape as :meth:`_record_cell` but tagged with
        ``metadata.kind`` of ``"probe_code"`` / ``"probe_stdout"`` and
        a ``probe-cell-`` id prefix so probe lookups are distinguishable
        from session-loop cells. ``extract_tool_trace`` reads from
        ``agent.log.entries[prev_log_count:]`` to populate
        ``ProbeResult.tool_trace``; without this mirror the trace is
        always empty.

        ``thought`` carries the agent's reasoning text emitted alongside
        the tool call (the assistant message's plain-text content
        block, written between cells). It's stored on the probe_code
        entry's ``content`` field so trajectory distillation captures
        THINK / DO / SEE triples, not just DO / SEE.
        """
        cell_id = f"probe-cell-{self._current_session}-{iter_idx}"
        self.log.append(LogEntry.make(
            session_idx=self._current_session,
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
            session_idx=self._current_session,
            role="tool",
            tool_result={
                "id": cell_id,
                "name": "code_exec",
                "output": output_blob.strip(),
            },
            metadata={"kind": "probe_stdout"},
        ))

    def _snapshot_history(self) -> list[str]:
        """Return a human-readable summary of this session's LM turns."""
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

        Used by ``memory_mode == "sliding"`` (Basic's `mem(E)` baseline).
        When over the char budget, picks the oldest batch (~budget / 4),
        asks the model to fold it into the running ``_compressed_summary``,
        then deletes those messages from ``_history``. The summary is
        re-injected into every model call by :meth:`_call_model`, so the
        LM still sees old context — just compacted.

        On LLM failure: messages are still dropped (matches the legacy
        ``SlidingMemory._compress`` behavior — better to truncate than
        to grow unbounded).
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
          which replaces the session-loop substrate rules (always reply
          with a code block; call ``wait_for_next_day()`` when done)
          with probe-specific rules (plain-text ``Answer:`` required;
          do not end the session). Without this swap the model frequently
          ends probes with ``wait_for_next_day()`` or buries the
          answer inside ``print("Answer: ...")``.
        - ``wait_for_next_day`` in the REPL globals is replaced with a
          no-op shim for the duration of the probe, so a stray call
          can't end the inner loop before the agent commits.
        - History handling is unified across memory modes: probes
          run *inside the same-session session* — we keep ``self._history``
          (today's session-loop turns: system + briefing + cell traces),
          replace the system prompt at index 0 with the probe-mode
          version, and append the question. This way step-mode agents
          see today's reasoning context while answering (matching what
          sliding always did), and the prompt's "you must query for
          factual probes" rule pushes them to re-fetch from memoryspace
          / log instead of trusting mid-session intermediate values
          they might recall. For sliding we additionally trim to
          context budget since its history
          spans days. ``self._history`` is restored after the probe
          so the Q+A doesn't leak into working memory.
        """
        old_day_ended = self._tool_state._session_ended
        self._tool_state._session_ended = False

        history_backup = list(self._history)
        probe_sys_msg = {
            "role": "system",
            "content": self._probe_sys_prompt(),
        }

        # Swap wait_for_next_day to a probe-safe no-op for the
        # duration of the probe. Restored in the finally block.
        old_wait = self._runtime.globals.get("wait_for_next_day")
        wait_was_present = "wait_for_next_day" in self._runtime.globals
        self._runtime.globals["wait_for_next_day"] = _probe_safe_wait_for_next_session

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
            self._tool_state._session_ended = old_day_ended
            if wait_was_present:
                self._runtime.globals["wait_for_next_day"] = old_wait
            else:
                self._runtime.globals.pop("wait_for_next_day", None)
            if pq_was_present:
                self._runtime.globals["probe_question"] = old_pq
            else:
                self._runtime.globals.pop("probe_question", None)
        return answer

    async def _answer_probe_inner(self, max_iters: int) -> str:
        last_text = ""
        last_stdout = ""
        cells_executed = 0
        zero_cell_abstain_forced = False
        for it in range(max_iters):
            if self.memory_mode == "sliding":
                await self._compress_history_to_budget()
            override = await self._on_pre_cell_async()
            response = await self._call_model(override)
            text = _extract_text_from_response(response)
            if not text:
                break
            code = extract_python_block(text)
            # Defensive: if the model wrote BOTH a code block AND a
            # final-answer / refusal in the same response, strip the
            # post-code conclusion. This is a critical fix for probe
            # answering — without it, a premature "Answer: I don't
            # have that information" written in the same turn as the
            # search code becomes ``last_text`` and gets returned to
            # the judge as the final answer when the loop ends, even
            # though the cell's actual stdout (which may contain the
            # answer) is never reasoned about.
            stripped_premature = False
            if code is not None:
                _, suffix = _split_after_code_block(text)
                if _is_premature_conclusion(suffix):
                    text = text[:len(text) - len(suffix)] + (
                        "\n\n[note: post-code conclusion stripped — "
                        "wait for the cell's stdout, then commit your "
                        "Answer in the NEXT turn based on what you "
                        "actually observed.]"
                    )
                    stripped_premature = True

            self._history.append({"role": "assistant", "content": text})
            last_text = text
            if code is None:
                # FORCE: if the agent is abstaining ("I don't have that
                # information") without ever running a memoryspace query,
                # reject the answer once and require at least one probe
                # cell. Models like qwen3-30b sometimes skim chat history
                # and abstain without using the structured store at all
                # — directly violating SEARCH BEFORE YOU REFUSE. Forcing
                # one retry recovers a fraction of these cases.
                # Only triggers once per probe (zero_cell_abstain_forced)
                # so a determined abstain doesn't infinite-loop.
                if (
                    cells_executed == 0
                    and not zero_cell_abstain_forced
                    and _is_abstention_text(text)
                ):
                    zero_cell_abstain_forced = True
                    self._history.append({
                        "role": "user",
                        "content": (
                            "[ABSTENTION REJECTED — zero probe cells.] "
                            "You answered \"I don't have that information\" "
                            "without running ANY memoryspace query. Per the "
                            "SEARCH BEFORE YOU REFUSE rule, a zero-cell "
                            "abstention is wrong by default — the data may "
                            "be in your store even when nothing in your "
                            "recent chat history mentions it.\n\n"
                            "Run at least ONE probe cell now: "
                            "``memoryspace.sql_exec(...)`` or "
                            "``log.semantic_search(...)`` or "
                            "``log.findall(...)`` using keywords from the "
                            "probe question. If the query genuinely returns "
                            "nothing relevant, you may abstain on your NEXT "
                            "response based on what you observed."
                        ),
                    })
                    last_text = ""  # discard the rejected abstention
                    continue
                break
            cells_executed += 1

            # Wrap probe-time cell execution in its own span so probes
            # don't lose trace coverage. Distinct prefix (``probe_cell``
            # vs ``code_exec``) so the session-loop and probe paths can be
            # filtered apart in Phoenix.
            ops = _detect_cell_ops(code)
            ops_label = ", ".join(ops) if ops else "no-ops"
            line_count = code.count("\n") + 1
            span_name = (
                f"probe_cell d{self._current_session}.i{it} "
                f"({line_count}L) [{ops_label}]"
            )
            code_preview = code if len(code) <= 3000 else (
                code[:2900] + "\n\n# ... [truncated; see cell.code_full]"
            )
            with _tracer.start_as_current_span(span_name) as span:
                span.set_attributes({
                    "tool.name": "code_exec",
                    "cell.session": self._current_session,
                    "cell.iter": it,
                    "cell.kind": "probe",
                    "cell.ops": ",".join(ops),
                    "cell.line_count": line_count,
                    "cell.code_chars": len(code),
                    "cell.code_full": code,
                    "cell.stripped_premature_conclusion": stripped_premature,
                    "input.value": code_preview,
                    "input.mime_type": "text/plain",
                })
                result = await self._runtime.execute_cell(code)
                output_parts = []
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
                    else f"[probe cell ran, no stdout]\n\nsummary: {cell_summary}\n\ncode (first 200c):\n{code[:200]}"
                )
                stdout_preview = (result.stdout or "")[:200].replace("\n", " ⏎ ")
                stdout_lines = (result.stdout.count("\n") if result.stdout else 0) + (1 if result.stdout else 0)
                span.set_attributes({
                    "openinference.span.kind": "TOOL",
                    "output.value": output_preview,
                    "output.mime_type": "text/plain",
                    "cell.summary": cell_summary,
                    "cell.stdout_preview": stdout_preview or "[empty]",
                    "cell.stdout_lines": stdout_lines,
                    "cell.stdout_chars": len(result.stdout or ""),
                    "code_exec.has_exception": result.exception is not None,
                })
                if result.exc is not None:
                    span.record_exception(result.exc)
                    span.set_status(Status(
                        StatusCode.ERROR,
                        f"{type(result.exc).__name__}: {result.exc}"[:200],
                    ))
            last_stdout = result.stdout or ""
            # Mirror probe-time cells into ConversationLog so that
            # ``probe_results.json`` ``tool_trace`` is populated by
            # ``extract_tool_trace`` (which reads from ``agent.log``).
            # Tagged with ``metadata.kind == "probe_code"`` /
            # ``"probe_stdout"`` so downstream consumers can tell session-
            # loop cells apart from probe-answering cells.
            self._record_probe_cell(code, result, it)
            # Cell stdout is appended verbatim. Earlier versions
            # tacked on a "[PROBE REMINDER]" instructing the model to
            # reply in plain text — the probe-mode system prompt and
            # the strategy's PROBE_EXAMPLE already cover that contract,
            # so the per-cell reminder was just noise. The
            # ``stripped_premature`` corrective message stays because
            # it's a conditional signal (only fires when the model
            # actually mixed code + answer in one turn) and conveys
            # information the system prompt can't preempt.
            cell_msg = result.to_user_message()
            if stripped_premature:
                cell_msg += (
                    "\n\n[your previous response wrote an Answer/refusal "
                    "in the SAME turn as the code, based on an IMAGINED "
                    "stdout. That conclusion has been stripped. The "
                    "cell's REAL stdout is above — read it carefully "
                    "and write a fresh plain-text Answer on a single "
                    "line starting with `Answer: `. Do NOT mix code "
                    "and Answer in one response again.]"
                )
            self._history.append({"role": "user", "content": cell_msg})
            if result.session_ended:
                break
        # Fallback for agents that put ``print(f"Answer: ...")`` inside
        # their final code cell instead of writing a separate plain-text
        # turn. The scorer's regex only sees the assistant text, so if
        # that text is a code block (no plain ``Answer:`` line) but the
        # cell's stdout DOES carry one, append the stdout so the scorer
        # can find it. The strategy prompt still teaches the two-turn
        # format; this is a safety net, not the recommended pattern.
        if last_stdout and not _ANSWER_LINE_RE.search(last_text):
            return last_text + "\n\n[stdout]\n" + last_stdout
        return last_text

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def to_checkpoint(self) -> dict:
        return {
            "tool_state": self._tool_state.to_checkpoint(),
            "pending_env_outcomes": list(self._pending_env_outcomes),
            "pending_datasource_notes": list(self._pending_datasource_notes),
            "iter_count": self._iter_count,
            "compressed_summary": self._compressed_summary,
        }

    def from_checkpoint(self, data: dict) -> None:
        self._tool_state.from_checkpoint(data["tool_state"])
        self._pending_env_outcomes = list(data.get("pending_env_outcomes", []))
        self._pending_datasource_notes = list(data.get("pending_datasource_notes", []))
        self._iter_count = int(data.get("iter_count", 0))
        self._compressed_summary = data.get("compressed_summary", "")


__all__ = [
    "CodeActAgent",
    "PROBE_SUBSTRATE_PROMPT",
    "SUBSTRATE_PROMPT",
    "_strategy_to_user_text",
    "_wrap_closure_for_repl",
    "make_wait_for_next_session",
    "make_wait_for_next_day",  # back-compat alias
]
