"""scroll_react — CodeAct-style ReAct with scroll context management.

The model has three tools: ``execute_python`` (runs Python in a persistent
namespace owned by the context manager's ``ScrollRuntime``), ``bash``
(canonical Harbor shell proxy, a top-level tool dispatched straight to
``ctx.environment`` — only present when an eval attaches a sandbox), and
``submit_answer`` (terminates). Inside ``execute_python`` the model reaches
``ms`` — a read-only query window onto its durable, cross-session
``conversation_history`` — and keeps its working data in plain Python
variables that persist across calls.

All context management is delegated to
:class:`scroll_context.ScrollContextManager`: write-through
persistence, observation aging, token-budget eviction folded into the pinned
in-context ``EvictionIndex`` map, the seeded prior-sessions ``[memory]`` map,
and the per-turn ``[working memory]`` digest. The model-facing protocol text
(``core.md`` / ``index.md``) is likewise owned by the ``scroll_context``
package and assembled per configuration by ``mgr.protocol_prompt()``; this
module contributes only the harness layers — ``prompts/system.md`` (the loop
preamble) and ``prompts/loop.md`` (per-turn batching + finishing policy) —
plus the *loop* itself: model calls (with retry/streaming collapse), tool
dispatch, budget gates, the reserved final submit turn, tracing, and the
trajectory record.

The conversation ``history`` is a plain OpenAI-format dict message list — the
manager's native format. AgentScope ``Msg`` objects exist only at the
model-call boundary: ``_to_agentscope`` reassembles each assistant dict and
its ``role:"tool"`` results into one ``AssistantMsg`` whose content carries
the ``ToolResultBlock``s (Pydantic forbids tool_result blocks in user
messages, following AgentScope's _save_to_context), reproducing the exact
wire format of the pre-refactor loop.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import time
import uuid
from typing import Any

from agentscope.message import (
    AssistantMsg,
    Msg,
    SystemMsg,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultState,
    UserMsg,
)

from opentelemetry import trace as _otel_trace

from scroll_eval._tools_common import (
    budget_notice,
    format_bash_observation,
    run_bash,
    select_tools,
)
from scroll_eval.base_agents.scroll_react import prompts
from scroll_context import ScrollContextManager
from scroll_eval.tracing import otel
from scroll_eval.types import LoopContext, Step, TaskSpec, TerminationReason, Trajectory


# Step cap. The 20-step default suits short retrieval probes; long-horizon
# coding benchmarks (e.g. EdgeBench) raise it via SCROLL_MAX_STEPS.
MAX_STEPS = int(os.environ.get("SCROLL_MAX_STEPS") or 20)
EXECUTE_PYTHON_TIMEOUT_S = 60.0
DEFAULT_TOOLS = ["execute_python", "bash", "submit_answer"]

# Reserved final answer: when the loop would otherwise spend its *last* affordable
# turn on yet another search and then stop with no answer, the probe scores a hard
# zero for never committing — even when it had gathered enough to answer. So we
# reserve that last turn: once a budget axis (steps/wall/tokens) is down to its
# final turn of headroom, the tool surface is narrowed to submit_answer and the
# model is told to commit. This stays *within* budget (no extra turn). The soft
# per-step nudge is advisory and reasoning models routinely ignore it; this is the
# binding version. Disable with SCROLL_FORCE_FINAL_ANSWER=0.
_RESERVE_WALL_S = 90.0       # keep ≥ this much wall-time for the reserved submit turn
_RESERVE_TOKENS = 8000       # keep ≥ this many tokens for the reserved submit turn
_FORCE_SUBMIT_DIRECTIVE = (
    "[budget] You are out of budget: you may NOT run any more searches or code. "
    "Using ONLY the information already gathered above, call submit_answer now "
    "with your single best answer. Best-effort is required — a partial answer is "
    "fine, but an empty or missing answer scores zero."
)

# Transient LLM-call resilience. Provider-side bursts (429 rate limits, 5xx,
# gateway timeouts, dropped thinking-mode streams) would otherwise raise out of
# the loop and abort the whole probe with no trajectory. The model call has no
# side effects until its response is processed, so a failed attempt is safe to
# repeat — retry a few times with capped exponential backoff before giving up.
LLM_MAX_ATTEMPTS = 5
LLM_BACKOFF_BASE_S = 2.0
LLM_BACKOFF_CAP_S = 30.0

# Per-command bash timeout: model-settable within a hard range. Decoupled from
# the *total* budget (which the loop-level gate enforces by stopping the run),
# so it never degenerates to a 1s timeout near the budget edge.
DEFAULT_COMMAND_TIMEOUT_S = 120
MIN_COMMAND_TIMEOUT_S = 5
MAX_COMMAND_TIMEOUT_S = 600

# Indices 0 (system) and 1 (task) of `history` are pinned and never evicted.
_PINNED = 2


_BASH_TIMEOUT_PROP = {
    "type": "integer",
    "description": (
        "Optional max seconds for this command (default "
        f"{DEFAULT_COMMAND_TIMEOUT_S}, max {MAX_COMMAND_TIMEOUT_S}). Set higher "
        "for likely slow commands, builds, or installs."
    ),
}


def _scroll_tools() -> list[dict]:
    """Default tool surface with a `timeout` param added to `bash`.

    Deep-copies the shared schema so the canonical ``_BASH_SCHEMA`` used by the
    legacy agents stays byte-identical — only scroll_react advertises the
    per-command timeout knob.
    """
    tools = [copy.deepcopy(t) for t in select_tools(DEFAULT_TOOLS)]
    for t in tools:
        fn = t.get("function", {})
        if fn.get("name") == "bash":
            fn["parameters"]["properties"]["timeout"] = _BASH_TIMEOUT_PROP
    return tools


def _parse_input(raw: Any) -> dict:
    """Parse ToolCallBlock.input (JSON string or already a dict) into a dict."""
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


async def _resolve_response(response: Any) -> Any:
    """Collapse a streamed model response down to its final ChatResponse.

    With thinking mode enabled the DashScope client runs in stream mode, so
    ``await model(...)`` yields an async generator of incremental ChatResponses;
    the terminal one (``is_last=True``) carries the fully accumulated content
    (thinking + text + tool calls) and the final usage. A non-stream response is
    a single ``ChatResponse`` and passes straight through unchanged.

    Detection uses ``inspect.isasyncgen`` rather than ``hasattr(__aiter__)``:
    ``ChatResponse`` is a ``dict`` subclass whose ``__getattr__`` raises
    ``KeyError`` (not ``AttributeError``) for missing keys, so ``hasattr`` would
    leak that ``KeyError`` instead of returning ``False``.
    """
    if not inspect.isasyncgen(response):
        return response
    final = None
    async for chunk in response:
        final = chunk
    return final


async def _call_model_with_retry(
    model: Any,
    call_messages: list[Any],
    tool_schema: Any,
    *,
    deadline: float | None = None,
) -> Any:
    """Invoke the model, retrying transient failures with exponential backoff.

    The call is effectively idempotent (no state changes until the response is
    processed downstream), so a failed attempt is safe to repeat. Retries up to
    ``LLM_MAX_ATTEMPTS`` times; backoff is capped at ``LLM_BACKOFF_CAP_S`` and is
    further clamped so we never sleep past ``deadline`` (the wall budget). The
    last exception is re-raised once attempts are exhausted, letting the loop end
    the run cleanly as ``TerminationReason.ERROR`` instead of crashing the probe.
    """
    last_exc: Exception | None = None
    for attempt in range(LLM_MAX_ATTEMPTS):
        try:
            return await _resolve_response(await model(call_messages, tools=tool_schema))
        except Exception as exc:  # noqa: BLE001 - any provider error is retryable
            last_exc = exc
            if attempt == LLM_MAX_ATTEMPTS - 1:
                break
            delay = min(LLM_BACKOFF_BASE_S * (2**attempt), LLM_BACKOFF_CAP_S)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                delay = min(delay, remaining)
            try:
                _otel_trace.get_current_span().add_event(
                    "llm_call.retry",
                    {
                        "attempt": attempt + 1,
                        "delay_s": round(delay, 2),
                        "error": str(exc)[:200],
                    },
                )
            except Exception:  # noqa: BLE001 - monitoring never breaks the loop
                pass
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _command_timeout(args: dict, remaining_s: float | None) -> int:
    """Resolve a bash command's timeout: model request, hard-capped + floored.

    The model's ``timeout`` (or the default) is clamped to
    ``[MIN, MAX]`` and, if a wall budget is set, to the time remaining — but
    floored at ``MIN`` so it never collapses to ~1s. The loop-level budget gate
    stops the run when the wall is reached, so a command is only ever issued
    while time remains; this just bounds how far past the wall it may run.
    """
    requested = args.get("timeout")
    try:
        requested = int(requested) if requested is not None else DEFAULT_COMMAND_TIMEOUT_S
    except (TypeError, ValueError):
        requested = DEFAULT_COMMAND_TIMEOUT_S
    requested = min(requested, MAX_COMMAND_TIMEOUT_S)
    if remaining_s is not None:
        requested = min(requested, int(remaining_s))
    return max(MIN_COMMAND_TIMEOUT_S, requested)


# --- OpenAI-dict ⇄ AgentScope conversion (model-call boundary only) -----------


def _to_agentscope(messages: list[dict]) -> list[Msg]:
    """Build the AgentScope view of an OpenAI-dict message list.

    An assistant dict and its contiguous ``role:"tool"`` results collapse into
    ONE ``AssistantMsg`` whose content carries the ``ToolResultBlock``s — the
    same wire format the pre-refactor loop sent (AgentScope forbids tool_result
    blocks in user messages). The dict list is the source of truth the manager
    mutates (aging, eviction); this view is rebuilt every call.
    """
    out: list[Msg] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role")
        content = m.get("content")
        text = content if isinstance(content, str) else ""
        if role == "system":
            out.append(SystemMsg(name=m.get("name") or "system", content=text))
            i += 1
        elif role == "assistant":
            blocks: list[Any] = []
            if text:
                blocks.append(TextBlock(text=text))
            tool_calls = m.get("tool_calls") or []
            ids = set()
            for tc in tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                args = fn.get("arguments", "{}")
                tc_id = tc.get("id") or uuid.uuid4().hex
                ids.add(tc_id)
                blocks.append(
                    ToolCallBlock(
                        id=tc_id,
                        name=fn.get("name", "") or "",
                        input=args if isinstance(args, str) else json.dumps(args),
                    )
                )
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool" and (
                messages[j].get("tool_call_id") in ids or not ids
            ):
                t = messages[j]
                blocks.append(
                    ToolResultBlock(
                        id=t.get("tool_call_id") or "",
                        name=t.get("name") or "",
                        output=t.get("content") or "",
                        state=ToolResultState.SUCCESS,
                    )
                )
                j += 1
            if not blocks:
                blocks.append(TextBlock(text="(no content)"))
            out.append(AssistantMsg(name=m.get("name") or "assistant", content=blocks))
            i = j
        else:  # user (task, memory placeholder, digest, nudges) — and any stray
            out.append(UserMsg(name=m.get("name") or "user", content=text))
            i += 1
    return out


def _messages_for_trace(call_messages: list[dict]) -> list[dict]:
    """OpenInference {role, content, tool_calls} view for the llm.call span."""
    out: list[dict] = []
    for m in call_messages:
        role = m.get("role") or "user"
        text = m.get("content") or ""
        if role == "tool":
            text = f"[tool_result {m.get('name', '')}] {text}"
        record: dict = {"role": role, "content": text}
        tool_calls = [
            {
                "name": (tc.get("function", {}) or {}).get("name", ""),
                "arguments": (tc.get("function", {}) or {}).get("arguments", ""),
            }
            for tc in m.get("tool_calls") or []
            if isinstance(tc, dict)
        ]
        if tool_calls:
            record["tool_calls"] = tool_calls
        out.append(record)
    return out


def _open_prompt_log(logs_dir: str | None):
    """Open the per-turn prompt dump (call_messages.jsonl), or None if no dir."""
    if not logs_dir:
        return None
    try:
        os.makedirs(logs_dir, exist_ok=True)
        return open(os.path.join(logs_dir, "call_messages.jsonl"), "w", encoding="utf-8")
    except OSError:
        return None


def _dump_system(prompt_log, system_msg: dict) -> None:
    """Write the (constant) system prompt once as the file's header line."""
    if prompt_log is None:
        return
    try:
        prompt_log.write(json.dumps({"system": system_msg}) + "\n")
        prompt_log.flush()
    except Exception:  # noqa: BLE001 - logging must never break the agent loop
        pass


def _dump_prompt(prompt_log, step_index: int, call_messages: list[dict]) -> None:
    """Append one turn's messages to the dump. Never raises into the loop.

    The system prompt is constant and written once via ``_dump_system``, so it
    is omitted here — each turn records only the messages that change. Messages
    are already plain dicts; they dump as-is.
    """
    if prompt_log is None:
        return
    try:
        messages = [m for m in call_messages if m.get("role") != "system"]
        line = json.dumps({"step": step_index, "messages": messages})
        prompt_log.write(line + "\n")
        prompt_log.flush()
    except Exception:  # noqa: BLE001 - logging must never break the agent loop
        pass


def _submit_only(tool_schema: list[dict]) -> list[dict]:
    """The tool surface narrowed to just submit_answer, for a reserved final turn."""
    only = [t for t in tool_schema if t.get("function", {}).get("name") == "submit_answer"]
    return only or tool_schema


def _reserve(budget: float, hard: float, frac: float = 0.15) -> float:
    """Headroom to keep for the reserved final submit turn.

    The smaller of a hard cap and a fraction of the budget, so a tiny budget
    doesn't end up reserving all of itself (which would force a submit on turn 0).
    """
    return min(hard, budget * frac)


async def run(task: TaskSpec, ctx: LoopContext) -> Trajectory:
    model = ctx.llm_agentscope
    tool_schema = ctx.tools or _scroll_tools()

    budget_wall_s = getattr(ctx.budget, "wall_time_s", None)
    budget_max_tokens = getattr(ctx.budget, "max_tokens", None)

    run_id = getattr(ctx, "run_id", None) or "local"
    session_id = f"{run_id}:{task.task_id}"
    history_db_path = getattr(ctx, "history_db_path", None)
    history_max_tokens = getattr(ctx, "history_max_tokens", None)

    # Feature knobs — overridable by env so the index ON/OFF, level-cap, and
    # forced-final ablations need no code change. The index flag is resolved
    # here (not left to the manager's own env default) because the *prompt*
    # assembly below must agree with it.
    enable_index = os.environ.get("SCROLL_EVICTION_INDEX", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )
    seed_on = os.environ.get("SCROLL_SEED_INDEX", "").strip().lower() in ("1", "true", "yes", "on")
    force_final = os.environ.get("SCROLL_FORCE_FINAL_ANSWER", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )

    # All ability-bearing context management lives in the manager: write-through
    # persistence, aging, eviction + index, the seed map, digest, REPL.
    # ``history_max_tokens=0`` disables eviction (no budget), matching the
    # pre-refactor behavior for a None budget.
    mgr = ScrollContextManager(
        history_db_path=history_db_path,
        session_id=session_id,
        run_id=run_id,
        task_id=task.task_id,
        history_max_tokens=int(history_max_tokens or 0),
        pinned=_PINNED,
        enable_index=enable_index,
        execute_timeout_s=EXECUTE_PYTHON_TIMEOUT_S,
        # Shared-tier run ids (e.g. an eval's seeded prior sessions) so a shared
        # history DB keeps scope='task' isolated to "shared tier + own session".
        shared_run_ids=tuple(getattr(ctx, "shared_run_ids", ()) or ()),
        repl_name="execute_python",
        placeholder_name="memory",
    )

    # System prompt = harness preamble (system.md: the one-tool loop contract)
    # + the manager's context-management protocol (core / index per its own
    # configuration, headline schema stripped when the index is off) + harness
    # finishing policy (loop.md: per-turn batching, submit/commit).
    base = (
        f"{prompts.load('system')}\n\n"
        f"{mgr.protocol_prompt()}\n\n"
        f"{prompts.load('loop')}"
    )
    # An eval may *append* task-specific framing via ctx.system_prompt (e.g.
    # BEAM's memory and grounding rules) — it augments the capability prompt
    # rather than replacing it, so capability guidance is never lost.
    extra = getattr(ctx, "system_prompt", None)
    system_content = f"{base}\n\n{extra}" if extra else base

    # Seed-index: fold this task's prior ``run_id='seed'`` sessions into an
    # in-context [L0]/[L1] memory map appended to the system prompt (which is
    # pinned, never evicted), so the agent sees the haystack's shape from turn
    # one. No-op when off or when the task has no seed rows.
    if seed_on:
        seed_map = mgr.seed_index_map()
        if seed_map:
            system_content = f"{system_content}\n\n{seed_map}"

    history: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": task.instruction},
    ]
    # Persist the task instruction so a later session can retrieve it.
    mgr.record_initial_prompt(history[1], step_index=-1, msg_index=1)

    steps: list[Step] = []
    tokens_in_total = 0
    tokens_out_total = 0
    terminated = TerminationReason.GAVE_UP
    final_answer: str | None = None
    forced_final_answer = False
    error_detail: str | None = None
    ms_ops: dict = {}
    totals: dict = {}
    start = time.monotonic()
    model_deadline = (start + budget_wall_s) if budget_wall_s is not None else None
    next_msg_index = 2  # 0=system, 1=task

    # Per-turn dump of the exact messages sent to the model, written next to
    # trajectory.json. Streamed (flushed) each turn so a hang/timeout still
    # leaves the prompts up to the point of failure. The constant system prompt
    # is written once as a header line; per-turn lines omit it. Best-effort.
    prompt_log = _open_prompt_log(getattr(ctx, "logs_dir", None))
    _dump_system(prompt_log, history[0])

    try:
        for step_index in range(MAX_STEPS):
            # Loop-level budget gate: stop the run cleanly when the total
            # wall-time or token budget is spent, instead of letting it grind
            # to MAX_STEPS issuing commands that can no longer make progress.
            elapsed = time.monotonic() - start
            if budget_wall_s is not None and elapsed >= budget_wall_s:
                terminated = TerminationReason.BUDGET
                break
            if budget_max_tokens is not None and (
                tokens_in_total + tokens_out_total
            ) >= budget_max_tokens:
                terminated = TerminationReason.BUDGET
                break

            # Reserve the last affordable turn for a submission: once any axis is
            # down to its final turn of headroom, restrict the tool surface to
            # submit_answer so the run commits an answer instead of burning its
            # last turn on a search and stopping empty. Stays within budget.
            tokens_used = tokens_in_total + tokens_out_total
            force_submit = force_final and step_index >= MAX_STEPS - 1
            if force_final and budget_wall_s is not None:
                force_submit = force_submit or elapsed >= budget_wall_s - _reserve(
                    budget_wall_s, _RESERVE_WALL_S
                )
            if force_final and budget_max_tokens is not None:
                force_submit = force_submit or tokens_used >= budget_max_tokens - _reserve(
                    budget_max_tokens, _RESERVE_TOKENS
                )

            with otel.loop_step(ctx.tracer, step_index=step_index):
                # The manager's pre-call pipeline: age old tool outputs, then
                # evict to the token budget (folding dropped turns into the
                # pinned index map), mutating `history` in place.
                events = mgr.manage(history)
                if events.get("evicted_msgs"):
                    try:
                        _otel_trace.get_current_span().add_event(
                            "history.evict",
                            {
                                "step_index": step_index,
                                "msgs_evicted": events["evicted_msgs"],
                                "tokens_evicted": events.get("evicted_tokens_est", 0),
                                "in_context_after": events.get(
                                    "in_context_after", len(history)
                                ),
                            },
                        )
                    except Exception:  # noqa: BLE001 - monitoring never breaks the loop
                        pass

                # Append a fresh, transient working-notes digest after each
                # assistant/tool turn (never stored in `history`, so it can't
                # accumulate). On a reserved final turn the digest carries a
                # hard submit-now directive; otherwise, only near a budget
                # limit, the soft nudge.
                budget_note = _FORCE_SUBMIT_DIRECTIVE if force_submit else budget_notice(
                    elapsed_s=elapsed,
                    wall_time_s=budget_wall_s,
                    tokens_used=tokens_used,
                    max_tokens=budget_max_tokens,
                    steps_used=step_index,
                    max_steps=MAX_STEPS,
                )
                call_messages = list(history)
                if history[-1].get("role") != "user":
                    call_messages.append(mgr.digest_message(budget_note))

                _dump_prompt(prompt_log, step_index, call_messages)

                try:
                    with otel.llm_call(
                        ctx.tracer,
                        model=ctx.model_name,
                        input_messages=_messages_for_trace(call_messages),
                    ) as span:
                        t0 = time.monotonic()
                        # Thinking mode runs the client in stream mode; collapse
                        # the async-generator response to its final accumulated
                        # chunk. Retries transient provider errors with backoff;
                        # a terminal failure raises and is handled just below.
                        response = await _call_model_with_retry(
                            model,
                            _to_agentscope(call_messages),
                            _submit_only(tool_schema) if force_submit else tool_schema,
                            deadline=model_deadline,
                        )
                        tokens_in, tokens_out = _usage(response)
                        otel.set_llm_output(
                            span,
                            prompt_tokens=tokens_in,
                            completion_tokens=tokens_out,
                            latency_ms=int((time.monotonic() - t0) * 1000),
                        )
                except Exception as exc:  # noqa: BLE001 - terminal model failure
                    # Retries are exhausted (or the wall budget is spent). End the
                    # run cleanly as ERROR so a trajectory (with the steps so far)
                    # is still written and the probe is distinguishable from a
                    # budget stop — instead of crashing out with no trajectory.
                    terminated = TerminationReason.ERROR
                    error_detail = f"{type(exc).__name__}: {exc}"
                    try:
                        _otel_trace.get_current_span().add_event(
                            "llm_call.failed", {"error": error_detail[:500]}
                        )
                    except Exception:  # noqa: BLE001 - monitoring never breaks the loop
                        pass
                    break

                tokens_in_total += tokens_in
                tokens_out_total += tokens_out

                blocks = list(getattr(response, "content", []) or [])
                thought_parts: list[str] = []
                reasoning_parts: list[str] = []
                tool_call_blocks: list[Any] = []
                for block in blocks:
                    btype = getattr(block, "type", None)
                    if btype == "text":
                        thought_parts.append(getattr(block, "text", "") or "")
                    elif btype == "thinking":
                        reasoning_parts.append(getattr(block, "thinking", "") or "")
                    elif btype == "tool_call":
                        tool_call_blocks.append(block)

                thought = "\n".join(thought_parts)
                # Reasoning content (thinking mode) is persisted for inspection
                # but deliberately NOT kept in `history` — re-sending chain of
                # thought bloats the window and isn't expected back by the API.
                reasoning = "\n".join(p for p in reasoning_parts if p)

                assistant_msg: dict = {"role": "assistant", "content": thought}
                if tool_call_blocks:
                    assistant_msg["tool_calls"] = []
                    for tc in tool_call_blocks:
                        tc_input = getattr(tc, "input", "{}")
                        assistant_msg["tool_calls"].append(
                            {
                                "id": getattr(tc, "id", None) or uuid.uuid4().hex,
                                "type": "function",
                                "function": {
                                    "name": getattr(tc, "name", "") or "",
                                    "arguments": tc_input
                                    if isinstance(tc_input, str)
                                    else json.dumps(tc_input),
                                },
                            }
                        )
                history.append(assistant_msg)
                msg_index = next_msg_index
                next_msg_index += 1
                # Write-through persist: a ⟦…⟧ fence in the text becomes the
                # turn's durable headline (and, on eviction, an index leaf);
                # reported usage re-anchors the manager's token estimate.
                mgr.record_assistant_turn(
                    assistant_msg,
                    usage={"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
                    step_index=step_index,
                    msg_index=msg_index,
                    reasoning=reasoning or None,
                )

                if force_submit:
                    # This was the reserved final turn (tools narrowed to
                    # submit_answer). If the model complied, dispatch only that
                    # submission below. If it still refused to submit, salvage its
                    # visible text as the answer rather than spending another turn.
                    submit_blocks = [
                        b for b in tool_call_blocks if getattr(b, "name", "") == "submit_answer"
                    ]
                    if submit_blocks:
                        tool_call_blocks = submit_blocks
                    else:
                        # No submit_answer even on the reserved turn. Salvage the
                        # model's visible text — or, in thinking mode where the
                        # text block is often empty and the answer sits in the
                        # chain of thought, its reasoning — so we commit *something*
                        # (the judge can extract it) rather than an empty answer.
                        final_answer = thought.strip() or reasoning.strip()
                        terminated = TerminationReason.BUDGET
                        forced_final_answer = True
                        steps.append(
                            Step(
                                index=len(steps),
                                thought=thought,
                                action=None,
                                observation=final_answer or "(no answer)",
                                tokens_in=tokens_in,
                                tokens_out=tokens_out,
                                reasoning=reasoning or None,
                            )
                        )
                        break

                if not tool_call_blocks:
                    # The model produced no tool call. Nudge it to call one next
                    # turn rather than terminating silently. Deliberately NOT
                    # persisted (the durable log records real turns, not scaffold
                    # nudges); the manager skips unrecorded messages when folding
                    # evictions into the index.
                    history.append(
                        {
                            "role": "user",
                            "content": "Call a tool (execute_python, bash, or submit_answer).",
                        }
                    )
                    next_msg_index += 1
                    steps.append(
                        Step(
                            index=len(steps),
                            thought=thought[:500],
                            action=None,
                            observation="(no tool call)",
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                            reasoning=reasoning or None,
                        )
                    )
                    continue

                call = tool_call_blocks[0]
                name = getattr(call, "name", "")
                args = _parse_input(getattr(call, "input", "{}"))
                call_id = getattr(call, "id", None) or uuid.uuid4().hex

                if name == "submit_answer":
                    final_answer = str(args.get("answer", ""))
                    terminated = TerminationReason.SUCCESS
                    forced_final_answer = force_submit
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": "Answer submitted.",
                        }
                    )
                    mgr.record_tool_call(
                        name,
                        final_answer,
                        tool_input=args,
                        tool_call_id=call_id,
                        msg_index=msg_index,
                    )
                    steps.append(
                        Step(
                            index=len(steps),
                            thought=thought,
                            action={"tool": name, "args": args},
                            observation=final_answer,
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                            reasoning=reasoning or None,
                        )
                    )
                    break

                if name == "bash":
                    command = str(args.get("command", ""))
                    remaining = (
                        budget_wall_s - (time.monotonic() - start)
                        if budget_wall_s is not None
                        else None
                    )
                    with otel.tool_call(ctx.tracer, tool_name="bash") as span:
                        t0 = time.monotonic()
                        bash_result = await run_bash(
                            ctx.environment,
                            command,
                            timeout_sec=_command_timeout(args, remaining),
                        )
                        observation = format_bash_observation(bash_result)
                        otel.set_tool_io(span, input_value=command, output_value=observation)
                        span.set_attribute("duration_ms", int((time.monotonic() - t0) * 1000))
                elif name == "execute_python":
                    source = str(args.get("source", ""))
                    with otel.tool_call(ctx.tracer, tool_name="execute_python") as span:
                        t0 = time.monotonic()
                        observation = await mgr.execute_python_async(source)
                        otel.set_tool_io(span, input_value=source, output_value=observation)
                        span.set_attribute("duration_ms", int((time.monotonic() - t0) * 1000))
                else:
                    observation = f"unknown tool: {name}"

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": observation,
                }
                history.append(tool_msg)
                mgr.record_tool_result(
                    tool_msg,
                    tool_name=name,
                    tool_input=args,
                    tool_state=ToolResultState.SUCCESS.value,
                    msg_index=msg_index,
                )
                steps.append(
                    Step(
                        index=len(steps),
                        thought=thought,
                        action={"tool": name, "args": args},
                        observation=observation,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        reasoning=reasoning or None,
                    )
                )
        else:
            terminated = TerminationReason.BUDGET

        # Backstop (no extra model call): if the reserved turn never got to fire —
        # e.g. a single turn overran the whole wall budget, so the hard gate broke
        # first — salvage the most recent visible model text as the answer rather
        # than scoring a guaranteed zero. `terminated` stays BUDGET and the
        # `forced_final_answer` flag marks it for analysis.
        if (
            force_final
            and final_answer is None
            and terminated in (TerminationReason.BUDGET, TerminationReason.GAVE_UP)
        ):
            salvaged = next(
                (s.thought for s in reversed(steps) if (s.thought or "").strip()), ""
            )
            if salvaged.strip():
                final_answer = salvaged.strip()
                forced_final_answer = True
    finally:
        totals = mgr.metrics()
        ms_ops = totals.get("ms_ops", {})
        mgr.close()
        if prompt_log is not None:
            try:
                prompt_log.close()
            except OSError:
                pass

    metrics: dict = {
        "tokens_in": tokens_in_total,
        "tokens_out": tokens_out_total,
        "wall_time_s": round(time.monotonic() - start, 2),
        "step_count": len(steps),
        "ms_ops": ms_ops,
        # Same schema as the pre-refactor loop (analysis tooling reads these):
        # `turns` = sweeps that dropped ≥1 message, `msgs` = messages dropped,
        # `max_in_context` = high-water mark of in-context messages.
        "eviction": {
            "turns": totals.get("evict_sweeps", 0),
            "msgs": totals.get("evicted_msgs", 0),
            "tokens_est": totals.get("evicted_tokens_est", 0),
            "max_in_context": max(totals.get("max_in_context", 0), len(history)),
        },
        "obs_aging": {
            "blocks": totals.get("aged_blocks", 0),
            "tokens_est": totals.get("aged_tokens_est", 0),
            "keep_turns": mgr.obs_keep_turns,
        },
    }
    if error_detail is not None:
        metrics["error"] = error_detail
    if forced_final_answer:
        metrics["forced_final_answer"] = True

    return Trajectory(
        task_id=task.task_id,
        steps=steps,
        final_answer=final_answer,
        terminated=terminated,
        metrics=metrics,
    )
