"""scroll_codact — CodeAct loop with scroll context management.

Where scroll_react is ReAct (one JSON tool call per action; ``bash`` is a
top-level tool), scroll_codact collapses the action surface into code: the
model has exactly two tools — ``execute_python`` (the transport: one Python
cell per turn, run in the persistent namespace owned by the context manager's
``ScrollRuntime``) and ``submit_answer`` (terminates). Effectful task tools
are *functions inside the namespace* (``await bash("make test")`` when the
eval attaches a sandbox), so one cell composes tool calls, control flow, and
data handling — and reaches ``ms`` for recall in the same breath.

The transport deliberately stays on the OpenAI tool-call wire format rather
than fenced code in assistant text: the manager's observation aging stubs
only ``role:"tool"`` messages, and its eviction groups "assistant +
contiguous tool results" atomically — keeping the format keeps the whole
scroll pipeline (aging, grouping, write-through persistence, FTS recall of
past actions) without library changes.

Context management is delegated to :class:`scroll_context.ScrollContextManager`
exactly as in scroll_react. Loop scaffolding (retry/streaming collapse,
AgentScope conversion, budget gates, the reserved final submit turn, prompt
dumps) is imported from scroll_react.agent — promote those helpers to a
shared ``_loop_common`` module when this graduates from sketch.
"""
from __future__ import annotations

import copy
import json
import os
import time
import uuid
from typing import Any

from agentscope.message import ToolResultState

from opentelemetry import trace as _otel_trace

from scroll_context import ScrollContextManager
from scroll_eval._tools_common import (
    budget_notice,
    format_bash_observation,
    run_bash,
    select_tools,
)
from scroll_eval.base_agents.scroll_codact import prompts
from scroll_eval.base_agents.scroll_react.agent import (
    _FORCE_SUBMIT_DIRECTIVE,
    _RESERVE_TOKENS,
    _RESERVE_WALL_S,
    DEFAULT_COMMAND_TIMEOUT_S,
    MAX_COMMAND_TIMEOUT_S,
    MIN_COMMAND_TIMEOUT_S,
    _call_model_with_retry,
    _dump_prompt,
    _dump_system,
    _messages_for_trace,
    _open_prompt_log,
    _parse_input,
    _reserve,
    _submit_only,
    _to_agentscope,
    _usage,
)
from scroll_eval.tracing import otel
from scroll_eval.types import LoopContext, Step, TaskSpec, TerminationReason, Trajectory


MAX_STEPS = int(os.environ.get("SCROLL_MAX_STEPS") or 20)

# A CodeAct cell may await long-running commands (builds, installs), so the
# cell timeout must cover the slowest awaited tool call — it matches the bash
# hard cap instead of scroll_react's 60s recall-cell budget. The loop-level
# wall gate remains the real backstop.
EXECUTE_CELL_TIMEOUT_S = float(MAX_COMMAND_TIMEOUT_S)

DEFAULT_TOOLS = ["execute_python", "submit_answer"]

# Indices 0 (system) and 1 (task) of `history` are pinned and never evicted.
_PINNED = 2

_CODACT_EXECUTE_PYTHON_DESCRIPTION = (
    "Run one Python cell in the persistent runtime namespace — this is how "
    "you act. Task tools are async functions already defined in the namespace "
    "(see 'Action tools' in the system prompt); call them with top-level "
    "await, e.g. out = await bash('ls'). Memory recall (ms.search / ms.expand "
    "/ ms.sql_query) is available in the same cell. Variables, imports and "
    "function defs persist across calls. Returns the captured stdout/stderr."
)


def _codact_tools(base: list[dict] | None) -> list[dict]:
    """The two-tool CodeAct surface, with execute_python retold as the act channel.

    Deep-copies so the registry's canonical schemas (shared with the ReAct
    agents) are never mutated. When an eval passes its own ``ctx.tools`` list,
    the same description override is applied to its execute_python entry.
    """
    tools = [copy.deepcopy(t) for t in (base or select_tools(DEFAULT_TOOLS))]
    for t in tools:
        fn = t.get("function", {})
        if fn.get("name") == "execute_python":
            fn["description"] = _CODACT_EXECUTE_PYTHON_DESCRIPTION
    return tools


def _install_action_tools(
    mgr: ScrollContextManager,
    ctx: LoopContext,
    *,
    wall_deadline: float | None,
) -> list[str]:
    """Inject the effectful task tools into the REPL namespace; return their docs.

    The returned lines become the system prompt's "Action tools" section, so
    the namespace and the prompt can't drift apart. Tools are host closures —
    this in-process wiring is exactly what couples the namespace to the host
    process; it must become an RPC shim if the executor ever moves out of
    process.
    """
    docs: list[str] = []
    env = getattr(ctx, "environment", None)
    if env is not None:

        async def bash(command: str, timeout: int = DEFAULT_COMMAND_TIMEOUT_S) -> str:
            try:
                t = int(timeout)
            except (TypeError, ValueError):
                t = DEFAULT_COMMAND_TIMEOUT_S
            t = min(t, MAX_COMMAND_TIMEOUT_S)
            if wall_deadline is not None:
                t = min(t, int(wall_deadline - time.monotonic()))
            t = max(MIN_COMMAND_TIMEOUT_S, t)
            return format_bash_observation(await run_bash(env, command, timeout_sec=t))

        mgr.runtime.namespace["bash"] = bash
        docs.append(
            f"- `await bash(command, timeout={DEFAULT_COMMAND_TIMEOUT_S})` — run a "
            "shell command in the task's Linux container; returns 'exit=N' plus "
            "combined stdout/stderr. Raise `timeout` (seconds, max "
            f"{MAX_COMMAND_TIMEOUT_S}) for slow commands, builds, or installs."
        )
    return docs


async def run(task: TaskSpec, ctx: LoopContext) -> Trajectory:
    model = ctx.llm_agentscope
    tool_schema = _codact_tools(ctx.tools)

    budget_wall_s = getattr(ctx.budget, "wall_time_s", None)
    budget_max_tokens = getattr(ctx.budget, "max_tokens", None)

    run_id = getattr(ctx, "run_id", None) or "local"
    session_id = f"{run_id}:{task.task_id}"
    history_db_path = getattr(ctx, "history_db_path", None)
    history_max_tokens = getattr(ctx, "history_max_tokens", None)

    # Feature knobs — same env surface as scroll_react so the ablations run
    # unchanged against this loop.
    enable_index = os.environ.get("SCROLL_EVICTION_INDEX", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )
    seed_on = os.environ.get("SCROLL_SEED_INDEX", "").strip().lower() in ("1", "true", "yes", "on")
    force_final = os.environ.get("SCROLL_FORCE_FINAL_ANSWER", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )

    mgr = ScrollContextManager(
        history_db_path=history_db_path,
        session_id=session_id,
        run_id=run_id,
        task_id=task.task_id,
        history_max_tokens=int(history_max_tokens or 0),
        pinned=_PINNED,
        enable_index=enable_index,
        execute_timeout_s=EXECUTE_CELL_TIMEOUT_S,
        shared_run_ids=tuple(getattr(ctx, "shared_run_ids", ()) or ()),
        repl_name="execute_python",
        placeholder_name="memory",
    )

    start = time.monotonic()
    model_deadline = (start + budget_wall_s) if budget_wall_s is not None else None

    # CodeAct wiring: effectful tools become namespace functions, and their
    # docs become the prompt's "Action tools" section. When the eval attaches
    # no sandbox (e.g. BEAM) the section is empty and the loop degrades to
    # pure recall+compute — same tool surface, nothing to remove.
    action_docs = _install_action_tools(mgr, ctx, wall_deadline=model_deadline)

    # System prompt = CodeAct preamble (system.md) + the manager's
    # context-management protocol + finishing policy (loop.md) + the action
    # tools available in the namespace this run.
    base = (
        f"{prompts.load('system')}\n\n"
        f"{mgr.protocol_prompt()}\n\n"
        f"{prompts.load('loop')}"
    )
    if action_docs:
        base = f"{base}\n\n# Action tools\n\n" + "\n".join(action_docs)
    extra = getattr(ctx, "system_prompt", None)
    system_content = f"{base}\n\n{extra}" if extra else base
    if seed_on:
        seed_map = mgr.seed_index_map()
        if seed_map:
            system_content = f"{system_content}\n\n{seed_map}"

    history: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": task.instruction},
    ]
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
    next_msg_index = 2  # 0=system, 1=task

    prompt_log = _open_prompt_log(getattr(ctx, "logs_dir", None))
    _dump_system(prompt_log, history[0])

    try:
        for step_index in range(MAX_STEPS):
            elapsed = time.monotonic() - start
            if budget_wall_s is not None and elapsed >= budget_wall_s:
                terminated = TerminationReason.BUDGET
                break
            if budget_max_tokens is not None and (
                tokens_in_total + tokens_out_total
            ) >= budget_max_tokens:
                terminated = TerminationReason.BUDGET
                break

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
                mgr.record_assistant_turn(
                    assistant_msg,
                    usage={"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
                    step_index=step_index,
                    msg_index=msg_index,
                    reasoning=reasoning or None,
                )

                if force_submit:
                    submit_blocks = [
                        b for b in tool_call_blocks if getattr(b, "name", "") == "submit_answer"
                    ]
                    if submit_blocks:
                        tool_call_blocks = submit_blocks
                    else:
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
                    history.append(
                        {
                            "role": "user",
                            "content": (
                                "Write a code cell (execute_python) or submit "
                                "(submit_answer)."
                            ),
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

                if name == "execute_python":
                    # The single dispatch branch: the cell IS the action. Any
                    # effectful tool runs inside it as an awaited namespace
                    # function, so there is nothing else to dispatch.
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
