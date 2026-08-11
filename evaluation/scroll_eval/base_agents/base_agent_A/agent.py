"""base_agent_A — ReAct on AgentScope primitives.

Uses agentscope.model.ChatModelBase for LLM calls and constructs Msg history
manually. Two tools: bash (runs in the Harbor container) and submit_answer
(ends the task). Minimal ReAct baseline: no scroll context management.

API shape notes (verified against local agentscope source):
- Msg is a Pydantic model: Msg(name=str, content=list[ContentBlock], role=str)
  Plain strings are auto-converted to [TextBlock(text=...)] via _to_blocks().
- ChatResponse.content is Sequence[TextBlock | ToolCallBlock | ...], NOT a
  plain string. There is no .tool_calls attribute.
- ToolCallBlock.input is a raw JSON string (not a dict).
- ToolResultBlock requires an `id` field matching the ToolCallBlock.id.
- ToolResultBlock has type="tool_result" — NOT allowed in user messages (only
  "text" and "data" are permitted). Tool results are appended to the last
  assistant message instead, following AgentScope's own _save_to_context pattern.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from agentscope.message import (
    Msg,
    SystemMsg,
    UserMsg,
    AssistantMsg,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultState,
)

from scroll_eval.tracing import otel
from scroll_eval.types import LoopContext, Step, TaskSpec, Trajectory, TerminationReason
from scroll_eval.base_agents.base_agent_A import prompts
from scroll_eval._tools_common import OPENAI_TOOLS_SCHEMA as _TOOLS_SCHEMA


MAX_STEPS = 30

# Default per-bash-command timeout (seconds); capped by the loop to the
# remaining wall-time budget.
DEFAULT_BASH_TIMEOUT_S = 300


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


async def _run_bash(environment: Any, command: str, timeout_sec: int) -> str:
    from scroll_eval._tools_common import run_bash, format_bash_observation
    return format_bash_observation(
        await run_bash(environment, command, timeout_sec=timeout_sec)
    )


def _args_str(raw: Any) -> str:
    """ToolCallBlock.input is a JSON string or dict; return it as a string."""
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw)
    except (TypeError, ValueError):
        return str(raw)


def _trace_messages_from(msg: Any) -> list[dict]:
    """Flatten one AgentScope Msg into one or more trace messages.

    A Msg's content is a str or a list of blocks. AgentScope packs tool calls
    (ToolCallBlock) and — per this family's design — tool *results*
    (ToolResultBlock) into the assistant message. For tracing we surface the
    assistant text + tool_calls as one message, and emit each tool result as a
    separate ``role="tool"`` message so Phoenix shows distinct tool rows
    instead of empty assistant rows.
    """
    role = getattr(msg, "role", "")
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return [{"role": role, "content": content}]

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    tool_results: list[str] = []
    for b in content or []:
        btype = getattr(b, "type", None)
        if btype == "text":
            text_parts.append(getattr(b, "text", "") or "")
        elif btype == "tool_result":
            out = getattr(b, "output", "")
            if out:
                tool_results.append(str(out))
        elif btype == "tool_call":
            tool_calls.append(
                {
                    "name": getattr(b, "name", ""),
                    "arguments": _args_str(getattr(b, "input", "")),
                }
            )

    out: list[dict] = []
    primary: dict = {"role": role, "content": "\n".join(p for p in text_parts if p)}
    if tool_calls:
        primary["tool_calls"] = tool_calls
    # Skip an empty assistant placeholder (it carried only a tool result).
    if primary["content"] or tool_calls:
        out.append(primary)
    out.extend({"role": "tool", "content": r} for r in tool_results)
    return out


def _trace_input_messages(history: list[Any]) -> list[dict]:
    """Simplify the Msg history into trace messages for tracing."""
    out: list[dict] = []
    for m in history:
        out.extend(_trace_messages_from(m))
    return out


def _trace_output_message(thought: str, tool_call_blocks: list[Any]) -> dict:
    """Build the assistant output message for tracing from the response blocks."""
    item: dict = {"role": "assistant", "content": thought}
    if tool_call_blocks:
        item["tool_calls"] = [
            {
                "name": getattr(tc, "name", ""),
                "arguments": _args_str(getattr(tc, "input", "")),
            }
            for tc in tool_call_blocks
        ]
    return item


async def run(task: TaskSpec, ctx: LoopContext) -> Trajectory:
    model = ctx.llm_agentscope

    # Build initial history using the factory helpers (SystemMsg / UserMsg)
    # which handle string -> [TextBlock] conversion internally.
    history: list[Msg] = [
        SystemMsg(name="system", content=prompts.load("system")),
        UserMsg(name="user", content=task.instruction),
    ]

    steps: list[Step] = []
    tokens_in_total = 0
    tokens_out_total = 0
    terminated = TerminationReason.GAVE_UP
    final_answer: str | None = None
    start = time.monotonic()

    budget = ctx.budget
    wall_time_s = getattr(budget, "wall_time_s", None)
    max_tokens = getattr(budget, "max_tokens", None)

    for i in range(MAX_STEPS):
        # Budget gates, checked before spending more on this step.
        elapsed = time.monotonic() - start
        if wall_time_s is not None and elapsed >= wall_time_s:
            terminated = TerminationReason.BUDGET
            break
        if max_tokens is not None and (tokens_in_total + tokens_out_total) >= max_tokens:
            terminated = TerminationReason.BUDGET
            break

        with otel.loop_step(ctx.tracer, step_index=i):
            # --- LLM call ---------------------------------------------------
            with otel.llm_call(
                ctx.tracer,
                model=ctx.model_name,
                input_messages=_trace_input_messages(history),
            ) as span:
                t0 = time.monotonic()
                response = await model(history, tools=_TOOLS_SCHEMA)
                tokens_in = (
                    getattr(getattr(response, "usage", None), "input_tokens", 0) or 0
                )
                tokens_out = (
                    getattr(getattr(response, "usage", None), "output_tokens", 0) or 0
                )

                # --- Parse response blocks ----------------------------------
                # ChatResponse.content is Sequence[TextBlock | ToolCallBlock | ...]
                response_blocks = list(getattr(response, "content", []) or [])
                thought_parts: list[str] = []
                tool_call_blocks: list[Any] = []
                for block in response_blocks:
                    btype = getattr(block, "type", None)
                    if btype == "text":
                        thought_parts.append(getattr(block, "text", "") or "")
                    elif btype == "tool_call":
                        tool_call_blocks.append(block)
                thought = "\n".join(thought_parts)

                otel.set_llm_output(
                    span,
                    output_messages=[
                        _trace_output_message(thought, tool_call_blocks)
                    ],
                    prompt_tokens=tokens_in,
                    completion_tokens=tokens_out,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )

            tokens_in_total += tokens_in
            tokens_out_total += tokens_out

            # Append assistant message with the response blocks.
            # We build real AgentScope TextBlock/ToolCallBlock objects only when
            # interacting with Msg — for reading we used duck-typed access above.
            assistant_content: list[Any] = []
            if thought:
                assistant_content.append(TextBlock(text=thought))
            for tc in tool_call_blocks:
                # ToolCallBlock requires id (str) and input (JSON string).
                tc_id = getattr(tc, "id", None) or uuid.uuid4().hex
                tc_input = getattr(tc, "input", "{}")
                if isinstance(tc_input, dict):
                    tc_input = json.dumps(tc_input)
                assistant_content.append(
                    ToolCallBlock(
                        id=tc_id,
                        name=getattr(tc, "name", ""),
                        input=tc_input,
                    )
                )

            if not assistant_content:
                assistant_content.append(TextBlock(text="(no content)"))

            history.append(AssistantMsg(name="assistant", content=assistant_content))

            # --- No tool call -----------------------------------------------
            if not tool_call_blocks:
                history.append(
                    UserMsg(
                        name="user",
                        content="Call a tool (bash or submit_answer).",
                    )
                )
                steps.append(
                    Step(
                        index=i,
                        thought=thought[:500],
                        action=None,
                        observation="(no tool call)",
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                    )
                )
                continue

            # --- Dispatch first tool call -----------------------------------
            call = tool_call_blocks[0]
            name = getattr(call, "name", "")
            args = _parse_input(getattr(call, "input", "{}"))
            call_id = getattr(call, "id", None) or uuid.uuid4().hex

            if name == "submit_answer":
                final_answer = str(args.get("answer", ""))
                terminated = TerminationReason.SUCCESS

                # Append tool result into the last assistant message.
                # ToolResultBlock must not go in a user message (Pydantic
                # validation forbids type="tool_result" in user content).
                history[-1].content.append(
                    ToolResultBlock(
                        id=call_id,
                        name=name,
                        output="Answer submitted.",
                        state=ToolResultState.SUCCESS,
                    )
                )
                steps.append(
                    Step(
                        index=i,
                        thought=thought,
                        action={"tool": name, "args": args},
                        observation=final_answer,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                    )
                )
                break

            if name == "bash":
                # Cap the command timeout to the remaining wall-time budget so
                # one command can't run past the task's total budget.
                cap = DEFAULT_BASH_TIMEOUT_S
                if wall_time_s is not None:
                    cap = max(1, min(cap, int(wall_time_s - (time.monotonic() - start))))
                command = str(args.get("command", ""))
                with otel.tool_call(ctx.tracer, tool_name=name) as span:
                    t1 = time.monotonic()
                    observation = await _run_bash(ctx.environment, command, cap)
                    span.set_attribute("duration_ms", int((time.monotonic() - t1) * 1000))
                    otel.set_tool_io(
                        span, input_value=command, output_value=observation
                    )
            else:
                observation = f"unknown tool: {name}"

            # Append tool result into the last assistant message (same pattern
            # as AgentScope's internal _save_to_context).
            history[-1].content.append(
                ToolResultBlock(
                    id=call_id,
                    name=name,
                    output=observation,
                    state=ToolResultState.SUCCESS,
                )
            )

            steps.append(
                Step(
                    index=i,
                    thought=thought,
                    action={"tool": name, "args": args},
                    observation=observation,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )
            )
    else:
        terminated = TerminationReason.BUDGET

    return Trajectory(
        task_id=task.task_id,
        steps=steps,
        final_answer=final_answer,
        terminated=terminated,
        metrics={
            "tokens_in": tokens_in_total,
            "tokens_out": tokens_out_total,
            "wall_time_s": round(time.monotonic() - start, 2),
            "step_count": len(steps),
        },
    )
