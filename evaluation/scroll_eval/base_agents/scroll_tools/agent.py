"""scroll_tools — the DB-without-REPL ablation arm.

A clone of scroll_react where the ONLY change is the retrieval surface: the
``execute_python`` REPL is replaced by two plain JSON tools —
``search_history`` and ``expand_turns`` — dispatched host-side to the SAME
``MemorySpace`` reads (``ms.search`` / ``ms.expand``). The model retrieves
without writing code: no persistent variables, no ``sql_query``, no in-code
aggregation — it cannot reorganize retrieved information across steps, which
is precisely the capability this arm removes. Everything else is identical to
scroll_react: ScrollContextManager construction (write-through persistence,
observation aging, token-budget eviction + index, seed-map priming, digest),
budget gates, the reserved final submit turn, tracing, and the trajectory
record.

Two deliberate redirections keep manager-owned recovery texts sensible without
touching the library: ``repl_name="expand_turns"`` (aged/cap stubs and the
digest name a tool that exists) and a tool-flavored ``index_header``. The
system prompt adds a translation note for the remaining ``ms.*`` idioms in
manager text.

NOTE: ``prompts/index.md`` here is a manual derivative of
``src/scroll_context/index.md`` with the REPL/SQL drill-down recipes rewritten
as tool calls — keep the two in sync when the library file changes.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from agentscope.message import ToolResultState

from opentelemetry import trace as _otel_trace

from scroll_context import ScrollContextManager
from scroll_eval._tools_common import budget_notice, select_tools
from scroll_eval.base_agents.scroll_tools import prompts
from scroll_eval.base_agents.scroll_react.agent import (
    _FORCE_SUBMIT_DIRECTIVE,
    _RESERVE_TOKENS,
    _RESERVE_WALL_S,
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
DEFAULT_TOOLS = ["search_history", "expand_turns", "submit_answer"]

# Indices 0 (system) and 1 (task) of `history` are pinned and never evicted.
_PINNED = 2

_SEARCH_K_DEFAULT = 10
_SEARCH_K_MAX = 50
_EXPAND_SEQS_MAX = 100

# Tool-flavored copy of the manager's _INDEX_HEADER_TMPL: same map semantics,
# but the drill-in instruction names this agent's tools instead of a REPL.
_TOOLS_INDEX_HEADER = (
    "<system-info>[memory] Your eviction index — earlier history (prior "
    "conversation turns with this user, and your own steps compacted out of "
    "this prompt) was folded into this map over rounds of compaction. "
    "Newest/finest entries are at the bottom; older spans are chunked upward "
    "into single lines whose endpoints bracket their era. The full rows are "
    "durable in your conversation history: to look up earlier history, find "
    "the relevant span here and drill into it with "
    "search_history(seq_range=[lo, hi]) or expand_turns."
)


# --- JSON tool dispatch (the arm's whole delta vs scroll_react) ---------------


def _int_pair(value: Any) -> tuple[int, int] | None:
    """Coerce a [lo, hi] JSON array to an int 2-tuple, else None."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _format_hits(rows: list[dict], k: int) -> str:
    """One line per hit — the same triage view a well-behaved REPL agent prints.

    Fields: seq, S<session>, date, kind/role, ⟦headline⟧ when present, the
    match-centred snippet (router markers like ``⟦via summary of S<n>⟧`` ride
    inside it), and a broadened marker. Appends the k-saturation note (more may
    match) or 0-hit guidance.
    """
    if not rows:
        return (
            "0 hits — try different terms, an OR of synonyms, or a seq_range "
            "sweep from the [memory] map."
        )
    provenance = getattr(rows, "provenance", "") or ""
    lines = [f"{len(rows)} hits" + (f" ({provenance})" if provenance else "") + ":"]
    for r in rows:
        step = r.get("step_index")
        head = f"seq={r.get('seq')}  S{step}  {r.get('date') or '?'}  " \
               f"{r.get('kind')}/{r.get('role')}"
        headline = r.get("headline")
        text = r.get("snippet") or r.get("content") or ""
        parts = [head, "—"]
        if headline:
            parts.insert(1, f"⟦{headline}⟧")
        parts.append(" ".join(str(text).split()))
        line = "  ".join(p for p in parts if p)
        if r.get("broadened"):
            line += "  [broadened — a lead, verify it]"
        lines.append(line)
    if len(rows) >= k:
        lines.append(
            f"[note] result filled k={k} — more turns may match; narrow the "
            "query, bound with seq_range, or raise k."
        )
    return "\n".join(lines)


def _format_expanded(rows: list[dict], requested: list[int]) -> str:
    """Full-content blocks in seq order, plus a note when seqs were not found."""
    if not rows:
        return "0 of {n} requested seqs found — check the seq ids.".format(
            n=len(requested)
        )
    blocks = []
    for r in rows:
        blocks.append(
            f"--- seq={r.get('seq')}  S{r.get('step_index')}  "
            f"{r.get('date') or '?'}  {r.get('kind')}/{r.get('role')} ---\n"
            f"{r.get('content') or ''}"
        )
    out = "\n\n".join(blocks)
    if len(rows) < len(set(requested)):
        out += (
            f"\n\n[note] {len(rows)} of {len(set(requested))} requested seqs "
            "found; the rest don't exist or aren't visible in this task."
        )
    return out


def _dispatch_search(ms, args: dict) -> str:
    """search_history: coerce args, call ms.search(scope='task'), render lines.

    scope is fixed host-side — the "forgot scope='task'" failure mode is not
    the capability under test. Argument errors come back as observations, never
    exceptions; FTS syntax errors need no handling here (ms.search sanitizes
    and falls back internally).
    """
    query = str(args.get("query") or "").strip()
    if not query:
        return "error: search_history requires a non-empty `query` string."
    try:
        k = int(args.get("k") or _SEARCH_K_DEFAULT)
    except (TypeError, ValueError):
        k = _SEARCH_K_DEFAULT
    k = max(1, min(k, _SEARCH_K_MAX))
    kind = args.get("kind")
    kind = str(kind) if kind else None
    notes: list[str] = []
    seq_range = _int_pair(args.get("seq_range"))
    if args.get("seq_range") is not None and seq_range is None:
        notes.append("[bad argument] seq_range must be [lo, hi] — ignored.")
    step_range = _int_pair(args.get("step_range"))
    if args.get("step_range") is not None and step_range is None:
        notes.append("[bad argument] step_range must be [lo, hi] — ignored.")
    try:
        rows = ms.search(
            query,
            scope="task",
            snippet=True,
            k=k,
            kind=kind,
            seq_range=seq_range,
            step_range=step_range,
        )
    except Exception as exc:  # noqa: BLE001 - argument errors become observations
        return f"error: {exc}. Check the tool's parameters."
    out = _format_hits(rows, k)
    return "\n".join(notes + [out]) if notes else out


def _dispatch_expand(ms, args: dict) -> str:
    """expand_turns: coerce seqs, call ms.expand, render full-content blocks."""
    raw = args.get("seqs")
    if not isinstance(raw, (list, tuple)) or not raw:
        return "error: expand_turns requires `seqs`, a non-empty array of seq ids."
    seqs: list[int] = []
    for s in raw:
        try:
            seqs.append(int(s))
        except (TypeError, ValueError):
            return f"error: seqs must be integers (got {s!r})."
    note = ""
    if len(seqs) > _EXPAND_SEQS_MAX:
        note = (
            f"[note] {len(seqs)} seqs requested; expanding the first "
            f"{_EXPAND_SEQS_MAX} — call again for the rest.\n"
        )
        seqs = seqs[:_EXPAND_SEQS_MAX]
    try:
        rows = ms.expand(seqs)
    except Exception as exc:  # noqa: BLE001 - argument errors become observations
        return f"error: {exc}. Check the tool's parameters."
    return note + _format_expanded(rows, seqs)


async def run(task: TaskSpec, ctx: LoopContext) -> Trajectory:
    model = ctx.llm_agentscope
    tool_schema = ctx.tools or select_tools(DEFAULT_TOOLS)

    budget_wall_s = getattr(ctx.budget, "wall_time_s", None)
    budget_max_tokens = getattr(ctx.budget, "max_tokens", None)

    run_id = getattr(ctx, "run_id", None) or "local"
    session_id = f"{run_id}:{task.task_id}"
    history_db_path = getattr(ctx, "history_db_path", None)
    history_max_tokens = getattr(ctx, "history_max_tokens", None)

    # Same env knobs as scroll_react so the index ON/OFF and forced-final
    # ablations run unchanged against this loop.
    enable_index = os.environ.get("SCROLL_EVICTION_INDEX", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )
    seed_on = os.environ.get("SCROLL_SEED_INDEX", "").strip().lower() in ("1", "true", "yes", "on")
    force_final = os.environ.get("SCROLL_FORCE_FINAL_ANSWER", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )

    # Manager construction identical to scroll_react's EXCEPT the two prompt
    # redirections (repl_name / index_header) — the ablation's parity anchor.
    mgr = ScrollContextManager(
        history_db_path=history_db_path,
        session_id=session_id,
        run_id=run_id,
        task_id=task.task_id,
        history_max_tokens=int(history_max_tokens or 0),
        pinned=_PINNED,
        enable_index=enable_index,
        shared_run_ids=tuple(getattr(ctx, "shared_run_ids", ()) or ()),
        repl_name="expand_turns",
        index_header=_TOOLS_INDEX_HEADER,
        placeholder_name="memory",
    )
    ms = mgr.runtime.memoryspace

    # System prompt = tools preamble + (index guidance when the index is on) +
    # finishing policy + the eval's task framing. Deliberately NOT
    # mgr.protocol_prompt(): core.md teaches Python-in-a-REPL, which this arm
    # removes.
    base_parts = [prompts.load("system")]
    if enable_index:
        base_parts.append(prompts.load("index"))
    base_parts.append(prompts.load("loop"))
    base = "\n\n".join(base_parts)
    extra = getattr(ctx, "system_prompt", None)
    system_content = f"{base}\n\n{extra}" if extra else base

    history: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": task.instruction},
    ]
    mgr.record_initial_prompt(history[1], step_index=-1, msg_index=1)

    # Prior-session priming, same as scroll_react: folds shared-tier (S<n>)
    # and own prior (P<n>) spans into the live index; inserts the pinned
    # [memory] placeholder (with this agent's tool-flavored header) at
    # history[_PINNED].
    if seed_on:
        mgr.prime_prior_sessions(history)

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
                                "Call a tool (search_history, expand_turns, or "
                                "submit_answer)."
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

                if name == "search_history":
                    with otel.tool_call(ctx.tracer, tool_name=name) as span:
                        t0 = time.monotonic()
                        observation = _dispatch_search(ms, args)
                        otel.set_tool_io(
                            span, input_value=json.dumps(args), output_value=observation
                        )
                        span.set_attribute("duration_ms", int((time.monotonic() - t0) * 1000))
                elif name == "expand_turns":
                    with otel.tool_call(ctx.tracer, tool_name=name) as span:
                        t0 = time.monotonic()
                        observation = _dispatch_expand(ms, args)
                        otel.set_tool_io(
                            span, input_value=json.dumps(args), output_value=observation
                        )
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
        # The ported manager.metrics() no longer bundles ms_ops; read the
        # retrieval-route counters straight from the memoryspace.
        ms_ops = totals.get("ms_ops") or ms.stats()
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
