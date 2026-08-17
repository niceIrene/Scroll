"""longctx_baseline — the vanilla long-context baseline (no DB, no REPL).

The BEAM paper's own "vanilla long-context" baseline, as an agent: the prior
conversation transcript is stuffed into the prompt; when it exceeds the token
budget, the earliest turns are dropped so the largest RECENT segment fits
(recency truncation). One LLM call — transcript + question → plain-text
answer. No tools, no loop, no retrieval.

"No DB / no REPL" describes the MODEL's capabilities, not the plumbing: the
transcript is reconstructed HOST-side from the seeded rows in
``ctx.history_db_path`` (read-only), so the runner needs no changes and the
seed ingest stays identical across ablation arms. The model itself gets no
query access and writes nothing.

Budget parity: ``ctx.history_max_tokens`` — the eviction budget in the scroll
arms — is the prompt-stuffing budget here. Override with
``SCROLL_LONGCTX_MAX_TOKENS`` when the serving model's real window is smaller.
"""
from __future__ import annotations

import os
import sqlite3
import time

from scroll_eval.base_agents.longctx_baseline import prompts
from scroll_eval.base_agents.scroll_react.agent import (
    _call_model_with_retry,
    _dump_prompt,
    _dump_system,
    _messages_for_trace,
    _open_prompt_log,
    _to_agentscope,
    _usage,
)
from scroll_eval.tracing import otel
from scroll_eval.types import LoopContext, Step, TaskSpec, TerminationReason, Trajectory

# Mirrors scroll_context.manager._CHARS_PER_TOKEN — the same estimate the
# scroll arms use for their eviction budget, for parity.
_CHARS_PER_TOKEN = 4
# chars/4 underestimates tokens on dense text; an API context-overflow error
# would zero the probe, so stuff conservatively.
_SAFETY = 0.9

_TRANSCRIPT_PREFIX = (
    "Below is the full transcript of your prior conversation with the user:\n\n"
)
_TRUNCATION_NOTICE = (
    "[NOTE: this transcript was truncated to fit the context window — the "
    "earliest {n} turns are omitted; it begins mid-conversation.]\n\n"
)


def _budget_tokens(ctx: LoopContext) -> int | None:
    """The prompt-stuffing token budget: env override, else history_max_tokens."""
    raw = os.environ.get("SCROLL_LONGCTX_MAX_TOKENS", "").strip()
    if raw.isdigit():
        return int(raw)
    return ctx.history_max_tokens


def _load_seed_turns(db_path: str) -> list[str]:
    """The seeded conversation turns, in order, from the shared history DB.

    Read-only URI connection: concurrent sibling probes share the file with no
    contention and no risk of writes. NOT MemorySpace.sql_query (row-capped at
    1000 — a 100K-tier conversation exceeds it) and NOT HistoryStore (opens
    read-write). Content rows are already formatted
    ``[Session N | <ISO date>] role: ...`` by the ingester.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT content FROM conversation_history "
            "WHERE run_id='seed' ORDER BY seq"
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows if r[0]]


def _fit_transcript(turns: list[str], budget_chars: int) -> tuple[str, int]:
    """Recency-truncate: drop whole turns from the HEAD until the join fits.

    Returns ``(transcript_text, n_dropped)``. When anything was dropped, the
    text begins with a notice naming the drop count. Degenerate case: if even
    the last turn alone exceeds the budget, keep that turn's tail slice rather
    than returning nothing.
    """
    total = sum(len(t) + 1 for t in turns)
    dropped = 0
    while turns and total > budget_chars and len(turns) > 1:
        total -= len(turns[0]) + 1
        turns = turns[1:]
        dropped += 1
    text = "\n".join(turns)
    if len(text) > budget_chars:
        text = text[-budget_chars:]
    if dropped:
        text = _TRUNCATION_NOTICE.format(n=dropped) + text
    return text, dropped


async def run(task: TaskSpec, ctx: LoopContext) -> Trajectory:
    model = ctx.llm_agentscope
    start = time.monotonic()

    base = prompts.load("system")
    extra = getattr(ctx, "system_prompt", None)
    system_content = f"{base}\n\n{extra}" if extra else base

    turns = _load_seed_turns(ctx.history_db_path) if ctx.history_db_path else []
    budget_tokens = _budget_tokens(ctx)
    if budget_tokens:
        overhead = len(system_content) + len(task.instruction) + len(_TRANSCRIPT_PREFIX)
        budget_chars = max(
            1000, int(budget_tokens * _CHARS_PER_TOKEN * _SAFETY) - overhead
        )
    else:
        budget_chars = sum(len(t) + 1 for t in turns) + 1  # unbounded: keep all
    transcript, dropped = _fit_transcript(list(turns), budget_chars)

    call_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": _TRANSCRIPT_PREFIX + transcript},
        {"role": "user", "content": task.instruction},
    ]

    prompt_log = _open_prompt_log(getattr(ctx, "logs_dir", None))
    _dump_system(prompt_log, call_messages[0])
    _dump_prompt(prompt_log, 0, call_messages)

    budget_wall_s = getattr(ctx.budget, "wall_time_s", None)
    deadline = (start + budget_wall_s) if budget_wall_s is not None else None

    terminated = TerminationReason.SUCCESS
    error_detail: str | None = None
    final_answer: str | None = None
    thought = ""
    reasoning = ""
    tokens_in = tokens_out = 0
    try:
        with otel.llm_call(
            ctx.tracer,
            model=ctx.model_name,
            input_messages=_messages_for_trace(call_messages),
        ) as span:
            t0 = time.monotonic()
            # tools=None: single-shot plain-text answer, no tool surface at all.
            response = await _call_model_with_retry(
                model, _to_agentscope(call_messages), None, deadline=deadline
            )
            tokens_in, tokens_out = _usage(response)
            otel.set_llm_output(
                span,
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        blocks = list(getattr(response, "content", []) or [])
        thought = "\n".join(
            getattr(b, "text", "") or "" for b in blocks if getattr(b, "type", None) == "text"
        )
        reasoning = "\n".join(
            getattr(b, "thinking", "") or ""
            for b in blocks
            if getattr(b, "type", None) == "thinking"
        )
        # Thinking-mode salvage: when the text block is empty the answer often
        # sits in the chain of thought (same posture as scroll_react's forced
        # final turn).
        final_answer = thought.strip() or reasoning.strip() or None
        if final_answer is None:
            terminated = TerminationReason.GAVE_UP
    except Exception as exc:  # noqa: BLE001 - terminal model failure after retries
        terminated = TerminationReason.ERROR
        error_detail = f"{type(exc).__name__}: {exc}"
    finally:
        if prompt_log is not None:
            try:
                prompt_log.close()
            except OSError:
                pass

    metrics: dict = {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "wall_time_s": round(time.monotonic() - start, 2),
        "step_count": 1,
        # Stubbed to the scroll agents' schema so run-analysis tooling
        # (scripts/beam_analysis.py) reads baseline runs without KeyErrors.
        "ms_ops": {},
        "eviction": {"turns": 0, "msgs": 0, "tokens_est": 0, "max_in_context": 3},
        "obs_aging": {"blocks": 0, "tokens_est": 0, "keep_turns": None},
        # Arm-specific: what the model actually saw.
        "transcript_turns": len(turns) - dropped,
        "transcript_dropped_turns": dropped,
        "transcript_chars": len(transcript),
        "truncated": dropped > 0,
    }
    if error_detail is not None:
        metrics["error"] = error_detail

    steps = [
        Step(
            index=0,
            thought=thought,
            action=None,
            observation=final_answer or "(no answer)",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            reasoning=reasoning or None,
        )
    ]
    return Trajectory(
        task_id=task.task_id,
        steps=steps,
        final_answer=final_answer,
        terminated=terminated,
        metrics=metrics,
    )
