"""summary_baseline — the rolling-summary baseline (no DB, no REPL).

The classic incremental-summarization memory agent, as a baseline: the seeded
conversation is consumed in chunks and folded into ONE continuation summary —
the first chunk creates it ("initial" mode), each later chunk updates it
("update" mode) — then a final single-shot QA call answers the question from
the summary alone. The model never sees the raw transcript at answer time;
whatever the summary failed to preserve is gone.

The summarization prompt is the English ("en") continuation-summary template
from QwenPaw (src/qwenpaw/agents/context/scroll/continuation_summary.py):
five fixed sections (Active Task / Current State / Constraints / Decisions /
Open Work), evidence-only claims, exact-identifier preservation, and a hard
"never exceed 4000 tokens" size contract. See prompts/summarize.md.

"No DB / no REPL" describes the MODEL's capabilities, not the plumbing: like
longctx_baseline, the transcript is reconstructed HOST-side from the seeded
rows in ``ctx.history_db_path`` (read-only), so the runner needs no changes
and the seed ingest stays identical across ablation arms. The model gets no
query access and writes nothing.

Budget: ``ctx.history_max_tokens`` does not bind here — the final QA prompt is
the summary itself, which the template caps at 4000 tokens. The arm-specific
knob is the summarizer's per-call input size, ``SCROLL_SUMMARY_CHUNK_TOKENS``
(default 50000): each summarization call sees one chunk of that many tokens
plus the previous summary.

Cost shape: the beam runner calls ``run()`` once PER PROBE, and every probe of
a task shares one seed DB — so the seed summary is computed ONCE per task and
cached in a sidecar file next to the DB (``<db>.summary.json``). A per-DB
asyncio lock makes concurrent sibling probes wait for the first one instead of
each re-summarizing; changing ``SCROLL_SUMMARY_CHUNK_TOKENS`` invalidates the
cache. The wall budget (``ctx.budget.wall_time_s``) is enforced between chunk
calls: past the deadline the loop stops, the run is marked BUDGET, and the QA
call still answers from the partial summary (which is never cached).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import time
from pathlib import Path

from scroll_eval.base_agents.summary_baseline import prompts
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
_DEFAULT_CHUNK_TOKENS = 50_000

_SUMMARY_PREFIX = (
    "Below is the final continuation summary of your prior conversation with "
    "the user:\n\n"
)
_EMPTY_SUMMARY = "(none)"

# The en-mode instructions from QwenPaw's build_update_prompt, verbatim.
_INSTRUCTIONS = {
    "initial": (
        "Create the first continuation summary from the newly "
        "archived context. Extract the effective task state; do not "
        "narrate the conversation."
    ),
    "update": (
        "Update the previous continuation summary using the newly "
        "archived context. Return one complete replacement state. "
        "Treat the previous summary as the baseline: preserve items "
        "that remain relevant and are not contradicted; incorporate "
        "new facts; remove items only when they are explicitly "
        "superseded, completed, withdrawn, or clearly obsolete; and "
        "move completed work out of Open Work. When source material "
        "conflicts with the previous summary, prefer the exact source "
        "material; when facts changed over time, prefer the newer "
        "state. Do not append a change log."
    ),
}

_FENCE_RE = re.compile(r"```(?:markdown|md)?\s*\n(.*?)\n```", re.DOTALL)
# Source links are code-managed in the QwenPaw design; strip any the model
# emits despite the prompt so they don't accumulate across updates.
_SOURCE_RE = re.compile(r"\[(?:seq:\d+(?:[-–]\d+)?|(?:artifact|file):[^\]]+)\]")


_CACHE_SUFFIX = ".summary.json"
# Per-(event loop, db path) locks so concurrent sibling probes summarize once.
# Keyed by the loop OBJECT (not its id — ids can be reused across loops)
# because asyncio.Lock binds to the loop that first acquires it, and tests
# (and the Harbor path) run separate asyncio.run() loops.
_summary_locks: dict[tuple[object, str], asyncio.Lock] = {}


def _chunk_tokens() -> int:
    """The summarizer's per-call input budget (tokens), env-overridable."""
    raw = os.environ.get("SCROLL_SUMMARY_CHUNK_TOKENS", "").strip()
    return int(raw) if raw.isdigit() else _DEFAULT_CHUNK_TOKENS


def _cache_lock(db_path: str) -> asyncio.Lock:
    key = (asyncio.get_running_loop(), db_path)
    lock = _summary_locks.get(key)
    if lock is None:
        lock = _summary_locks[key] = asyncio.Lock()
    return lock


def _load_summary_cache(db_path: str, chunk_tokens: int) -> dict | None:
    """The completed seed summary for this DB, or None when absent/stale."""
    try:
        data = json.loads(
            Path(db_path + _CACHE_SUFFIX).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if data.get("chunk_tokens") != chunk_tokens or not data.get("summary"):
        return None
    return data


def _write_summary_cache(db_path: str, payload: dict) -> None:
    """Atomic write (tmp + rename) so a reader never sees a partial file."""
    path = Path(db_path + _CACHE_SUFFIX)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass  # cache is an optimization; never fail the probe over it


def _load_seed_rows(db_path: str) -> list[tuple[int, str]]:
    """The seeded conversation turns as ``(seq, content)``, in seq order.

    Same read-only URI posture as longctx_baseline._load_seed_turns; the seq
    is kept because the summary template reports the covered sequence range.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT seq, content FROM conversation_history "
            "WHERE run_id='seed' ORDER BY seq"
        ).fetchall()
    finally:
        conn.close()
    return [(int(r[0]), r[1]) for r in rows if r[1]]


def _chunk_rows(
    rows: list[tuple[int, str]], chunk_chars: int
) -> list[list[tuple[int, str]]]:
    """Greedily pack whole turns into chunks of at most ``chunk_chars``.

    A single turn larger than the budget becomes its own chunk — turns are
    never split, so every summarization call sees complete turns.
    """
    chunks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    size = 0
    for seq, content in rows:
        length = len(content) + 1
        if current and size + length > chunk_chars:
            chunks.append(current)
            current, size = [], 0
        current.append((seq, content))
        size += length
    if current:
        chunks.append(current)
    return chunks


def _build_summary_prompt(
    mode: str,
    previous: str | None,
    archived: str,
    covered_seq: tuple[int, int],
) -> str:
    return prompts.load("summarize").format(
        instructions=_INSTRUCTIONS[mode],
        covered_lo=covered_seq[0],
        covered_hi=covered_seq[1],
        previous=previous if previous is not None else "(none)",
        archived=archived,
    )


def _clean_summary(text: str) -> str:
    value = text.strip()
    match = _FENCE_RE.fullmatch(value)
    if match:
        value = match.group(1).strip()
    return _SOURCE_RE.sub("", value).strip()


def _response_text(response) -> tuple[str, str]:
    """Extract ``(text, reasoning)`` from an AgentScope ChatResponse."""
    blocks = list(getattr(response, "content", []) or [])
    text = "\n".join(
        getattr(b, "text", "") or "" for b in blocks if getattr(b, "type", None) == "text"
    )
    reasoning = "\n".join(
        getattr(b, "thinking", "") or ""
        for b in blocks
        if getattr(b, "type", None) == "thinking"
    )
    return text, reasoning


async def run(task: TaskSpec, ctx: LoopContext) -> Trajectory:
    model = ctx.llm_agentscope
    start = time.monotonic()

    base = prompts.load("system")
    extra = getattr(ctx, "system_prompt", None)
    system_content = f"{base}\n\n{extra}" if extra else base

    rows = _load_seed_rows(ctx.history_db_path) if ctx.history_db_path else []
    chunk_tokens = _chunk_tokens()
    chunks = _chunk_rows(rows, chunk_tokens * _CHARS_PER_TOKEN)

    prompt_log = _open_prompt_log(getattr(ctx, "logs_dir", None))

    budget_wall_s = getattr(ctx.budget, "wall_time_s", None)
    deadline = (start + budget_wall_s) if budget_wall_s is not None else None

    terminated = TerminationReason.SUCCESS
    error_detail: str | None = None
    final_answer: str | None = None
    summary: str | None = None
    failed_updates = 0
    updates_made = 0
    cached = False
    budget_cut = False
    tokens_in = tokens_out = 0
    steps: list[Step] = []
    covered_lo = rows[0][0] if rows else 0

    async def _fold_chunks() -> None:
        """Phase 1 — fold each chunk into the rolling continuation summary."""
        nonlocal summary, failed_updates, updates_made, budget_cut
        nonlocal tokens_in, tokens_out
        for chunk in chunks:
            # Wall-budget gate between calls: chunk calls are the expensive
            # part of this arm, and _call_model_with_retry only clamps retry
            # SLEEPS to the deadline — without this check a probe would keep
            # summarizing indefinitely past wall_time_s.
            if deadline is not None and time.monotonic() >= deadline:
                budget_cut = True
                return
            mode = "initial" if summary is None else "update"
            archived = "\n".join(content for _, content in chunk)
            prompt = _build_summary_prompt(
                mode, summary, archived, (covered_lo, chunk[-1][0])
            )
            call_messages = [{"role": "user", "content": prompt}]
            _dump_prompt(prompt_log, len(steps), call_messages)
            with otel.llm_call(
                ctx.tracer,
                model=ctx.model_name,
                input_messages=_messages_for_trace(call_messages),
            ) as span:
                t0 = time.monotonic()
                response = await _call_model_with_retry(
                    model, _to_agentscope(call_messages), None, deadline=deadline
                )
                step_in, step_out = _usage(response)
                tokens_in += step_in
                tokens_out += step_out
                otel.set_llm_output(
                    span,
                    prompt_tokens=step_in,
                    completion_tokens=step_out,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
            updates_made += 1
            text, reasoning = _response_text(response)
            candidate = _clean_summary(text)
            if candidate:
                summary = candidate
            else:
                # Keep the last good summary rather than replacing it with
                # nothing — the QwenPaw design's fail-closed posture.
                failed_updates += 1
            steps.append(
                Step(
                    index=len(steps),
                    thought=f"summary {mode} over seq {covered_lo}-{chunk[-1][0]}",
                    action=None,
                    observation=summary or "(no summary)",
                    tokens_in=step_in,
                    tokens_out=step_out,
                    reasoning=reasoning or None,
                )
            )

    try:
        if rows:
            # Every probe of a task shares one seed DB, so the seed summary is
            # computed once and reused. The lock makes concurrent sibling
            # probes wait for the first computation instead of duplicating it;
            # the recheck under the lock is what turns waiters into cache hits.
            db_path = ctx.history_db_path
            cache = _load_summary_cache(db_path, chunk_tokens)
            if cache is None:
                async with _cache_lock(db_path):
                    cache = _load_summary_cache(db_path, chunk_tokens)
                    if cache is None:
                        await _fold_chunks()
                        # Only a COMPLETE summary is cached — a budget-cut or
                        # fully-failed phase 1 leaves the next probe to retry.
                        if summary is not None and not budget_cut:
                            _write_summary_cache(db_path, {
                                "chunk_tokens": chunk_tokens,
                                "summary": summary,
                                "summary_updates": updates_made,
                                "summary_failed_updates": failed_updates,
                                "transcript_turns": len(rows),
                            })
            if cache is not None:
                summary = cache["summary"]
                failed_updates = int(cache.get("summary_failed_updates", 0))
                cached = True

        # Phase 2 — single-shot QA from the summary alone.
        call_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": _SUMMARY_PREFIX + (summary or _EMPTY_SUMMARY)},
            {"role": "user", "content": task.instruction},
        ]
        _dump_system(prompt_log, call_messages[0])
        _dump_prompt(prompt_log, len(steps), call_messages)
        with otel.llm_call(
            ctx.tracer,
            model=ctx.model_name,
            input_messages=_messages_for_trace(call_messages),
        ) as span:
            t0 = time.monotonic()
            # tools=None: plain-text answer, no tool surface at all.
            response = await _call_model_with_retry(
                model, _to_agentscope(call_messages), None, deadline=deadline
            )
            step_in, step_out = _usage(response)
            tokens_in += step_in
            tokens_out += step_out
            otel.set_llm_output(
                span,
                prompt_tokens=step_in,
                completion_tokens=step_out,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        thought, reasoning = _response_text(response)
        # Thinking-mode salvage: when the text block is empty the answer often
        # sits in the chain of thought (same posture as longctx_baseline).
        final_answer = thought.strip() or reasoning.strip() or None
        if budget_cut:
            # Answered from a partial summary — degraded, and marked as such.
            terminated = TerminationReason.BUDGET
        elif final_answer is None:
            terminated = TerminationReason.GAVE_UP
        steps.append(
            Step(
                index=len(steps),
                thought=thought,
                action=None,
                observation=final_answer or "(no answer)",
                tokens_in=step_in,
                tokens_out=step_out,
                reasoning=reasoning or None,
            )
        )
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
        "step_count": len(steps),
        # Stubbed to the scroll agents' schema so run-analysis tooling
        # (scripts/beam_analysis.py) reads baseline runs without KeyErrors.
        "ms_ops": {},
        "eviction": {"turns": 0, "msgs": 0, "tokens_est": 0, "max_in_context": 3},
        "obs_aging": {"blocks": 0, "tokens_est": 0, "keep_turns": None},
        # Arm-specific: the rolling-summary shape of the run.
        # summary_updates counts calls made by THIS probe (0 on a cache hit);
        # summary_chunks is the task's full chunk count either way.
        "transcript_turns": len(rows),
        "summary_updates": updates_made,
        "summary_chunks": len(chunks),
        "summary_failed_updates": failed_updates,
        "summary_cached": cached,
        "summary_chars": len(summary or ""),
        "chunk_tokens": chunk_tokens,
    }
    if error_detail is not None:
        metrics["error"] = error_detail

    if not steps:  # no seed rows and the QA call itself raised
        steps = [
            Step(
                index=0,
                thought="",
                action=None,
                observation="(no answer)",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        ]
    return Trajectory(
        task_id=task.task_id,
        steps=steps,
        final_answer=final_answer,
        terminated=terminated,
        metrics=metrics,
    )
