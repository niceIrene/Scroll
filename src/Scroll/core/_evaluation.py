"""Generic evaluation framework: probe injection and result types.

Environment-specific probes (questions, ground truth, scoring) live in
each environment's tasks/ directory. This module provides the shared
infrastructure for injecting probes and collecting results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from opentelemetry import trace as otel_trace

_tracer = otel_trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProbeToolCall:
    """A single tool call made by the agent while answering a probe."""

    tool_name: str
    arguments: dict
    result: str


@dataclass
class ProbeResult:
    """Result of injecting a single probe question."""

    turn_idx: int
    question_id: str
    question: str
    agent_answer: str
    ground_truth: str
    score: float  # 0.0–1.0
    tool_calls_used: int
    tool_trace: list[ProbeToolCall] = field(default_factory=list)
    # Probe-only LM cost — diffed from the agent's lifetime usage
    # counter (``ToolState._lm_calls`` / ``_prompt_tokens`` /
    # ``_completion_tokens``) across the probe call boundary.
    # Lets reports separate "probe cost" from "turn-loop cost" cleanly.
    probe_lm_calls: int = 0
    probe_prompt_tokens: int = 0
    probe_completion_tokens: int = 0


@dataclass
class EnvSnapshot:
    """Lightweight per-turn snapshot kept by the benchmark loop.

    Env-agnostic core (``turn_idx``, ``logs``, ``extra``). Concrete envs
    may subclass to add typed fields (e.g. :class:`VendingSnapshot`); other
    envs populate ``extra`` directly. The turn-log serializer dumps
    every field via :func:`dataclasses.asdict`, so subclass fields and
    ``extra`` flow through automatically.
    """

    turn_idx: int
    logs: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class ProbeSpec:
    """Specification for a single probe question.

    When ``passive`` is True the probe does not call ``agent.answer_probe``;
    the score is taken directly from ``ground_truth_fn`` (which is expected
    to return a string parseable as a float, e.g. an env-computed reward).
    Used by envs whose evaluation is purely state-based rather than
    asking the agent a question.
    """

    question_id: str
    turn_idx: int
    question: str
    ground_truth_fn: Callable[[list[EnvSnapshot], Any], str]
    scoring_fn: Callable[[str, str], float]
    passive: bool = False


# ---------------------------------------------------------------------------
# Probe injection
# ---------------------------------------------------------------------------

def extract_tool_trace(agent, prev_log_count: int) -> list[ProbeToolCall]:
    """Extract tool calls and results from log entries added after `prev_log_count`.

    Reads from ``agent.log.entries`` (the unified ConversationLog) so the
    trace shape is uniform across the legacy ReAct path (where
    SlidingMemory mirrored every Msg into the log) and the RLM-style
    CodeAct path (where ``CodeActAgent`` appends LogEntry records
    directly).
    """
    trace: list[ProbeToolCall] = []
    try:
        entries = agent.log.entries[prev_log_count:]
    except Exception:
        return trace

    pending: dict[str, ProbeToolCall] = {}
    for entry in entries:
        if entry.tool_call:
            call_id = entry.tool_call.get("id", "")
            tc = ProbeToolCall(
                tool_name=entry.tool_call.get("name", "?"),
                arguments=entry.tool_call.get("arguments", {}) or {},
                result="",
            )
            pending[call_id] = tc
            trace.append(tc)
        if entry.tool_result:
            call_id = entry.tool_result.get("id", "")
            output = entry.tool_result.get("output", "")
            output = str(output).strip()
            if len(output) > 1000:
                output = output[:1000] + "..."
            if call_id in pending:
                pending[call_id].result = output
    return trace


def inject_probe(
    agent,
    probe: ProbeSpec,
    snapshots: list[EnvSnapshot],
    data,
) -> ProbeResult:
    """Inject a probe question into the agent and score the response.

    Uses `agent.answer_probe()` to get the agent's answer, keeping probe
    logic decoupled from agent internals.
    """

    with _tracer.start_as_current_span(f"probe.{probe.question_id}") as probe_span:
        probe_span.set_attributes({
            "probe.question_id": probe.question_id,
            "probe.question": probe.question,
        })

        ground_truth = probe.ground_truth_fn(snapshots, data)

        if probe.passive:
            try:
                score = float(ground_truth)
            except (TypeError, ValueError):
                score = probe.scoring_fn(ground_truth, ground_truth)
            probe_span.set_attributes({
                "probe.score": score,
                "probe.ground_truth": ground_truth,
                "probe.passive": True,
            })
            return ProbeResult(
                turn_idx=probe.turn_idx,
                question_id=probe.question_id,
                question=probe.question,
                agent_answer="",
                ground_truth=ground_truth,
                score=score,
                tool_calls_used=0,
                tool_trace=[],
            )

        # Equalize visibility across agents: the turn-loop fires probes
        # after ``env.step_turn()`` but any outcome logs the env
        # produced (env-specific — for vending: sales / deliveries /
        # fees; chat-memory envs typically have none) are only buffered
        # in ``_pending_env_outcomes`` for the next turn's prompt.
        # Surface them to log-based agents here so the GT's window
        # (which includes turn N) is answerable regardless of whether
        # the agent has already auto-ingested the turn.
        outcomes_block = ""
        pending = getattr(agent, "_pending_env_outcomes", None)
        if pending:
            outcomes_lines = "\n".join(f"  - {line}" for line in pending)
            outcomes_block = (
                f"Events from end of turn {probe.turn_idx} "
                f"(just completed; these are within this probe's "
                f"reporting window):\n{outcomes_lines}\n\n"
            )

        # Per-env probe-format reminders ride on the user-message
        # postscript (``BaseEnvironment.probe_user_postscript``) so
        # they sit RIGHT NEXT TO the question text where the model's
        # instruction-following attention is highest.
        env = (
            getattr(agent, "_tool_state", None)
            and agent._tool_state.env
        )
        postscript = ""
        if env is not None:
            try:
                postscript = (env.probe_user_postscript() or "").strip()
            except Exception:  # noqa: BLE001
                postscript = ""

        # Optional per-probe wrapper that lands IMMEDIATELY before
        # the question text — e.g. LongMemEvalAgent uses this to
        # prepend "Today's Date: X (latest session_idx=N)\nQuestion: ".
        # Default ("") leaves question rendering unchanged. NOT stripped:
        # the agent owns the exact whitespace so it can drop the question
        # inline or on a new line.
        try:
            question_prefix = agent.probe_user_question_prefix(probe) or ""
        except TypeError:
            question_prefix = agent.probe_user_question_prefix() or ""
        except Exception:  # noqa: BLE001
            question_prefix = ""

        # No extra separator — the prefix carries its own trailing
        # punctuation/whitespace so it can drop the question inline
        # (mem0 style: "...\nQuestion: <text>") or on a new line, as
        # the agent chooses.
        question_block = (question_prefix + probe.question) if question_prefix else probe.question

        # Strategy-level retrieval hint (per-agent override of
        # :meth:`BaseAgent.probe_user_hint`). Placed BEFORE the
        # [PROBE — qid] header / question, mirroring Anthropic's
        # claude-code <system-reminder> pattern: instructions are
        # priors the model reads, then it processes the question
        # through that lens. (Postscripts that are scorer-format
        # reminders still go AFTER the question — those are tied to
        # the specific Q.)
        #
        # ``probe`` is passed so per-question-type routers (e.g.
        # ``LongMemEvalAgent``) can dispatch the recipe by
        # ``probe.question_type``. Backward-compat: agents whose
        # override still has the zero-arg signature get a no-arg call
        # via the TypeError fallback.
        try:
            try:
                agent_hint = (agent.probe_user_hint(probe) or "").strip()
            except TypeError:
                agent_hint = (agent.probe_user_hint() or "").strip()
        except Exception:  # noqa: BLE001
            agent_hint = ""

        # Horizontal rule around the PROBE block so it's visually distinct
        # from the SYSTEM_REMINDERS above (agent_hint) and the postscript
        # below — matches the in-prompt ``──`` separator style used in
        # ``_LME_PROBE_HINT``.
        _SEP = "─" * 66
        question_text = outcomes_block
        if agent_hint:
            question_text += agent_hint + "\n\n"
        question_text += (
            f"{_SEP}\n"
            f"[PROBE — {probe.question_id}]\n"
            f"{question_block}\n"
            f"{_SEP}"
        )
        if postscript:
            question_text += "\n\n" + postscript

        prev_count = getattr(agent, "message_count", 0)

        prev_log_count = 0
        try:
            prev_log_count = len(agent.log.entries)
        except Exception:
            pass

        # Snapshot the agent's lifetime LM usage counters BEFORE the
        # probe so we can diff them after to get probe-only cost.
        # Only the CodeActAgent path populates these via the
        # ``_UsageTrackingModel`` wrapper; non-tracking agents will
        # report 0 (cleanly, no special-casing).
        ts = getattr(agent, "_tool_state", None)
        pre_lm_calls = getattr(ts, "_lm_calls", 0) if ts else 0
        pre_prompt = getattr(ts, "_prompt_tokens", 0) if ts else 0
        pre_completion = getattr(ts, "_completion_tokens", 0) if ts else 0

        agent_answer = agent.answer_probe(question_text)

        tool_calls_used = getattr(agent, "message_count", 0) - prev_count
        probe_lm_calls = (getattr(ts, "_lm_calls", 0) if ts else 0) - pre_lm_calls
        probe_prompt_tokens = (getattr(ts, "_prompt_tokens", 0) if ts else 0) - pre_prompt
        probe_completion_tokens = (getattr(ts, "_completion_tokens", 0) if ts else 0) - pre_completion

        tool_trace = extract_tool_trace(agent, prev_log_count)

        score = probe.scoring_fn(agent_answer, ground_truth)

        # Post-probe lifecycle hook. Subclasses (currently
        # ``LongMemEvalAgent``) override ``_on_probe_complete`` to distill
        # success/failure patterns from this trajectory and persist
        # them to a cross-task store for future probes. Default no-op
        # on ``BaseAgent``, so unmigrated agents are unaffected. We
        # absorb errors here — distillation is opportunistic and must
        # not break probe scoring.
        try:
            import asyncio as _asyncio
            _asyncio.run(agent._on_probe_complete(
                probe=probe,
                agent_answer=agent_answer,
                score=score,
                ground_truth=ground_truth,
            ))
        except Exception:  # noqa: BLE001
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "_on_probe_complete failed for %s", probe.question_id,
                exc_info=True,
            )

        # Output length is bounded at the model call via
        # `AgentConfig.max_output_tokens`, so we don't truncate here — the
        # stored reply matches what the scorer actually saw.
        probe_span.set_attributes({
            "probe.score": score,
            "probe.agent_answer": agent_answer,
            "probe.ground_truth": ground_truth,
            "probe.tool_calls_used": tool_calls_used,
        })

        return ProbeResult(
            session_idx=probe.session_idx,
            question_id=probe.question_id,
            question=probe.question,
            agent_answer=agent_answer,
            ground_truth=ground_truth,
            score=score,
            tool_calls_used=tool_calls_used,
            tool_trace=tool_trace,
            probe_lm_calls=probe_lm_calls,
            probe_prompt_tokens=probe_prompt_tokens,
            probe_completion_tokens=probe_completion_tokens,
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_probe_results(
    results: list[ProbeResult],
    efficiency: dict[str, Any],
    output_dir: str | Path,
    extra_summary: dict[str, Any] | None = None,
) -> None:
    """Write probe results and efficiency metrics to probe_results.json.

    ``extra_summary`` lets the caller (typically ``benchmark.py``) splice
    in env-specific summary fields obtained from
    :meth:`BaseEnvironment.summarize_probes` — e.g. vending's per-category
    A/B averages. Core keeps only the env-agnostic ``total_probes`` and
    ``avg_score`` aggregates.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "total_probes": len(results),
        "avg_score": round(
            sum(r.score for r in results) / len(results), 3
        ) if results else 0,
    }
    if extra_summary:
        summary.update(extra_summary)

    data = {
        "probes": [asdict(r) for r in results],
        "efficiency": efficiency,
        "summary": summary,
    }
    (out / "probe_results.json").write_text(json.dumps(data, indent=2, default=str))
