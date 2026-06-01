"""LongMemEval probe construction + LLM-judge scoring.

A LongMemEval QA item produces exactly one probe — fired after the
last haystack session — whose ``scoring_fn`` is an LLM judge using
the same templates as the official ``evaluate_qa.py``. Because the
judge call goes out to OpenAI (or an OpenAI-compatible vLLM server),
the score for one probe is a single network call wrapped by
``backoff`` against rate limits. Costs are small: ~200 input tokens
+ 5 output tokens per probe.

The active item / config pair is registered per-run via
:func:`set_active_probe` (called from
:meth:`LongMemEvalEnv.__init__`); ``get_probes_for_turn`` returns
the probe only on the final turn. Single-process re-runs are safe because
each :func:`run_single` constructs a fresh env, which overwrites the
active probe.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from opentelemetry import trace as otel_trace

from Scroll.core import EnvSnapshot, ProbeSpec
from Scroll.benchmarks.longmemeval.env import LongMemEvalEnvConfig
from Scroll.benchmarks.longmemeval.datasource import LongMemEvalItem

_log = logging.getLogger(__name__)
_tracer = otel_trace.get_tracer(__name__)


from Scroll.benchmarks.longmemeval.agents.prompts import (  # noqa: E402
    LME_PROBE_USER_POSTSCRIPT,
    _ABSTENTION_POSTSCRIPT,
    _QTYPE_POSTSCRIPT,
)

__all__ = [
    "LME_PROBE_USER_POSTSCRIPT",
    "PROBES",
    "active_is_abstention",
    "active_question_type",
    "compose_user_postscript",
    "compute_efficiency_metrics",
    "get_probes_for_turn",
    "set_active_probe",
]


# ---------------------------------------------------------------------------
# Active-probe registry (per-run, single-process)
# ---------------------------------------------------------------------------

_active_probe: ProbeSpec | None = None
_active_qtype: str | None = None
_active_is_abstention: bool = False
# Mirrors ``LongMemEvalEnvConfig.agent_during_ingestion`` for the active
# run. Under the SCROLL-pure path (``False``) the probe fires
# end-of-task via :meth:`LongMemEvalEnv.get_end_of_task_probes` and
# the per-turn registry returns ``[]``. Under the legacy path
# (``True``) the per-turn registry returns the probe on the ``+1``
# turn.
_active_agent_during_ingestion: bool = False


def active_question_type() -> str | None:
    """Question type of the currently registered probe (or ``None``)."""
    return _active_qtype


def active_is_abstention() -> bool:
    """Whether the currently registered probe is an abstention item."""
    return _active_is_abstention


def compose_user_postscript() -> str:
    """User-turn postscript = universal hint + per-qtype nudge if any.

    Reads from the module-level active-probe registry; safe to call
    only after :func:`set_active_probe` has run (i.e. inside an
    LME run loop). Falls back to the universal hint alone if the
    active question type has no registered nudge.
    """
    parts = [LME_PROBE_USER_POSTSCRIPT]
    if _active_is_abstention:
        parts.append(_ABSTENTION_POSTSCRIPT)
    elif _active_qtype and _active_qtype in _QTYPE_POSTSCRIPT:
        parts.append(_QTYPE_POSTSCRIPT[_active_qtype])
    return "\n\n".join(parts)


def set_active_probe(item: LongMemEvalItem, cfg: LongMemEvalEnvConfig) -> None:
    """Register the probe for this run; called from env.__init__."""
    global _active_probe, _active_qtype, _active_is_abstention
    global _active_agent_during_ingestion
    _active_probe = _build_probe(item, cfg)
    _active_qtype = item.question_type
    _active_is_abstention = item.is_abstention
    _active_agent_during_ingestion = bool(
        getattr(cfg, "agent_during_ingestion", False)
    )


PROBES: list[ProbeSpec] = []  # filled per-run via set_active_probe


def get_probes_for_turn(turn_idx: int) -> list[ProbeSpec]:
    # New SCROLL-pure path: probe fires end-of-task (see
    # :meth:`LongMemEvalEnv.get_end_of_task_probes`), not on a turn.
    if not _active_agent_during_ingestion:
        return []
    # Legacy path: probe fires on the +1 probe-only turn.
    if _active_probe is None or _active_probe.turn_idx != turn_idx:
        return []
    return [_active_probe]


def compute_efficiency_metrics(daily_action_logs: list[list[str]]) -> dict:
    """Per-turn action stats. LME has no env actions (the agent only
    acts at probe time), so ``avg_actions_per_day`` is always 0; we
    keep ``total_days`` for run-shape sanity. Token / LM-call cost is
    added downstream by ``benchmark.py`` from ``self._tool_state``.
    """
    total_days = len(daily_action_logs)
    if total_days == 0:
        return {"total_days": 0, "avg_actions_per_day": 0}
    total_actions = sum(len(actions) for actions in daily_action_logs)
    return {
        "total_days": total_days,
        "avg_actions_per_day": round(total_actions / total_days, 2),
    }


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------


def _build_probe(item: LongMemEvalItem, cfg: LongMemEvalEnvConfig) -> ProbeSpec:
    qid = item.question_id
    qtype = item.question_type
    answer = item.answer
    is_abs = item.is_abstention
    judge_cfg = {
        "model": cfg.judge_model,
        "api_key_env": cfg.judge_api_key_env,
        "api_base": cfg.judge_api_base,
        "api_base_env": cfg.judge_api_base_env,
    }

    def gt_fn(snapshots: list[EnvSnapshot], data) -> str:
        return answer

    def scoring_fn(agent_answer: str, ground_truth: str) -> float:
        return _judge_score(
            qtype=qtype,
            question=item.question,
            answer=ground_truth,
            response=agent_answer,
            abstention=is_abs,
            judge_cfg=judge_cfg,
        )

    question_text = (
        f"As of {item.question_date}, please answer the following based on "
        f"everything you've observed across the chat sessions so far.\n\n"
        f"Question: {item.question}\n\n"
        "Answer from what you find in the chat. Connecting two stated "
        "facts to reach a third is fine (e.g. 'user uses Cartwheel app' "
        "+ 'Cartwheel = Target' → 'redeemed at Target') — that's reading "
        "context, not fabricating. Only abstain when the chat truly "
        "contains nothing relevant; do NOT abstain just because the "
        "answer requires one inferential step from stated facts."
    )

    return ProbeSpec(
        question_id=qid,
        # ``turn_idx`` only matters under the legacy
        # ``agent_during_ingestion=True`` path: the probe rides on the
        # dedicated probe-only turn ``total_sessions+1`` (one iteration
        # past the last haystack chat session), whose ``run_turn``
        # exposes "no chat session today" so the agent's turn-loop
        # history at probe time isn't dominated by the last ingest's
        # transcript. The SCROLL-pure path fires the probe end-of-task
        # and ignores this field.
        turn_idx=item.total_sessions + 1,
        question=question_text,
        ground_truth_fn=gt_fn,
        scoring_fn=scoring_fn,
        passive=False,
    )


# ---------------------------------------------------------------------------
# Judge — mirrors the per-task templates in
# external/longmemeval/src/evaluation/evaluate_qa.py
# ---------------------------------------------------------------------------


# Judge templates — relaxed beyond LME upstream defaults to handle:
#   (1) FALSE NEGATIVES on format/tense/article variation
#       (e.g., GT="The GR-90 trail." vs response="GR-90"; GT="$2,000" vs
#       response cites "$2,000" verbatim in Notes evidence). Upstream
#       template said "answer yes if response contains the correct answer"
#       which qwen3-max interpreted strictly as substring-equal.
#   (2) FALSE POSITIVES on abstention masquerading as correct answer
#       (e.g., agent says "I don't have that information" but GT is a
#       concrete answer — qwen3-max sometimes scored this YES).
#

_DEFAULT_PREAMBLE = (
    "You are grading a model's answer to a memory-recall question. "
    "The model has access to past chat sessions and must recall facts "
    "from them.\n\n"
    "CORE PRINCIPLE — semantic equivalence: judge by MEANING, not "
    "exact words. Answer YES if the response addresses the same "
    "factual claim as the ground truth, even with different "
    "vocabulary, units, formatting, or extra surrounding context.\n\n"
    "BIAS CHECK: you tend to say 'no' too quickly. Before concluding "
    "'no', verify the answer is truly wrong — not just differently "
    "worded. When in doubt, lean YES.\n\n"
    "Score YES if the response contains the correct answer or its key "
    "information — including when:\n"
    "  - phrased differently / paraphrased (synonyms, re-wording)\n"
    "  - using different tense, articles, or punctuation "
    "(e.g., 'has' vs 'had'; 'GR-90' vs 'The GR-90 trail.')\n"
    "  - formatted differently ('$2000' vs '$2,000'; '9:45am' vs '9:45 AM'; "
    "'Feb 1st' = 'February 1, 2023' = 'on Feb 1')\n"
    "  - approximate unit-equivalent ('14 weeks' ≈ '3 months'; "
    "'6 months' ≈ 'half a year'; '22 days' matches '3 weeks'; "
    "'8 months and 20 days' ≈ '9 months')\n"
    "  - generously rounded ('7 months and 16 days' ≈ '8 months'; "
    "'about nine months' ≈ '9 months')\n"
    "  - hedged ('at least 3', 'approximately', a range that includes "
    "the correct value)\n"
    "  - a SUPERSET (correct answer plus extra details / qualifiers); "
    "extra context is fine unless it's a direct contradiction\n"
    "  - more specific than the ground truth (captures the exact "
    "item/place/name but omits a broader container — still correct)\n"
    "  - presented inside a quote / Notes / stdout block (the model "
    "retrieved + surfaced the correct value as evidence; this counts)\n"
    "  - if GT='0' or 'nothing found' and the response says 'not "
    "enough information' (or vice versa) — these are equivalent on "
    "counting questions\n\n"
    "Score NO if the response:\n"
    "  - gives a wrong factual value (e.g., GT='$400,000' but model "
    "says '$350,000') that is NOT covered by the equivalences above\n"
    "  - **abstains** ('I don't have that information', 'no "
    "information found', 'cannot answer') when the question has a "
    "definite answer — abstention is WRONG on questions that are "
    "answerable from the chat history\n"
    "  - misses the key element of the ground truth (the core claim "
    "is unaddressed or contradicted)"
)


_JUDGE_OUTPUT_INSTRUCTION = (
    "Reason step-by-step inside <judge_thinking> tags first. Walk "
    "through:\n"
    "  1. What is the core factual claim of the correct answer?\n"
    "  2. Does the response address that same claim (possibly in "
    "different words)?\n"
    "  3. Is the response a superset / more-specific variant / unit "
    "conversion that's still equivalent?\n"
    "  4. For numbers: does the core number match, ignoring "
    "hedging/qualifiers/units?\n"
    "  5. For abstention questions: does the response effectively "
    "decline to answer?\n"
    "Only conclude 'no' if, after this check, a core concept is "
    "unaddressed or contradicted.\n\n"
    "After the closing </judge_thinking> tag, output exactly one "
    "word on a new line: ``yes`` or ``no`` (lowercase, no other "
    "text)."
)

_PREFERENCE_PREAMBLE = (
    "You are grading a model's RECOMMENDATION on a preference question. "
    "The 'rubric' below describes what the user would prefer (their stated "
    "tastes, constraints, or recurring interests).\n\n"
    "CORE PRINCIPLE — judge by overall thrust, not keyword scanning. The "
    "response is correct if its MAIN suggestions align with what the user "
    "wants. A minor incidental mention of a 'not-preferred' thing is fine "
    "if the bulk of the response respects the rubric.\n\n"
    "BIAS CHECK: you tend to say 'no' too quickly on preference questions. "
    "Acknowledge: the rubric is a GUIDE, not a checklist; the model "
    "does NOT need to name every specific tool / mention every preference "
    "to pass.\n\n"
    "Score YES if the response:\n"
    "  - demonstrates awareness of the user's personal context\n"
    "    (preferences, habits, interests) — even if only some rubric\n"
    "    points are addressed\n"
    "  - is grounded in user-stated info (not generic best-practice)\n"
    "  - mentions a phone/app as a MEANS to a preferred activity (e.g.,\n"
    "    meditation app for sleep) — judge by activity, not delivery\n"
    "    mechanism\n"
    "  - includes adjacent/related interests alongside the preferred\n"
    "    ones (additive, not contradictory)\n"
    "  - 'may not prefer' rubric items: secondary/context-dependent\n"
    "    inclusion is fine; only flat violations of strong dislikes hurt\n\n"
    "Score NO if the response:\n"
    "  - gives a generic recommendation that ignores the user's preferences\n"
    "  - abstains ('I don't have that info') — the question expects a\n"
    "    personalized recommendation, even from related (not-identical)\n"
    "    topics\n"
    "  - centrally suggests something the user explicitly DISLIKED\n"
    "    (occasional incidental references are fine)"
)


_ABSTENTION_PREAMBLE = (
    "You are grading whether a model correctly identified an UNANSWERABLE "
    "question. The user never discussed this topic in the chat history, so "
    "the correct behavior is to abstain.\n\n"
    "KEY TEST: does the response REFUSE to answer the question? If yes, "
    "it matches the abstention ground truth — regardless of what partial "
    "context is mentioned or omitted alongside.\n\n"
    "Score YES if the model:\n"
    "  - explicitly says it doesn't have the information\n"
    "  - says the information is incomplete / missing\n"
    "  - says the topic was never discussed\n"
    "  - declines to answer for lack of evidence\n"
    "  - says 'no record of X' or 'only have plans for X, not actual\n"
    "    dates' — any phrasing that conveys 'I can't answer this'\n"
    "  - mentions partial related context but still refuses the\n"
    "    specific question asked\n\n"
    "Score NO if the model gives a concrete answer (named X, picked a value, "
    "made a count) — that's fabrication on an unanswerable question. The "
    "presence of hedging words doesn't help if a definite value is committed."
)


_JUDGE_TEMPLATES = {
    # single-session-user / single-session-assistant / multi-session
    "default": (
        _DEFAULT_PREAMBLE + "\n\n"
        + _JUDGE_OUTPUT_INSTRUCTION + "\n\n"
        "Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
    ),
    "temporal-reasoning": (
        _DEFAULT_PREAMBLE + "\n\n"
        "Additional rule for this temporal question: do NOT penalize "
        "off-by-one errors on day/week/month counts. If GT='18 days' and "
        "model says '19 days', score YES. Same-day event ordering swaps "
        "are acceptable. References like 'last weekend' / 'last Wednesday' "
        "are imprecise — accept the response if it falls in a ~10-day "
        "window around the literal interpretation.\n\n"
        + _JUDGE_OUTPUT_INSTRUCTION + "\n\n"
        "Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
    ),
    "knowledge-update": (
        _DEFAULT_PREAMBLE + "\n\n"
        "Additional rule for this knowledge-update question: the user has "
        "REVISED an earlier answer. Score YES as long as the model's final "
        "answer is the LATEST/UPDATED value. Mentioning the older value "
        "alongside is fine — the test is which value the model COMMITS to "
        "as the answer, not which values it mentions.\n\n"
        + _JUDGE_OUTPUT_INSTRUCTION + "\n\n"
        "Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
    ),
    "single-session-preference": (
        _PREFERENCE_PREAMBLE + "\n\n"
        + _JUDGE_OUTPUT_INSTRUCTION + "\n\n"
        "Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}"
    ),
    "abstention": (
        _ABSTENTION_PREAMBLE + "\n\n"
        + _JUDGE_OUTPUT_INSTRUCTION + "\n\n"
        "Question: {question}\n\nExplanation (why unanswerable): {answer}\n\nModel Response: {response}"
    ),
}


def _judge_prompt(qtype: str, question: str, answer: str, response: str, abstention: bool) -> str:
    if abstention:
        tmpl = _JUDGE_TEMPLATES["abstention"]
    elif qtype in _JUDGE_TEMPLATES:
        tmpl = _JUDGE_TEMPLATES[qtype]
    else:
        # Covers single-session-user / single-session-assistant / multi-session
        tmpl = _JUDGE_TEMPLATES["default"]
    return tmpl.format(question=question, answer=answer, response=response)


def _judge_score(
    qtype: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool,
    judge_cfg: dict,
) -> float:
    """Call the configured judge and return 1.0 / 0.0.

    Network failures => 0.0 with a warning. We prefer scoring the run
    as 0.0 over crashing the entire benchmark — the run can be
    re-judged offline by re-reading ``probe_results.json`` if needed.
    """
    prompt = _judge_prompt(qtype, question, answer, response, abstention)

    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover
        _log.warning("openai not installed — judge skipped, scoring 0.0")
        return 0.0

    api_key = os.getenv(judge_cfg["api_key_env"], "")
    if not api_key:
        _log.warning(
            "judge api key env %s is empty — scoring 0.0; set %s to enable judging",
            judge_cfg["api_key_env"], judge_cfg["api_key_env"],
        )
        return 0.0

    _base_env = judge_cfg.get("api_base_env")
    base_url = (
        judge_cfg.get("api_base")
        or (os.environ.get(_base_env) if _base_env else None)
    )
    client = OpenAI(api_key=api_key, base_url=base_url)

    # `enable_thinking` is a Dashscope-only extra_body field; sending it to
    # the OpenAI endpoint returns HTTP 400. Gate on the base_url / model.
    judge_extra: dict = {}
    _is_dashscope = (
        "dashscope" in (judge_cfg.get("api_base") or "").lower()
        or (judge_cfg.get("model") or "").lower().startswith(("qwen", "deepseek"))
    )
    if _is_dashscope:
        judge_extra["extra_body"] = {"enable_thinking": False}

    # Wrap the OpenAI judge call in a clearly-named parent span so
    # Phoenix doesn't show it as just another "ChatCompletion" mixed
    # in with the agent's own LLM calls. The OpenInference instrumentor
    # still creates the underlying ChatCompletion span as a CHILD of
    # this one, so we keep the LLM-kinded data without losing the
    # human-readable label.
    with _tracer.start_as_current_span(f"judge.{qtype}") as judge_span:
        judge_span.set_attributes({
            "openinference.span.kind": "CHAIN",
            "judge.model": judge_cfg.get("model", ""),
            "judge.qtype": qtype,
            "judge.question": question,
            "judge.ground_truth": str(answer)[:500],
            "judge.agent_response": str(response)[:500],
            "judge.abstention": bool(abstention),
        })
        try:
            completion = client.chat.completions.create(
                model=judge_cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                n=1,
                temperature=0,
                # Bumped from 10 — prompt now mandates a
                # <judge_thinking> block before the verdict.
                max_tokens=800,
                **judge_extra,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("judge call failed (%s) — scoring 0.0", exc)
            judge_span.set_attribute("judge.error", str(exc)[:200])
            judge_span.set_attribute("judge.score", 0.0)
            return 0.0

        text = (completion.choices[0].message.content or "").strip().lower()
        score = _parse_judge_verdict(text)
        judge_span.set_attribute("judge.verdict", text[-200:])
        judge_span.set_attribute("judge.score", score)
        return score


_JUDGE_THINKING_CLOSE_RE = re.compile(r"</judge_thinking\s*>", re.IGNORECASE)
_VERDICT_TOKEN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _parse_judge_verdict(text: str) -> float:
    """Extract yes/no verdict from a judge response with a
    ``<judge_thinking>`` block.

    Strategy:
      1. If a ``</judge_thinking>`` close tag exists, look only at
         text AFTER it (the thinking block can legitimately contain
         the words ``yes`` / ``no``, which would confuse a naive
         substring check).
      2. Else fall back to the LAST ``yes``/``no`` word in the
         whole response — the verdict is conventionally the last
         thing.
      3. If the response was truncated mid-thinking (no close tag,
         no final yes/no word), default to 0.0 (treat as failure
         rather than mis-credit a pass).
    """
    if not text:
        return 0.0
    m_close = _JUDGE_THINKING_CLOSE_RE.search(text)
    search_region = text[m_close.end():] if m_close else text
    matches = _VERDICT_TOKEN_RE.findall(search_region)
    if not matches:
        # No verdict in the expected region — fall back to scanning
        # the whole response for the LAST occurrence (handles judges
        # that forget the close tag).
        matches = _VERDICT_TOKEN_RE.findall(text)
    if not matches:
        return 0.0
    return 1.0 if matches[-1].lower() == "yes" else 0.0
