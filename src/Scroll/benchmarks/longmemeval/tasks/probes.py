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
:meth:`LongMemEvalEnv.__init__`); ``get_probes_for_session`` returns
the probe only on the final session. Single-process re-runs are safe because
each :func:`run_single` constructs a fresh env, which overwrites the
active probe.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from opentelemetry import trace as otel_trace

from Scroll.core import EnvSnapshot, ProbeSpec
from Scroll.benchmarks.longmemeval.catalog import LongMemEvalEnvConfig
from Scroll.benchmarks.longmemeval.dataset import LongMemEvalItem

_log = logging.getLogger(__name__)
_tracer = otel_trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# Probe-mode prompt — appended to the universal PROBE_SUBSTRATE_PROMPT
# ---------------------------------------------------------------------------

# LME probes are graded by an LLM judge (see :data:`_JUDGE_TEMPLATES`)
# with a per-question-type template. The judge is lenient about extra
# context as long as the correct answer is contained, so format rules
# are looser than vending's — no required ``Answer:`` line, no unit
# tolerances, no last-line-wins behavior. Abstention questions DO need
# explicit "I don't know" phrasing for the judge to score them
# correct.
LME_PROBE_FORMAT = """\
PROBE FORMAT — LONGMEMEVAL:

- Between turns you write Python cells — workspace queries, log
  lookups, arithmetic, ``rlm`` calls, or any
  combination — and print what you need to see. Each cell's stdout
  comes back as the next user message. Commit by replying with a
  plain-text answer when you have enough. 

- Format: direct prose answer is fine — no required ``Answer:``
  line. Extra context (citations, brief reasoning) is OK as long
  as the correct answer is in the reply.

- ABSTENTION: when the search genuinely returns nothing, say so
  explicitly — e.g. "I don't have that information from our
  conversations". The judge scores abstention correct ONLY when
  you actually decline. Guessing is wrong; refusing-with-evidence
  is also wrong (you have evidence, compose).

- Don't paste the entire transcript back; cite only the facts
  relevant to the question.
"""


# Short reminder appended to the user-turn message inside
# ``inject_probe`` (sits next to the question text). Heavy format
# rules live in :data:`LME_PROBE_FORMAT` (system prompt).
LME_PROBE_USER_POSTSCRIPT = (
    "Write Python cells to query, compute, or call ``rlm`` — "
    "whatever combination you need to gather and combine "
    "evidence. Commit by replying with a plain-text answer when you "
    "have enough. If the search genuinely returns nothing, abstain "
    "explicitly: \"I don't have that information from our "
    "conversations\" — the judge scores that correct."
)


# Per-question-type nudges, layered on top of LME_PROBE_USER_POSTSCRIPT.
# These mirror the dimensions the paper identifies as separately
# tested (TR, KU, IE-preference) and lift the abstention judge's
# explicit-refusal requirement up to the user-turn message so the
# agent can see it without re-reading the system prompt.
_QTYPE_POSTSCRIPT: dict[str, str] = {
    "temporal-reasoning": (
        "Time-sensitive question. Pattern: extract a date range from "
        "the question first, filter sessions by that range "
        "(``session_ts_iso``) BEFORE reading content — don't keyword-"
        "scan the whole haystack. Off-by-one on day/week/month is not "
        "penalized."
    ),
    "knowledge-update": (
        "Recency-sensitive question. The user may have stated multiple "
        "values over time; the MOST RECENT statement is the current "
        "truth. Pattern: order matching sessions by ``session_ts_iso`` "
        "descending and take the latest value. Mentioning prior values "
        "is fine, but do not state an out-of-date value as current."
    ),
    "single-session-preference": (
        "Preference question. The literal subject of the question may "
        "never have been discussed verbatim — that is the point. Exact "
        "keyword / SQL match will miss; use semantic retrieval to "
        "surface related preferences (likes, dislikes, constraints, "
        "recurring interests) and ground the recommendation in those. "
        "\"I have no information about <topic>\" is the wrong frame "
        "here."
    ),
}

# Abstention applies to ANY qtype when the question_id has the `_abs`
# suffix (per dataset convention). It overrides the qtype postscript.
_ABSTENTION_POSTSCRIPT = (
    "This may be an unanswerable question — the user may never have "
    "stated the relevant information. After ≥2 distinct retrieval "
    "queries (different keywords AND different surfaces) return "
    "nothing useful, abstain EXPLICITLY using the phrasing: "
    "\"I don't have that information from our conversations.\" The "
    "judge requires explicit refusal to score abstention correct."
)


# ---------------------------------------------------------------------------
# Active-probe registry (per-run, single-process)
# ---------------------------------------------------------------------------

_active_probe: ProbeSpec | None = None
_active_qtype: str | None = None
_active_is_abstention: bool = False


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
    _active_probe = _build_probe(item, cfg)
    _active_qtype = item.question_type
    _active_is_abstention = item.is_abstention


PROBES: list[ProbeSpec] = []  # filled per-run via set_active_probe


def get_probes_for_session(session_idx: int) -> list[ProbeSpec]:
    if _active_probe is None or _active_probe.session_idx != session_idx:
        return []
    return [_active_probe]


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
        # Probe fires on the dedicated probe-only session = total_sessions+1
        # (one iteration past the last haystack session). The probe
        # session's run_session exposes "no session today" so the agent's
        # session-loop history at probe time isn't dominated by the last
        # ingest's transcript.
        session_idx=item.total_sessions + 1,
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
# The new templates explicitly call out both edge cases via a shared
# preamble; per-type wrapper just adds task-specific allowances.
_DEFAULT_PREAMBLE = (
    "You are grading a model's answer to a memory-recall question. "
    "The model has access to past chat sessions and must recall facts "
    "from them.\n\n"
    "Score YES if the response contains the correct answer or its key "
    "information — including when:\n"
    "  - phrased differently / paraphrased (synonyms, re-wording)\n"
    "  - using different tense, articles, or punctuation "
    "(e.g., 'has' vs 'had'; 'GR-90' vs 'The GR-90 trail.')\n"
    "  - formatted differently ('$2000' vs '$2,000'; '9:45am' vs '9:45 AM')\n"
    "  - presented inside a quote / Notes block (the model retrieved + "
    "surfaced the correct value as evidence; this counts)\n"
    "  - surrounded by extra context, qualifiers, or reasoning steps\n\n"
    "Score NO if the response:\n"
    "  - gives a wrong factual value (e.g., GT='$400,000' but model says '$350,000')\n"
    "  - **abstains** ('I don't have that information', 'no information found', "
    "'cannot answer') when the question has a definite answer — abstention is "
    "WRONG on questions that are answerable from the chat history\n"
    "  - misses the key element of the ground truth"
)

_JUDGE_TEMPLATES = {
    # single-session-user / single-session-assistant / multi-session
    "default": (
        _DEFAULT_PREAMBLE + "\n\n"
        "Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
        "Output exactly 'yes' or 'no' (lowercase, no other text)."
    ),
    "temporal-reasoning": (
        _DEFAULT_PREAMBLE + "\n\n"
        "Additional rule for this temporal question: do NOT penalize off-by-one "
        "errors on day/week/month counts. If GT='18 days' and model says '19 days', "
        "score YES.\n\n"
        "Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
        "Output exactly 'yes' or 'no' (lowercase, no other text)."
    ),
    "knowledge-update": (
        _DEFAULT_PREAMBLE + "\n\n"
        "Additional rule for this knowledge-update question: the user has REVISED "
        "an earlier answer. Score YES as long as the model's final answer is the "
        "LATEST/UPDATED value. Mentioning the older value alongside is fine.\n\n"
        "Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
        "Output exactly 'yes' or 'no' (lowercase, no other text)."
    ),
    "single-session-preference": (
        "You are grading a model's RECOMMENDATION on a preference question. "
        "The 'rubric' below describes what the user would prefer (their stated "
        "tastes, constraints, or recurring interests).\n\n"
        "Score YES if the response recalls and utilizes the user's personal "
        "information correctly to ground the recommendation. The model does NOT "
        "need to reflect every point in the rubric — partial use of the user's "
        "stated preferences is fine, as long as the recommendation is grounded "
        "in them rather than generic.\n\n"
        "Score NO if the response:\n"
        "  - gives a generic recommendation that ignores the user's preferences\n"
        "  - abstains ('I don't have that info') — the question expects a "
        "personalized recommendation, even from related (not-identical) topics\n"
        "  - misuses the user's stated preferences (e.g., recommending something "
        "the user explicitly disliked)\n\n"
        "Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
        "Output exactly 'yes' or 'no' (lowercase, no other text)."
    ),
    "abstention": (
        "You are grading whether a model correctly identified an UNANSWERABLE "
        "question. The user never discussed this topic in the chat history, so "
        "the correct behavior is to abstain.\n\n"
        "Score YES if the model:\n"
        "  - explicitly says it doesn't have the information\n"
        "  - says the information is incomplete / missing\n"
        "  - says the topic was never discussed\n"
        "  - declines to answer for lack of evidence\n\n"
        "Score NO if the model gives a concrete answer (named X, picked a value, "
        "made a count) — that's fabrication on an unanswerable question.\n\n"
        "Question: {question}\n\nExplanation (why unanswerable): {answer}\n\nModel Response: {response}\n\n"
        "Output exactly 'yes' or 'no' (lowercase, no other text)."
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

    client = OpenAI(api_key=api_key, base_url=judge_cfg.get("api_base"))

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
                max_tokens=10,
                **judge_extra,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("judge call failed (%s) — scoring 0.0", exc)
            judge_span.set_attribute("judge.error", str(exc)[:200])
            judge_span.set_attribute("judge.score", 0.0)
            return 0.0

        text = (completion.choices[0].message.content or "").strip().lower()
        score = 1.0 if "yes" in text else 0.0
        judge_span.set_attribute("judge.verdict", text[:50])
        judge_span.set_attribute("judge.score", score)
        return score
