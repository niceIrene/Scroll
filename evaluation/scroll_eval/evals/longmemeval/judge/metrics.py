"""LongMemEval judge templates + scoring.

Ported verbatim from the original Scroll LongMemEval ``tasks/probes.py`` judge
(the relaxed templates + ``<judge_thinking>`` verdict parser). The judge-model
call resolves its endpoint/model from the same env the agent uses (so a run is
graded on the same model by default) and mirrors beam's transient-error retry.
"""
from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

# --------------------------------------------------------------------------- #
# Judge templates (relaxed beyond LME upstream — see the original docstring).  #
# --------------------------------------------------------------------------- #

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


def judge_prompt(qtype: str, question: str, answer: str, response: str, abstention: bool) -> str:
    if abstention:
        tmpl = _JUDGE_TEMPLATES["abstention"]
    elif qtype in _JUDGE_TEMPLATES:
        tmpl = _JUDGE_TEMPLATES[qtype]
    else:  # single-session-user / single-session-assistant / multi-session
        tmpl = _JUDGE_TEMPLATES["default"]
    return tmpl.format(question=question, answer=answer, response=response)


_JUDGE_THINKING_CLOSE_RE = re.compile(r"</judge_thinking\s*>", re.IGNORECASE)
_VERDICT_TOKEN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_judge_verdict(text: str) -> float:
    """Extract a 1.0/0.0 verdict from a judge response with a ``<judge_thinking>`` block.

    Look only AFTER the ``</judge_thinking>`` close tag (the thinking block can
    legitimately contain the words yes/no); else fall back to the LAST yes/no in
    the whole response; else 0.0 (treat a truncated/absent verdict as failure).
    """
    if not text:
        return 0.0
    m_close = _JUDGE_THINKING_CLOSE_RE.search(text)
    region = text[m_close.end():] if m_close else text
    matches = _VERDICT_TOKEN_RE.findall(region) or _VERDICT_TOKEN_RE.findall(text)
    if not matches:
        return 0.0
    return 1.0 if matches[-1].lower() == "yes" else 0.0


# --------------------------------------------------------------------------- #
# Judge model shim (OpenAI client + transient-error retry, env-resolved).      #
# --------------------------------------------------------------------------- #

_RETRY_EXC = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
_MAX_ATTEMPTS = 6
_BACKOFF_BASE_S = 2.0
_BACKOFF_CAP_S = 60.0


@dataclass
class _Response:
    content: str


def _retry_after_seconds(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


class LmeJudgeModel:
    """``invoke(prompt) -> obj.content`` over the OpenAI client, env-resolved.

    Model resolves: explicit ``model`` > ``SCROLL_JUDGE_MODEL`` > ``SCROLL_MODEL``.
    Endpoint: explicit ``base_url`` > ``SCROLL_JUDGE_BASE_URL`` > ``OPENAI_BASE_URL``.
    Mirrors the original judge: temperature 0, ``max_tokens`` room for the
    ``<judge_thinking>`` block, and DashScope's ``enable_thinking=False`` gate.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 800,
        timeout: float = 60.0,
    ) -> None:
        raw = (
            model
            or os.environ.get("SCROLL_JUDGE_MODEL")
            or os.environ.get("SCROLL_MODEL")
            or os.environ.get("OPENAI_MODEL_NAME")
            or ""
        )
        self.model = raw.split("/", 1)[-1] if "/" in raw else raw
        self.max_tokens = max_tokens
        self._base_url = base_url or os.environ.get("SCROLL_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or "",
            base_url=self._base_url,
            max_retries=2,
            timeout=timeout,
        )
        # ``enable_thinking`` is a DashScope-only extra_body field; sending it to a
        # plain OpenAI endpoint returns HTTP 400. Gate on endpoint/model.
        self._extra: dict[str, Any] = {}
        if "dashscope" in (self._base_url or "").lower() or self.model.lower().startswith(
            ("qwen", "deepseek")
        ):
            self._extra["extra_body"] = {"enable_thinking": False}

    def invoke(self, prompt: str) -> _Response:
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=self.max_tokens,
                    **self._extra,
                )
                return _Response(content=resp.choices[0].message.content or "")
            except _RETRY_EXC as exc:
                last_exc = exc
                if attempt == _MAX_ATTEMPTS - 1:
                    break
                capped = min(_BACKOFF_BASE_S * (2 ** attempt), _BACKOFF_CAP_S)
                hinted = _retry_after_seconds(exc)
                time.sleep(hinted if hinted is not None else capped / 2 + random.uniform(0, capped / 2))
        assert last_exc is not None
        raise last_exc


def score_one(model: LmeJudgeModel, gold: dict, response: str) -> float:
    """Grade one answer against one gold rubric → 1.0/0.0.

    ``gold`` = ``{question, answer, question_type, is_abstention}``. A judge-call
    failure scores 0.0 (never crash grading; the run can be re-judged offline).
    """
    prompt = judge_prompt(
        qtype=gold.get("question_type", ""),
        question=gold.get("question", ""),
        answer=gold.get("answer", ""),
        response=response or "",
        abstention=bool(gold.get("is_abstention")),
    )
    try:
        text = model.invoke(prompt).content.strip().lower()
    except Exception as exc:  # noqa: BLE001 — never crash grading over one probe
        print(f"[longmemeval.judge] judge call failed ({exc}) — scoring 0.0")
        return 0.0
    return parse_judge_verdict(text)
