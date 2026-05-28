"""BEAM probe construction + LLM-judge scoring.

Each BEAM chat carries 10 question categories × ~2 questions each
(~20 probes per chat). Every probe is scored by averaging an LLM-judge
verdict across the question's rubric items, using BEAM's own
``unified_llm_judge_base_prompt`` (vendored here so we don't depend on
the external repo's evaluation module at runtime).

All probes fire on the probe day (= ``num_batches + 1``); the framework
calls ``inject_probe`` for each in sequence and records per-probe
scores into ``probe_results.json``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from opentelemetry import trace as otel_trace

from Scroll.core import EnvSnapshot, ProbeSpec
from Scroll.benchmarks.beam.catalog import BeamEnvConfig
from Scroll.benchmarks.beam.dataset import BeamItem, BeamProbingQuestion

_log = logging.getLogger(__name__)
_tracer = otel_trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# Probe-mode prompt — system-side rules appended to the SCROLL substrate.
# ---------------------------------------------------------------------------

BEAM_PROBE_FORMAT = """\
PROBE FORMAT — BEAM:

- Multiple probing questions fire in sequence at the end of this
  run, one per the question categories below. Each is graded
  independently by an LLM judge using a per-question rubric, so
  answer EACH question on its own terms — don't bundle answers or
  carry over context that doesn't apply.

- TWO PHASES per probe — code cells THEN plain-text answer, in
  SEPARATE responses:
    • RETRIEVAL TURNS (1-N): each response is ONLY a python block
      that prints evidence to stdout (or a tool call to
      ``execute_python``). NO answer prose mixed with code.
    • ANSWER TURN: plain text only, no code block. Compose a
      direct answer based on the stdout you've observed.

- Question categories (10 memory abilities, paraphrased):
    information_extraction, multi_session_reasoning, knowledge_update,
    temporal_reasoning, abstention, contradiction_resolution,
    event_ordering, instruction_following, preference_following,
    summarization

- ABSTENTION: when the topic genuinely wasn't discussed, say so
  explicitly — e.g. "Based on the provided chat, there is no
  information related to <topic>". The judge rubric for abstention
  questions REQUIRES that explicit phrasing pattern.

- INSTRUCTION_FOLLOWING / PREFERENCE_FOLLOWING: pull constraints
  from the user's stated preferences across batches; ground the
  answer in those rather than giving generic advice.

- EVENT_ORDERING / TEMPORAL_REASONING: BEAM batches are
  time-anchored (the ``time_anchor`` of each batch is recorded with
  every turn). Use ``session_date_iso`` / ``session_ts_iso`` for
  chronological queries.

- Don't paste the full transcript back — cite only the spans
  relevant to each question.
"""


BEAM_PROBE_USER_POSTSCRIPT = (
    "TWO-PHASE PROBE. Phase 1 (this turn and possibly a few more): "
    "write python cells (via ``execute_python``) that retrieve "
    "evidence and print it. NO answer prose this turn — the cell "
    "hasn't run yet, so anything you'd write would be guessed. "
    "Phase 2 (a LATER turn, after you've seen real stdout): plain "
    "text answer grounded in what the stdout actually showed. "
    "If the topic genuinely isn't covered in the chat, say so "
    "explicitly with the phrase 'Based on the provided chat, there "
    "is no information related to <topic>' — the abstention "
    "category's rubric expects that exact pattern."
)


# ---------------------------------------------------------------------------
# Per-category nudges, layered on top of BEAM_PROBE_USER_POSTSCRIPT.
# ---------------------------------------------------------------------------

_CATEGORY_POSTSCRIPT: dict[str, str] = {
    "abstention": (
        "This question may NOT be answerable from the chat. After "
        "≥2 distinct retrieval queries (different keywords AND "
        "different surfaces) return nothing relevant, abstain "
        "explicitly: \"Based on the provided chat, there is no "
        "information related to <topic>.\""
    ),
    "knowledge_update": (
        "If the user has revised a value across batches, the MOST "
        "RECENT statement is the current truth. Sort by "
        "session_ts_iso DESC (or session_idx DESC); the latest value "
        "wins. Mentioning older values alongside is fine."
    ),
    "temporal_reasoning": (
        "Time-sensitive. Extract a date scope from the question "
        "first, filter by ``session_date_iso`` BEFORE reading "
        "content. Off-by-one day/week/month errors are not "
        "penalized."
    ),
    "event_ordering": (
        "Output events in chronological order using the recorded "
        "``time_anchor`` (= ``session_date_iso``) of each batch as "
        "the anchor for events discussed within it."
    ),
    "preference_following": (
        "Ground the answer in the user's stated preferences "
        "(likes, dislikes, constraints, recurring interests) "
        "discoverable in ``user_preferences`` or via ``chat_turns`` "
        "LIKE queries on 'I prefer / I'd rather / my favorite / I "
        "avoid'."
    ),
    "instruction_following": (
        "Honor any explicit user-stated constraints (format, length, "
        "exclusions). Cite the batch where the instruction was given."
    ),
    "contradiction_resolution": (
        "The user stated conflicting things across batches. Note "
        "both statements, then commit to the resolution the user "
        "most recently made (or, when the user explicitly resolved "
        "the contradiction, that resolution)."
    ),
    "multi_session_reasoning": (
        "Combine evidence from MULTIPLE batches. A single batch is "
        "rarely enough. Cite each batch you draw from."
    ),
    "summarization": (
        "Synthesize across the whole chat — short, faithful, no "
        "fabrication. Cite the batches you summarize from."
    ),
}


# ---------------------------------------------------------------------------
# Active item registry (per-run, single-process).
# ---------------------------------------------------------------------------

_active_item: BeamItem | None = None
_active_cfg: BeamEnvConfig | None = None
_active_probes: list[ProbeSpec] = []
_active_category: str | None = None  # last-injected probe's category
# Mirrors ``BeamEnvConfig.agent_during_ingestion`` for the active run.
# Under the default ``False`` path probes fire end-of-task (see
# :meth:`BeamEnv.get_end_of_task_probes`) and the per-turn registry
# returns ``[]``. Under the legacy ``True`` path the per-turn registry
# returns all probes on the ``+1`` turn (today's behavior).
_active_agent_during_ingestion: bool = False


def active_category() -> str | None:
    """Category of the most recently fired probe (used by the agent's
    probe_user_hint to inject the right nudge)."""
    return _active_category


def compose_user_postscript() -> str:
    """Default user-turn postscript. Per-category nudge gets prepended
    by the agent's ``probe_user_hint`` if it has the active probe."""
    return BEAM_PROBE_USER_POSTSCRIPT


def set_active_item(item: BeamItem, cfg: BeamEnvConfig) -> None:
    """Register the item + build ProbeSpecs for all its probing questions."""
    global _active_item, _active_cfg, _active_probes
    global _active_agent_during_ingestion
    _active_item = item
    _active_cfg = cfg
    _active_probes = [
        _build_probe(item, pq, cfg) for pq in item.probing_questions
    ]
    _active_agent_during_ingestion = bool(
        getattr(cfg, "agent_during_ingestion", False)
    )


PROBES: list[ProbeSpec] = []  # populated per-run via set_active_item


def get_probes_for_turn(turn_idx: int) -> list[ProbeSpec]:
    """Per-turn probe registry.

    Under the new SCROLL-pure path (``agent_during_ingestion=False``,
    default) returns ``[]`` — probes fire end-of-task via
    :meth:`BeamEnv.get_end_of_task_probes`.

    Under the legacy path all probes fire on the probe turn
    (= num_batches + 1).
    """
    if not _active_agent_during_ingestion:
        return []
    if not _active_probes:
        return []
    probe_turn = _active_probes[0].turn_idx
    if turn_idx != probe_turn:
        return []
    return list(_active_probes)


# Back-compat alias; drop in PR #6.
get_probes_for_session = get_probes_for_turn


def compute_efficiency_metrics(daily_action_logs: list[list[str]]) -> dict:
    """BEAM has no action-cost metrics — the agent runs passively."""
    return {}


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------


def _build_probe(item: BeamItem, pq: BeamProbingQuestion, cfg: BeamEnvConfig) -> ProbeSpec:
    qid = f"{item.chat_id}_{pq.category}_{pq.index}"
    judge_cfg = {
        "model": cfg.judge_model,
        "api_key_env": cfg.judge_api_key_env,
        "api_base": cfg.judge_api_base,
    }

    def gt_fn(snapshots: list[EnvSnapshot], data) -> str:
        # Surface both the ideal response and the rubric so
        # ``probe_results.json`` carries the grading signal that was
        # actually applied.
        return json.dumps({
            "ideal_response": pq.ideal_response,
            "rubric": pq.rubric,
        }, ensure_ascii=False)

    def scoring_fn(agent_answer: str, ground_truth: str) -> float:
        global _active_category
        _active_category = pq.category
        return _rubric_judge_score(
            category=pq.category,
            question=pq.question,
            rubric=pq.rubric,
            response=agent_answer,
            judge_cfg=judge_cfg,
        )

    # Build the prompt with category + question text. The probe turn is
    # num_batches + 1 (same shape as LongMemEval).
    nudge = _CATEGORY_POSTSCRIPT.get(pq.category, "")
    question_text = (
        f"[{pq.category}] {pq.question}"
        + (f"\n\n{nudge}" if nudge else "")
        + "\n\nAnswer from what you find in the chat history. Connecting "
        "two stated facts to reach a third is fine. Only abstain when "
        "the chat truly contains nothing relevant."
    )

    return ProbeSpec(
        question_id=qid,
        turn_idx=item.num_batches + 1,
        question=question_text,
        ground_truth_fn=gt_fn,
        scoring_fn=scoring_fn,
        passive=False,
    )


# ---------------------------------------------------------------------------
# Judge — BEAM's per-rubric-item LLM-judge averaged.
#
# Vendored from external/beam/src/prompts.py at submodule pin. Keep the
# wording 1:1 with upstream so SCROLL's scores are comparable.
# ---------------------------------------------------------------------------

_UNIFIED_JUDGE_PROMPT = """\
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): {question}
- RUBRIC CRITERION (what to check): {rubric_item}
- RESPONSE TO EVALUATE: {response}

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept paraphrases and synonyms that preserve intent.
- Case/punctuation/whitespace differences must be ignored.
- Numbers/currencies/dates may appear in equivalent forms (e.g., "$68,000" vs "68k" vs "68,000 USD") and should be treated as equal when numerically equivalent.

## STYLE NEUTRALITY:
Ignore tone, politeness, length, and verbosity unless the rubric explicitly requires a format/structure.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
- **0.5 (Partial Compliance)**: Captures the rubric's intent but misses a key element or contains a minor inaccuracy.
- **0.0 (Non-Compliance)**: Does not satisfy the rubric criterion, or is non-responsive to the QUESTION.

Return STRICT JSON, no prose, exactly: {{"score": <0.0|0.5|1.0>, "rationale": "<one short sentence>"}}
"""


_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9.]+)')


def _rubric_judge_score(
    *,
    category: str,
    question: str,
    rubric: list[str],
    response: str,
    judge_cfg: dict,
) -> float:
    """Average a per-rubric-item LLM-judge score across the rubric.

    Network failures => 0.0 with a warning. We prefer scoring the run
    as 0.0 over crashing — failed probes can be re-judged offline.
    """
    if not rubric:
        return 0.0

    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover
        _log.warning("openai not installed — judge skipped, scoring 0.0")
        return 0.0

    api_key = os.getenv(judge_cfg["api_key_env"], "")
    if not api_key:
        _log.warning(
            "judge api key env %s is empty — scoring 0.0",
            judge_cfg["api_key_env"],
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

    total = 0.0
    count = 0
    with _tracer.start_as_current_span(f"judge.beam.{category}") as judge_span:
        judge_span.set_attributes({
            "openinference.span.kind": "CHAIN",
            "judge.model": judge_cfg.get("model", ""),
            "judge.category": category,
            "judge.question": question[:500],
            "judge.rubric_items": len(rubric),
            "judge.agent_response": str(response)[:500],
        })
        for item in rubric:
            prompt = _UNIFIED_JUDGE_PROMPT.format(
                question=question,
                rubric_item=item,
                response=response,
            )
            try:
                completion = client.chat.completions.create(
                    model=judge_cfg["model"],
                    messages=[{"role": "user", "content": prompt}],
                    n=1,
                    temperature=0,
                    max_tokens=200,
                    **judge_extra,
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "BEAM judge call failed (category=%s): %s — item scored 0",
                    category, exc,
                )
                continue

            text = (completion.choices[0].message.content or "").strip()
            item_score = _parse_item_score(text)
            total += item_score
            count += 1

        avg = total / count if count else 0.0
        judge_span.set_attribute("judge.score", avg)
        judge_span.set_attribute("judge.items_judged", count)
        return avg


def _parse_item_score(raw: str) -> float:
    """Pull a numeric score out of the judge's response. Falls back
    to regex if JSON parsing fails."""
    if not raw:
        return 0.0
    try:
        obj = json.loads(raw)
        v = obj.get("score") if isinstance(obj, dict) else None
        if v is not None:
            return max(0.0, min(1.0, float(v)))
    except (ValueError, json.JSONDecodeError):
        pass
    m = _SCORE_RE.search(raw)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    # Word-level fallback for non-JSON judges.
    lower = raw.lower()
    if "1.0" in lower or "yes" in lower or "compliant" in lower:
        return 1.0
    if "0.5" in lower or "partial" in lower:
        return 0.5
    return 0.0
