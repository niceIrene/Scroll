"""Ported BEAM grading metrics (the code path run_evaluation actually uses).

Faithful to ``src/evaluation/compute_metrics.py``:
- ``judge_rubric`` is the shared body of all 9 non-ordering evaluators (they are
  byte-identical upstream): per rubric item, ask the judge for ``{score}`` and
  average. Only ``<rubric_item>`` and ``<llm_response>`` are substituted —
  ``<question>`` is left literal, matching upstream exactly.
- ``evaluate_event_ordering`` adds ``event_ordering_score`` (Kendall-tau + F1)
  over an LLM-aligned event list. Documented divergence: upstream builds that
  list with ``extract_facts`` (atomic semantic facts) and then accidentally
  overwrites it with a raw ``llm_response.split("\\n")`` — a dead assignment
  that ships the metric with its intended input disabled, so formatting
  (blank lines, preambles, paragraph answers) dominates the score.
  ``align_input`` selects the behavior: ``"facts"`` (default) restores the
  intended fact-extraction pipeline; ``"lines"`` reproduces upstream's
  *executed* line-split for comparability with published BEAM numbers.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Tuple

from json_repair import repair_json

from scroll_eval.evals.beam.judge.prompts import (
    EQUIVALENCE_CATEGORY_SYSTEM_PROMPT,
    EQUIVALENCE_SYSTEM_PROMPT,
    break_paragraph_to_facts_detailed_prompt,
    unified_llm_judge_base_prompt,
)

# Concurrency for the (independent) LLM calls within one probe: rubric-item
# scoring and event_ordering's per-line equivalence checks. The judge model's
# OpenAI client is thread-safe, so these run on a thread pool. Probe-level
# concurrency is layered on top in judge/__main__.grade (a separate pool), so
# peak in-flight calls ~= this x that pool's size. Parallelism does not change
# scores: event_ordering's greedy alignment still takes the lowest-index match.
_WITHIN_PROBE_WORKERS = 8


def _pmap(fn, items: list, max_workers: int) -> list:
    """Order-preserving parallel map; falls back to serial for trivial cases."""
    if max_workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as ex:
        return list(ex.map(fn, items))


def parse_json_response(response: str) -> Any:
    """Extract a JSON object/array from a model response (BEAM verbatim)."""
    response = response.strip()
    if response.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", response, re.DOTALL)
        if match:
            response = match.group(1).strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*?\}|\[.*?\])", response, re.DOTALL)
    if match:
        json_part = match.group(1)
        try:
            return json.loads(json_part)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Found possible JSON but failed to parse it: {e}")
    raise ValueError("No valid JSON found in response.")


def _parse_or_repair(text: str) -> Any:
    try:
        return parse_json_response(text)
    except Exception:  # noqa: BLE001 - mirror BEAM's bare-except repair fallback
        return json.loads(repair_json(text))


def judge_rubric(
    rubric: list, llm_response: str, model, *, max_workers: int = _WITHIN_PROBE_WORKERS
) -> dict:
    """Score a response against each rubric item; average. Shared by 9 types.

    Rubric items are independent LLM calls, scored concurrently. ``_pmap``
    preserves order, so ``llm_judge_responses`` still aligns with ``rubric``.
    """
    if not rubric:
        # Upstream would ZeroDivision here; guard so a rubric-less question
        # scores 0.0 rather than crashing the whole run.
        return {"llm_judge_score": 0.0, "llm_judge_responses": []}

    def _score(item: str) -> Any:
        prompt = unified_llm_judge_base_prompt.replace("<rubric_item>", item).replace(
            "<llm_response>", llm_response
        )
        return _parse_or_repair(model.invoke(prompt).content.strip())

    parsed_list = _pmap(_score, list(rubric), max_workers)
    score = sum(float(p["score"]) for p in parsed_list)
    return {
        "llm_judge_score": score / len(rubric),
        "llm_judge_responses": parsed_list,
    }


# --- event_ordering: Kendall-tau + F1 over the response (align_type="llm") ----

def _llm_equivalence(
    first: str, second: str, llm, *, system_prompt: str = EQUIVALENCE_SYSTEM_PROMPT
) -> bool:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"First snippet: {first} \n\n                       Second snippet: {second}\n                    "},
    ]
    return "yes" in llm.invoke(messages).content.lower()


def _align_with_llm(
    reference: List[str],
    system: List[str],
    llm,
    *,
    system_prompt: str = EQUIVALENCE_SYSTEM_PROMPT,
    max_workers: int = _WITHIN_PROBE_WORKERS,
) -> Tuple[List[str], List[str]]:
    """Greedy 1:1 alignment of system items to reference (BEAM, faithful).

    Upstream scans reference in order and takes the FIRST unused equivalent
    match. We check a line's unused candidates concurrently and keep the
    lowest matching index — identical result to "first match wins", just
    parallel. (It can issue a few more calls than the serial short-circuit,
    but they overlap.) The outer loop stays sequential: ``used`` carries over.
    """
    used: set = set()
    system_out: list = []
    for s in system:
        candidates = [i for i in range(len(reference)) if i not in used]
        if not candidates:
            system_out.append(s)
            continue
        flags = _pmap(
            lambda i: _llm_equivalence(
                first=reference[i], second=s, llm=llm, system_prompt=system_prompt
            ),
            candidates,
            max_workers,
        )
        matched = [i for i, ok in zip(candidates, flags) if ok]
        if matched:
            idx = min(matched)
            system_out.append(reference[idx])
            used.add(idx)
        else:
            system_out.append(s)
    return reference, system_out


def event_ordering_score(
    reference_list: List[str],
    system_list: List[str],
    llm,
    *,
    equivalence_prompt: str = EQUIVALENCE_SYSTEM_PROMPT,
    max_workers: int = _WITHIN_PROBE_WORKERS,
) -> dict:
    """Kendall-tau-b (normalized) × F1 over LLM-aligned event lists (BEAM)."""
    reference_canon, system_canon = _align_with_llm(
        reference_list,
        system_list,
        llm,
        system_prompt=equivalence_prompt,
        max_workers=max_workers,
    )

    tp = len(set(reference_canon) & set(system_canon))
    fp = len([x for x in system_canon if x not in reference_canon])
    fn = len([x for x in reference_canon if x not in system_canon])

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    union = list(dict.fromkeys(reference_canon + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq):
        r = {item: i + 1 for i, item in enumerate(seq)}
        return [r.get(u, tie_rank) for u in union]

    # scipy is imported lazily (and only here) so the judge module loads — and
    # every non-event_ordering question type still scores — without it. Only
    # event_ordering needs Kendall-tau; install with `uv sync --extra beam`.
    try:
        from scipy.stats import kendalltau
    except ImportError as exc:  # pragma: no cover - clear, actionable failure
        raise ImportError(
            "event_ordering scoring requires scipy; install the beam extra: "
            "`uv sync --extra beam` (or `uv pip install 'scipy>=1.11'`)."
        ) from exc

    tau_b, _ = kendalltau(to_rank(reference_canon), to_rank(system_canon), variant="b", method="auto")
    tau_b_norm = (tau_b + 1) / 2 if tau_b is not None else 0
    return dict(
        precision=precision,
        recall=recall,
        f1=f1,
        tau_norm=tau_b_norm,
        final_score=tau_b_norm * f1,
    )


# Leading enumeration/bullet markers on extracted fact lines ("1. ", "2)",
# "- ", "* ") and the quotes the extraction prompt's examples use — stripped so
# the equivalence check compares fact content, not list decoration.
_FACT_LINE_MARKER = re.compile(r"^\s*(?:\d+\s*[\.\):]|[-*•])\s*")

# Some chats' golden rubric items embed the judge instruction ("LLM response
# should mention: X") in the reference string itself; in the fixed alignment
# path we strip it so the aligner compares event content, not judge phrasing.
_RUBRIC_INSTRUCTION_PREFIX = re.compile(r"^\s*LLM response should (?:mention|state)\s*:\s*", re.IGNORECASE)


def extract_facts(paragraph: str, question: str, model) -> List[str]:
    """Break a response into atomic semantic fact units via the judge LLM.

    Upstream BEAM's ``extract_facts`` (compute_metrics.py), restored: it is the
    input ``event_ordering_score`` was designed around, but upstream's caller
    dead-assigns over its result. One divergence from upstream's raw
    ``response.split("\\n")``: blank lines are dropped and enumeration markers/
    wrapping quotes trimmed — the prompt asks for a numbered list, and feeding
    decoration to the aligner is exactly the pathology this path exists to fix.
    """
    prompt = break_paragraph_to_facts_detailed_prompt.replace(
        "<question>", question
    ).replace("<input_text>", paragraph)
    lines = model.invoke(prompt).content.split("\n")
    facts = []
    for line in lines:
        fact = _FACT_LINE_MARKER.sub("", line).strip().strip('"').strip()
        if fact:
            facts.append(fact)
    return facts


def evaluate_event_ordering(
    rubric: list,
    llm_response: str,
    model,
    *,
    question: str = "",
    align_input: str = "facts",
    max_workers: int = _WITHIN_PROBE_WORKERS,
) -> dict:
    """Score an event_ordering answer: tau/F1 over aligned events + rubric judge.

    ``align_input`` picks the alignment pipeline:
    - ``"facts"`` (default): the fixed pipeline. Aligner input = atomic facts
      from ``extract_facts`` (BEAM's intended input, robust to markdown/prose
      formatting); alignment reference = rubric items with any leading
      "LLM response should mention:" judge-instruction prefix stripped; the
      equivalence classifier uses the category-aware prompt (rubric labels are
      often abstract categories that the strict SAME-event prompt rejects even
      when the rubric judge scores the same content compliant).
    - ``"lines"``: upstream's executed behavior, verbatim — raw
      ``llm_response.split("\\n")`` against raw rubric strings with the strict
      SAME-event prompt; use for comparability with published BEAM scores.
    """
    if align_input == "facts":
        system_list = extract_facts(llm_response, question, model)
        if not system_list:  # extraction hiccup — fall back rather than zero out
            system_list = llm_response.split("\n")
        reference_list = [_RUBRIC_INSTRUCTION_PREFIX.sub("", str(r)).strip() for r in rubric]
        equivalence_prompt = EQUIVALENCE_CATEGORY_SYSTEM_PROMPT
    elif align_input == "lines":
        system_list = llm_response.split("\n")
        reference_list = rubric
        equivalence_prompt = EQUIVALENCE_SYSTEM_PROMPT
    else:
        raise ValueError(f"align_input must be 'facts' or 'lines', got {align_input!r}")
    score = event_ordering_score(
        reference_list=reference_list,
        system_list=system_list,
        llm=model,
        equivalence_prompt=equivalence_prompt,
        max_workers=max_workers,
    )
    judged = judge_rubric(rubric, llm_response, model, max_workers=max_workers)
    score["llm_judge_score"] = judged["llm_judge_score"]
    score["llm_judge_responses"] = judged["llm_judge_responses"]
    score["align_input"] = align_input
    return score


# The 10 BEAM question types. All but event_ordering use the shared judge body.
QUESTION_TYPES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)


def evaluate(
    qtype: str,
    rubric: list,
    llm_response: str,
    model,
    *,
    question: str = "",
    align_input: str = "facts",
    max_workers: int = _WITHIN_PROBE_WORKERS,
) -> dict:
    """Dispatch one question to its evaluator. Returns BEAM's result dict.

    ``question``/``align_input`` only affect event_ordering (fact extraction is
    question-conditioned); the other nine types ignore them.
    """
    if qtype == "event_ordering":
        return evaluate_event_ordering(
            rubric,
            llm_response,
            model,
            question=question,
            align_input=align_input,
            max_workers=max_workers,
        )
    return judge_rubric(rubric, llm_response, model, max_workers=max_workers)


def primary_score(qtype: str, result: dict) -> float:
    """The headline score per type (BEAM report: tau_norm for ordering, else judge)."""
    if qtype == "event_ordering":
        return float(result.get("tau_norm", 0.0))
    return float(result.get("llm_judge_score", 0.0))
