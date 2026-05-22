"""Vending-specific scoring functions and efficiency metrics."""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Probe-mode prompts
# ---------------------------------------------------------------------------

# Appended to the universal ``PROBE_SUBSTRATE_PROMPT`` (in
# ``Scroll.core._codeact_agent``) when the active env is
# :class:`Scroll.benchmarks.vending.env.VendingEnv`. Captures everything the
# deterministic regex scorer (``score_numeric`` / ``score_keyword``,
# below) cares about: an ``Answer:`` line on its own at the end of
# the reply, the same unit symbol the question uses, and the
# per-unit tolerances the scorer applies.
VENDING_PROBE_FORMAT = """\
PROBE FORMAT — VENDING:
- DO NOT call ``wait_for_next_day()`` while answering a probe. It
  has been swapped for a no-op shim for the duration of this probe
  — calling it does NOT end today; today resumes after you submit
  the answer. (You also will not see a "day N+1" briefing as a
  result of calling it.)
- The scorer is a deterministic regex with per-unit tolerances:
  ±5% relative on dollar amounts, ±5 percentage points on
  percentages, ±10% relative on bare numbers. Use the same unit
  symbol the question uses ("$", "%", or bare).
- Reply IN PLAIN TEXT (no code block) on your final turn. Reasoning
  goes FIRST; the ``Answer: ...`` line goes LAST, on its own line.
  The Answer line must contain every value the question asks for.
- Only the LAST ``Answer:`` line in your reply is scored. If you
  recompute mid-reply, write a fresh ``Answer:`` line at the end —
  earlier ones are ignored.
- ``print("Answer: ...")`` inside a code cell is NOT scored. Only
  the plain-text reply is.
- Match the question's value TYPE: "which day(s)" → list day
  numbers; "how much" → list the amount; "which SKU(s)" → list SKU
  names. If the question asks for an extremum (highest, most,
  cheapest, etc.) and there is a tie, list ALL tied items.
- Worked Answer lines: ``Answer: 6 deliveries``, ``Answer: $282.00``,
  ``Answer: 42% margin``, ``Answer: machine=0 storage=27``.
"""


# Per-question reminder appended to the user-turn message inside
# ``inject_probe``. Kept short — the heavy format rules live in
# :data:`VENDING_PROBE_FORMAT` (system prompt). This block exists so
# the format reminder sits *next to* the question text, where the
# attention-budget for instruction-following is highest.
VENDING_PROBE_USER_POSTSCRIPT = (
    "Answer line format: a single line starting with ``Answer: `` "
    "containing every value the question asks for (numbers, names, "
    "units, dollar signs). If the question asks for an extremum "
    "(highest, most, cheapest, etc.) and there is a tie, list ALL "
    "tied items on the Answer line. Use the same units the question "
    "implies — e.g. ``6 deliveries``, ``$282.00``, ``42% margin``, "
    "``machine=0 storage=27``. Reasoning goes on lines AFTER the "
    "Answer line. This does not count as a business action."
)


# ``$1,234.56`` → ``$1234.56`` so the digit regex doesn't split them
# into ``[1.0, 234.56]``. Only commas between digits are stripped —
# list separators (``a, b``) stay alone.
_THOUSAND_SEP_RE = re.compile(r"(?<=\d),(?=\d)")

# (``$``)? + (signed number) + (``%``)?. Whitespace between the
# prefix and the number is allowed (``$ 481.50``); the trailing ``%``
# must be adjacent. ``finditer`` walks left-to-right and consumes
# each number once.
_NUM_WITH_UNIT_RE = re.compile(r"(\$)?\s*(-?\d+(?:\.\d+)?)(%)?")


def _extract_typed_numbers(text: str) -> list[tuple[float, str]]:
    """Extract ``(value, unit_tag)`` pairs from text.

    ``unit_tag`` is one of ``"$"`` (``$`` prefix), ``"%"`` (``%``
    suffix), or ``""`` (bare). The tag is what makes ``score_numeric``
    refuse to match a percent against a dollar amount — without it,
    a probe like B5 (whose GT contains ``37%`` and whose agent mentions
    ``$38.75`` somewhere on the Answer line) gets false credit because
    37 lies within ±10% of 38.75.
    """
    text = _THOUSAND_SEP_RE.sub("", text)
    out: list[tuple[float, str]] = []
    for match in _NUM_WITH_UNIT_RE.finditer(text):
        dollar, num, percent = match.groups()
        if num is None:
            continue
        try:
            value = float(num)
        except ValueError:
            continue
        unit = "$" if dollar else ("%" if percent else "")
        out.append((value, unit))
    return out


# Accept the agent's ``Answer:`` line even when wrapped in markdown
# emphasis / heading / bullet / blockquote characters — Qwen and many
# OpenAI models emit ``**Answer:** ...``, ``## Answer: ...``, or
# ``> Answer: ...`` despite the prompt asking for a bare line.
_ANSWER_LINE_RE = re.compile(
    r"(?im)^[\s>#\-*•]*\**\s*answer\s*\**\s*[:\-]\**\s*(.+?)\s*$"
)


def _scoring_text(answer: str) -> str | None:
    """Return the content of the agent's `Answer: ...` line, or None if the
    agent never committed to one. Returning None is the signal for scorers
    to award 0.0 — the prompt instructs the agent to use this format, so
    a missing Answer line is a real failure mode rather than a scoring
    fallback that should be papered over by token-matching the prose.

    When the model emits multiple ``Answer:`` lines (common with Basic,
    which has no retrieval tool and reasons entirely in prose — it
    typically commits to a quick first answer and then self-corrects
    while writing), we take the LAST one. The strategy prompt is paired
    with this: it instructs the agent to put reasoning first and the
    final ``Answer:`` line at the end."""
    matches = _ANSWER_LINE_RE.findall(answer)
    return matches[-1] if matches else None


# Per-unit tolerances for ``score_numeric``. Earlier we ran a single
# ±10% relative tolerance against ALL extracted numbers (currency,
# percent, bare counts) pooled together — that let A7 ($3.75 vs $4.00,
# 6.7% diff) and B5 (GT 37% accidentally matching agent's $38.75)
# claim full credit for wrong answers. Tightening currency to ±5%
# rejects A7's $0.25 price miss; the percentage rule (±5 percentage
# points absolute) rejects 16% vs 37% but still accepts 49% vs 48%
# (B6's actual rounding); bare numbers keep the original ±10%
# relative tolerance for count-style probes that the harness's
# upstream rounding can perturb.
_DOLLAR_REL_TOL = 0.05
_PERCENT_ABS_TOL = 5.0
_BARE_REL_TOL = 0.10


def _numeric_match(gt_value: float, gt_unit: str, ans_value: float) -> bool:
    """Single-pair tolerance check, dispatched by unit."""
    if gt_unit == "$":
        if gt_value == 0:
            return abs(ans_value) <= 0.50  # nominal abs floor near zero
        return abs(ans_value - gt_value) / abs(gt_value) <= _DOLLAR_REL_TOL
    if gt_unit == "%":
        return abs(ans_value - gt_value) <= _PERCENT_ABS_TOL
    if gt_value == 0:
        return abs(ans_value) < 1.0
    return abs(ans_value - gt_value) / abs(gt_value) <= _BARE_REL_TOL


def score_numeric(answer: str, ground_truth: str, tolerance: float = 0.10) -> float:
    """Match each GT number against an unused answer number with the same unit.

    The ``tolerance`` parameter is kept for backwards compat but is no
    longer the only knob: per-unit thresholds (currency / percent /
    bare) live in :func:`_numeric_match`. Greedy first-fit matching is
    preserved — for each GT number we walk answer numbers in order and
    consume the first compatible one.
    """
    text = _scoring_text(answer)
    if text is None:
        return 0.0
    gt_pairs = _extract_typed_numbers(ground_truth)
    ans_pairs = _extract_typed_numbers(text)
    if not gt_pairs:
        return 0.0
    used = [False] * len(ans_pairs)
    matched = 0
    for gt_value, gt_unit in gt_pairs:
        for i, (ans_value, ans_unit) in enumerate(ans_pairs):
            if used[i] or ans_unit != gt_unit:
                continue
            if _numeric_match(gt_value, gt_unit, ans_value):
                matched += 1
                used[i] = True
                break
    return matched / len(gt_pairs)


_PUNCT_RE = re.compile(r'[(){}\[\];:,!?]')

# Characters models use as key/value glue (``cola=32``, ``cola: 32``).
# We want these split into separate tokens so set-overlap matching
# treats ``cola=32`` as ``{cola, 32}`` rather than the literal
# ``cola32`` (which never matches a GT word).
_KV_GLUE_RE = re.compile(r'[=]')


def _normalize_token(token: str) -> str:
    """Strip surrounding punctuation but preserve internal chars like '.' and '$'."""
    token = _PUNCT_RE.sub('', token)
    # Strip leading/trailing periods (sentence punctuation) but keep internal
    # ones so decimals like "$28.30" survive.
    token = token.strip('.')
    return token


def score_keyword(answer: str, ground_truth: str) -> float:
    text = _scoring_text(answer)
    if text is None:
        return 0.0
    gt_tokens = {_normalize_token(t) for t in _KV_GLUE_RE.sub(' ', ground_truth.lower()).split()}
    ans_tokens = {_normalize_token(t) for t in _KV_GLUE_RE.sub(' ', text.lower()).split()}
    stop = {"the", "a", "an", "is", "was", "were", "are", "for", "on", "in", "to", "of", "and", "or", "it", "that", "this", ""}
    gt_tokens -= stop
    ans_tokens -= stop
    if not gt_tokens:
        return 0.0
    overlap = gt_tokens & ans_tokens
    return len(overlap) / len(gt_tokens)


# action_log entries are env-action tool names (substrate agents do
# their data ops *inside* code cells, which don't reach action_log;
# only env tools tick state._action_log). So efficiency metrics here
# can only reason about decision frequency and overall env-tool usage.
_DECISION_TOOLS = {"send_email", "run_sub_agent"}
_NOOP_TOOLS = {"wait_for_next_day"}


def compute_efficiency_metrics(
    daily_action_logs: list[list[str]],
) -> dict[str, Any]:
    """Aggregate env-action efficiency signals derived from action_log.

    Under the RLM substrate the agent's data ops (``db.query``,
    ``ms.sql_exec``, ``log.search`` …) happen inside code cells and
    are NOT mirrored into the ``ToolState._action_log`` — only env
    actions (send_email, run_sub_agent, etc.) are. So this reports
    only env-tool-level signals; for cell-level metrics use the
    ``conversation_log.jsonl`` or trace data.
    """
    total_days = len(daily_action_logs)
    if total_days == 0:
        return {"total_days": 0}

    total_decisions = 0
    total_actions = 0
    for actions in daily_action_logs:
        for action in actions:
            tool_name = action.split()[0].rstrip(":") if action else ""
            if tool_name in _NOOP_TOOLS:
                continue
            total_actions += 1
            if tool_name in _DECISION_TOOLS:
                total_decisions += 1

    return {
        "total_days": total_days,
        "avg_decisions_per_day": round(total_decisions / total_days, 2),
        "avg_actions_per_day": round(total_actions / total_days, 2),
    }
