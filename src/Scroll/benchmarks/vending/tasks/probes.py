"""Vending-specific probe definitions, ground truth + LLM-judge scoring.

Scoring is via an LLM judge (mirrors LongMemEval / BEAM): the
deterministic ground-truth string is built first by ``_gt_*`` from
:class:`VendingSnapshot` / :class:`DataSourceManager` state, then the
agent's free-text answer + the ground truth are handed to
``judge_model`` (configured in ``EnvConfig``). The judge returns a
``<judge_thinking>...</judge_thinking>`` reasoning block followed by
``yes`` / ``no``, which we map to 1.0 / 0.0.

Each scoring call gets its own ``judge.probe.{qid}`` trace span so
Phoenix can render the judge thinking alongside the agent's probe
trace.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Callable

from opentelemetry import trace as otel_trace

from Scroll.core import EnvSnapshot, ProbeSpec
from Scroll.benchmarks.vending.datasource import DataSourceManager

_log = logging.getLogger(__name__)
_tracer = otel_trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# Ground truth extraction functions
# ---------------------------------------------------------------------------

def _all_argmax(d: dict) -> list:
    """Return all keys whose value equals the max (sorted, FP-tolerant).

    Used so probes about "the highest X" enumerate every tied winner instead
    of arbitrarily picking the first one — otherwise an agent that correctly
    reports the tie is penalized by token-overlap scoring.
    """
    if not d:
        return []
    best = max(d.values())
    return sorted(k for k, v in d.items() if abs(v - best) < 1e-6)


def _all_argmin(d: dict) -> list:
    if not d:
        return []
    best = min(d.values())
    return sorted(k for k, v in d.items() if abs(v - best) < 1e-6)


def _gt_a1(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    total = 0.0
    for s in snapshots:
        if 6 <= s.day <= 9:
            for log in s.logs:
                m = re.search(r"rev=([\d.]+)", log)
                if m:
                    total += float(m.group(1))
    return f"{total:.2f}"


def _gt_a2(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    sku_units: dict[str, int] = {}
    for s in snapshots:
        if s.day > 15:
            continue
        for log in s.logs:
            m_sku = re.search(r"sku=(\w+)", log)
            m_units = re.search(r"units=(\d+)", log)
            if m_sku and m_units and log.startswith("sale"):
                sku = m_sku.group(1)
                sku_units[sku] = sku_units.get(sku, 0) + int(m_units.group(1))
    if not sku_units:
        return "no sales"
    best_skus = _all_argmax(sku_units)
    best_units = sku_units[best_skus[0]]
    if len(best_skus) == 1:
        return f"{best_skus[0]} ({best_units} units)"
    return f"{', '.join(best_skus)} (tied at {best_units} units each)"


def _gt_a3(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    price = data._supplier_emails.get("metro_wholesale@example.com", {}).get("cola")
    if price is not None:
        return f"${price:.2f}"
    return "unknown"


def _gt_a4(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    count = 0
    total_cost = 0.0
    for s in snapshots:
        if s.day > 25:
            continue
        for log in s.logs:
            if "delivery_arrived" in log:
                count += 1
                m = re.search(r"(?:booked_cost|cost)=([\d.]+)", log)
                if m:
                    total_cost += float(m.group(1))
    return f"{count} deliveries, total cost ${total_cost:.2f}"


def _gt_a5(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    for s in snapshots:
        if s.day == 14:
            return f"${s.net_worth:.2f}"
    return "unknown"


def _gt_b1(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    day_rev: dict[int, float] = {}
    for s in snapshots:
        if s.day > 12:
            continue
        rev = 0.0
        for log in s.logs:
            m = re.search(r"rev=([\d.]+)", log)
            if m:
                rev += float(m.group(1))
        day_rev[s.day] = rev
    if not day_rev:
        return "no revenue data"
    best_days = _all_argmax(day_rev)
    # Question asks "which day"; revenue value is documentation, not part of
    # the answer. Use raw day numbers (no "day"/"days" prefix) so a wrong day
    # scores 0 instead of getting half credit for echoing the word "day".
    return ", ".join(str(d) for d in best_days)


def _gt_b2(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    sku_stockouts: dict[str, int] = {}
    for s in snapshots:
        if s.day > 18:
            continue
        for log in s.logs:
            if log.startswith("stockout"):
                m = re.search(r"sku=(\w+)", log)
                if m:
                    sku = m.group(1)
                    sku_stockouts[sku] = sku_stockouts.get(sku, 0) + 1
    if not sku_stockouts:
        return "no stockouts recorded"
    frequent = sorted(sku_stockouts.items(), key=lambda x: -x[1])
    return ", ".join(f"{sku} ({count} stockouts)" for sku, count in frequent[:3])


def _gt_b3(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    if not snapshots:
        return "unknown"
    relevant = [s for s in snapshots if s.day <= 22]
    if len(relevant) < 2:
        return "insufficient data"
    first = relevant[0]
    last = relevant[-1]
    days = last.day - first.day
    if days <= 0:
        return "insufficient data"
    total_profit = last.net_worth - first.net_worth
    avg = total_profit / days
    return f"${avg:.2f}/day"


def _gt_a6(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    for s in reversed(snapshots):
        if s.day <= 16 and s.inventory:
            cola = s.inventory.get("cola", {})
            return f"machine={cola.get('machine', 0)} storage={cola.get('storage', 0)}"
    return "unknown"


def _gt_a7(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    for s in reversed(snapshots):
        if s.day <= 23 and s.inventory:
            energy = s.inventory.get("energy", {})
            price = energy.get("price")
            if price is not None:
                return f"${price:.2f}"
    return "unknown"


def _gt_b4(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    catalogs = data._supplier_emails
    all_skus: dict[str, dict[str, float]] = {}
    for supplier, catalog in catalogs.items():
        for sku, price in catalog.items():
            all_skus.setdefault(sku, {})[supplier] = price
    overlapping = {sku: suppliers for sku, suppliers in all_skus.items() if len(suppliers) > 1}
    if not overlapping:
        return "no overlapping products"
    parts: list[str] = []
    for sku, suppliers in sorted(overlapping.items()):
        cheapest = _all_argmin(suppliers)
        cheapest_price = suppliers[cheapest[0]]
        short_names = [s.split("@")[0] for s in cheapest]
        parts.append(f"{sku}: {', '.join(short_names)} (${cheapest_price:.2f})")
    return "; ".join(parts)


def _gt_b5(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    sku_rev: dict[str, float] = {}
    total_rev = 0.0
    for s in snapshots:
        if s.day > 14:
            continue
        for log in s.logs:
            if not log.startswith("sale"):
                continue
            m_sku = re.search(r"sku=(\w+)", log)
            m_rev = re.search(r"rev=([\d.]+)", log)
            if m_sku and m_rev:
                sku = m_sku.group(1)
                rev = float(m_rev.group(1))
                sku_rev[sku] = sku_rev.get(sku, 0) + rev
                total_rev += rev
    if not sku_rev or total_rev == 0:
        return "no revenue"
    top_skus = _all_argmax(sku_rev)
    pct = sku_rev[top_skus[0]] / total_rev * 100
    if len(top_skus) == 1:
        return f"total ${total_rev:.2f}, top SKU {top_skus[0]} at {pct:.0f}%"
    return f"total ${total_rev:.2f}, top SKUs {', '.join(top_skus)} tied at {pct:.0f}% each"


def _gt_b6(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    selling_price = None
    for s in reversed(snapshots):
        if s.day <= 24 and s.inventory:
            cola = s.inventory.get("cola", {})
            selling_price = cola.get("price")
            if selling_price is not None:
                break
    wholesale = data._supplier_emails.get(
        "metro_wholesale@example.com", {}
    ).get("cola")
    if selling_price is None or wholesale is None:
        return "unknown"
    margin = (selling_price - wholesale) / selling_price * 100
    # Question asks for the margin percentage only; selling/cost are inputs the
    # agent may show in its work but aren't required to be in the final answer.
    return f"{margin:.0f}% margin"


# ---------------------------------------------------------------------------
# Extended-horizon probes (30-180 days)
# ---------------------------------------------------------------------------

def _sum_revenue(snapshots: list[EnvSnapshot], lo: int, hi: int) -> float:
    total = 0.0
    for s in snapshots:
        if lo <= s.day <= hi:
            for log in s.logs:
                m = re.search(r"rev=([\d.]+)", log)
                if m:
                    total += float(m.group(1))
    return total


def _min_wholesale(data: DataSourceManager) -> dict[str, float]:
    out: dict[str, float] = {}
    for catalog in data._supplier_emails.values():
        for sku, price in catalog.items():
            if sku not in out or price < out[sku]:
                out[sku] = price
    return out


def _gt_a8(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    return f"{_sum_revenue(snapshots, 21, 30):.2f}"


def _gt_a9(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    count = 0
    for s in snapshots:
        if s.day > 40:
            continue
        for log in s.logs:
            if "delivery_arrived" in log:
                count += 1
    return f"{count} deliveries"


def _gt_b7(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    first = _sum_revenue(snapshots, 1, 15) / 15
    second = _sum_revenue(snapshots, 16, 30) / 15
    # Token-only label so the keyword scorer awards 1.0 on a right answer and
    # 0.0 on a wrong one (no shared "days" giving partial credit either way).
    if first >= second:
        return "1-15"
    return "16-30"


def _gt_a10(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    for s in snapshots:
        if s.day == 30:
            return f"${s.net_worth:.2f}"
    return "unknown"


def _gt_b8(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    wholesale = _min_wholesale(data)
    profit: dict[str, float] = {}
    for s in snapshots:
        if s.day > 55:
            continue
        for log in s.logs:
            if not log.startswith("sale"):
                continue
            m_sku = re.search(r"sku=(\w+)", log)
            m_units = re.search(r"units=(\d+)", log)
            m_rev = re.search(r"rev=([\d.]+)", log)
            if m_sku and m_units and m_rev:
                sku = m_sku.group(1)
                units = int(m_units.group(1))
                rev = float(m_rev.group(1))
                cost = wholesale.get(sku, 0) * units
                profit[sku] = profit.get(sku, 0) + (rev - cost)
    if not profit:
        return "no sales"
    top_skus = _all_argmax(profit)
    # Question asks "which SKU"; the dollar profit value is bonus info we
    # don't want to grade against, so the GT lists SKU name(s) only.
    if len(top_skus) == 1:
        return top_skus[0]
    return ", ".join(top_skus)


def _gt_a11(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    for s in snapshots:
        if s.day == 30 and s.inventory:
            water = s.inventory.get("water", {})
            return f"machine={water.get('machine', 0)} storage={water.get('storage', 0)}"
    return "unknown"


def _gt_b9(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    total_units = 0
    days = 0
    for s in snapshots:
        if 46 <= s.day <= 75:
            total_units += s.units_sold
            days += 1
    if days == 0:
        return "no data"
    return f"{total_units / days:.1f} units/day"


def _gt_a12(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    return f"${_sum_revenue(snapshots, 1, 60):.2f}"


def _gt_b10(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    seen: set[str] = set()
    for s in snapshots:
        if s.day > 100:
            continue
        for sku, info in s.inventory.items():
            if info.get("machine", 0) > 0:
                seen.add(sku)
    if not seen:
        return "0 SKUs"
    return f"{len(seen)} SKUs: {', '.join(sorted(seen))}"


def _gt_a13(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    count = 0
    for s in snapshots:
        if 81 <= s.day <= 110:
            for log in s.logs:
                if log.startswith("stockout"):
                    count += 1
    return f"{count} stockouts"


def _gt_b11(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    window = [s for s in snapshots if 91 <= s.day <= 120]
    if len(window) < 2:
        return "insufficient data"
    first = window[0].net_worth
    last = window[-1].net_worth
    diff = last - first
    # Categorical answer only — exact net-worth values were noise for the
    # keyword scorer; the question is about the trend, not the magnitudes.
    if abs(diff) < 0.05 * max(abs(first), 1.0):
        return "flat"
    return "up" if diff > 0 else "down"


def _gt_a14(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    for s in snapshots:
        if s.day == 60:
            return f"${s.net_worth:.2f}"
    return "unknown"


def _gt_b12(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    counts: dict[str, int] = {}
    for m in data.inbox:
        if m.day <= 145 and m.subject.startswith("Order confirmation"):
            counts[m.source] = counts.get(m.source, 0) + 1
    if not counts:
        return "no orders"
    top = _all_argmax(counts)
    shorts = [t.split("@")[0] for t in top]
    # Question asks "which supplier"; the order count is bonus info we don't
    # want to grade against, so the GT lists supplier name(s) only.
    if len(top) == 1:
        return shorts[0]
    return ", ".join(shorts)


def _gt_a15(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    day_rev: dict[int, float] = {}
    for s in snapshots:
        if s.day > 160:
            continue
        rev = 0.0
        for log in s.logs:
            m = re.search(r"rev=([\d.]+)", log)
            if m:
                rev += float(m.group(1))
        day_rev[s.day] = rev
    if not day_rev:
        return "no data"
    best_days = _all_argmax(day_rev)
    best_rev = day_rev[best_days[0]]
    if len(best_days) == 1:
        return f"day {best_days[0]} (${best_rev:.2f})"
    days_str = ", ".join(str(d) for d in best_days)
    return f"days {days_str} (${best_rev:.2f})"


def _gt_b13(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    early = _sum_revenue(snapshots, 1, 30) / 30
    recent = _sum_revenue(snapshots, 146, 175) / 30
    diff = recent - early
    # Categorical answer only — magnitudes were noise for the keyword scorer.
    if abs(diff) < 0.05 * max(abs(early), 1.0):
        return "flat"
    return "up" if diff > 0 else "down"


def _gt_a16(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    total = 0.0
    for s in snapshots:
        if s.day > 90:
            continue
        for log in s.logs:
            if "delivery_arrived" in log:
                m = re.search(r"(?:booked_cost|cost)=([\d.]+)", log)
                if m:
                    total += float(m.group(1))
    return f"${total:.2f}"


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are evaluating whether an agent's free-text response to a
vending-machine business question matches the deterministic ground
truth. The agent typically commits a final ``Answer: ...`` line; the
content of THAT line is what should be checked. Reasoning above the
Answer line is for your context only.

QUESTION:
{question}

GROUND TRUTH:
{answer}

AGENT RESPONSE:
{response}

DECISION RULES:
- Numeric values: allow modest tolerance — ±5% relative on dollar
  amounts (``$282.00`` vs ``$280.00`` is acceptable), ±5 percentage
  points absolute on percentages (``42%`` vs ``45%`` is acceptable),
  ±10% relative on bare counts.
- Multi-value answers: every value the ground truth lists must be
  matched (e.g. "cola, water (tied at 40 units)" requires both SKUs
  AND the count). Missing a tied item is wrong.
- Phrasing: synonyms / unit symbols ($ vs USD) / minor wording
  variations are fine if the underlying values match.
- Extra prose around the Answer line is fine. Only judge the Answer
  line's content.
- If the agent gave no committed Answer or abstained without trying
  to compute, mark wrong.

Output a brief ``<judge_thinking>...</judge_thinking>`` block with
your reasoning, then on a NEW LINE write either ``yes`` (response
matches ground truth) or ``no`` (it doesn't).
"""


_JUDGE_THINKING_CLOSE_RE = re.compile(r"</judge_thinking\s*>", re.IGNORECASE)
_VERDICT_TOKEN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _parse_judge_verdict(text: str) -> float:
    """Extract yes/no verdict from a judge response.

    Strategy: look at text after the ``</judge_thinking>`` close tag
    (the thinking block can legitimately contain yes/no); if missing,
    fall back to the LAST yes/no in the whole text. Returns 1.0 for
    yes, 0.0 otherwise.
    """
    if not text:
        return 0.0
    m_close = _JUDGE_THINKING_CLOSE_RE.search(text)
    search_region = text[m_close.end():] if m_close else text
    matches = _VERDICT_TOKEN_RE.findall(search_region)
    if not matches:
        matches = _VERDICT_TOKEN_RE.findall(text)
    if not matches:
        return 0.0
    return 1.0 if matches[-1].lower() == "yes" else 0.0


def _judge_score(
    qid: str,
    question: str,
    ground_truth: str,
    agent_answer: str,
    judge_cfg: dict,
) -> float:
    """Call the configured judge and return 1.0 / 0.0.

    Network / quota failures => 0.0 with a warning. Each call gets its
    own ``judge.probe.{qid}`` trace span so Phoenix renders the judge
    thinking alongside the agent's probe trace.
    """
    prompt = _JUDGE_PROMPT.format(
        question=question, answer=ground_truth, response=agent_answer,
    )

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

    # Dashscope-only extra_body. Gate on base_url / model name so we
    # don't send it to the real OpenAI endpoint (which returns HTTP 400).
    judge_extra: dict = {}
    _is_dashscope = (
        "dashscope" in (judge_cfg.get("api_base") or "").lower()
        or (judge_cfg.get("model") or "").lower().startswith(("qwen", "deepseek"))
    )
    if _is_dashscope:
        judge_extra["extra_body"] = {"enable_thinking": False}

    with _tracer.start_as_current_span(f"judge.probe.{qid}") as judge_span:
        judge_span.set_attributes({
            "openinference.span.kind": "CHAIN",
            "judge.model": judge_cfg.get("model", ""),
            "judge.qid": qid,
            "judge.question": question,
            "judge.ground_truth": str(ground_truth)[:500],
            "judge.agent_response": str(agent_answer)[:500],
        })
        try:
            completion = client.chat.completions.create(
                model=judge_cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                n=1,
                temperature=0,
                max_tokens=800,
                **judge_extra,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("judge call failed (%s) — scoring 0.0", exc)
            judge_span.set_attribute("judge.error", str(exc)[:200])
            judge_span.set_attribute("judge.score", 0.0)
            return 0.0

        text = (completion.choices[0].message.content or "").strip()
        score = _parse_judge_verdict(text)
        judge_span.set_attribute("judge.verdict", text[-200:])
        judge_span.set_attribute("judge.score", score)
        return score


def _make_judge_scorer(qid: str, question: str, judge_cfg: dict) -> Callable[[str, str], float]:
    """Build a ``(agent_answer, ground_truth) -> float`` closure that
    routes through :func:`_judge_score` with this probe's identity and
    the shared judge config.
    """
    def scoring_fn(agent_answer: str, ground_truth: str) -> float:
        return _judge_score(
            qid=qid,
            question=question,
            ground_truth=ground_truth,
            agent_answer=agent_answer,
            judge_cfg=judge_cfg,
        )
    return scoring_fn


# ---------------------------------------------------------------------------
# Probe definitions
#
# Static metadata table (qid, turn_idx, question, gt_fn). The
# scoring_fn is built per-probe at ``set_judge_config`` time so each
# closure captures its own (qid, question, judge_cfg). The module-level
# ``PROBES`` list starts empty and is populated when
# :class:`VendingEnv.__init__` calls :func:`set_judge_config`.
# ---------------------------------------------------------------------------


_PROBE_META: list[tuple] = [
    # (qid, turn_idx, question, gt_fn)
    # Early phase (days 10-28): short-horizon recall + basic analysis
    ("A1", 10, "What was the total sales revenue for days 6 through 9?", _gt_a1),
    ("A2", 15, "Which SKU has sold the most units so far? If multiple SKUs are tied for the most, list every one of them with the unit count.", _gt_a2),
    ("A3", 20, "What is the wholesale price of cola from metro_wholesale?", _gt_a3),
    ("A4", 25, "How many delivery shipments have arrived so far (count one shipment per `delivery_arrived` event in the env logs, regardless of how many SKUs were inside), and what was their total cost?", _gt_a4),
    ("A5", 28, "What was your net worth on day 14?", _gt_a5),
    ("A6", 16, "At end of day 16 (after today's sales have already executed), how many units of cola are in the vending machine, and how many are in storage?", _gt_a6),
    ("A7", 23, "At end of day 23 (after today's sales have already executed), what is the selling price of energy drinks?", _gt_a7),
    ("B1", 12, "Which day so far had the highest sales revenue? If multiple days are tied at the same revenue, list every one of them.", _gt_b1),
    ("B2", 18, "Are any SKUs running out completely (zero units in BOTH the vending machine AND storage on the same day)? How many distinct stockout events occurred for each SKU that is actively sold?", _gt_b2),
    ("B3", 22, "What has been the average daily change in your net worth so far? (Take the net worth at end of the latest day, subtract the net worth at end of day 1, and divide by the number of elapsed days.)", _gt_b3),
    ("B4", 26, "For products carried by multiple suppliers, which supplier is cheaper?", _gt_b4),
    ("B5", 14, "What is your total sales revenue so far, and what percentage of that revenue came from your single top-revenue SKU? If multiple SKUs are tied for top revenue, list each with the same shared percentage.", _gt_b5),
    ("B6", 24, "What is your approximate profit margin on cola (selling price minus wholesale cost, as a percentage of selling price)?", _gt_b6),
    # Medium phase (days 35-60)
    ("A8", 35, "What was your total sales revenue for days 21 through 30?", _gt_a8),
    ("A9", 40, "How many delivery shipments have arrived so far (count one shipment per `delivery_arrived` event in the env logs, regardless of how many SKUs were inside)?", _gt_a9),
    ("B7", 45, "Comparing days 1-15 vs days 16-30, which period had higher average daily revenue? Answer with the period range only — exactly `1-15` or `16-30`.", _gt_b7),
    ("A10", 50, "What was your net worth on day 30?", _gt_a10),
    ("B8", 55, "Which SKU has contributed the most cumulative profit so far, where profit per unit = selling revenue minus the CHEAPEST available wholesale price for that SKU across all known suppliers? If multiple SKUs are tied for top profit, list every one of them.", _gt_b8),
    # Long phase (days 65-120)
    ("A11", 65, "How many units of water were in the machine and in storage on day 30?", _gt_a11),
    ("B9", 75, "Over the past 30 days (days 46-75), what was the average daily units sold?", _gt_b9),
    ("A12", 90, "What was your cumulative sales revenue across days 1 through 60?", _gt_a12),
    ("B10", 100, "How many distinct SKUs have you stocked in the vending machine at any point so far?", _gt_b10),
    ("A13", 110, "How many full-depletion stockout events have occurred in the last 30 days (days 81-110)? Count one event per (day, SKU) pair where the SKU had zero units in BOTH the vending machine AND storage at end of day, and only count SKUs that have been stocked at some point.", _gt_a13),
    ("B11", 120, "Over the past 30 days (days 91-120), is your net worth trending up, down, or flat? Answer with one word: `up`, `down`, or `flat` (treat changes within ±5% of the starting net worth as flat).", _gt_b11),
    # Full-horizon phase (days 130-180)
    ("A14", 130, "What was your net worth on day 60?", _gt_a14),
    ("B12", 145, "Which supplier have you placed the most orders with so far, counted by the number of `Order confirmation` emails received from each supplier (NOT by price inquiries, replies, or rejection emails)? If multiple suppliers are tied, list every one of them.", _gt_b12),
    ("A15", 160, "Which day so far had the highest sales revenue, and what was that revenue? If multiple days are tied at the same peak revenue, list every one of them.", _gt_a15),
    ("B13", 175, "Comparing the first 30 days (days 1-30) and the most recent 30 days (days 146-175), has average daily revenue gone up, down, or stayed flat? Answer with one word: `up`, `down`, or `flat` (treat changes within ±5% of the early average as flat).", _gt_b13),
    ("A16", 180, "What was your total delivery spend across the first 90 days?", _gt_a16),
]


# Populated by ``set_judge_config(cfg)``. Empty at module-load time
# because the judge config comes from ``EnvConfig`` (passed in by the
# benchmark harness at env init).
PROBES: list[ProbeSpec] = []


def set_judge_config(cfg) -> None:
    """Build the ``PROBES`` list using ``cfg``'s judge fields.

    Called from :class:`VendingEnv.__init__`. Idempotent — re-calling
    rebuilds the list (useful for tests that spin up multiple envs).
    """
    judge_cfg = {
        "model": cfg.judge_model,
        "api_key_env": cfg.judge_api_key_env,
        "api_base": cfg.judge_api_base,
    }
    PROBES.clear()
    for qid, turn_idx, question, gt_fn in _PROBE_META:
        scoring_fn = _make_judge_scorer(qid, question, judge_cfg)
        PROBES.append(
            ProbeSpec(qid, turn_idx, question, gt_fn, scoring_fn)
        )


def get_probes_for_turn(turn_idx: int) -> list[ProbeSpec]:
    return [p for p in PROBES if p.turn_idx == turn_idx]
