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
    units = _day_sku_units(snapshots, 6, "cola")
    return f"{units} units of cola on day 6"


def _gt_a3(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    price = data._supplier_emails.get("metro_wholesale@example.com", {}).get("cola")
    if price is not None:
        return f"${price:.2f}"
    return "unknown"


def _gt_a4(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    first_day: int | None = None
    costs: list[float] = []
    for s in snapshots:
        for log in s.logs:
            if "delivery_arrived" not in log:
                continue
            m = re.search(r"(?:booked_cost|cost)=([\d.]+)", log)
            cost = float(m.group(1)) if m else 0.0
            if first_day is None:
                first_day = s.day
            if s.day == first_day:
                costs.append(cost)
        if first_day is not None:
            break
    if first_day is None:
        return "no deliveries yet"
    cost_str = ", ".join(f"${c:.2f}" for c in costs)
    return f"day {first_day}, cost(s) {cost_str}"


def _gt_a5(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    val = _day_field(snapshots, 14, "machine_cash")
    if val is None:
        return "unknown"
    return f"${val:.2f}"


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
    quotes: list[tuple[int, str, float]] = []
    for mail in data.inbox:
        if "price list" not in mail.subject.lower():
            continue
        match = re.search(r"cola:\s*\$?([\d.]+)", mail.body, re.IGNORECASE)
        if match:
            quotes.append((mail.day, mail.source.split("@")[0], float(match.group(1))))
    if not quotes:
        return "no cola quotes received yet"
    quotes.sort()
    return "; ".join(f"{src} day {day} ${price:.2f}" for day, src, price in quotes)


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
    sales: list[int] = []
    for s in snapshots:
        for log in s.logs:
            if not log.startswith("sale"):
                continue
            m_sku = re.search(r"sku=(\w+)", log)
            m_units = re.search(r"units=(\d+)", log)
            if m_sku and m_units and m_sku.group(1) == "cola":
                u = int(m_units.group(1))
                if u > 0:
                    sales.append(u)
    if not sales:
        return "no cola sales yet"
    return f"{max(sales) - min(sales)} units (max {max(sales)} minus min {min(sales)})"


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
    return f"${_day_revenue(snapshots, 22):.2f}"


def _gt_a9(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    seen = 0
    for s in snapshots:
        for log in s.logs:
            if "delivery_arrived" not in log:
                continue
            seen += 1
            if seen == 2:
                m_items = re.search(r"items=(\{[^}]+\})", log)
                items_str = m_items.group(1) if m_items else "{}"
                return f"day {s.day}, items {items_str}"
    return "fewer than 2 deliveries so far"


def _gt_b7(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    first = _sum_revenue(snapshots, 1, 15) / 15
    second = _sum_revenue(snapshots, 16, 30) / 15
    # Token-only label so the keyword scorer awards 1.0 on a right answer and
    # 0.0 on a wrong one (no shared "days" giving partial credit either way).
    if first >= second:
        return "1-15"
    return "16-30"


def _gt_a10(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    units = _day_sku_units(snapshots, 17, "chips")
    return f"{units} units of chips on day 17"


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
    return f"${_day_revenue(snapshots, 33):.2f}"


def _gt_b10(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    days = [s.day for s in snapshots if s.day <= 100 and any(
        log.startswith("stockout") for log in s.logs
    )]
    if not days:
        return "no stockouts"
    return f"{len(days)} days: {', '.join(str(d) for d in days)}"


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
    val = _day_field(snapshots, 50, "cash")
    if val is None:
        return "unknown"
    return f"${val:.2f}"


def _gt_b12(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    best: tuple[int, int, str] | None = None  # (sku_count, day, source)
    for mail in data.inbox:
        if not mail.subject.startswith("Order confirmation"):
            continue
        items_match = re.search(r"items=\{([^}]*)\}", mail.body)
        if not items_match:
            continue
        sku_count = len(re.findall(r"'(\w+)'", items_match.group(1)))
        if sku_count == 0:
            continue
        if best is None or sku_count < best[0]:
            best = (sku_count, mail.day, mail.source)
    if best is None:
        return "no order confirmations yet"
    sku_count, day, source = best
    return f"day {day}, supplier {source.split('@')[0]}, {sku_count} SKU(s)"


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
    shipments: list[tuple[int, float]] = []
    for s in snapshots:
        for log in s.logs:
            if "delivery_arrived" not in log:
                continue
            m = re.search(r"(?:booked_cost|cost)=([\d.]+)", log)
            if m:
                shipments.append((s.day, float(m.group(1))))
    if not shipments:
        return "no deliveries"
    best_cost = max(c for _, c in shipments)
    tied_days = sorted({d for d, c in shipments if abs(c - best_cost) < 0.01})
    if len(tied_days) == 1:
        return f"day {tied_days[0]}, cost ${best_cost:.2f}"
    days_str = ", ".join(str(d) for d in tied_days)
    return f"cost ${best_cost:.2f}, tied across days {days_str}"


# ---------------------------------------------------------------------------
# Extended coverage probes — fill medium/long/late-horizon gaps and
# introduce new question shapes (volatility/spread, abandoned SKUs,
# price-change detection, fee totals).
# ---------------------------------------------------------------------------

_CATALOG_SKUS = frozenset({"cola", "water", "chips", "choco", "energy", "gum", "juice", "nuts"})


def _day_logs(snapshots: list[EnvSnapshot], day: int) -> list[str]:
    for s in snapshots:
        if s.day == day:
            return s.logs
    return []


def _day_field(snapshots: list[EnvSnapshot], day: int, field: str):
    for s in snapshots:
        if s.day == day:
            return getattr(s, field, None)
    return None


def _day_revenue(snapshots: list[EnvSnapshot], day: int) -> float:
    total = 0.0
    for log in _day_logs(snapshots, day):
        m = re.search(r"rev=([\d.]+)", log)
        if m:
            total += float(m.group(1))
    return total


def _day_sku_units(snapshots: list[EnvSnapshot], day: int, sku: str) -> int:
    for log in _day_logs(snapshots, day):
        if not log.startswith("sale"):
            continue
        m_sku = re.search(r"sku=(\w+)", log)
        m_units = re.search(r"units=(\d+)", log)
        if m_sku and m_units and m_sku.group(1) == sku:
            return int(m_units.group(1))
    return 0


def _day_total_units(snapshots: list[EnvSnapshot], day: int) -> int:
    total = 0
    for log in _day_logs(snapshots, day):
        if not log.startswith("sale"):
            continue
        m = re.search(r"units=(\d+)", log)
        if m:
            total += int(m.group(1))
    return total


def _daily_revenue(snapshots: list[EnvSnapshot], lo: int, hi: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for s in snapshots:
        if not (lo <= s.day <= hi):
            continue
        rev = 0.0
        for log in s.logs:
            m = re.search(r"rev=([\d.]+)", log)
            if m:
                rev += float(m.group(1))
        out[s.day] = rev
    return out


def _sold_skus(snapshots: list[EnvSnapshot], lo: int, hi: int) -> set[str]:
    sold: set[str] = set()
    for s in snapshots:
        if not (lo <= s.day <= hi):
            continue
        for log in s.logs:
            if not log.startswith("sale"):
                continue
            m = re.search(r"sku=(\w+)", log)
            if m:
                sold.add(m.group(1))
    return sold


def _sku_revenue(snapshots: list[EnvSnapshot], lo: int, hi: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in snapshots:
        if not (lo <= s.day <= hi):
            continue
        for log in s.logs:
            if not log.startswith("sale"):
                continue
            m_sku = re.search(r"sku=(\w+)", log)
            m_rev = re.search(r"rev=([\d.]+)", log)
            if m_sku and m_rev:
                out[m_sku.group(1)] = out.get(m_sku.group(1), 0.0) + float(m_rev.group(1))
    return out


def _gt_a17(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    return f"{_day_total_units(snapshots, 14)} units on day 14"


def _gt_a18(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    for s in snapshots:
        if s.day == 35 and s.inventory:
            cola = s.inventory.get("cola", {})
            return f"storage={cola.get('storage', 0)}"
    return "unknown"


def _gt_b14(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    day_rev = _daily_revenue(snapshots, 21, 40)
    if not day_rev:
        return "no data"
    spread = max(day_rev.values()) - min(day_rev.values())
    return f"${spread:.2f}"


def _gt_b15(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    sold = _sold_skus(snapshots, 16, 45)
    unsold = sorted(_CATALOG_SKUS - sold)
    if not unsold:
        return "none"
    return ", ".join(unsold)


def _gt_a19(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    count = sum(1 for m in data.inbox if m.day <= 50)
    return f"{count} emails"


def _gt_b16(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    prev: float | None = None
    for s in snapshots:
        if s.day > 70:
            break
        if not s.inventory:
            continue
        price = s.inventory.get("cola", {}).get("price")
        if price is None:
            continue
        if prev is not None and abs(price - prev) > 0.01:
            return f"day {s.day}"
        prev = price
    return "never"


def _gt_a20(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    total = 0.0
    for s in snapshots:
        if not (31 <= s.day <= 60):
            continue
        for log in s.logs:
            if "delivery_arrived" in log:
                m = re.search(r"(?:booked_cost|cost)=([\d.]+)", log)
                if m:
                    total += float(m.group(1))
    return f"${total:.2f}"


def _gt_b17(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    day_rev = _daily_revenue(snapshots, 61, 85)
    if not day_rev:
        return "no data"
    spread = max(day_rev.values()) - min(day_rev.values())
    return f"${spread:.2f}"


def _gt_a21(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    day_rev = _daily_revenue(snapshots, 1, 95)
    count = sum(1 for rev in day_rev.values() if rev > 50)
    return f"{count} days"


def _gt_b18(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    sku_so: dict[str, int] = {}
    for s in snapshots:
        if s.day > 100:
            continue
        for log in s.logs:
            if log.startswith("stockout"):
                m = re.search(r"sku=(\w+)", log)
                if m:
                    sku_so[m.group(1)] = sku_so.get(m.group(1), 0) + 1
    if not sku_so:
        return "no stockouts"
    top = _all_argmax(sku_so)
    if len(top) == 1:
        return top[0]
    return ", ".join(top)


def _gt_a22(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    val = _day_field(snapshots, 50, "machine_cash")
    if val is None:
        return "unknown"
    return f"${val:.2f}"


def _gt_b19(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    sold = _sold_skus(snapshots, 86, 115)
    unsold = sorted(_CATALOG_SKUS - sold)
    if not unsold:
        return "none"
    return ", ".join(unsold)


def _gt_a23(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    return f"${_day_revenue(snapshots, 75):.2f}"


def _gt_a24(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    val = _day_field(snapshots, 100, "cash")
    if val is None:
        return "unknown"
    return f"${val:.2f}"


def _gt_a25(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    units = _day_sku_units(snapshots, 50, "water")
    return f"{units} units of water on day 50"


def _gt_b20(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    first = _sum_revenue(snapshots, 91, 120) / 30
    second = _sum_revenue(snapshots, 121, 150) / 30
    if abs(first - second) < 0.05 * max(first, 1.0):
        return "tied"
    if first > second:
        return "91-120"
    return "121-150"


def _gt_a26(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    total = 0.0
    for s in snapshots:
        if s.day > 150:
            continue
        for log in s.logs:
            if "delivery_arrived" in log:
                m = re.search(r"(?:booked_cost|cost)=([\d.]+)", log)
                if m:
                    total += float(m.group(1))
    return f"${total:.2f}"


def _gt_b21(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    sku_rev = _sku_revenue(snapshots, 121, 150)
    if not sku_rev:
        return "no sales"
    top = _all_argmax(sku_rev)
    if len(top) == 1:
        return top[0]
    return ", ".join(top)


def _gt_a27(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    return f"${_day_revenue(snapshots, 137):.2f}"


def _gt_b22(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    day_rev = _daily_revenue(snapshots, 141, 170)
    if len(day_rev) < 2:
        return "insufficient data"
    days = sorted(day_rev.keys())
    max_jump = max(
        abs(day_rev[days[i + 1]] - day_rev[days[i]]) for i in range(len(days) - 1)
    )
    return f"${max_jump:.2f}"


def _gt_b23(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    sku_rev = _sku_revenue(snapshots, 1, 178)
    if not sku_rev:
        return "no sales"
    top = _all_argmax(sku_rev)
    if len(top) == 1:
        return top[0]
    return ", ".join(top)


# ---------------------------------------------------------------------------
# Business-analytics probes — trend + aggregation queries phrased as
# realistic stakeholder asks (investor reviews, finance audits, ops
# debriefs). SQL-friendly, summary-hostile.
# ---------------------------------------------------------------------------


def _gt_b24(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    week_rev: dict[int, float] = {}
    for s in snapshots:
        if s.day > 168:
            continue
        week = (s.day - 1) // 7 + 1
        rev = 0.0
        for log in s.logs:
            m = re.search(r"rev=([\d.]+)", log)
            if m:
                rev += float(m.group(1))
        week_rev[week] = week_rev.get(week, 0.0) + rev
    if not week_rev:
        return "no data"
    best = max(week_rev.values())
    tied = sorted(w for w, r in week_rev.items() if abs(r - best) < 0.01)
    if len(tied) == 1:
        return f"week {tied[0]} (${best:.2f})"
    return f"weeks {', '.join(str(w) for w in tied)} tied at ${best:.2f}"


def _gt_b25(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    weekend_rev, weekend_days = 0.0, 0
    weekday_rev, weekday_days = 0.0, 0
    for s in snapshots:
        if s.day > 88:
            continue
        rev = 0.0
        for log in s.logs:
            m = re.search(r"rev=([\d.]+)", log)
            if m:
                rev += float(m.group(1))
        if s.day % 7 in (5, 6):
            weekend_rev += rev
            weekend_days += 1
        else:
            weekday_rev += rev
            weekday_days += 1
    if weekend_days == 0 or weekday_days == 0:
        return "insufficient data"
    wk_avg = weekend_rev / weekend_days
    wd_avg = weekday_rev / weekday_days
    diff_pct = ((wk_avg - wd_avg) / wd_avg * 100) if wd_avg > 0 else 0.0
    return f"weekend ${wk_avg:.2f}/day vs weekday ${wd_avg:.2f}/day ({diff_pct:+.0f}%)"


def _gt_b26(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    cum = 0.0
    for s in snapshots:
        if s.day > 60:
            continue
        for log in s.logs:
            m = re.search(r"rev=([\d.]+)", log)
            if m:
                cum += float(m.group(1))
        if cum >= 1000:
            return f"day {s.day}"
    return "not yet reached"


def _gt_b27(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    day_rev = _daily_revenue(snapshots, 1, 158)
    if len(day_rev) < 7:
        return "insufficient data"
    days_sorted = sorted(day_rev.keys())
    best_sum, starts = -1.0, []
    for i in range(len(days_sorted) - 6):
        window = days_sorted[i:i + 7]
        if window[6] - window[0] != 6:
            continue
        s_sum = sum(day_rev[d] for d in window)
        if s_sum > best_sum + 0.01:
            best_sum = s_sum
            starts = [window[0]]
        elif abs(s_sum - best_sum) < 0.01:
            starts.append(window[0])
    if not starts:
        return "no contiguous window"
    if len(starts) == 1:
        return f"start day {starts[0]} (${best_sum:.2f})"
    return f"start days {', '.join(str(s) for s in starts)} tied at ${best_sum:.2f}"


def _gt_b28(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    h1 = _sum_revenue(snapshots, 1, 60) / 60
    h2 = _sum_revenue(snapshots, 61, 120) / 60
    if h1 <= 0:
        return "no early data"
    pct = (h2 - h1) / h1 * 100
    if abs(pct) < 10:
        label = "stable"
    elif pct > 0:
        label = "accelerating"
    else:
        label = "decelerating"
    return f"{label} (H1 avg ${h1:.2f}/day, H2 avg ${h2:.2f}/day, {pct:+.0f}%)"


def _gt_a28(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    sku_units = {sku: 0 for sku in _CATALOG_SKUS}
    for s in snapshots:
        if s.day > 78:
            continue
        for log in s.logs:
            if not log.startswith("sale"):
                continue
            m_sku = re.search(r"sku=(\w+)", log)
            m_units = re.search(r"units=(\d+)", log)
            if m_sku and m_units:
                sku = m_sku.group(1)
                sku_units[sku] = sku_units.get(sku, 0) + int(m_units.group(1))
    ranked = sorted(sku_units.items(), key=lambda x: (-x[1], x[0]))
    return ", ".join(f"{sku} {u}" for sku, u in ranked)


def _gt_a29(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    spend: dict[str, float] = {}
    for mail in data.inbox:
        if mail.day > 148:
            continue
        if not mail.subject.startswith("Order confirmation"):
            continue
        m = re.search(r"cost=([\d.]+)", mail.body)
        if not m:
            continue
        src = mail.source.split("@")[0]
        spend[src] = spend.get(src, 0.0) + float(m.group(1))
    if not spend:
        return "no orders placed yet"
    ranked = sorted(spend.items(), key=lambda x: (-x[1], x[0]))
    return ", ".join(f"{s} ${c:.2f}" for s, c in ranked)


def _gt_a30(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    count = 0
    for s in snapshots:
        if s.day > 102:
            continue
        rev = 0.0
        for log in s.logs:
            m = re.search(r"rev=([\d.]+)", log)
            if m:
                rev += float(m.group(1))
        if rev < 15:
            count += 1
    return f"{count} days"


def _gt_a31(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    revs: list[float] = []
    for s in snapshots:
        if s.day > 113:
            continue
        rev = 0.0
        for log in s.logs:
            m = re.search(r"rev=([\d.]+)", log)
            if m:
                rev += float(m.group(1))
        revs.append(rev)
    if not revs:
        return "no data"
    revs.sort()
    n = len(revs)
    med = revs[n // 2] if n % 2 == 1 else (revs[n // 2 - 1] + revs[n // 2]) / 2
    return f"${med:.2f}"


def _gt_a32(snapshots: list[EnvSnapshot], data: DataSourceManager) -> str:
    count = 0
    for s in snapshots:
        if s.day > 138:
            continue
        has_delivery = any("delivery_arrived" in log for log in s.logs)
        has_stockout = any(log.startswith("stockout") for log in s.logs)
        if has_delivery and has_stockout:
            count += 1
    return f"{count} days"


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
    # Every probe is phrased as a realistic stakeholder ask (investor
    # update, ops debrief, finance audit, etc.) followed by a precise
    # computable specification.
    #
    # Early phase (days 10-28): short-horizon recall + basic analysis
    ("A1", 10, "Quick sanity check before tonight's team standup — what was our total sales revenue for days 6 through 9 combined?", _gt_a1),
    ("A2", 15, "Cola is our flagship and I want to know how it did on its first big day on the floor. How many units of cola did you sell on day 6 specifically?", _gt_a2),
    ("A3", 20, "Procurement is running a margin check on our top SKU. What is the wholesale price that `metro_wholesale` charges per unit of cola?", _gt_a3),
    ("A4", 25, "I'm putting together our supply-chain timeline for the deck. On which day did your FIRST delivery shipment arrive, and what was the `booked_cost` (if multiple shipments arrived on that first day, list every shipment's cost)?", _gt_a4),
    ("A5", 28, "Finance is reconciling cash positions. What was your `machine_cash` balance at end of day 14 — that's the uncollected sales cash physically sitting inside the vending machine, NOT net worth and NOT your `cash` account?", _gt_a5),
    ("A6", 16, "End-of-day inventory snapshot for ops. At end of day 16 (after today's sales have already cleared), how many units of cola are in the vending machine, and how many are sitting in back-room storage?", _gt_a6),
    ("A7", 23, "Pricing audit check — what is the current selling price on energy drinks at end of day 23 (after today's sales executed)?", _gt_a7),
    ("B1", 12, "I want to highlight our best day so far in the team channel. From day 1 through day 12, which day had the highest total sales revenue? If multiple days are tied at the same peak, list every one of them.", _gt_b1),
    ("B2", 18, "Ops debrief — I'm worried we're running SKUs to zero. From day 1 through day 18, identify whether any SKUs depleted completely (zero units in BOTH the vending machine AND storage on the same day). For each SKU that's actively sold, report how many distinct stockout events have occurred so far.", _gt_b2),
    ("B3", 22, "Investor update prep — what's our average daily change in net worth so far? (Take net worth at end of the latest day, subtract net worth at end of day 1, divide by the number of elapsed days.)", _gt_b3),
    ("B4", 26, "Procurement wants every cola price quote we've ever received. List every cola wholesale-price quote that has actually arrived in your inbox so far (only count `Re: Price List` reply emails). For each, give the supplier short-name, the day the quote arrived, and the quoted price. Answer `no cola quotes received yet` if no such email is in the inbox.", _gt_b4),
    ("B5", 14, "Quick investor blurb: what's our total sales revenue so far, and what percentage of it came from our single top-earning SKU? If multiple SKUs are tied at top revenue, list each at the same shared percentage.", _gt_b5),
    ("B6", 24, "Demand-variance check on our flagship — looking only at days where cola actually sold (non-zero `units` in the sale log line), what is the difference (in units) between your single LARGEST cola sale day and your single SMALLEST cola sale day?", _gt_b6),
    # Medium phase (days 30-55)
    ("A17", 32, "I'm reconstructing a daily throughput chart. How many total units (summed across every SKU) did you sell on day 14 specifically?", _gt_a17),
    ("A8", 35, "Spot-check on a single historical day for the model: what was your EXACT total sales revenue on day 22? Single-day value — not the 21-30 window.", _gt_a8),
    ("A18", 38, "Back-room inventory check: at end of day 35 (after that day's sales executed), how many units of cola were sitting in storage?", _gt_a18),
    ("A9", 40, "Supply-chain timeline part 2: look at the SECOND delivery shipment you ever received (the second `delivery_arrived` line in your env logs). On which day did it arrive and what items did it contain?", _gt_a9),
    ("B14", 42, "Variance check for the ops team — across days 21 through 40, what was the revenue SPREAD (highest single-day total revenue minus lowest single-day total revenue in that window)?", _gt_b14),
    ("B7", 45, "Did the second half of our first month beat the first half? Comparing days 1-15 vs days 16-30, which window had higher average daily revenue? Answer with the period range only — exactly `1-15` or `16-30`.", _gt_b7),
    ("B15", 48, "Catalog rationalization meeting — which SKUs are basically dead inventory? From day 16 through day 45, list every SKU in your catalog that had ZERO sales in that window (comma-separated). Answer `none` if every SKU sold at least one unit.", _gt_b15),
    ("A10", 50, "Spot-check on a single SKU on a specific past day. How many units of chips did you sell on day 17?", _gt_a10),
    ("A19", 52, "Inbox hygiene check — how many emails were sitting in your inbox at end of day 50? Count every received email regardless of read/unread status.", _gt_a19),
    ("B8", 55, "Margin analysis: which SKU has contributed the most cumulative PROFIT so far (not revenue — profit), where profit per unit = selling revenue minus the CHEAPEST known wholesale price for that SKU across all suppliers we've heard from? If multiple SKUs are tied at top profit, list every one of them.", _gt_b8),
    ("B26", 60, "We want to tell the team when the business hit its first big milestone. Looking at cumulative sales revenue (just gross sales, not profit, summed from day 1 onward), on which day did cumulative revenue first cross or reach $1000? Give the exact day number. If we haven't hit it yet, answer `not yet reached`.", _gt_b26),
    # Long phase (days 65-120)
    ("A11", 65, "Historical inventory snapshot for the deck: how many units of water were in the vending machine and how many were in storage on day 30?", _gt_a11),
    ("B16", 70, "Pricing change log: on which day did the selling price of cola first change from its initial value? Answer with the day number (e.g. `day 17`), or `never` if the price has not changed.", _gt_b16),
    ("B9", 75, "Rolling 30-day throughput KPI — over the past 30 days (days 46-75), what was your average daily units sold?", _gt_b9),
    ("A28", 78, "Q3 SKU performance review — we're deciding which SKUs to keep stocking next quarter. From day 1 through day 78, list every SKU in the catalog along with its cumulative units sold across that window. Include all 8 SKUs (`cola`, `water`, `chips`, `choco`, `energy`, `gum`, `juice`, `nuts`) — even SKUs with zero sales should appear with `0`. Order from highest to lowest unit count.", _gt_a28),
    ("A20", 80, "Q2 procurement spend check — what was the total delivery cost (sum of `booked_cost` across all `delivery_arrived` events) for days 31 through 60?", _gt_a20),
    ("B17", 85, "Variance check on the late-summer window for ops: across days 61 through 85, what was the revenue spread — highest single-day total revenue minus lowest single-day total revenue?", _gt_b17),
    ("B25", 88, "We're thinking about a weekend promotion but first need to know how much weekend traffic is already lifting us. From day 1 through day 88, treat days where `day % 7` is 5 or 6 as weekends and the rest as weekdays. Report (a) average daily sales revenue on weekend days, (b) average daily sales revenue on weekday days, and (c) the percentage uplift of weekend over weekday (signed, e.g. `+15%`).", _gt_b25),
    ("A12", 90, "Another model spot-check on a specific past day. What was your EXACT total sales revenue on day 33? Single-day value — not a window.", _gt_a12),
    ("A21", 95, "Strong-day counter for the year-in-review slide: from day 1 through day 95, how many days generated more than $50 in total sales revenue?", _gt_a21),
    ("B10", 100, "Stockout audit before the ops monthly: list EVERY distinct day from day 1 through day 100 on which at least one `stockout` event was recorded. Report the count of such days plus the comma-separated list of day numbers.", _gt_b10),
    ("A30", 102, "Internal audit — we want to flag underperforming days for review. From day 1 through day 102, how many days had total daily sales revenue strictly below $15? ($15 is roughly our daily breakeven once you account for the daily fee and inventory carrying cost, so a day below that is a day we essentially lost money operationally.) Just report the count.", _gt_a30),
    ("B18", 105, "Worst-offender SKU for stockouts so far: from day 1 through day 100, which SKU has had the most stockout events? If multiple SKUs are tied at the same number of stockouts, list every one of them.", _gt_b18),
    ("A22", 108, "Cash reconciliation back to mid-Q2 — what was your `machine_cash` balance at end of day 50 (the uncollected cash sitting inside the machine, NOT net worth, NOT the `cash` account)?", _gt_a22),
    ("A13", 110, "Trailing 30-day stockout review: how many full-depletion stockout events occurred from day 81 through day 110? Count one event per (day, SKU) pair where the SKU had zero units in BOTH the vending machine AND storage at end of day; only count SKUs that have been stocked at some point.", _gt_a13),
    ("A31", 113, "Average daily revenue keeps getting skewed by a few good days, so I want a number that better reflects a 'typical day'. From day 1 through day 113, take every day's total sales revenue, sort them from low to high, and report the MEDIAN (50th percentile). For even sample sizes, use the average of the two middle values.", _gt_a31),
    ("B19", 115, "Rolling 30-day dead-inventory check — which SKUs in your catalog had ZERO sales over the past 30 days (days 86 through 115)? List every such SKU (comma-separated), or answer `none` if every SKU sold at least one unit.", _gt_b19),
    ("B11", 120, "Last 30 days net-worth direction for the investor email — is net worth trending up, down, or flat over days 91-120? Answer with one word: `up`, `down`, or `flat` (treat changes within ±5% of the starting net worth as flat).", _gt_b11),
    ("A23", 122, "Another spot-check on a specific past day. What was your EXACT total sales revenue on day 75? Single-day value, not a window.", _gt_a23),
    ("B28", 125, "The board wants a single-word read on our trajectory — are we accelerating, decelerating, or basically flat? Compare the first half (days 1-60) and second half (days 61-120) of the first 120 days. Compute average daily sales revenue for each half, then answer with one word: `accelerating` (H2 is >10% higher than H1), `decelerating` (H2 is >10% lower than H1), or `stable` (within ±10%). Also report both averages.", _gt_b28),
    # Full-horizon phase (days 130-180)
    ("A14", 130, "Finance reconciliation — what was your `cash` balance (the cash-account field returned by `get_money_balance`, NOT net worth and NOT `machine_cash`) at end of day 50?", _gt_a14),
    ("A24", 135, "Quarterly cash position check — what was your `cash` balance (the cash-account field, NOT net worth and NOT `machine_cash`) at end of day 100?", _gt_a24),
    ("A32", 138, "Ops review — I want to know whether our restocking is reactive (i.e. we wait until we're already out of stock and a delivery happens to land the same day). From day 1 through day 138, how many days had BOTH a `delivery_arrived` event AND at least one `stockout` event recorded on the same day? Just report the count.", _gt_a32),
    ("A25", 140, "Spot-check on a single SKU on a specific day: how many units of water did you sell on day 50?", _gt_a25),
    ("B12", 145, "Procurement order audit — across every `Order confirmation` email in your inbox so far, identify the order with the SMALLEST number of distinct SKUs (count distinct SKUs in that confirmation's `items` dict). Report the day the confirmation arrived, the supplier short-name, and the SKU count.", _gt_b12),
    ("A29", 148, "Finance is doing annual procurement review. From day 1 through day 148, what is your total spend with each supplier (sum the `cost=` field across every `Order confirmation` email's body, grouped by the email's `source` supplier)? Report the breakdown using supplier short-names, ordered from highest to lowest spend.", _gt_a29),
    ("B20", 150, "Last-60-days month-over-month comparison: did average daily revenue rise from days 91-120 to days 121-150? Answer with one of — `91-120`, `121-150`, or `tied` (treat changes within ±5% as tied).", _gt_b20),
    ("A26", 155, "Year-to-date procurement spend KPI — what was your total delivery cost (sum of `booked_cost` across all `delivery_arrived` events) from day 1 through day 150?", _gt_a26),
    ("B27", 158, "For the pitch deck I want to feature our single best week of operations. Considering every possible 7-consecutive-day window from day 1 through day 158 (so windows start on day 1, 2, 3, ..., 152), which window had the highest total sales revenue? Report the starting day and the window's total revenue. If multiple windows are tied at the same peak, list every tied starting day.", _gt_b27),
    ("A15", 160, "Best day in business so far — from day 1 through day 160, which day had the highest total sales revenue, and what was that revenue? If multiple days are tied at the same peak, list every one of them.", _gt_a15),
    ("B21", 165, "Top SKU for the last quarter window (days 121-150): which SKU generated the most cumulative sales revenue in that window? If multiple SKUs are tied at the same total, list every one of them.", _gt_b21),
    ("B24", 168, "Weekly performance review for the investor deck. Group day 1 through day 168 into 7-day weeks (week 1 = days 1-7, week 2 = days 8-14, ..., week 24 = days 162-168). Which week had the highest total sales revenue? Report the week number and that week's total. If multiple weeks are tied, list every tied week number.", _gt_b24),
    ("A27", 170, "Another single-day spot-check: what was your EXACT total sales revenue on day 137? Single-day value — not a window.", _gt_a27),
    ("B22", 172, "Stability check on recent operations — across days 141 through 170, what was the LARGEST absolute change in total daily revenue between any two consecutive days?", _gt_b22),
    ("B13", 175, "Full-horizon trajectory check — comparing the first 30 days (days 1-30) and the most recent 30 days (days 146-175), has average daily revenue gone up, down, or stayed flat? Answer with one word: `up`, `down`, or `flat` (treat changes within ±5% of the early average as flat).", _gt_b13),
    ("B23", 178, "All-time winner SKU: which SKU has generated the most cumulative sales revenue across days 1 through 178? If multiple SKUs are tied at the same total, list every one of them.", _gt_b23),
    ("A16", 180, "End-of-horizon procurement review — across all 180 days, which SINGLE delivery shipment had the highest `booked_cost`? Report the day it arrived and the cost. If multiple shipments are tied at the same peak cost, list every tied day.", _gt_a16),
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
