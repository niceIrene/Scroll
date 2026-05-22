"""Vending-specific probe definitions and ground truth functions."""

from __future__ import annotations

import re

from Scroll.core import EnvSnapshot, ProbeSpec
from Scroll.benchmarks.vending.datasource import DataSourceManager
from Scroll.benchmarks.vending.tasks.rewards import score_numeric, score_keyword


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
# Probe definitions
# ---------------------------------------------------------------------------

PROBES: list[ProbeSpec] = [
    # Early phase (days 10-28): short-horizon recall + basic analysis
    ProbeSpec("A1", 10, "What was the total sales revenue for days 6 through 9?", _gt_a1, score_numeric),
    ProbeSpec("A2", 15, "Which SKU has sold the most units so far? If multiple SKUs are tied for the most, list every one of them with the unit count.", _gt_a2, score_keyword),
    ProbeSpec("A3", 20, "What is the wholesale price of cola from metro_wholesale?", _gt_a3, score_numeric),
    ProbeSpec("A4", 25, "How many delivery shipments have arrived so far (count one shipment per `delivery_arrived` event in the env logs, regardless of how many SKUs were inside), and what was their total cost?", _gt_a4, score_numeric),
    ProbeSpec("A5", 28, "What was your net worth on day 14?", _gt_a5, score_numeric),
    ProbeSpec("A6", 16, "At end of day 16 (after today's sales have already executed), how many units of cola are in the vending machine, and how many are in storage?", _gt_a6, score_numeric),
    ProbeSpec("A7", 23, "At end of day 23 (after today's sales have already executed), what is the selling price of energy drinks?", _gt_a7, score_numeric),
    ProbeSpec("B1", 12, "Which day so far had the highest sales revenue? If multiple days are tied at the same revenue, list every one of them.", _gt_b1, score_keyword),
    ProbeSpec("B2", 18, "Are any SKUs running out completely (zero units in BOTH the vending machine AND storage on the same day)? How many distinct stockout events occurred for each SKU that is actively sold?", _gt_b2, score_keyword),
    ProbeSpec("B3", 22, "What has been the average daily change in your net worth so far? (Take the net worth at end of the latest day, subtract the net worth at end of day 1, and divide by the number of elapsed days.)", _gt_b3, score_numeric),
    ProbeSpec("B4", 26, "For products carried by multiple suppliers, which supplier is cheaper?", _gt_b4, score_keyword),
    ProbeSpec("B5", 14, "What is your total sales revenue so far, and what percentage of that revenue came from your single top-revenue SKU? If multiple SKUs are tied for top revenue, list each with the same shared percentage.", _gt_b5, score_numeric),
    ProbeSpec("B6", 24, "What is your approximate profit margin on cola (selling price minus wholesale cost, as a percentage of selling price)?", _gt_b6, score_numeric),

    # Medium phase (days 35-60): 10-day window aggregations, past-state recall
    ProbeSpec("A8", 35, "What was your total sales revenue for days 21 through 30?", _gt_a8, score_numeric),
    ProbeSpec("A9", 40, "How many delivery shipments have arrived so far (count one shipment per `delivery_arrived` event in the env logs, regardless of how many SKUs were inside)?", _gt_a9, score_numeric),
    ProbeSpec("B7", 45, "Comparing days 1-15 vs days 16-30, which period had higher average daily revenue? Answer with the period range only — exactly `1-15` or `16-30`.", _gt_b7, score_keyword),
    ProbeSpec("A10", 50, "What was your net worth on day 30?", _gt_a10, score_numeric),
    ProbeSpec("B8", 55, "Which SKU has contributed the most cumulative profit so far, where profit per unit = selling revenue minus the CHEAPEST available wholesale price for that SKU across all known suppliers? If multiple SKUs are tied for top profit, list every one of them.", _gt_b8, score_keyword),

    # Long phase (days 65-120): 30-day rolling windows, deep-past recall
    ProbeSpec("A11", 65, "How many units of water were in the machine and in storage on day 30?", _gt_a11, score_numeric),
    ProbeSpec("B9", 75, "Over the past 30 days (days 46-75), what was the average daily units sold?", _gt_b9, score_numeric),
    ProbeSpec("A12", 90, "What was your cumulative sales revenue across days 1 through 60?", _gt_a12, score_numeric),
    ProbeSpec("B10", 100, "How many distinct SKUs have you stocked in the vending machine at any point so far?", _gt_b10, score_numeric),
    ProbeSpec("A13", 110, "How many full-depletion stockout events have occurred in the last 30 days (days 81-110)? Count one event per (day, SKU) pair where the SKU had zero units in BOTH the vending machine AND storage at end of day, and only count SKUs that have been stocked at some point.", _gt_a13, score_numeric),
    ProbeSpec("B11", 120, "Over the past 30 days (days 91-120), is your net worth trending up, down, or flat? Answer with one word: `up`, `down`, or `flat` (treat changes within ±5% of the starting net worth as flat).", _gt_b11, score_keyword),

    # Full-horizon phase (days 130-180): run-wide extrema, long-term comparisons
    ProbeSpec("A14", 130, "What was your net worth on day 60?", _gt_a14, score_numeric),
    ProbeSpec("B12", 145, "Which supplier have you placed the most orders with so far, counted by the number of `Order confirmation` emails received from each supplier (NOT by price inquiries, replies, or rejection emails)? If multiple suppliers are tied, list every one of them.", _gt_b12, score_keyword),
    ProbeSpec("A15", 160, "Which day so far had the highest sales revenue, and what was that revenue? If multiple days are tied at the same peak revenue, list every one of them.", _gt_a15, score_numeric),
    ProbeSpec("B13", 175, "Comparing the first 30 days (days 1-30) and the most recent 30 days (days 146-175), has average daily revenue gone up, down, or stayed flat? Answer with one word: `up`, `down`, or `flat` (treat changes within ±5% of the early average as flat).", _gt_b13, score_keyword),
    ProbeSpec("A16", 180, "What was your total delivery spend across the first 90 days?", _gt_a16, score_numeric),
]


def get_probes_for_session(session_idx: int) -> list[ProbeSpec]:
    return [p for p in PROBES if p.session_idx == session_idx]
