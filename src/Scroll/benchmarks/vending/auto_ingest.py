"""Vending-environment auto-ingestion helpers — `D = ingest(E)`.

These functions read vending-specific env state (``env.cash``,
``env.machine``, ``env.storage``, ``env.catalog``, ``env.prices``)
and parse vending-specific log strings (``sale sku=...``,
``delivery_arrived ...``, supplier price-list email replies, etc.)
into the seven canonical vending DB tables:

    sales, deliveries, orders, daily_finances, daily_inventory,
    supplier_prices, notes

They are NOT called by the LLM. Two call sites populate ``D``:

  - **Boundary ingest** — :func:`auto_ingest_context` (morning) and
    :func:`auto_ingest_outcomes` (overnight) fire from
    ``DatabaseAgent.receive_*`` hooks for the natural-event channels
    (sales, deliveries, supplier prices, daily snapshots).

  - **Action ingest** — :func:`ingest_orders`, :func:`ingest_finances`,
    :func:`ingest_inventory` fire as tap-around hooks inside
    :func:`vending.agents.agent._make_env_namespace`,
    so D reflects agent-driven state changes (sending an order email,
    polling money balance, asking sub-agent to check inventory) the
    same turn they happen.

Lives under ``vending/`` because every assumption here — log format,
env attribute names, table names — is vending-specific. The
env-independent SQLite plumbing (``SQLiteFacade``, ``init_schema``)
stays in ``tools/sql_tools.py``.
"""

from __future__ import annotations

import ast
import re
import sqlite3
from typing import Any


# ---------------------------------------------------------------------------
# Per-table ingest
# ---------------------------------------------------------------------------

def ingest_sales(conn: sqlite3.Connection, day: int, logs: list[str]) -> None:
    """Parse ``sale sku=... units=...`` lines and INSERT into ``sales``."""
    for line in logs:
        if not line.startswith("sale "):
            continue
        parsed = _parse_sale(line)
        if parsed is None:
            continue
        conn.execute(
            "INSERT INTO sales (day, sku, units, price, revenue) VALUES (?, ?, ?, ?, ?)",
            (day, parsed["sku"], parsed["units"], parsed.get("price", 0.0), parsed.get("revenue", 0.0)),
        )
    conn.commit()


def ingest_deliveries(conn: sqlite3.Connection, day: int, logs: list[str]) -> None:
    """Parse ``delivery_arrived items={...} cost=... supplier=...`` lines into ``deliveries``.

    Each ``delivery_arrived`` env-log line is one shipment that may
    carry several SKUs. Every row from the same event shares a fresh
    ``shipment_id`` so ``COUNT(DISTINCT shipment_id)`` recovers the
    number of shipments without needing to assume one shipment per
    day. We assign ids from ``MAX(shipment_id) + 1`` to keep them
    monotonic across days and ingest passes.
    """
    cur = conn.execute("SELECT COALESCE(MAX(shipment_id), 0) FROM deliveries")
    next_id = int(cur.fetchone()[0] or 0) + 1
    for line in logs:
        if "delivery_arrived" not in line:
            continue
        rows = _parse_delivery(line)
        if rows is None:
            continue
        sid = next_id
        next_id += 1
        for row in rows:
            conn.execute(
                "INSERT INTO deliveries "
                "(day, shipment_id, sku, units, cost, supplier) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (day, sid, row["sku"], row["units"], row["cost"], row.get("supplier")),
            )
    conn.commit()


def ingest_finances(conn: sqlite3.Connection, day: int, env) -> None:
    """Snapshot ``env.cash / machine_cash / inventory_value() / net_worth()`` into ``daily_finances``."""
    conn.execute(
        "INSERT OR REPLACE INTO daily_finances (day, cash, machine_cash, inventory_value, net_worth) "
        "VALUES (?, ?, ?, ?, ?)",
        (day, round(env.cash, 2), round(env.machine_cash, 2),
         round(env.inventory_value(), 2), round(env.net_worth(), 2)),
    )
    conn.commit()


def ingest_inventory(conn: sqlite3.Connection, day: int, env) -> None:
    """Snapshot ``env.machine[sku] / env.storage[sku] / env.prices[sku]`` for every SKU."""
    for sku in env.catalog:
        machine_qty = env.machine.get(sku, 0)
        storage_qty = env.storage.get(sku, 0)
        price = env.prices.get(sku)
        conn.execute(
            "INSERT OR REPLACE INTO daily_inventory (day, sku, machine_qty, storage_qty, price) "
            "VALUES (?, ?, ?, ?, ?)",
            (day, sku, machine_qty, storage_qty, price),
        )
    conn.commit()


def _short_supplier(addr: str | None) -> str | None:
    """Strip the ``@example.com`` domain for table-facing supplier ids.

    The simulation routes orders by full email address (``send_email``
    needs that), but every other consumer — probe ground truth, LM
    queries, dashboards — refers to suppliers by the short name
    (``metro_wholesale``). Storing the short form in tables keeps the
    SQL store consistent with how the world labels them and lets the
    agent's natural ``WHERE supplier = 'metro_wholesale'`` queries
    work without having to know the email suffix.
    """
    if not addr:
        return addr
    return addr.split("@", 1)[0]


def ingest_orders(conn: sqlite3.Connection, day: int, to: str, subject: str, body: str) -> None:
    """Parse outgoing order email (``cola=20 water=15`` style) and INSERT into ``orders``."""
    subject_l = subject.lower()
    body_l = body.lower()
    is_order = any(
        kw in subject_l or kw in body_l
        for kw in ("order", "purchase", "buy")
    )
    if not is_order:
        return
    supplier = _short_supplier(to)
    items = _parse_order_items(body)
    for sku, qty in items.items():
        conn.execute(
            "INSERT INTO orders (day, supplier, sku, qty) VALUES (?, ?, ?, ?)",
            (day, supplier, sku, qty),
        )
    if items:
        conn.commit()


def ingest_supplier_prices(conn: sqlite3.Connection, inbox) -> None:
    """Scan inbox for ``Re: Price List`` replies and INSERT OR REPLACE supplier prices."""
    for mail in inbox:
        if "Price List" not in getattr(mail, "subject", ""):
            continue
        supplier = _short_supplier(mail.source)
        for match in re.finditer(r"(\w+):\s*\$([0-9.]+)/unit", mail.body):
            sku = match.group(1)
            price = float(match.group(2))
            conn.execute(
                "INSERT OR REPLACE INTO supplier_prices (supplier, sku, price) "
                "VALUES (?, ?, ?)",
                (supplier, sku, price),
            )
    conn.commit()


def ingest_context_notes(conn: sqlite3.Connection, day: int, notes: list[str]) -> None:
    """Stamp the day's free-form datasource notes (weather / news) into ``notes``."""
    if not notes:
        return
    value = "; ".join(notes)
    conn.execute(
        "INSERT OR REPLACE INTO notes (key, value, day) VALUES (?, ?, ?)",
        (f"context_day_{day}", value, day),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Boundary composers — call from DatabaseAgent.receive_*
# ---------------------------------------------------------------------------

def auto_ingest_context(
    conn: sqlite3.Connection,
    day: int,
    notes: list[str],
    *,
    inbox=None,
) -> None:
    """Refresh ``D`` from the morning-of-day inputs (``E``'s context channel).

    Pulls supplier prices from the inbox and stamps datasource notes
    into the ``notes`` table.
    """
    if inbox is not None:
        ingest_supplier_prices(conn, inbox)
    ingest_context_notes(conn, day, notes)


def auto_ingest_outcomes(
    conn: sqlite3.Connection,
    day: int,
    logs: list[str],
    *,
    env=None,
) -> None:
    """Refresh ``D`` from the end-of-day outcome logs (``E``'s env channel).

    Parses sales / deliveries from the day's env logs and snapshots
    financial & inventory state when ``env`` is available.
    """
    ingest_sales(conn, day, logs)
    ingest_deliveries(conn, day, logs)
    if env is not None:
        ingest_finances(conn, day, env)
        ingest_inventory(conn, day, env)


# ---------------------------------------------------------------------------
# Parsing helpers (vending log + email body formats)
# ---------------------------------------------------------------------------

def _parse_sale(text: str) -> dict[str, Any] | None:
    m_sku = re.search(r"sku=(\w+)", text)
    m_units = re.search(r"units=(\d+)", text)
    m_price = re.search(r"price=([\d.]+)", text)
    m_rev = re.search(r"rev=([\d.]+)", text)
    if not (m_sku and m_units):
        return None
    result: dict[str, Any] = {"sku": m_sku.group(1), "units": int(m_units.group(1))}
    if m_price:
        result["price"] = float(m_price.group(1))
    if m_rev:
        result["revenue"] = float(m_rev.group(1))
    return result


def _parse_delivery(text: str) -> list[dict[str, Any]] | None:
    m_items = re.search(r"items=(\{[^}]+\})", text)
    if not m_items:
        return None
    try:
        items = ast.literal_eval(m_items.group(1))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(items, dict) or not items:
        return None

    m_cost = re.search(r"(?:booked_cost|cost)=([\d.]+)", text)
    total_cost = float(m_cost.group(1)) if m_cost else 0.0
    total_units = sum(items.values()) or 1

    m_supplier = re.search(r"supplier=(\S+)", text)
    supplier = m_supplier.group(1) if m_supplier else None

    rows: list[dict[str, Any]] = []
    for sku, units in items.items():
        row: dict[str, Any] = {
            "sku": str(sku),
            "units": int(units),
            "cost": round(total_cost * int(units) / total_units, 2),
        }
        if supplier:
            row["supplier"] = supplier
        rows.append(row)
    return rows


def _parse_order_items(body: str) -> dict[str, int]:
    """Parse ``sku=qty`` pairs from an order email body (mirrors DataSourceManager)."""
    items: dict[str, int] = {}
    for part in body.replace(",", " ").split():
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            q = int(v)
        except ValueError:
            continue
        if q > 0:
            items[k.strip().lower()] = q
    return items


__all__ = [
    "auto_ingest_context",
    "auto_ingest_outcomes",
    "ingest_sales",
    "ingest_deliveries",
    "ingest_finances",
    "ingest_inventory",
    "ingest_orders",
    "ingest_supplier_prices",
    "ingest_context_notes",
]
