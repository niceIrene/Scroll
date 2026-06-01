"""Vending ``E → W`` ingestor.

Consumes four entry kinds from the conversation log (E) and
populates the vending memoryspace (W):

  - ``kind="briefing_note"`` — one per datasource note (weather, news)
  - ``kind="env_log"``       — one per env outcome line (sales, deliveries)
  - ``kind="inbox_mail"``    — one per email landed in the inbox
                              (metadata = source / subject / body / idx)
  - ``kind="env_snapshot"``  — one per end-of-session env state snapshot
                              (metadata = cash / machine_cash /
                              inventory_value / net_worth / skus)

W is the seven canonical tables (sales, deliveries, orders,
daily_finances, daily_inventory, supplier_prices, notes), the five
analytic views, plus the json/vector aggregates that the agent reads.

Unlike the LME ingestor (which reads chat_turn from E), the Vending
ingestor expects env state to be SERIALIZED into entry metadata at
emit time — :meth:`VendingAgent._emit_context_entries` /
:meth:`_emit_outcome_entries` are responsible for that serialization.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from Scroll.core._ingestor import Ingestor
from Scroll.core._models import LogEntry
from Scroll.tools.memoryspace import Memoryspace

from Scroll.benchmarks.vending.auto_ingest import (
    _short_supplier,
    ingest_context_notes,
    ingest_deliveries,
    ingest_finances,
    ingest_inventory,
    ingest_sales,
)


# ---------------------------------------------------------------------------
# Schema (relocated from agents/agent.py)
# ---------------------------------------------------------------------------

_VENDING_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS sales (
    day INTEGER NOT NULL, sku TEXT NOT NULL,
    units INTEGER NOT NULL, price REAL NOT NULL, revenue REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS deliveries (
    day INTEGER NOT NULL, shipment_id INTEGER, sku TEXT NOT NULL,
    units INTEGER NOT NULL, cost REAL NOT NULL, supplier TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    day INTEGER NOT NULL, supplier TEXT NOT NULL,
    sku TEXT NOT NULL, qty INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_finances (
    day INTEGER PRIMARY KEY, cash REAL NOT NULL,
    machine_cash REAL NOT NULL, inventory_value REAL NOT NULL, net_worth REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_inventory (
    day INTEGER NOT NULL, sku TEXT NOT NULL,
    machine_qty INTEGER NOT NULL DEFAULT 0, storage_qty INTEGER NOT NULL DEFAULT 0,
    price REAL, PRIMARY KEY (day, sku)
);

CREATE TABLE IF NOT EXISTS supplier_prices (
    supplier TEXT NOT NULL, sku TEXT NOT NULL,
    price REAL NOT NULL, PRIMARY KEY (supplier, sku)
);

CREATE TABLE IF NOT EXISTS notes (
    key TEXT PRIMARY KEY, value TEXT NOT NULL, day INTEGER NOT NULL
);

CREATE VIEW IF NOT EXISTS weekly_revenue_by_sku AS
SELECT (day - 1) / 7 AS week, sku,
       SUM(units) AS units, SUM(revenue) AS revenue
FROM sales
GROUP BY week, sku;

CREATE VIEW IF NOT EXISTS rolling_7d_units_by_sku AS
SELECT day, sku,
       SUM(units) OVER (
           PARTITION BY sku ORDER BY day
           RANGE BETWEEN 6 PRECEDING AND CURRENT ROW
       ) AS units_7d
FROM (SELECT day, sku, SUM(units) AS units FROM sales GROUP BY day, sku);

CREATE VIEW IF NOT EXISTS cogs_by_supplier AS
SELECT supplier, sku, SUM(units) AS units, SUM(cost) AS cost
FROM deliveries
WHERE supplier IS NOT NULL
GROUP BY supplier, sku;

CREATE VIEW IF NOT EXISTS inventory_turnover AS
SELECT i.day, i.sku,
       (i.machine_qty + i.storage_qty) AS total_qty,
       COALESCE(s.units_sold, 0) AS units_sold,
       CASE WHEN (i.machine_qty + i.storage_qty) > 0
            THEN ROUND(CAST(COALESCE(s.units_sold, 0) AS REAL) / (i.machine_qty + i.storage_qty), 4)
            ELSE NULL END AS turnover_ratio
FROM daily_inventory i
LEFT JOIN (SELECT day, sku, SUM(units) AS units_sold FROM sales GROUP BY day, sku) s
    ON s.day = i.day AND s.sku = i.sku;

CREATE VIEW IF NOT EXISTS daily_pnl AS
SELECT d.day,
       COALESCE(s.revenue, 0) AS revenue,
       COALESCE(c.cogs, 0) AS cogs,
       COALESCE(s.revenue, 0) - COALESCE(c.cogs, 0) AS gross_profit
FROM daily_finances d
LEFT JOIN (SELECT day, SUM(revenue) AS revenue FROM sales GROUP BY day) s
    ON s.day = d.day
LEFT JOIN (SELECT day, SUM(cost) AS cogs FROM deliveries GROUP BY day) c
    ON c.day = d.day;
"""


def ensure_schema(ms: Memoryspace) -> None:
    ms.sqlite.executescript(_VENDING_SCHEMA_SQL)
    ms.sqlite.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PRICE_LIST_RE = re.compile(r"(\w+):\s*\$([0-9.]+)/unit")


def _ingest_supplier_prices_from_mail(
    conn, source: str | None, subject: str, body: str,
) -> None:
    if "Price List" not in str(subject):
        return
    supplier = _short_supplier(source)
    for match in _PRICE_LIST_RE.finditer(body):
        sku = match.group(1)
        price = float(match.group(2))
        conn.execute(
            "INSERT OR REPLACE INTO supplier_prices (supplier, sku, price) "
            "VALUES (?, ?, ?)",
            (supplier, sku, price),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# VendingIngestor
# ---------------------------------------------------------------------------


class VendingIngestor(Ingestor):
    """E → W for the vending env.

    Buffers entries per (kind, turn_idx) so the existing
    boundary-shaped helpers in :mod:`auto_ingest` keep working
    unchanged (they're written to take ``day`` + ``list[str]``).
    """

    def __init__(self, ms: Memoryspace) -> None:
        self.ms = ms

    def consume(self, entries: Iterable[LogEntry]) -> None:
        # Group by (turn_idx, kind) so we can re-use the
        # existing day-shaped helpers (ingest_sales, ingest_deliveries,
        # ingest_context_notes) which expect a list per day.
        ctx_notes: dict[int, list[str]] = {}
        env_logs: dict[int, list[str]] = {}
        snapshots: dict[int, LogEntry] = {}
        mails: list[LogEntry] = []

        for entry in entries:
            meta = entry.metadata or {}
            kind = meta.get("kind")
            day = entry.turn_idx
            if kind == "briefing_note":
                ctx_notes.setdefault(day, []).append(str(entry.content))
            elif kind == "env_log":
                env_logs.setdefault(day, []).append(str(entry.content))
            elif kind == "env_snapshot":
                # Last snapshot per day wins (deterministic — we only
                # emit one per day at receive_outcomes time).
                snapshots[day] = entry
            elif kind == "inbox_mail":
                mails.append(entry)
            # Silently ignore other kinds (code_exec, rlm_*, etc.).

        conn = self.ms.sqlite

        # --- briefing notes ---
        for day, notes in ctx_notes.items():
            ingest_context_notes(conn, day, notes)
            self.ms.json_store[f"context_day_{day}"] = list(notes)
            for i, note in enumerate(notes):
                self.ms.vectors.add(f"note_day_{day}_{i}", note)

        # --- env outcome logs ---
        for day, logs in env_logs.items():
            ingest_sales(conn, day, logs)
            ingest_deliveries(conn, day, logs)
            self.ms.json_store[f"outcomes_day_{day}"] = list(logs)
            for i, line in enumerate(logs):
                self.ms.vectors.add(f"outcome_day_{day}_{i}", line)

        # --- inbox mail (supplier price lists + vector embed) ---
        for mail_entry in mails:
            md = mail_entry.metadata or {}
            source = md.get("source")
            subject = str(md.get("subject", ""))
            body = str(md.get("body", ""))
            idx = int(md.get("idx", 0))
            day = mail_entry.turn_idx
            _ingest_supplier_prices_from_mail(conn, source, subject, body)
            self.ms.vectors.add(
                f"email_day_{day}_idx_{idx}",
                f"from={source} subject={subject}\n{body}",
            )

        # --- env snapshots (finances + per-SKU inventory) ---
        for day, snap in snapshots.items():
            self._apply_snapshot(conn, day, snap.metadata or {})

        conn.commit()

    def _apply_snapshot(self, conn, day: int, meta: dict) -> None:
        # Adapter: snapshot metadata reproduces the shape expected by
        # ingest_finances / ingest_inventory (which originally took an
        # ``env`` object). We construct a duck-typed proxy here so the
        # boundary helpers can stay unchanged.
        skus_raw = meta.get("skus") or {}
        if isinstance(skus_raw, str):
            try:
                skus_raw = json.loads(skus_raw)
            except (TypeError, ValueError):
                skus_raw = {}

        class _SnapEnv:
            cash = float(meta.get("cash", 0.0))
            machine_cash = float(meta.get("machine_cash", 0.0))
            _inv_value = float(meta.get("inventory_value", 0.0))
            _net_worth = float(meta.get("net_worth", 0.0))
            catalog = list(skus_raw.keys())
            machine = {k: int(v.get("machine_qty", 0)) for k, v in skus_raw.items()}
            storage = {k: int(v.get("storage_qty", 0)) for k, v in skus_raw.items()}
            prices = {k: v.get("price") for k, v in skus_raw.items()}

            def inventory_value(self):
                return self._inv_value

            def net_worth(self):
                return self._net_worth

        ingest_finances(conn, day, _SnapEnv())
        ingest_inventory(conn, day, _SnapEnv())


# ---------------------------------------------------------------------------
# Helpers for the agent: serialize env state into LogEntry metadata
# ---------------------------------------------------------------------------


def serialize_env_snapshot(env) -> dict:
    """Capture the env's bankable state as a JSON-able dict.

    Used by :meth:`VendingAgent._emit_outcome_entries` to attach to a
    ``kind="env_snapshot"`` log entry. Anything used by the boundary
    helpers (cash, machine_cash, inventory_value, net_worth, per-SKU
    machine/storage/price) must be included here.
    """
    skus: dict[str, dict] = {}
    for sku in env.catalog:
        skus[sku] = {
            "machine_qty": int(env.machine.get(sku, 0)),
            "storage_qty": int(env.storage.get(sku, 0)),
            "price": env.prices.get(sku),
        }
    return {
        "cash": float(env.cash),
        "machine_cash": float(env.machine_cash),
        "inventory_value": float(env.inventory_value()),
        "net_worth": float(env.net_worth()),
        "skus": skus,
    }


def serialize_inbox_mail(mail, idx: int) -> dict:
    """Capture an inbox mail object as a JSON-able dict for log metadata."""
    return {
        "source": getattr(mail, "source", None),
        "subject": getattr(mail, "subject", ""),
        "body": getattr(mail, "body", ""),
        "idx": idx,
    }


__all__ = [
    "VendingIngestor",
    "ensure_schema",
    "serialize_env_snapshot",
    "serialize_inbox_mail",
]
