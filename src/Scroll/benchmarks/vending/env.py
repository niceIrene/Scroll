from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from Scroll.core import (
    BaseEnvironment,
    EnvSnapshot,
    SessionResult,
    serialize_rng,
    restore_rng,
)
from Scroll.benchmarks.vending.catalog import EnvConfig, Product


@dataclass
class VendingSnapshot(EnvSnapshot):
    """Vending-machine snapshot fields consumed by vending probes."""

    net_worth: float = 0.0
    cash: float = 0.0
    machine_cash: float = 0.0
    revenue: float = 0.0
    units_sold: int = 0
    inventory: dict[str, dict] = field(default_factory=dict)


_VENDING_SUBSTRATE_ENDGAME = """\
RUN STRUCTURE — VENDING:
- Each iteration is one business day. The simulation runs for many
  consecutive days; cash, inventory, and pending deliveries persist
  across days, so what you do today directly affects tomorrow.
- ``today`` (1-indexed int) is bound in your REPL as the current day
  number. Use it to label or slice your data, e.g. ``today - 1`` for
  yesterday or ``log.range_by_day(today - 7, today - 1)`` for the
  past week.
- When you have finished all the actions you want to take today
  (restocking, ordering, pricing, replying to suppliers, etc.), call
  ``wait_for_next_day()`` to advance to the next morning. Code after
  that call in the same cell will not run.
"""


def default_catalog() -> list[Product]:
    return [
        Product("cola", "Cola", 1.00, 7.0, 2.50, 1.3),
        Product("water", "Water", 0.55, 8.0, 1.75, 0.8),
        Product("chips", "Chips", 0.90, 6.0, 2.25, 1.1),
        Product("choco", "Chocolate Bar", 0.70, 5.0, 1.95, 1.0),
        Product("energy", "Energy Drink", 1.20, 4.0, 2.95, 1.5),
        Product("gum", "Gum", 0.35, 4.0, 1.25, 0.6),
        Product("juice", "Juice", 0.85, 5.0, 2.15, 1.0),
        Product("nuts", "Nuts", 1.10, 3.5, 2.75, 1.2),
    ]


class VendingEnv(BaseEnvironment):
    def __init__(self, cfg: EnvConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.session_idx = 0
        self.cash = cfg.start_cash
        self.machine_cash = 0.0
        self.catalog = {p.sku: p for p in default_catalog()}
        self.storage = defaultdict(int)
        self.machine = defaultdict(int)
        self.prices = {sku: p.reference_price for sku, p in self.catalog.items()}
        self.pending_deliveries: list[tuple[int, dict[str, int], float]] = []
        self.total_units_sold = 0
        self.bankrupt_days = 0
        self._last_revenue = 0.0
        self._last_sold_units = 0

    def visible_state(self) -> dict:
        return {"session_idx": self.session_idx,
            "cash": round(self.cash, 2),
            "machine_cash": round(self.machine_cash, 2),
            "storage": dict(self.storage),
            "machine": dict(self.machine),
            "prices": {k: round(v, 2) for k, v in self.prices.items()},
            "pending_deliveries": [
                {"arrive_day": d, "items": items, "cost": round(cost, 2)}
                for d, items, cost in self.pending_deliveries
            ],
        }

    def set_price(self, sku: str, price: float) -> str:
        if sku not in self.catalog:
            return f"unknown sku {sku}"
        self.prices[sku] = max(0.5, float(price))
        return f"price_set {sku}={self.prices[sku]:.2f}"

    def order(self, items: dict[str, int]) -> str:
        cleaned = {k: int(v) for k, v in items.items() if k in self.catalog and v > 0}
        if not cleaned:
            return "order_rejected empty"
        total_cost = sum(self.catalog[sku].wholesale_cost * qty for sku, qty in cleaned.items())
        if total_cost > self.cash:
            return f"order_rejected insufficient_cash need={total_cost:.2f} have={self.cash:.2f}"
        self.cash -= total_cost
        arrive = self.session_idx + self.cfg.lead_time_days
        self.pending_deliveries.append((arrive, cleaned, total_cost))
        return f"order_placed arrive_day={arrive} cost={total_cost:.2f}"

    def restock(self, units_per_sku: int) -> str:
        active_skus = [sku for sku, qty in self.machine.items() if qty > 0]
        if len(active_skus) < self.cfg.max_skus_in_machine:
            for sku in self.catalog:
                if sku not in active_skus:
                    active_skus.append(sku)
                if len(active_skus) >= self.cfg.max_skus_in_machine:
                    break

        remaining_slots = self.cfg.machine_slots - sum(self.machine.values())
        if remaining_slots <= 0:
            return "restock_skipped machine_full"

        moved_total = 0
        for sku in active_skus:
            if remaining_slots <= 0:
                break
            need = min(units_per_sku, remaining_slots)
            take = min(need, self.storage[sku])
            if take > 0:
                self.storage[sku] -= take
                self.machine[sku] += take
                moved_total += take
                remaining_slots -= take

        return f"restock_done moved_units={moved_total}"

    def restock_sku(self, sku: str, qty: int) -> str:
        """Restock a specific SKU from storage into the machine."""
        if sku not in self.catalog:
            return f"restock_failed unknown_sku={sku}"
        remaining_slots = self.cfg.machine_slots - sum(self.machine.values())
        if remaining_slots <= 0:
            return "restock_skipped machine_full"
        active_skus = {s for s, q in self.machine.items() if q > 0}
        if sku not in active_skus and len(active_skus) >= self.cfg.max_skus_in_machine:
            return f"restock_failed max_skus_reached ({self.cfg.max_skus_in_machine})"
        need = min(qty, remaining_slots)
        take = min(need, self.storage[sku])
        if take <= 0:
            return f"restock_failed sku={sku} nothing_in_storage"
        self.storage[sku] -= take
        self.machine[sku] += take
        return f"restock_done sku={sku} moved={take}"

    def collect_cash(self) -> str:
        val = self.machine_cash
        self.cash += val
        self.machine_cash = 0.0
        return f"cash_collected amount={val:.2f}"

    def _deliver_orders(self) -> list[str]:
        delivered_logs = []
        next_pending = []
        for arrive_day, items, cost in self.pending_deliveries:
            if arrive_day <= self.session_idx:
                for sku, qty in items.items():
                    self.storage[sku] += qty
                delivered_logs.append(f"delivery_arrived items={items} booked_cost={cost:.2f}")
            else:
                next_pending.append((arrive_day, items, cost))
        self.pending_deliveries = next_pending
        return delivered_logs

    def _simulate_sales(self) -> tuple[int, float, list[str]]:
        sold = 0
        revenue = 0.0
        logs = []

        weekday_mult = 1.15 if self.session_idx % 7 in (5, 6) else 1.0
        month_mult = 1.1 if (self.session_idx // 30) in (5, 6, 7) else 0.95

        active_sku_count = sum(1 for qty in self.machine.values() if qty > 0)
        if active_sku_count == 0:
            choice_mult = 0.0
        elif active_sku_count <= 4:
            choice_mult = 1.1
        elif active_sku_count <= 7:
            choice_mult = 1.0
        else:
            choice_mult = 0.8

        for sku, qty in list(self.machine.items()):
            if qty <= 0:
                continue
            p = self.catalog[sku]
            price = self.prices[sku]
            delta = (price - p.reference_price) / max(0.01, p.reference_price)
            demand_mult = max(0.1, 1.0 - p.elasticity * delta)
            noise = self.rng.uniform(0.85, 1.15)
            predicted = p.base_daily_demand * demand_mult * weekday_mult * month_mult * choice_mult * noise
            units = max(0, min(qty, int(round(predicted))))
            if units > 0:
                self.machine[sku] -= units
                sold += units
                rev = units * price
                revenue += rev
                logs.append(f"sale sku={sku} units={units} price={price:.2f} rev={rev:.2f}")

        # Stockout = SKU that has been stocked at some point but is now fully
        # depleted (zero in both machine and storage). Emitted once per day per
        # depleted SKU after the sales loop.
        for sku in list(self.machine.keys()):
            if self.machine[sku] == 0 and self.storage.get(sku, 0) == 0:
                logs.append(f"stockout sku={sku}")

        self.machine_cash += revenue
        self.total_units_sold += sold
        return sold, revenue, logs

    def step_session(self) -> SessionResult:
        self.session_idx += 1
        delivery_logs = self._deliver_orders()
        sold_units, revenue, sales_logs = self._simulate_sales()
        self._last_revenue = revenue
        self._last_sold_units = sold_units

        self.cash -= self.cfg.daily_fee
        if self.cash < 0:
            self.bankrupt_days += 1
        else:
            self.bankrupt_days = 0

        self._today_logs = delivery_logs + sales_logs + [f"fee_charged {self.cfg.daily_fee:.2f}"]

        return SessionResult(
            session_idx=self.session_idx,
            sold_units=sold_units,
            revenue=revenue,
            machine_cash=self.machine_cash,
            cash=self.cash,
        )

    def today_logs(self) -> list[str]:
        return list(getattr(self, "_today_logs", []))

    def inventory_value(self) -> float:
        storage_val = sum(self.catalog[sku].wholesale_cost * qty for sku, qty in self.storage.items())
        machine_val = sum(self.catalog[sku].wholesale_cost * qty for sku, qty in self.machine.items())
        return storage_val + machine_val

    def net_worth(self) -> float:
        pending_val = sum(cost for _, _, cost in self.pending_deliveries)
        return self.cash + self.machine_cash + self.inventory_value() + pending_val

    def is_terminal(self) -> bool:
        """Bankrupt after ten consecutive days unable to cover the daily fee."""
        return self.bankrupt_days >= 10

    def report_run_metrics(self, agent) -> dict:
        """Vending's headline outcome signals: net_worth, units_sold, bankrupt."""
        return {
            "net_worth": round(self.net_worth(), 2),
            "units_sold": self.total_units_sold,
            "bankrupt": self.is_terminal(),
        }

    def summarize_probes(self, results: list) -> dict:
        """Vending-specific A/B category split.

        Vending probes use ``A*`` (factual recall) and ``B*`` (multi-step
        reasoning) prefixes on ``question_id``; we surface per-category
        averages in ``probe_results.json`` and the run span. Cores envs
        without this convention inherit the base ``{}`` default.
        """
        out: dict = {}
        a_scores = [r.score for r in results if r.question_id.startswith("A")]
        b_scores = [r.score for r in results if r.question_id.startswith("B")]
        if a_scores:
            out["category_a_avg"] = round(sum(a_scores) / len(a_scores), 3)
        if b_scores:
            out["category_b_avg"] = round(sum(b_scores) / len(b_scores), 3)
        return out

    def substrate_endgame_prompt(self) -> str:
        return _VENDING_SUBSTRATE_ENDGAME

    def probe_substrate_prompt(self) -> str:
        from Scroll.benchmarks.vending.tasks.rewards import VENDING_PROBE_FORMAT
        return VENDING_PROBE_FORMAT

    def probe_user_postscript(self) -> str:
        from Scroll.benchmarks.vending.tasks.rewards import VENDING_PROBE_USER_POSTSCRIPT
        return VENDING_PROBE_USER_POSTSCRIPT

    def to_checkpoint(self) -> dict:
        return {"session_idx": self.session_idx,
            "cash": self.cash,
            "machine_cash": self.machine_cash,
            "storage": dict(self.storage),
            "machine": dict(self.machine),
            "prices": dict(self.prices),
            "pending_deliveries": [
                {"arrive_day": d, "items": items, "cost": cost}
                for d, items, cost in self.pending_deliveries
            ],
            "total_units_sold": self.total_units_sold,
            "bankrupt_days": self.bankrupt_days,
            "rng_state": serialize_rng(self.rng),
            "_last_revenue": self._last_revenue,
            "_last_sold_units": self._last_sold_units,
        }

    @classmethod
    def from_checkpoint(cls, data: dict, cfg: EnvConfig) -> "VendingEnv":
        env = cls(cfg, seed=0)  # seed doesn't matter, RNG is restored
        env.session_idx = data["session_idx"]
        env.cash = data["cash"]
        env.machine_cash = data["machine_cash"]
        env.storage = defaultdict(int, data["storage"])
        env.machine = defaultdict(int, data["machine"])
        env.prices = data["prices"]
        env.pending_deliveries = [
            (d["arrive_day"], d["items"], d["cost"])
            for d in data["pending_deliveries"]
        ]
        env.total_units_sold = data["total_units_sold"]
        env.bankrupt_days = data["bankrupt_days"]
        restore_rng(env.rng, data["rng_state"])
        env._last_revenue = data.get("_last_revenue", 0.0)
        env._last_sold_units = data.get("_last_sold_units", 0)
        return env

    def build_snapshot(self) -> EnvSnapshot:
        return VendingSnapshot(
            session_idx=self.session_idx,
            logs=list(self.today_logs()),
            net_worth=round(self.net_worth(), 2),
            cash=round(self.cash, 2),
            machine_cash=round(self.machine_cash, 2),
            revenue=round(self._last_revenue, 2),
            units_sold=self._last_sold_units,
            inventory={
                sku: {
                    "machine": self.machine.get(sku, 0),
                    "storage": self.storage.get(sku, 0),
                    "price": round(self.prices.get(sku, 0), 2),
                }
                for sku in self.catalog
            },
        )
