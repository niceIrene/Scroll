"""Vending-specific data models: Product and EnvConfig."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    wholesale_cost: float
    base_daily_demand: float
    reference_price: float
    elasticity: float


@dataclass
class EnvConfig:
    num_sessions: int                 # number of simulated business days
    start_cash: float
    daily_fee: float
    machine_slots: int
    max_skus_in_machine: int
    lead_time_days: int               # vending domain: real calendar lead time

    @classmethod
    def from_dict(cls, d: dict) -> EnvConfig:
        """Construct from a raw config dict, using dataclass defaults for missing keys.

        Accepts the legacy ``"days"`` key as a synonym for ``"num_sessions"``
        so older configs continue to parse.
        """
        known = {f.name for f in fields(cls)}
        normalized = dict(d)
        if "num_sessions" not in normalized and "days" in normalized:
            normalized["num_sessions"] = normalized["days"]
        return cls(**{k: v for k, v in normalized.items() if k in known})
