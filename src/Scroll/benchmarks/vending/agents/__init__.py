"""Vending agents — single SCROLL strategy.

The vending env's agent applies the :class:`Scroll.core.ScrollAgent`
harness with vending-specific schema (7 tables + 5 analytic views),
ingest pipeline, and action tools.
"""

from __future__ import annotations

from Scroll.benchmarks.vending.agents._common import VENDING_CONTEXT, vending_day_prompt
from Scroll.benchmarks.vending.agents.agent import VendingAgent


def create_agent(policy: str, cfg, log, data):
    if policy in ("scroll", "code_auto"):
        return VendingAgent(cfg, log, data)
    raise ValueError(f"unknown vending policy {policy!r}")


__all__ = ["VendingAgent", "VENDING_CONTEXT", "vending_day_prompt", "create_agent"]
