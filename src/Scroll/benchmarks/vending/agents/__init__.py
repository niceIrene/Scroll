"""Vending agents.

Two policies:

- ``scroll`` — :class:`VendingAgent`, the SCROLL strategy with the
  7-table memoryspace + auto-ingest + ``ms`` / ``log`` / ``rlm``
  REPL surface.
- ``baseline`` — :class:`BaselineVendingAgent`, the context-only
  counterpart: same CodeAct REPL and env action tools, no off-context
  memory; long-horizon recall falls back to chat history +
  sliding-window summary compaction.
"""

from __future__ import annotations

from Scroll.benchmarks.vending.agents.agent import VendingAgent
from Scroll.benchmarks.vending.agents.baseline_agent import BaselineVendingAgent


def create_agent(policy: str, cfg, log, data):
    if policy == "scroll":
        return VendingAgent(cfg, log, data)
    if policy == "baseline":
        return BaselineVendingAgent(cfg, log, data)
    raise ValueError(f"unknown vending policy {policy!r}")


__all__ = ["VendingAgent", "BaselineVendingAgent", "create_agent"]
