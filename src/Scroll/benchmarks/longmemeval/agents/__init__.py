"""LongMemEval agents — single SCROLL strategy.

The LongMemEval env's agent applies the :class:`Scroll.core.ScrollAgent`
harness with LongMemEval-specific schema, regex extractors, and probe-time
hint composition.
"""

from __future__ import annotations

from Scroll.benchmarks.longmemeval.agents.agent import LongMemEvalAgent


def create_agent(policy: str, cfg, log, data):
    if policy in ("scroll", "code_auto_v2"):
        return LongMemEvalAgent(cfg, log, data)
    raise ValueError(f"unknown longmemeval policy {policy!r}")


__all__ = ["LongMemEvalAgent", "create_agent"]
