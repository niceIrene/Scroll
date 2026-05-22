"""BEAM agents — single SCROLL strategy.

The BEAM env's agent applies :class:`Scroll.core.ScrollAgent` with a
LongMemEval-like schema (chat-turn auto-ingest + regex extractors) and
a probe-time ``execute_python`` tool-call loop. No cross-task
procedural memory (BEAM is single-pass; LME's L3 distillation doesn't
apply).
"""

from __future__ import annotations

from Scroll.benchmarks.beam.agents.agent import BeamAgent


def create_agent(policy: str, cfg, log, data):
    if policy in ("scroll",):
        return BeamAgent(cfg, log, data)
    raise ValueError(f"unknown beam policy {policy!r}")


__all__ = ["BeamAgent", "create_agent"]
