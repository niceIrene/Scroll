"""Cheap bag-of-words embedding + cosine helpers.

Shared by :class:`Scroll.tools._log_handle.LogHandle` (for
``semantic_search`` over E) and :class:`Scroll.tools.memoryspace.Memoryspace`
(for the vector index inside W).

Bag-of-words, not neural: cheap and dependency-free, but only
captures token overlap. Synonymy (``dog`` ↔ ``puppy``) is not
modeled. Treat it as a recall-rerank surface, not an oracle.
"""

from __future__ import annotations

import math
import re
from collections import Counter


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _embed(text: str) -> dict[str, float]:
    """Return a normalized bag-of-words vector for ``text``."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    if not tokens:
        return {}
    counts = Counter(tokens)
    norm = math.sqrt(sum(c * c for c in counts.values())) or 1.0
    return {tok: c / norm for tok, c in counts.items()}


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two normalized sparse vectors."""
    if not a or not b:
        return 0.0
    # iterate over the smaller dict
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())
