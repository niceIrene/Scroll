"""Ingestor protocol — the ``f: E → W`` derivation.

A SCROLL agent's memoryspace ``W`` is a derived view of the event log
``E``. The ingestor is the pure function that does the derivation,
applied incrementally: every time new entries land in ``E``, the
ingestor consumes them and writes the resulting rows into ``W``.

Contract:
  - Constructed with a :class:`Memoryspace`.
  - :meth:`consume` is called with a slice of ``E`` (the tail past the
    last watermark). It must write only into the bound memoryspace
    and never read from anything other than the entries it was given.
  - Idempotency is not required at the per-entry level — the
    memoryspace tracks a watermark and only feeds each entry once
    in normal operation. The offline ``rebuild-w`` path starts from
    an empty memoryspace, so it also feeds each entry once.

This decouples ingestion from the agent class: tests can replay
``conversation_log.jsonl`` through an ingestor without instantiating
the agent or its environment.
"""

from __future__ import annotations

from typing import Iterable, Protocol

from Scroll.core._models import LogEntry


class Ingestor(Protocol):
    """Pure E → W derivation, bound to a single :class:`Memoryspace`."""

    def consume(self, entries: Iterable[LogEntry]) -> None:
        """Apply the entries to the bound memoryspace.

        Called with the tail of ``E`` past the memoryspace's watermark.
        Implementations should filter by ``entry.metadata.get("kind")``
        and ignore entry kinds they don't recognize so that adding new
        kinds in the future doesn't break existing ingestors.
        """
        ...
