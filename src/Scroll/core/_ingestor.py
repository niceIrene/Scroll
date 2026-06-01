"""Ingestor protocol — owns the env → E → W pipeline.

A SCROLL agent's memoryspace ``W`` is a derived view of the event log
``E``. The ingestor owns the data-flow into both:

  - :meth:`bootstrap` (once per task, optional) — read env's task-wide
    data (LME haystack chat sessions, BEAM batches) and append them
    into ``E`` as LogEntries. Vending doesn't need this — its data
    arrives turn-by-turn via ``receive_context`` / ``receive_outcomes``,
    so it inherits the default no-op.
  - :meth:`consume` (incremental, per ``ms`` access via the
    memoryspace watermark) — pure ``f: E → W``: read a slice of E
    (the tail past the last watermark) and write the resulting rows
    into ``W``.

Contract:
  - Constructed with a :class:`Memoryspace`.
  - :meth:`bootstrap` may read from ``env`` (its data accessors, e.g.
    ``env.item.haystack_sessions``) and write to ``log``. It is
    called once per task, before any agent session starts. Default:
    no-op. Subclasses for envs with task-wide pre-loaded data
    override it; ``write_chat_turn_entries`` from
    :mod:`Scroll.tools.chat_memory` is the canonical helper.
  - :meth:`consume` is called with a slice of ``E`` past the
    memoryspace's watermark. It must write only into the bound
    memoryspace and never read from anything other than the entries
    it was given. Idempotency is not required at the per-entry level
    — the memoryspace tracks a watermark and only feeds each entry
    once in normal operation. The offline ``rebuild-w`` path starts
    from an empty memoryspace, so it also feeds each entry once.

This decouples ingestion from the agent class: tests can replay
``conversation_log.jsonl`` through an ingestor without instantiating
the agent or its environment.
"""

from __future__ import annotations

from typing import Iterable, Protocol

from Scroll.core._models import LogEntry


class Ingestor(Protocol):
    """Owns env → E → W for a single :class:`Memoryspace`."""

    def consume(self, entries: Iterable[LogEntry]) -> None:
        """Apply the entries to the bound memoryspace.

        Called with the tail of ``E`` past the memoryspace's watermark.
        Implementations should filter by ``entry.metadata.get("kind")``
        and ignore entry kinds they don't recognize so that adding new
        kinds in the future doesn't break existing ingestors.
        """
        ...

    def bootstrap(self, env, log) -> None:
        """Bulk-load env's task-wide data into ``E``.

        Called once per task, before any agent session starts. Default
        no-op. Envs whose data is given upfront (LME haystack, BEAM
        batches) override this; vending uses turn-by-turn ingestion
        via the agent's ``receive_context`` / ``receive_outcomes``
        hooks and inherits the default.

        ``env`` is the :class:`BaseEnvironment` instance for this
        task; the ingestor reads from its data accessors
        (``env.item``, ``env.begin_turn(idx)``, ``env.current_session``,
        ...). ``log`` is the agent's :class:`ConversationLog` — the
        sink for new ``LogEntry`` instances.

        ``W`` is NOT built here — the appended entries are picked up
        on the next ``ms`` read via the memoryspace's lazy
        ``_maybe_catch_up()``. Implementations should not call
        ``consume`` themselves.
        """
        return None
