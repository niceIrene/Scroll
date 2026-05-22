"""``ScrollAgent`` — the SCROLL harness base class.

SCROLL (Session-as-Context: Recoverable Off-context LLM Log) frames
long-term memory as two complementary surfaces that live OUTSIDE the
LLM context:

  - ``E`` — an append-only event log (:class:`Scroll.log.ConversationLog`)
  - ``W`` — a structured memoryspace derived from ``E``
    (:class:`Scroll.tools.memoryspace.Memoryspace`)

The agent uses **CodeAct** to bridge memory back into context on
demand: it writes Python in a sandboxed REPL with ``log`` (a
:class:`LogHandle` over E) and ``ms`` (the memoryspace) already bound,
runs the snippet, and the stdout becomes the next turn's tool result.
Nothing is "stuffed" into the prompt; everything is retrieved by code.

``ScrollAgent`` owns the memoryspace lifecycle, the REPL bindings, and
the checkpoint plumbing. Per-env subclasses provide the schema,
ingestion hooks, and prompt text.

Relationship to ``CodeActAgent``
--------------------------------
The class hierarchy is intentionally::

    BaseAgent  →  CodeActAgent  →  ScrollAgent  →  <env agent>

with the layers carrying distinct concepts:

  - :class:`CodeActAgent` (substrate)  — "agent talks to an LLM and runs
    Python in a sandboxed REPL". Owns the LLM call loop, history
    management, memory modes (``step`` / ``sliding``), probe-mode
    machinery, and checkpoint plumbing. Knows nothing about a
    memoryspace.
  - :class:`ScrollAgent` (this class)  — "CodeAct + ``W = build(E)``".
    Adds the memoryspace lifecycle, log+ms REPL bindings, and ingest
    hooks. This is the SCROLL pattern: ``M = build(E)`` is the memory
    abstraction, retrieval is the CodeAct loop.

**Current usage**: every concrete agent shipped with this repo
(:class:`LongMemEvalAgent`, :class:`VendingAgent`, :class:`BeamAgent`)
inherits ``ScrollAgent`` rather than ``CodeActAgent`` directly. The
intermediate layer is preserved on purpose: it lets a future
CodeAct-only baseline (no memoryspace, log-as-memory only) plug in at
the ``CodeActAgent`` layer for comparison against SCROLL. If you do
not need that comparison story, treating ``CodeActAgent`` and
``ScrollAgent`` as a single concept is defensible — they are split
here for the paper-narrative reason that ``SCROLL = CodeAct + W``
should be readable in the code.
"""

from __future__ import annotations

import logging
from typing import Any

from Scroll.core._codeact_agent import CodeActAgent
from Scroll.core._ingestor import Ingestor
from Scroll.tools._log_handle import LogHandle
from Scroll.tools.memoryspace import Memoryspace


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Read-only memoryspace proxy
#
# Used when the SCROLL contract under test is "harness writes, agent reads"
# (e.g. LongMemEval's pure-retrieval evaluation). Every mutator raises
# :class:`PermissionError`; reads pass through to the underlying Memoryspace.
# ---------------------------------------------------------------------------

_READ_ONLY_SQL_PREFIXES = ("SELECT", "PRAGMA", "EXPLAIN", "WITH")


class _ReadOnlyMemoryspace:
    """Read-only proxy over a :class:`Memoryspace`.

    The harness writes via the raw memoryspace (``agent.memoryspace``); the
    agent's REPL sees this proxy under ``ms`` / ``memoryspace``. Any write
    attempt (``INSERT``, ``UPDATE``, ``json_write``, ``vector_store_add``,
    ``file_write``) raises ``PermissionError``. Scratch state should live
    in Python locals (cleared at day boundary).
    """

    def __init__(self, ms: Memoryspace) -> None:
        self._ms = ms
        self.sqlite = ms.sqlite

    def sql_exec(self, statement: str, params: tuple | None = None):
        first = (
            statement.lstrip().split(None, 1)[0].upper()
            if statement.strip()
            else ""
        )
        if first not in _READ_ONLY_SQL_PREFIXES:
            raise PermissionError(
                f"memoryspace is read-only: SQL '{first}' not allowed "
                f"(only SELECT/PRAGMA/EXPLAIN/WITH). The harness owns "
                f"ingestion; query the auto-built tables."
            )
        return self._ms.sql_exec(statement, params)

    def vector_query(self, query, top_k: int = 5):
        return self._ms.vector_query(query, top_k)

    def json_read(self, key):
        return self._ms.json_read(key)

    def json_list(self):
        return self._ms.json_list()

    def file_read(self, name):
        return self._ms.file_read(name)

    def file_list(self):
        return self._ms.file_list()

    def schema_inspect(self):
        return self._ms.schema_inspect()

    def _denied(self, method: str) -> None:
        raise PermissionError(
            f"memoryspace.{method} is not available: the memoryspace is "
            f"fully READ-ONLY (the harness auto-ingests every session). "
            f"For within-cell scratch state, use Python variables — they "
            f"live until the next day boundary."
        )

    def json_write(self, *_a, **_kw): self._denied("json_write")
    def vector_store_add(self, *_a, **_kw): self._denied("vector_store_add")
    def vector_delete(self, *_a, **_kw): self._denied("vector_delete")
    def file_write(self, *_a, **_kw): self._denied("file_write")


# ---------------------------------------------------------------------------
# ScrollAgent — the harness base class
# ---------------------------------------------------------------------------


class ScrollAgent(CodeActAgent):
    """SCROLL-pattern agent.

    Inherits from :class:`CodeActAgent` (the LLM loop, REPL runtime,
    history management, checkpoint plumbing) and adds the
    ``(E, W, CodeAct retrieval)`` contract:

    - Owns ``self.memoryspace`` — the env's :class:`Memoryspace` (W).
    - Binds ``log`` (LogHandle over E) and ``ms`` (memoryspace) into the
      REPL namespace on every day's reset.
    - Optionally exposes ``rlm`` (recursive sub-agent via dspy.RLM).
    - Implements ``W = build(E)``: env data flows through
      :meth:`_emit_context_entries` / :meth:`_emit_outcome_entries` into
      ``E``, and the attached :class:`Ingestor` derives ``W`` from those
      entries. Every ``ms.*`` read first calls ``_maybe_catch_up`` on
      the memoryspace, guaranteeing fresh ``W`` at query time.

    Subclasses provide the env layer:

    Required overrides:
      - :meth:`_ensure_schema` — env-specific CREATE TABLE / CREATE VIEW
      - :attr:`ingestor_cls` — the env's :class:`Ingestor` class
      - :meth:`day_prompt` (already abstract on ``CodeActAgent``)

    Optional overrides:
      - :meth:`_emit_context_entries` — extra env data → ``E`` at
        ``receive_context`` time (inbox mail, calendar context, ...)
      - :meth:`_emit_outcome_entries` — extra env data → ``E`` at
        ``receive_outcomes`` time (env snapshot, batch metadata, ...)
      - :meth:`_base_namespace` — env-specific tool closures
      - :meth:`extra_namespace` — extra REPL globals beyond
        ``log`` / ``ms`` / ``rlm``
      - :meth:`namespace_docs` (already on ``CodeActAgent``) — docs
        about the REPL primitives

    Toggle harness features via class-level flags:
      - ``readonly_memoryspace`` — wrap ``ms`` with :class:`_ReadOnlyMemoryspace`
      - ``expose_rlm`` — bind ``rlm`` into the REPL (requires ``dspy`` extra)
    """

    # ----- harness feature toggles (override in subclasses) -----
    readonly_memoryspace: bool = False
    expose_rlm: bool = False

    # ----- Ingestor class (subclass override) -----
    # The per-env ``f: E → W`` derivation. Constructed as
    # ``ingestor_cls(self.memoryspace)`` and attached to the
    # memoryspace so reads auto-catch-up against E. Subclasses MUST
    # override this. Default ``None`` is a no-op ingestor (W stays
    # empty) — used only by tests that don't exercise ingest.
    ingestor_cls: type[Ingestor] | None = None

    # ----- CodeActAgent defaults that fit the SCROLL pattern -----
    memory_mode: str = "step"

    # ----- lifecycle -----

    def __init__(self, cfg, storage, data) -> None:
        self.memoryspace = Memoryspace()
        self._ensure_schema(self.memoryspace)
        self.memoryspace.sqlite.commit()
        super().__init__(cfg, storage, data)
        # Wire E → W: now that ``self.log`` exists (set in
        # ``CodeActAgent.__init__``), bind it to the memoryspace
        # alongside the env's ingestor. Every subsequent ms read /
        # write will first consume any unprocessed tail of E.
        if self.ingestor_cls is not None:
            self.memoryspace.attach(self.log, self.ingestor_cls(self.memoryspace))

    def receive_context(self, session_idx: int, notes: list[str]) -> None:
        """Append briefing notes to E and let the ingestor catch up.

        Subclasses that need to land extra env-side data into E (Vending's
        inbox / env snapshot, etc.) should override
        :meth:`_emit_context_entries` rather than this method — the
        framework guarantees the ingestor catches up afterwards.
        """
        self._emit_context_entries(session_idx, notes)
        # Force a catch-up here so the post-receive_context state of W
        # reflects the just-appended entries, even if no ms read fires
        # before the next phase.
        try:
            self.memoryspace._maybe_catch_up()
        except Exception:  # noqa: BLE001
            _log.warning(
                "%s ingest catch-up failed (session_idx=%s)",
                type(self).__name__, session_idx, exc_info=True,
            )
        super().receive_context(session_idx, notes)

    def receive_outcomes(self, session_idx: int, logs: list[str]) -> None:
        """Append outcome logs to E and let the ingestor catch up."""
        self._emit_outcome_entries(session_idx, logs)
        try:
            self.memoryspace._maybe_catch_up()
        except Exception:  # noqa: BLE001
            _log.warning(
                "%s ingest catch-up failed (session_idx=%s)",
                type(self).__name__, session_idx, exc_info=True,
            )
        super().receive_outcomes(session_idx, logs)

    def to_checkpoint(self) -> dict:
        data = super().to_checkpoint()
        data["memoryspace"] = self.memoryspace.to_checkpoint()
        return data

    def from_checkpoint(self, data: dict) -> None:
        super().from_checkpoint(data)
        self.memoryspace.from_checkpoint(data["memoryspace"])
        self._rebuild_namespace()

    # ----- REPL namespace construction -----

    def init_namespace(self) -> dict:
        ns = dict(self._base_namespace())
        ns["log"] = LogHandle(self.log)
        ms_handle: Any = (
            _ReadOnlyMemoryspace(self.memoryspace)
            if self.readonly_memoryspace
            else self.memoryspace
        )
        ns["memoryspace"] = ms_handle
        ns["ms"] = ms_handle
        if self.expose_rlm:
            from Scroll.tools._rlm import make_dspy_rlm
            ns["rlm"] = make_dspy_rlm(
                self.cfg,
                log=self.log,
                day_provider=lambda: self._current_session,
            )
        ns.update(self.extra_namespace())
        return ns

    # ----- subclass hooks -----

    def _ensure_schema(self, memoryspace: Memoryspace) -> None:
        """Override to issue CREATE TABLE / CREATE VIEW / CREATE INDEX
        statements against ``memoryspace.sqlite``.

        Called once at ``__init__`` time before ``super().__init__``.
        ``commit()`` runs after this returns, so subclasses should
        ``executescript`` and not bother committing themselves.
        """
        return None

    def _emit_context_entries(self, session_idx: int, notes: list[str]) -> None:
        """Append env briefing data to ``E`` as one or more LogEntries.

        Default behavior: emit one ``kind="briefing_note"`` entry per
        note string. Subclasses override to additionally serialize
        env-side state (inbox mail, calendar context, etc.) that the
        ingestor needs to derive ``W``.

        After this returns, the ingestor catches up automatically.
        """
        from Scroll.core._models import LogEntry
        for note in notes:
            self.log.append(LogEntry.make(
                session_idx=session_idx, role="system",
                content=str(note),
                metadata={"kind": "briefing_note"},
            ))

    def _emit_outcome_entries(self, session_idx: int, logs: list[str]) -> None:
        """Append env outcome data to ``E`` as one or more LogEntries.

        Default behavior: emit one ``kind="env_log"`` entry per log
        string. Subclasses override to additionally serialize env-side
        snapshots (financial / inventory state, batch metadata, etc.).
        """
        from Scroll.core._models import LogEntry
        for line in logs:
            self.log.append(LogEntry.make(
                session_idx=session_idx, role="system",
                content=str(line),
                metadata={"kind": "env_log"},
            ))

    def _base_namespace(self) -> dict:
        """Override to provide env-specific tool closures (action tools,
        ``wait_for_next_day``, etc.). Default: empty (pure-retrieval
        envs like LongMemEval have no actions).
        """
        return {}

    def extra_namespace(self) -> dict:
        """Override to bind extra REPL globals beyond the SCROLL core
        primitives (``log``, ``ms``, ``rlm``).
        """
        return {}
