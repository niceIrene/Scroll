"""Vending SCROLL agent.

Applies the :class:`Scroll.core.ScrollAgent` harness to the vending
machine simulation. The harness owns ingestion of every iteration's
events; the agent reads from ``ms`` / ``log`` and reaches for the
recursive ``rlm`` sub-agent only when the structured surfaces miss.
(Module-level prompts still reference ``sub_llm`` — pending rewrite.)

Layout:
  - 7 base tables (``sales``, ``deliveries``, ``orders``,
    ``daily_finances``, ``daily_inventory``, ``supplier_prices``,
    ``notes``)
  - 5 analytic views (``weekly_revenue_by_sku``, ``rolling_7d_units_by_sku``,
    ``cogs_by_supplier``, ``inventory_turnover``, ``daily_pnl``)
  - Vector store over inbox emails, datasource notes, and env outcome
    lines.
"""

from __future__ import annotations

from Scroll.core import ScrollAgent
from Scroll.core._models import LogEntry
from Scroll.benchmarks.vending.ingestor import (
    VendingIngestor,
    ensure_schema as _vending_ensure_schema,
    serialize_env_snapshot,
    serialize_inbox_mail,
)
from Scroll.benchmarks.vending.agents._common import VENDING_CONTEXT, vending_day_prompt
from Scroll.benchmarks.vending.agents._namespace import make_env_namespace


# ---------------------------------------------------------------------------
# Strategy prompt
# ---------------------------------------------------------------------------

_STRATEGY_PROMPT = """\
DATA MANAGEMENT:
- You have THREE complementary recall surfaces, ranked by cost:
    1. memoryspace — multi-format store W (SQL + JSON + vectors +
       files), AUTO-POPULATED by the harness from each iteration's
       events. You read; the harness writes. Vending layout below.
    2. log — read-only handle on the raw event log E (LogHandle).
       Useful when memoryspace doesn't capture a field, when you want
       paraphrase-tolerant text search across E, or when you want to
       reuse a past ``sub_llm`` answer (filter by
       kind="sub_llm_result").
    3. sub_llm — async stateless one-shot sub-LM. Last resort; reach
       for it ONLY when neither memoryspace nor log resolves the
       question.
- Within-iteration chat memory is wiped at the iteration boundary.
  Across iterations, memoryspace + log are the only channels carrying
  past information.
- Be efficient: every tool call costs message budget.

YOUR JOB:
- Query memoryspace + log to make decisions. Trust the auto-populated
  rows and views — you don't need to re-derive them.
- Focus on PLANNING, not on data entry. The harness already ingests
  everything that flows through the env into memoryspace.
- Use ``memoryspace.schema_inspect()`` to remember what tables / views
  exist when unsure.

GENERIC MEMORYSPACE READS — SHAPE, NOT SPECIFIC TABLES:

  # Parameterized SELECT against an auto-populated table.
  rows = memoryspace.sql_exec(
      "SELECT <cols> FROM <table> WHERE <col> = ? ORDER BY day",
      params=(value,),
  )
  for r in rows:
      print(r["<col>"])      # rows is list[dict]

  # Vector recall for a free-text question.
  for key, text, score in memoryspace.vector_query(
      "<your natural-language query>", top_k=5,
  ):
      print(score, key, text[:80])

  # Schema introspection — use when unsure what tables / views exist.
  print(memoryspace.schema_inspect())

ROUTING — pick the cheapest tool that fits:

  | Question shape                                  | Tool                          |
  |-------------------------------------------------|-------------------------------|
  | Aggregates / counts / rankings over rows        | memoryspace.sql_exec            |
  | Structured fields embedded in free text         | log.findall (regex, 0 LLM)    |
  | Recall by paraphrase / topic / multi-word       | memoryspace.vector_query OR     |
  |                                                 | log.semantic_search           |
  | Free-text trend / qualitative / synthesis       | semantic-recall + sub_llm     |

  Auto-built analytic views (when the env has them) usually beat
  re-deriving from base tables — check ``memoryspace.schema_inspect()``
  first.

SUB-LM ESCAPE HATCH:
- ``sub_llm(prompt)`` is async, stateless, one-shot. NO memory, NO
  log/memoryspace access, NO tools — embed everything it needs in the
  prompt. Reach for it ONLY when no memoryspace / log primitive
  resolves the question.
- Every call is mirrored into E with kind="sub_llm_call" /
  "sub_llm_result". Recover prior inferences BEFORE re-paying:

      prior = log.semantic_search(
          "<your sub-question>", k=3, kind="sub_llm_result",
      )
      if prior and prior[0][1] > 0.3:
          print("reusing:", prior[0][0].tool_result["output"][:300])
      else:
          hits = log.semantic_search("<topic>", k=10, role="user")
          text = log.text([e for e, _ in hits])
          ans = await sub_llm(f"Given:\\n{text}\\n\\n<question>")
          print(ans)
"""


_VENDING_LAYOUT = """\
VENDING — AUTO-BUILT MEMORYSPACE LAYOUT:

  Base tables (mirror the auto-ingested D):
    sales, deliveries, orders, daily_finances, daily_inventory,
    supplier_prices, notes.

  Analytic views (don't re-derive these — query them directly):
    weekly_revenue_by_sku   (week, sku, units, revenue)
    rolling_7d_units_by_sku (day, sku, units_7d) — last 7 days incl. today
    cogs_by_supplier        (supplier, sku, units, cost)
    inventory_turnover      (day, sku, total_qty, units_sold, turnover_ratio)
    daily_pnl               (day, revenue, cogs, gross_profit)

  JSON store: ``context_day_N`` / ``outcomes_day_N`` keys hold the
    raw briefing / outcome lines for chronological free-text fallback.

  Vector store: each datasource note, inbox email, and env outcome
    line is auto-embedded — use ``vector_query`` for natural-language
    recall (e.g. "what did the supplier say about cola pricing?").
"""

_VENDING_EXAMPLES = """\
WORKED EXAMPLES — vending auto-built memoryspace:

  # 1. SELECT against a base table — params= for binds
  rows = memoryspace.sql_exec(
      "SELECT day, revenue FROM sales "
      "WHERE day BETWEEN ? AND ? ORDER BY day",
      params=(21, 30),
  )
  for r in rows:
      print(r["day"], r["revenue"])

  # 2. Hit a pre-built analytic view (no need to re-derive)
  rows = memoryspace.sql_exec(
      "SELECT week, SUM(revenue) AS r FROM weekly_revenue_by_sku "
      "WHERE sku = ? GROUP BY week",
      params=("cola",),
  )
  print(rows)

  # 3. Vector recall for a free-text question
  for key, text, score in memoryspace.vector_query(
      "supplier reply about cola wholesale price", top_k=5
  ):
      print(score, key, text[:80])

  # 4. JSON store for the raw briefing of a past day
  ctx = memoryspace.json_read(f"context_day_{today - 1}")
  print(ctx)
"""


BASE_PROMPT = (
    VENDING_CONTEXT + "\n"
    + _STRATEGY_PROMPT
    + "\n" + _VENDING_LAYOUT
    + "\n" + _VENDING_EXAMPLES
)


_NAMESPACE_DOCS = """\
AVAILABLE OBJECTS (in your REPL globals):

  log: LogHandle — read-only view of E (raw event log).
    log.slice(day=None, role=None, kind=None) -> list[LogEntry]
    log.range_by_day(start: int, end: int) -> list[LogEntry]
    log.search(query: str, k: int = 20) -> list[LogEntry]   # substring
    log.semantic_search(
        query: str, k: int = 10, *,
        day=None, role=None, kind=None,
        entries=None, min_score: float = 0.0,
    ) -> list[(LogEntry, float)]                            # cosine BoW
    log.findall(pattern: str, entries=None, *, flags: int = 0) -> list
    log.text(entries=None) -> str

    Filter sub_llm cache via:
      log.slice(kind="sub_llm_result")
      log.semantic_search("<topic>", kind="sub_llm_result")

  memoryspace (alias: ms): auto-built data store W. The harness
    populates W on every ``receive_context`` / ``receive_outcomes``;
    you only READ from W (though you may write your own scratch
    tables / json keys / files / vectors if useful).

    Auto-built relational tables (mirror D):
      sales, deliveries, orders, daily_finances, daily_inventory,
      supplier_prices, notes
    Auto-built analytic views:
      weekly_revenue_by_sku, rolling_7d_units_by_sku,
      cogs_by_supplier, inventory_turnover, daily_pnl
    Auto-built vector index entries:
      email_day_N_idx_*, note_day_N_*, outcome_day_N_*

    Methods return native Python values — errors raise (no string error codes):

      ms.sql_exec(sql)
          SELECT -> list[dict]; INSERT/UPDATE/DELETE -> int (rowcount);
          CREATE/DROP -> None; sqlite3.Error on SQL failure.
      ms.json_write(key, data) -> None
      ms.json_read(key)        -> Any         # raises KeyError if missing
      ms.json_list()           -> list[str]
      ms.vector_store_add(key, text)  -> int  # new total count
      ms.vector_query(q, top_k=5)     -> list[tuple[key, text, similarity]]
      ms.vector_delete(key)           -> int
      ms.file_write(name, content)    -> None
      ms.file_read(name)              -> str  # raises KeyError if missing
      ms.file_list()                  -> dict[name, size_in_chars]
      ms.schema_inspect()             -> dict {tables, views, json_keys, vector_count, files}

  sub_llm(prompt: str) -> str  (async — must `await`)
    Stateless one-shot sub-LM. NO memory, NO log/ms access. Mirrored
    to E with kind="sub_llm_call" / "sub_llm_result"; recover via
    log.semantic_search before re-paying.

  Action tools (each returns str — NOT a list[dict]; use json.loads
    or .split("\\n---\\n") if you need structured pieces):
      send_email(to, subject, body)
      read_email()                    # one email as text, or "No unread emails."
      read_email_inbox()              # up to 20 emails joined by "\\n---\\n",
                                      # or "Inbox is empty."
      ai_web_search(query)            # "\\n"-joined results or "No results found."
      get_money_balance()             # JSON string {cash, machine_cash,
                                      # inventory_value, net_worth}
      sub_agent_specs()
      run_sub_agent(instruction)      # e.g. "check_inventory",
                                      # "restock cola=3 water=3", "collect_cash"
      chat_with_sub_agent(question)
      wait_for_next_day()
"""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class VendingAgent(ScrollAgent):
    """SCROLL agent for the vending environment.

    Writable memoryspace (the contract is "harness auto-ingests, but the
    agent may also write its own scratch state"; differs from
    LongMemEval's read-only contract). The recursive ``rlm`` sub-agent
    is exposed in the REPL for free-text synthesis over recalled rows.
    """

    # SCROLL feature flags — vending exposes the writable memoryspace +
    # the recursive ``rlm`` sub-agent for free-text synthesis over recalled
    # rows. (Pre-refactor this was a one-shot ``sub_llm``; rlm subsumes it.
    # NOTE: prompts in this file still reference ``sub_llm`` — pending
    # rewrite. They must be updated before this agent can run.)
    readonly_memoryspace = False
    expose_rlm = True

    sys_prompt = BASE_PROMPT

    def __init__(self, cfg, storage, data) -> None:
        # Cursor over the env's inbox: how many mails we've already
        # mirrored into E as ``kind="inbox_mail"`` entries. Each
        # ``receive_context`` advances it past the new tail. Set before
        # super().__init__ because receive_context can fire during
        # checkpoint resume.
        self._emitted_inbox_count = 0
        super().__init__(cfg, storage, data)

    # ----- SCROLL hooks -----

    ingestor_cls = VendingIngestor

    def _ensure_schema(self, memoryspace) -> None:
        _vending_ensure_schema(memoryspace)

    def _base_namespace(self) -> dict:
        # Env-specific tool closures (send_email, read_email,
        # get_money_balance, run_sub_agent, wait_for_next_day, ...)
        return make_env_namespace(self._tool_state)

    def namespace_docs(self) -> str:
        return _NAMESPACE_DOCS

    def session_prompt(self, session_idx: int) -> str:
        # Vending: session_idx maps 1:1 to the calendar day, so we pass
        # it as the `day` arg to the domain helper.
        return vending_day_prompt(session_idx, self._tool_state.env)

    def _emit_context_entries(self, session_idx: int, notes: list[str]) -> None:
        """Land per-day briefing notes AND any new inbox mails into E.

        Vending: framework session == calendar day (1:1). The ingestor
        will pick these up on the next ``ms`` read and write them into
        the seven canonical tables + vector store.
        """
        # Briefing notes — one entry per note.
        for note in notes:
            self.log.append(LogEntry.make(
                session_idx=session_idx, role="system",
                content=str(note),
                metadata={"kind": "briefing_note"},
            ))
        # Inbox tail — emit one entry per newly-arrived mail.
        inbox = getattr(self._tool_state.data, "inbox", None)
        if inbox is not None:
            inbox_list = list(inbox)
            for idx in range(self._emitted_inbox_count, len(inbox_list)):
                mail = inbox_list[idx]
                self.log.append(LogEntry.make(
                    session_idx=session_idx, role="system",
                    content=f"from={getattr(mail, 'source', None)} "
                            f"subject={getattr(mail, 'subject', '')}",
                    metadata={
                        "kind": "inbox_mail",
                        **serialize_inbox_mail(mail, idx),
                    },
                ))
            self._emitted_inbox_count = len(inbox_list)

    def _emit_outcome_entries(self, session_idx: int, logs: list[str]) -> None:
        """Land per-day env outcome lines AND a single env snapshot into E."""
        for line in logs:
            self.log.append(LogEntry.make(
                session_idx=session_idx, role="system",
                content=str(line),
                metadata={"kind": "env_log"},
            ))
        env = self._tool_state.env
        if env is not None:
            self.log.append(LogEntry.make(
                session_idx=session_idx, role="system",
                content=f"env_snapshot day={session_idx}",
                metadata={
                    "kind": "env_snapshot",
                    **serialize_env_snapshot(env),
                },
            ))

    # ----- Checkpoint (inbox cursor; memoryspace handled by base) -----

    def to_checkpoint(self) -> dict:
        data = super().to_checkpoint()
        data["emitted_inbox_count"] = self._emitted_inbox_count
        return data

    def from_checkpoint(self, data: dict) -> None:
        super().from_checkpoint(data)
        # Back-compat: old checkpoints stored ``embedded_inbox_count``.
        self._emitted_inbox_count = data.get(
            "emitted_inbox_count",
            data.get("embedded_inbox_count", 0),
        )
