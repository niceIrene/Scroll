# SCROLL: Session-as-Context, Recoverable Off-context LLM Log

## TL;DR

SCROLL is a memory-system pattern for LLM agents on long-horizon tasks. It says:

- Don't stuff history into the LLM's context window.
- Keep the full session as an **append-only event log** `E` outside the context.
- Maintain a **derived memory memoryspace** `W = build(E)` (SQL tables, vector index, JSON store, files) populated by the harness as events flow through.
- At decision time, let the agent write **CodeAct** snippets — Python that queries `E` and `W` — to retrieve into context only what it needs.

The acronym: **S**ession-as-**C**ontext, **R**ecoverable **O**ff-context **L**LM **L**og.

The framework in this repo (the `Scroll` Python package) is a reference implementation. Three benchmarks plug into it: LongMemEval, the Vending Machine simulation, and BEAM (Stanford long-horizon).

---

## The pattern, formally

```
E:  append-only event log
W:  memory memoryspace, W = build(E)
f:  Python the LM writes against (E, W) → retrieved context
```

Three claims hold the pattern together:

1. **Memory lives outside the prompt.** Past sessions / past observations are persisted to `E` and indexed into `W`. The LLM's prompt carries only its CodeAct REPL globals plus what its last code cell printed.
2. **Memoryspace is a function of the log.** Anything in `W` is reproducible by replaying `E` through a `build()` pipeline. Different envs have different `build()` functions (LME's regex extractors over chat turns; vending's SQL ingest over sales/deliveries/finances). `E` is the source of truth.
3. **Retrieval is code.** The agent never gets "shown more context" via prompt-stuffing. To use a piece of memory the agent writes Python — a SQL query, a vector lookup, a regex over the log, or a sub-LM call on a slice — and the stdout becomes the next turn's tool result.

---

## Why this shape

The standard alternatives all show specific failure modes on long-horizon tasks:

- **Fitting it in the prompt** breaks past a few thousand tokens of history; the model attends unevenly to the middle.
- **RAG over chunks** loses structure (counts, dates, joins) and depends on embedding quality matching the question's phrasing.
- **Tool-using agents with thin function-call surfaces** force the model to invent retrieval verbs it might not have been trained on.

SCROLL trades these for a different bet: the model already knows how to write Python that reads from SQL tables and a log handle. If the harness pre-builds the right indices, retrieval reduces to writing the right query. The Python interpreter — not the LLM — is the retrieval mechanism.

The code-execution part is not novel (CodeAct, `dspy.RLM`, and others have shown the substrate works). What SCROLL adds is the discipline that **memory is the memoryspace**, not the prompt — and the contract that `W = build(E)` is recoverable.

---

## The substrate (`Scroll.core`)

The `ScrollAgent` base class ([src/Scroll/core/_scroll_agent.py](../src/Scroll/core/_scroll_agent.py)) inherits from `CodeActAgent` and owns the pattern:

```
ScrollAgent
├── self.memoryspace          : Memoryspace (W)
├── self.log                : ConversationLog (E)
└── init_namespace()        binds REPL globals:
                              ├── log  = LogHandle(self.log)
                              ├── ms   = self.memoryspace
                              ├── sub_llm (optional)
                              └── rlm     (optional)
```

Subclasses provide three hooks:

| Hook                        | Purpose                                                  |
|-----------------------------|----------------------------------------------------------|
| `_ensure_schema(memoryspace)` | Issue `CREATE TABLE` / `CREATE VIEW` for the env.        |
| `_ingest_context(session_idx, notes)` | Populate `W` from per-session briefing notes.              |
| `_ingest_outcomes(session_idx, logs)` | Populate `W` from end-of-session outcome lines.            |

Plus class-level feature toggles:

| Flag                  | Default | Effect                                                       |
|-----------------------|---------|--------------------------------------------------------------|
| `expose_sub_llm`      | `False` | Bind a stateless one-shot LM as `sub_llm` in the REPL.       |
| `expose_rlm`          | `False` | Bind a recursive sub-agent (`dspy.RLM`) as `rlm` in the REPL.|

These flags let each env tune the contract under test without forking the base class. The agent's REPL `ms` is always a minimal query-only view (`sql_exec` + `vector_query`); the harness owns all writes via `self.memoryspace` directly.

### One retrieval cycle

```
                                                ┌───────────────────────┐
   model writes ───►  execute_python(code)  ─►  │  REPL                 │
                                                │   log, ms, sub_llm…   │
                                                │   ↓ Python runs       │
                                                │   stdout captured     │
                                                └───────────────────────┘
                                                          ▲
                                                          │
                          ◄─── next user turn = stdout ───┘
```

The agent's "memory" is whatever its previous `print()` calls put into the next turn's tool result. Anything not printed is gone from context — but still in `E` and `W`, retrievable next turn.

---

## Per-benchmark instantiations

### LongMemEval (`Scroll.benchmarks.longmemeval`)

The original chat-memory benchmark (Wu et al., 2024). Each task is one QA item with N+1 days: days 1..N stream past chat sessions; day N+1 asks a single probe.

- `E` is populated with `chat_turn` entries (one per turn of every haystack session).
- `W` has five tables (`chat_turns`, `user_preferences`, `event_dates`, `facts`, `sessions`) populated by regex extractors over each turn, plus a `rounds` view that pre-joins user + assistant turns.
- The harness writes; the agent's REPL only sees the query-only `ms` view. This is the pure-retrieval contract.
- `expose_rlm=True` — the agent's most expensive primitive; for span extraction, ranking, and synthesis over candidate rows.
- Probe time uses an OpenAI-style `execute_python` tool call so the model can reason in message content while passing code through a structured `code` parameter.
- **L3 procedural memory**: after the judge scores each probe, `_on_probe_complete` distills a 1-3 case-study hint via a one-shot sub-LM and writes it to the per-task memoryspace under `procedural_hints`. Subsequent probes of the same `question_type` in the same task see the top-K most-relevant hints injected into their prompt.

Class: [`LongMemEvalAgent`](../src/Scroll/benchmarks/longmemeval/agents/agent.py).

### Vending Machine (`Scroll.benchmarks.vending`)

A long-horizon economic simulation: 30-180 day inventory + pricing + supplier-negotiation task.

- `E` is the full event log (briefings, agent actions, env outcomes).
- `W` has 7 base tables (`sales`, `deliveries`, `orders`, `daily_finances`, `daily_inventory`, `supplier_prices`, `notes`) plus 5 analytic views (`weekly_revenue_by_sku`, `rolling_7d_units_by_sku`, `cogs_by_supplier`, `inventory_turnover`, `daily_pnl`).
- Agent only queries the auto-ingested `W`; harness owns all writes (matches LME / BEAM).
- `expose_rlm=True` — recursive `dspy.RLM` sub-agent for free-text synthesis over recalled rows.
- Action tools (`send_email`, `read_email`, `get_money_balance`, `run_sub_agent`) live alongside `log` / `ms` in the REPL.

Class: [`VendingAgent`](../src/Scroll/benchmarks/vending/agents/agent.py).

### BEAM (`Scroll.benchmarks.beam`)

"Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs" (Tavakoli, ICLR 2026, arXiv:2510.27246). Long-context chat-memory benchmark — same shape as LongMemEval (chat haystack + end-of-conversation probes), scaled to 100K / 500K / 1M / 10M tokens per chat.

- `E` is populated with `chat_turn` entries (one per turn of every BEAM batch).
- `W` uses the same five-table schema as LongMemEval (`chat_turns`, `user_preferences`, `event_dates`, `facts`, `sessions`) plus the `rounds` view. Regex extractors are shared with LME (chat memory is chat memory).
- `expose_rlm=True` — same flag as LME (pure-retrieval contract).
- ~20 probing questions per chat across 10 ability categories (information_extraction, multi_session_reasoning, knowledge_update, temporal_reasoning, abstention, contradiction_resolution, event_ordering, instruction_following, preference_following, summarization). All fire on the probe day.
- Scoring: per-rubric-item LLM-judge averaged. Vendored prompt template from BEAM's own evaluator so scores are comparable with the upstream numbers.
- Dataset lives at `external/beam/` (git submodule). Time anchors (`Month-Day-Year` strings on each message) are normalized to ISO `YYYY-MM-DD` at load time so the memoryspace's `session_date_iso` column is sortable.

Class: [`BeamAgent`](../src/Scroll/benchmarks/beam/agents/agent.py).

The BEAM agent is a *leaner* version of `LongMemEvalAgent` — no procedural-hints distillation (BEAM is single-pass; no cross-task store), no qtype-conditional probe-hint composition (categories ride on the env's `probe_user_postscript` instead). Shared chat-memory primitives (time-range extractor, probe/handle session bodies) live in `Scroll.tools.chat_memory`; the BEAM-side ingestor still subclasses `LMEIngestor` for the regex extractors.

---

## Per-turn context management

History is kept continuous across sessions (one task = one
conversation). Each model call runs `_compress_history_to_budget` as
a pre-call hook: if total chars exceed `cfg.context_max_tokens * 4`,
the oldest batch is summarized into a running `_compressed_summary`
and dropped from `self._history`. The summary is re-injected as an
extra system message at every `_call_model`, so the LM still sees
old context — just compacted.

LME / BEAM auto-advance their session loops (no per-session LM call),
so their history grows trivially across haystack sessions and rarely
triggers compression. Vending exercises the compression path
naturally as the agent runs cells each day to plan inventory /
orders.

---

## Why this is publishable

Three points:

1. **The (E, W, CodeAct) decomposition is a clean primitive.** It separates *what the memory contains* (env-specific schema in `W`) from *how the agent uses it* (env-agnostic CodeAct + REPL). The same base class drives chat-memory probes and multi-day planning without per-task harness changes.
2. **The contract is testable.** The agent's REPL is always a minimal query-only view of `ms` (SQL + vector); the harness owns ingestion. This cleanly isolates "agent's retrieval skill" from "agent's planning skill" — every env follows the same pure-retrieval contract for `ms`, with planning surfaced through env-action tools when relevant (Vending) or omitted entirely (LME / BEAM).
3. **The pattern generalizes.** Adding a benchmark = define a schema + ingest pipeline + prompt. The substrate (LLM loop, REPL, checkpoint, tracing) is reused. We demonstrate this with three benchmarks of meaningfully different shape.

---

## Results

### LongMemEval (full 500-QA `_s` split incl. abstention twins)

Agent: `qwen3.7-max` (Dashscope CN). Judge: `qwen3.6-plus` with a `<judge_thinking>` step. Pipeline: type-specific qtype templates (multi-session count / KU stale-value / temporal) + grace-turn rescue + mem0-style synthesis rules at commit time.

| Type | Acc | n |
|---|---:|---:|
| single-session-assistant | 0.982 | 56 |
| knowledge-update | 0.974 | 78 |
| single-session-user | 0.971 | 70 |
| temporal-reasoning | 0.917 | 133 |
| single-session-preference | 0.900 | 30 |
| multi-session | 0.789 | 133 |
| **OVERALL** | **0.906** | **500** |

**Provenance of the +10.6pp delta from a naive baseline:** qwen3.6-plus + no fixes ≈ 0.80 → + grace-turn commit-rescue + abstention fallback (0.866) → + qwen3.7-max upgrade (0.888) → + per-qtype forcing templates + mem0-style synthesis rules (0.906). The biggest single move is the KU stale-value template + synthesis rules, which together took knowledge-update from 0.821 to 0.974 (+15pp on the subset).

## File map

```
src/Scroll/
├── core/
│   ├── _scroll_agent.py       ScrollAgent base + _ReadOnlyMemoryspace
│   ├── _codeact_agent.py      CodeActAgent (LLM loop, REPL)
│   ├── _rlm.py                make_dspy_rlm helper
│   ├── _environment.py        BaseEnvironment / BaseDataSource
│   └── _registry.py           env auto-discovery
├── longmemeval/
│   ├── agents/agent.py       LongMemEvalAgent
│   ├── env.py                 LongMemEvalEnv
│   └── tasks/probes.py        ProbeSpec list + LLM judge
├── vending/
│   ├── agents/agent.py       VendingAgent
│   ├── env.py                 VendingEnv
│   ├── auto_ingest.py         sales/deliveries/finances ingest
│   └── tasks/                 ProbeSpec list + deterministic scorer
└── beam/
    ├── agents/agent.py        BeamAgent
    ├── env.py                  BeamEnv (LME-shaped: batches → probe day)
    ├── dataset.py              chat.json + probing_questions loader
    └── tasks/probes.py         20 ProbeSpecs per chat + rubric judge

docs/scroll.md                 (this file)
configs/<env>/                  one JSON per (model, variant)
```
