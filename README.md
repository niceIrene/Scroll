# scroll-context

Scroll is a context-management substrate for LLM agents: every turn of the
conversation is **write-through persisted** to a durable SQLite history, the
in-context window is a **token-bounded view** (old turns age out and evict into
a compact in-context index map), and the model **retrieves what it needs on
demand** by writing Python against a read-only memory API (`ms`) inside a
persistent REPL.

This repository contains the runtime, a reusable context-management library,
two reference agents, and the two benchmarks used to evaluate them.

## Layout

The repo is a uv workspace: the root is the **releasable `scroll-context`
package** (PyPI-able, one lightweight dependency), and `evaluation/` is an
independent, never-published project holding everything experiment-related.

```
pyproject.toml               The scroll-context package (src-layout).
src/scroll_context/          Public API: ScrollContextManager (+ the prompt
                             protocol core.md / index.md, the scroll_repl tool
                             schema, and HistoryStore/LogEntry for seeding
                             prior-session tiers).
  _runtime/                  PRIVATE internals — never import directly:
                             HistoryStore impl (SQLite conversation_history +
                             FTS5), MemorySpace (ms.search / ms.expand /
                             ms.sql_query), Executor (persistent REPL),
                             EvictionIndex, ScrollRuntime.
tests/                       Package tests.
evaluation/                  Independent evaluation project (scroll-eval):
  pyproject.toml             Depends on scroll-context via the workspace.
  scroll_eval/
    base_agents/
      base_agent_A/          Minimal ReAct loop (no scroll) — the baseline.
      scroll_agent_A/        The same loop with scroll context management,
                             fully delegated to ScrollContextManager.
    evals/beam/              BEAM long-term-memory benchmark (+ LLM judge).
    evals/terminal_bench/    Terminal-Bench integration (Harbor sandbox).
    harness/                 Run orchestration: config, run dirs, summaries.
    cli.py                   The `scroll-eval` command.
    edgebench_entry.py       Standalone workspace-agent entrypoint.
  tests/                     Evaluation tests.
configs/                     beam.yaml, terminal-bench.yaml
local-tasks/                 Benchmark task data (beam/, terminal-bench-2.1/).
scripts/                     Run-analysis utilities (beam_analysis.py, ...).
runs/                        Run outputs (gitignored).
```

## Quick start

```bash
uv sync --all-packages --all-extras   # package + evaluation project

# BEAM (long-term memory): ingest + run + judge, per configs/beam.yaml
uv run scroll-eval beam configs/beam.yaml

# Terminal-Bench (Harbor/docker sandbox), per configs/terminal-bench.yaml
uv run scroll-eval run configs/terminal-bench.yaml --task <task-id>

# Tests (package + evaluation)
uv run --with pytest pytest tests evaluation/tests -q

# Build the releasable package (wheel ships only scroll_context/)
uv build
```

Model endpoint/key come from `.env.local` / env (`OPENAI_BASE_URL`,
`OPENAI_MODEL_NAME`, `OPENAI_API_KEY` — `DASHSCOPE_API_KEY` accepted for
DashScope endpoints). Tracing goes to Phoenix (`docker compose up` for a local
instance).

## Using the library in your own agent loop

```python
# pip install scroll-context
from scroll_context import ScrollContextManager, SCROLL_REPL_TOOL_SCHEMA

mgr = ScrollContextManager(
    history_db_path="history.db", session_id="run1:task1",
    history_max_tokens=100_000, pinned=2, repl_name="scroll_repl",
)
system = {"role": "system", "content": PREAMBLE + mgr.protocol_prompt() + FINISH}
messages = [system, {"role": "user", "content": task}]
mgr.record_initial_prompt(messages[1])

while not done:
    mgr.manage(messages)                                  # age + evict + map
    call = messages + [mgr.digest_message()]              # ephemeral digest
    assistant = llm(call, tools=[SCROLL_REPL_TOOL_SCHEMA, ...])
    messages.append(assistant)
    mgr.record_assistant_turn(assistant, usage=...)
    if repl_call(assistant):
        out = mgr.execute_python(source_of(assistant))    # ms.* available
        tool_msg = {"role": "tool", "tool_call_id": ..., "content": out}
        messages.append(tool_msg)
        mgr.record_tool_result(tool_msg, tool_name="scroll_repl")
mgr.close()
```

Inside the REPL the model has: `ms.search(...)`, `ms.expand(...)`,
`ms.sql_query(...)`, `ms.session_id` / `ms.task_id`, `or_terms([...])`,
`days_between(d1, d2)`, a persistent variable namespace, and the `⟦ headline ⟧`
fence that turns a milestone line into a durable, indexable headline.

Feature knobs (env): `SCROLL_EVICTION_INDEX` (index map on/off),
`SCROLL_INDEX_LEVEL_CAP`, `SCROLL_SEED_INDEX` (seeded [memory] map),
`SCROLL_OBS_KEEP_TURNS` (observation aging window),
`SCROLL_FORCE_FINAL_ANSWER`, `SCROLL_MAX_STEPS`.
