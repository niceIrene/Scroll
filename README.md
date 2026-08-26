# scroll-context

Scroll is a context-management substrate for LLM agents: every turn of the
conversation is **write-through persisted** to a durable SQLite history, the
in-context window is a **token-bounded view** (old turns age out and evict into
a compact in-context index map), and the model **retrieves what it needs on
demand** by writing Python against a read-only memory API (`ms`) inside a
persistent REPL.

This repository contains the runtime, a reusable context-management library,
the reference scroll agents plus baseline/ablation agents, and the three
benchmarks used to evaluate them.

## Layout

The repo is a uv workspace: the root is the **releasable `scroll-context`
package** (PyPI-able, one lightweight dependency), and `evaluation/` is an
independent, never-published project holding everything experiment-related.

```
pyproject.toml               The scroll-context package (src-layout).
src/scroll_context/          Public API: ScrollContextManager (+ the prompt
                             protocol core.md / index.md / index-dense.md /
                             vars.md, the scroll_repl tool schema, and
                             HistoryStore/LogEntry for seeding prior-session
                             tiers).
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
      scroll_react/          ReAct loop with scroll context management, fully
                             delegated to ScrollContextManager (the reference
                             integration).
      scroll_codact/         CodeAct variant: task tools are functions inside
                             the REPL namespace instead of top-level tools.
      scroll_tools/          Ablation arm: DB retrieval without the REPL
                             (search_history / expand_turns as plain tools).
      base_agent_A/          Minimal ReAct loop (no scroll) — the baseline.
      longctx_baseline/      Vanilla long-context baseline (transcript
                             stuffing + recency truncation; no tools).
      summary_baseline/      Rolling-summary baseline (incremental
                             continuation summary; no DB, no REPL).
    evals/beam/              BEAM long-term-memory benchmark (+ LLM judge).
    evals/longmemeval/       LongMemEval memory-QA benchmark (+ LLM judge).
    evals/terminal_bench/    Terminal-Bench integration (Harbor sandbox).
    harness/                 Run orchestration: config, run dirs, summaries.
    runner.py                The shared agent-loop runner.
    cli.py                   The `scroll-eval` command (run / beam /
                             longmemeval / summary / compare).
    edgebench_entry.py       Standalone workspace-agent entrypoint.
  tests/                     Evaluation tests.
configs/                     beam.yaml, longmemeval.yaml, longmemeval-m.yaml,
                             terminal-bench.yaml, ablation/*.yaml.
local-tasks/                 Materialized benchmark task data (beam/,
                             longmemeval/, longmemeval-m/, terminal-bench-2.1/).
scripts/                     Task materialization (migrate_beam.py,
                             gen_longmemeval_tasks.py) and run-analysis
                             utilities (beam_analysis.py, lme_analysis.py,
                             ablation_compare.py, dump_run_cost.py, ...).
runs/                        Run outputs (gitignored).
```

## Quick start

### Setting up the BEAM experiment

```bash
git submodule update --init --progress external/beam
#  The BEAM repo is ~700 MB, so it will take some time to download

uv sync --all-packages --all-extras   # package + evaluation project

# Put your Model endpoint/key into .env.local
touch .env.local

# Materialize the BEAM tasks into local-tasks/beam/ (gitignored, so this
# must be run once per clone). Reads external/beam/chats/<scale>;
# --scale picks the tier (100K default; also 500K, 1M, 10M).
uv run python scripts/migrate_beam.py --scale 100K
```

### Setting up the LongMemEval experiment

```bash
git submodule update --init --progress external/longmemeval

uv sync --all-packages --all-extras   # package + evaluation project

# Put your Model endpoint/key into .env.local
touch .env.local

# The dataset JSONs are NOT in the submodule repo — download from HuggingFace
# (~277 MB for the "s" tier; also longmemeval_m_cleaned / longmemeval_oracle):
curl -L -o external/longmemeval/data/longmemeval_s_cleaned.json \
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# Generate native tasks into local-tasks/longmemeval/ (gitignored, so this
# must be run once per clone) from the dataset file. --limit 0 = all 500
# instances (the default is 10); --qids / --question-type narrow further.
uv run python scripts/gen_longmemeval_tasks.py \
    --src external/longmemeval/data/longmemeval_s_cleaned.json \
    --dataset longmemeval --limit 0

# The "m" split (~1.5M-token haystacks, 2.7 GB download) lives side by side —
# the config's dataset.name picks the split (see configs/longmemeval-m.yaml):
#   curl -L -o external/longmemeval/data/longmemeval_m_cleaned.json \
#     https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json
#   uv run python scripts/gen_longmemeval_tasks.py \
#     --src external/longmemeval/data/longmemeval_m_cleaned.json \
#     --dataset longmemeval-m --limit 0
```

### Running

```bash
# BEAM (long-term memory): ingest + run + judge, per configs/beam.yaml
uv run scroll-eval beam configs/beam.yaml

# LongMemEval (memory QA): ingest + run + judge, per configs/longmemeval.yaml
uv run scroll-eval longmemeval configs/longmemeval.yaml

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
    history_db_path="history.db",
    session_id="run1:task1",
    history_max_tokens=100_000,
    pinned=2,
    repl_name="scroll_repl",
)
system = {
    "role": "system",
    "content": PREAMBLE + mgr.protocol_prompt() + FINISH,
}
messages = [system, {"role": "user", "content": task}]
mgr.record_initial_prompt(messages[1])
mgr.prime_prior_sessions(messages)  # fold prior-session spans into the map

while not done:
    mgr.manage(messages)  # age/virtualize + evict + refresh map
    call = messages + [mgr.digest_message()]  # ephemeral digest
    assistant = llm(call, tools=[SCROLL_REPL_TOOL_SCHEMA, ...])
    messages.append(assistant)
    mgr.record_assistant_turn(assistant, usage=...)
    if repl_call(assistant):
        out = mgr.execute_python(source_of(assistant))  # ms.* available
        tool_msg = {"role": "tool", "tool_call_id": ..., "content": out}
        messages.append(tool_msg)
        mgr.record_tool_result(tool_msg, tool_name="scroll_repl")
mgr.close_session(final_answer)  # durable session_record for future sessions
mgr.close()
```

The full integration contract, by lifecycle stage (all messages are plain
OpenAI chat dicts, and the SAME dict objects passed to `record_*` must be the
ones kept in the message list — bookkeeping is by object identity):

- **Setup** — construct `ScrollContextManager`; embed `mgr.protocol_prompt()`
  in the system prompt; expose `SCROLL_REPL_TOOL_SCHEMA` (or your own tool
  named `repl_name`); `mgr.record_initial_prompt(task_msg)`; optionally
  `mgr.prime_prior_sessions(messages)` to seed the map from shared tiers /
  this agent's own prior `session_record` rows (or pass explicit spans).
- **Every loop step** — `mgr.manage(messages)` before the API call (mutates
  the list in place: aging or var-context virtualization/distillation,
  budget eviction, index placeholder); append `mgr.digest_message()`
  ephemerally (never persist it); after the response,
  `mgr.record_assistant_turn(msg, usage)`; answer REPL calls with
  `mgr.execute_python(source)` (or `execute_python_async` inside an event
  loop) and record every tool result — REPL or external — with
  `mgr.record_tool_result(tool_msg, tool_name=...)`.
- **As they occur** — `mgr.record_user_message(msg, messages=messages)` for
  an interleaved user turn (this folds the previous turn via `close_turn`);
  `mgr.record_tool_call(...)` for a terminal call with no result message
  (e.g. `submit_answer`).
- **Teardown** — `mgr.close_session(final_answer)` (writes the durable
  `session_record` future sessions prime from), `mgr.metrics()` if you want
  totals, then `mgr.close()`.

Inside the REPL the model has: `ms.search(...)`, `ms.expand(...)`,
`ms.sql_query(...)`, `ms.session_id` / `ms.task_id`, `or_terms([...])`,
`days_between(d1, d2)`, a persistent variable namespace, the `⟦ headline ⟧`
fence that turns a milestone line into a durable, indexable headline, and — in
var-context mode — the `pin` / `note` / `show` variable-curation ops.

Package feature knobs (env, read by `scroll_context` itself; constructor
params take precedence): `SCROLL_EVICTION_INDEX` (index map on/off),
`SCROLL_INDEX_LEVEL_CAP`, `SCROLL_OBS_KEEP_TURNS` (observation aging window),
`SCROLL_VAR_CONTEXT` (var-context mode) and its tuning family
(`SCROLL_VAR_KEEP_THOUGHTS`, `SCROLL_VAR_FALLBACK_CHARS`,
`SCROLL_KEEP_TURNS_VERBATIM`, `SCROLL_TURN_ASK_CHARS`,
`SCROLL_TURN_ANS_CHARS`), `SCROLL_SEED_DENSE_HEADLINES` (dense-headline index
prompt variant).

Evaluation-harness knobs (env, read by `scroll-eval`, not the package):
`SCROLL_SEED_INDEX` (seeded [memory] map), `SCROLL_MAX_STEPS`,
`SCROLL_FORCE_FINAL_ANSWER`, `SCROLL_MODEL` / `SCROLL_JUDGE_MODEL`, and
others — see `evaluation/scroll_eval/harness/`.
