# Project: SCROLL

Reference framework for the SCROLL pattern: **S**ession-as-**C**ontext, **R**ecoverable **O**ff-context **L**LM **L**og. Long-term memory lives outside the LLM context as an event log `E` plus a derived memoryspace `W = build(E)`; the agent uses CodeAct (Python in a sandboxed REPL with `log` and `ms` bound) to retrieve into context on demand.

Start with [`docs/scroll.md`](docs/scroll.md) for the design rationale.

## Working in this repo

- **Always use the project venv.** Activate `.venv` (`source .venv/bin/activate`) or invoke binaries explicitly via `.venv/bin/python` / `.venv/bin/Scroll`. Do not fall back to system Python or create ad-hoc envs — dependencies are pinned to this venv via `uv pip install -e .`.
- **Keep this file updated.** When you change layout, agent hierarchy, session/turn vocabulary, config schema, conventions, or the workflow commands, update CLAUDE.md in the same change so it stays in sync with the codebase.

## Code layout

```
src/Scroll/                     # framework
├── core/                       env-agnostic substrate
│   ├── _scroll_agent.py        ScrollAgent base + _ReadOnlyMemoryspace
│   ├── _codeact_agent.py       CodeActAgent (LLM loop)
│   ├── _codeact_runtime.py     persistent Python REPL substrate
│   ├── _ingestor.py            Ingestor protocol — pure f: E → W
│   ├── _tool_state.py          ToolState (harness bookkeeping)
│   ├── _environment.py         BaseEnvironment / BaseDataSource ABCs
│   ├── _registry.py            env auto-discovery via dataclass registry
│   ├── _evaluation.py          ProbeSpec, inject_probe
│   └── _agent.py, _models.py, _checkpoint.py, _runner.py, _chat_model_adapters.py
├── tools/                      agent-facing REPL primitives (log / ms / rlm)
│   ├── _log_handle.py          LogHandle (read API over ConversationLog)
│   ├── memoryspace.py          Memoryspace — auto-catches-up from E on every read/write
│   ├── _embed.py               bag-of-words cosine (cheap; LME ms.vector_query backend)
│   ├── _rlm.py                 make_dspy_rlm (dspy.RLM sub-agent factory)
│   └── chat_memory.py          shared LME+BEAM primitives: session bodies,
│                               make_chat_memory_namespace, write_chat_turn_entries,
│                               make_time_range_extractor
├── log.py                      ConversationLog (E, append-only JSONL)
├── benchmark.py                run_single, _run_task, aggregate
├── cli.py                      `Scroll` CLI entry point (incl. `rebuild-w`)
├── _tracing.py                 OTel setup
├── __init__.py, __main__.py
└── benchmarks/                 # plug-in benchmarks (one subpkg per env)
    ├── longmemeval/            env: chat-memory probes (Wu et al., 2024)
    │   ├── ingestor.py        LMEIngestor + 2-table schema + FTS5 + optional
    │   │                        typed-extraction (gated by ``extract_typed``,
    │   │                        default False for LME)
    │   ├── env.py             LongMemEvalEnv + LongMemEvalEnvConfig (merged
    │   │                        from old catalog.py)
    │   ├── datasource.py     dataset loader (LongMemEvalItem + load_items)
    │   │                        + runtime DataSource (merged from old dataset.py)
    │   ├── _time_utils.py     session-metadata parser + free-text date-phrase
    │   │                        resolver (merged from old agents/_date_utils.py)
    │   ├── agents/agent.py    LongMemEvalAgent (qtype templates + synthesis rules
    │   │                        + domain equivalences; grace-turn rescue)
    │   └── tasks/probes.py     ProbeSpec list + LLM judge + compute_efficiency_metrics
    ├── vending/                env: long-horizon vending sim
    │   ├── ingestor.py        VendingIngestor + schema + env-snapshot serializer
    │   ├── env.py             VendingEnv + EnvConfig + Product (merged from
    │   │                        old catalog.py)
    │   ├── datasource.py     DataSourceManager + Mail (inbox + scheduled events)
    │   ├── auto_ingest.py    per-table parse/INSERT helpers shared by the
    │   │                        ingestor and any boundary hooks
    │   ├── tools.py          env-action tool closures (send_email, restock,
    │   │                        get_money_balance, run_sub_agent, ...)
    │   ├── agents/agent.py    VendingAgent + the env-namespace builder
    │   ├── agents/prompts.py  VENDING_CONTEXT, BASE_PROMPT, NAMESPACE_DOCS,
    │   │                        vending_day_prompt (mirrors LME's prompts.py)
    │   └── tasks/probes.py     deterministic regex scorer
    └── beam/                   env: BEAM long-context chat-memory
        ├── ingestor.py        BeamIngestor + 5-table schema; subclasses
        │                        LMEIngestor with extract_typed=True
        ├── agents/agent.py    BeamAgent
        ├── env.py, datasource.py, catalog.py, dataset.py
        └── tasks/probes.py     per-rubric-item LLM-judge scorer
```

## The agent class hierarchy

```
BaseAgent (abc)
  └── CodeActAgent           generic LLM-loop + REPL substrate
        └── ScrollAgent      adds memoryspace lifecycle + ingestor attachment
              ├── LongMemEvalAgent     expose_rlm=True
              ├── VendingAgent          expose_rlm=True
              └── BeamAgent             expose_rlm=True
```

Subclasses provide three SCROLL hooks:
- `_ensure_schema(memoryspace)` — `CREATE TABLE` / `CREATE VIEW` for the env.
- `ingestor_cls` (class attribute) — the env's `Ingestor` subclass; `f: E → W`. Attached to `memoryspace` at agent init.
- `_emit_context_entries(turn_idx, notes)` / `_emit_outcome_entries(turn_idx, logs)` — append env data to `E` as typed LogEntries (defaults: one `kind="briefing_note"` per note, one `kind="env_log"` per log line). Override to additionally serialize env-side state (inbox mail, env snapshot, etc.) — see `VendingAgent` for the canonical pattern.

`W = build(E)` is enforced by the substrate: every public method on `Memoryspace` calls `_maybe_catch_up()` first, consuming any unprocessed tail of `E` into `W`. The CLI `Scroll rebuild-w --log conv.jsonl --env <id> --output-dir <dir>` runs the same ingestor offline and is the canonical invariant test.

Plus the standard `CodeActAgent` overrides: `turn_prompt`, `sys_prompt`, `namespace_docs`, `extra_namespace`, `_base_namespace`.

## Task / session / turn vocabulary

Three nesting levels, all named consistently in the code (renamed in PR #2 of the loop redesign; old names — `run_session`, `step_session`, `num_sessions`, `SessionResult`, `LogEntry.session_idx`, `BaseEnvironment.session_idx`, `get_probes_for_session`, etc. — are kept as back-compat aliases that drop in PR #6):

- **Task** — one Scroll run, one `run_single` invocation (one LME QA, one Vending sim, one BEAM chat). One persisted `E`, one derived `W`.
- **Session** — one agent-instance lifetime. Crossing this boundary = spawn a new `Agent(...)`; in-context history is wiped; only persisted `E` / `W` survive. Today every env runs as exactly **one** session per task. The substrate API for that boundary is `start_session` / `end_session` (added in PR #3).
- **Turn** — one CodeAct exchange: one user prompt → agent commits via `done()` or hits `max_iters_per_turn`. Carries `turn_idx` on `LogEntry`, `BaseEnvironment.turn_idx`, `_run_task` (the per-task driver), `step_turn`, `run_turn`. For LME today = one past chat session (PR #4 makes ingestion turn-less); for Vending = one calendar day (vending's SQL still uses a `day` column as a domain field, mapped 1:1 from `turn_idx` inside `VendingIngestor`); for BEAM today = one chat batch (PR #5 same). The agent config field is **`max_iters_per_turn`** — bounds the inner CodeAct loop, i.e. how many LLM calls (+ tool execs) the agent may take inside a single turn before being forced to commit.

`_run_task` drives one task as `env.ingest_all()` → `agent.start_session()` → per-turn loop (begin_turn / receive_context / run_turn / step_turn / receive_outcomes / per-turn probes / checkpoint) → `env.get_end_of_task_probes()` → `agent.end_session()` (in `finally`). Today `ingest_all` / `get_end_of_task_probes` / `start_session` / `end_session` all default to no-ops, so only the per-turn loop body runs. PRs #4 / #5 wire the new hooks per-env (LME / BEAM move ingestion into `ingest_all` and the probe into `get_end_of_task_probes`).

## Workflow

```bash
# Install (uses uv; Python 3.11+)
uv pip install -e .

# Activate the venv (prompt label is "Scroll")
source .venv/bin/activate

# Single-task smoke for each env (runs the QA / day / batch the config points at)
Scroll --config configs/longmemeval/scroll_qwen37max.json --seed 1   # production (0.906 on _s 500-QA)
Scroll --config configs/vending/scroll.json              --seed 1
Scroll --config configs/beam/scroll.json                 --seed 1

# Note: configs/longmemeval/ also has ``scroll.json`` (qwen3.6-plus baseline,
# ~0.866) and ``scroll_qwen37max_m.json`` (production config pointed at the
# M-split dataset).

# Offline rebuild — re-derives W from a persisted conversation_log.jsonl.
# Result should be functionally equivalent to the W the agent saw at runtime;
# this is the W = build(E) invariant test.
Scroll rebuild-w --log output/<env>/<run>/conversation_log.jsonl \
                 --env <longmemeval|vending|beam> \
                 --output-dir output/<env>/<run>/rebuilt
```

### Running LongMemEval

Two entry points:

- `Scroll --config <cfg>` — runs **one** QA, the one pinned in the config (`simulation.question_index` or `simulation.question_id`). Use when you want to debug one item; switch items by editing the config.
- `scripts/run_longmemeval.py --config <cfg>` — the orchestrator. Drives many QAs in their own subprocesses under one config; produces an aggregated `hypotheses.jsonl` + per-type `summary.json`. Use this for everything except single-item debugging.

**Task shape** (PR #4 of loop redesign): under the default `simulation.agent_during_ingestion=false`, an LME task is `env.ingest_all(log)` (bulk-writes all ~50 historical chat sessions into `E` in one shot) → `num_turns=0` (per-turn loop skipped) → `env.get_end_of_task_probes()` fires the single QA probe. Set `simulation.agent_during_ingestion=true` to fall back to the legacy per-turn ingestion path (`num_turns = total_sessions + 1`, agent's `run_turn` mirrors one chat session per iteration, probe on the `+1` turn) — useful if the new path regresses on a particular config; PR #6 drops the flag once the production sweep validates parity.

BEAM has the same shape (PR #5 of loop redesign): default `simulation.agent_during_ingestion=false`, `env.ingest_all(log)` bulk-writes every batch into `E`, all M probing questions fire end-of-task via `env.get_end_of_task_probes()`. Set `simulation.agent_during_ingestion=true` for the legacy `num_batches + 1` path.

Output for the orchestrator lands under
`output/longmemeval/<policy>_<seed>_<hash8>/qa_<question_id>/` (per QA) plus
`hypotheses.jsonl` + `summary.json` at the run root.

Common invocations (all flags via `.venv/bin/python scripts/run_longmemeval.py --config configs/longmemeval/scroll.json`):

| Goal | Flags |
|---|---|
| **Full 500 QA** (production sweep) | `--max-parallel 8 --include-abstention` |
| **One specific QA by id** | `--question-ids 32a9c8...` |
| **N specific QAs by id** | `--question-ids id1 id2 id3` (space-separated) |
| **First N items** (any type) | `--limit N` |
| **All items of one type** | `--question-types multi-session` |
| **N items per type** (stratified — recommended for fast sanity) | `--per-type 5` |
| **N items of a few types** | `--question-types multi-session temporal-reasoning --per-type 5` |
| **Parallel across cores** | `--max-parallel 8` (each QA in its own subprocess, ~5s startup overhead per subprocess) |
| **Include abstention items** | `--include-abstention` (default excludes `_abs`-suffixed items; full reproduction includes them) |
| **Force fresh** (ignore checkpoints) | `--fresh` |
| **Custom output root** | `--output-root output/lme_experiment_42` |

The six `question_type` values in the `_s` (500-QA) split:
`knowledge-update` (78), `multi-session` (133), `single-session-assistant` (56),
`single-session-preference` (30), `single-session-user` (70),
`temporal-reasoning` (133). The full 500 includes abstention twins (suffix
`_abs`) for some; `--include-abstention` adds them.

`--limit` and `--per-type` are mutually exclusive. Without either, all items
matching `--question-types` (or the full 500 if no filter) are run.

### Tracing

Both entry points accept `--tracing-url <OTLP-HTTP-endpoint>` to ship spans
to a Phoenix backend. The substrate installs the OpenAI instrumentor and an
OpenInference rewriter that translates AgentScope's `gen_ai.*` spans into
Phoenix-renderable `TOOL` / `AGENT` / `CHAIN` / `LLM` kinds (see
`src/Scroll/_tracing.py`).

```bash
# 1. Start Phoenix locally (one-off; it persists traces in ~/.phoenix)
.venv/bin/phoenix serve              # UI at http://localhost:6006
# Or: docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest

# 2. Add the flag to either entry point.
Scroll --config configs/longmemeval/scroll.json --seed 1 \
       --tracing-url http://localhost:6006/v1/traces

.venv/bin/python scripts/run_longmemeval.py \
       --config configs/longmemeval/scroll.json \
       --per-type 3 \
       --tracing-url http://localhost:6006/v1/traces
```

Open `http://localhost:6006` and pick the project to see the trace tree:
each session's CodeAct loop becomes one chain, with `code_exec`
tool spans, `rlm` sub-agent spans, and `chat` LLM spans nested
underneath. Span attrs include the cell's full code (`cell.code_full`),
stdout preview (`cell.stdout_preview`), and tracked ops
(`cell.ops` — e.g. `ms.sql_exec,rlm`).

## Configs

Each run takes one JSON config under `configs/<env>/`. Schema (matches the
shipped LME production config):

```json
{
  "environment": "longmemeval",        // dispatches to Scroll.benchmarks.<env>
  "simulation": { ... },               // parsed by <env>.parse_env_config
  "agent": {
    "policy": "scroll",                // dispatches to benchmarks/<env>/agents/__init__.create_agent
    "qwen_model_name": "qwen3.7-max",
    "qwen_api_key_env": "CN_DASHSCOPE_API_KEY",
    "qwen_api_base_env": "CN_DASHSCOPE_BASE_URL",
    "enable_thinking": false,          // qwen-family CoT toggle (no-op on qwen3.7-max via compat-mode)
    "thinking_budget": null,           // qwen-native budget in tokens; pairs with enable_thinking
    "max_iters_per_turn": 10,          // cap on CodeAct loop iters inside one session
    "max_output_tokens": 4096,
    "context_max_tokens": 60000,
    "enable_playbook": false,          // LME-only: append distilled playbook block to probe prompt
    "enable_distillation": false       // LME-only: write new procedural hints back after each probe
  },
  "data_sources": {},
  "benchmark": {}
}
```

## Conventions

- **Module names**: env-agnostic substrate lives in `core/` with leading underscore (`_scroll_agent.py`, `_codeact_agent.py`); public classes (`ScrollAgent`, `CodeActAgent`) re-exported from `Scroll.core`.
- **Per-env agent file**: always `benchmarks/<env>/agents/agent.py`, class name `<Env>Agent`.
- **Per-env config files**: `configs/<env>/scroll<_variant>.json` (variant = model, dataset split, ablation tag).
- **Output dirs**: `output/<env>/<policy>_<seed>_<config-hash8>/` for single runs; LongMemEval multi-QA sweeps nest `qa_<question_id>/` under that.
- **Tracing**: optional via `--tracing-url`. Phoenix-compatible OTLP. See `Scroll._tracing` and the CLI `--help`.
- **Single policy per env**: LME and Vending both expose only `"policy": "scroll"` (old `code_auto_v2` / `code_auto` aliases were removed). Adding a new policy means adding a branch to that env's `create_agent`.

## What's intentionally not here

- **Baselines**: the earlier consolidation removed every non-SCROLL agent (basic, code_agent, agentscope_reme, rlm, database_*, adaptive*). Restoring one for paper comparison is a future task; see `docs/scroll.md` → "Why this is publishable" for context.
- **vending/taubench-era scripts**: deleted (`extract_sft_pairs.py`, `generate_sweep_report.py`, sweep_*.sh, etc.). Current `scripts/` contains: `run_longmemeval.py` (multi-QA orchestrator), `run_vending.py` (Docker-parallel sweep for Vending), `shard_lme_dataset.py` (one-time preprocessing), and `test_rlm.py` (RLM wrapper smoke).

## When editing the substrate

The substrate (`core/_codeact_agent.py`, `core/_scroll_agent.py`, `tools/memoryspace.py`, `log.py`) is env-agnostic. Any change there should be tested against all three envs:

```bash
.venv/bin/python -c "
from Scroll.benchmarks.longmemeval.agents import LongMemEvalAgent
from Scroll.benchmarks.vending.agents import VendingAgent
from Scroll.benchmarks.beam.agents import BeamAgent
from Scroll.core import get_env
for eid in ('longmemeval', 'vending', 'beam'):
    print(eid, get_env(eid).env_cls.__name__)
print(LongMemEvalAgent.__mro__[1:4])
"
```

For behavior-parity checks on the LME side, pin the judge model to `gpt-4o-mini` with `temperature=0` and diff `probe_results.json` scores across branches.

## Citation

License is Apache-2.0 (`LICENSE`).
