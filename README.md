# Scroll

**S**ession-as-**C**ontext, **R**ecoverable **O**ff-context **L**LM **L**og.

A reference implementation of a memory-system pattern for LLM agents on long-horizon tasks: long-term memory lives outside the LLM context as an append-only event log `E` plus a derived memoryspace `W = build(E)`; the agent uses CodeAct (writing Python in a sandboxed REPL with `log` and `ms` bound) to retrieve into context on demand.

Three benchmarks ship with the framework: **LongMemEval** (chat-memory probes), **Vending Machine** (long-horizon planning), and **BEAM** (long-context chat-memory at 100K–10M tokens, [Tavakoli ICLR 2026](https://arxiv.org/abs/2510.27246)).

See [`docs/scroll.md`](docs/scroll.md) for the design rationale and the (E, W, CodeAct) decomposition.

## Headline result

**LongMemEval `_s` split, 500 QA (incl. abstention twins): `acc = 0.906`**
SCROLL framework (E + W + CodeAct) + qwen3.7-max agent + qwen3.6-plus judge. Config: [`configs/longmemeval/scroll_qwen37max.json`](configs/longmemeval/scroll_qwen37max.json). Full provenance and failure analysis: [`output/longmemeval_qwen37max_v4/FAILURE_REPORT.md`](output/longmemeval_qwen37max_v4/FAILURE_REPORT.md).

| Type | Acc | n |
|---|---:|---:|
| single-session-assistant | 0.982 | 56 |
| knowledge-update | 0.974 | 78 |
| single-session-user | 0.971 | 70 |
| temporal-reasoning | 0.917 | 133 |
| single-session-preference | 0.900 | 30 |
| multi-session | 0.789 | 133 |
| **OVERALL** | **0.906** | **500** |

What the framework actually contributes vs the model: a naive `qwen3.6-plus` agent answering the same questions ends at ~0.80; replacing it with `qwen3.7-max` alone only gets to 0.888. The remaining +1.8 pp comes from the SCROLL-side pipeline — per-qtype forcing templates (multi-session count, KU stale-value, temporal-reasoning), a grace-turn rescue against budget-exhausted termination, and a commit-time synthesis block. The breakdown is in the failure report above.

## Install

```bash
git clone --recurse-submodules <repo> Scroll && cd Scroll
uv venv .venv --python 3.11 --prompt Scroll
source .venv/bin/activate
uv pip install -e .
```

If you already cloned without `--recurse-submodules`, pull the submodules in afterwards:

```bash
git submodule update --init --recursive
```

The LongMemEval dataset (`external/longmemeval/data/longmemeval_*.json`) is **not** in the submodule — download it from the [HuggingFace mirror](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) into `external/longmemeval/data/` before running LongMemEval configs.

To keep submodules in sync when pulling, either run `git pull --recurse-submodules` ad-hoc, or set it as the default once:

```bash
git config --global submodule.recurse true
```

Python 3.11+. Tested on macOS arm64; should work on Linux.

## Quickstart

```bash
# LongMemEval — one task to verify the install
Scroll --config configs/longmemeval/scroll_qwen37max.json --seed 1

# Run the full 500-QA sweep in parallel (the 0.906 number above)
python scripts/run_longmemeval.py \
    --config configs/longmemeval/scroll_qwen37max.json \
    --seed 1 --max-parallel 8 --include-abstention

# Vending — 30-day single run
Scroll --config configs/vending/scroll.json --seed 1
```

Outputs land under `output/<env>/<policy>_<seed>_<config-hash8>/`:

```
output/longmemeval/scroll_1_<hash>/
├── qa_<question_id>/             # one per LongMemEval QA
│   ├── conversation_log.jsonl    # the event log E
│   ├── probe_results.json        # judge scores + agent answers
│   └── ...
├── _shared_memoryspace.json        # cross-task procedural_hints
└── hypotheses.jsonl              # aggregate (LME-evaluator format)
```

## Reproducing paper numbers

Each benchmark has a single canonical config and a reproduction script:

```bash
bash scripts/reproduce_longmemeval.sh   # LongMemEval s-split, qwen3.7-max → 0.906
bash scripts/reproduce_vending.sh        # Vending, GPT-5-mini
bash scripts/reproduce_beam.sh           # BEAM, 100K scale, qwen3.6-plus
```

## Configuration

Each run takes one JSON config file. Skeleton (matches the 0.906 LongMemEval config):

```json
{
  "environment": "longmemeval",
  "simulation": {
    "dataset_path": "external/longmemeval/data/longmemeval_s_cleaned.json",
    "judge_model": "qwen3.6-plus",
    "judge_api_key_env": "US_DASHSCOPE_API_KEY",
    "judge_api_base": "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
  },
  "agent": {
    "policy": "scroll",
    "qwen_model_name": "qwen3.7-max",
    "qwen_api_key_env": "CN_DASHSCOPE_API_KEY",
    "qwen_api_base_env": "CN_DASHSCOPE_BASE_URL",
    "enable_thinking": false,
    "max_iters_per_turn": 10,
    "max_output_tokens": 4096,
    "context_max_tokens": 60000,
    "enable_playbook": false,
    "enable_distillation": true
  }
}
```

The `agent.policy` field selects the per-env agent class. Both LongMemEval and Vending currently expose a single policy: `"scroll"`.

### Provider routing

Set the right env vars before running:

```bash
export US_DASHSCOPE_API_KEY=sk-...
export US_DASHSCOPE_BASE_URL=https://dashscope-us.aliyuncs.com/compatible-mode/v1
```

A `.env` file at repo root is sourced automatically.

## Tracing (optional)

`arize-phoenix` is already installed by `uv pip install -e .`, so the
`phoenix` CLI is on the venv path — no extra install step. Start a local
Phoenix backend either way:

```bash
# Option A: local process (uses the venv's phoenix CLI)
.venv/bin/phoenix serve                              # UI at http://localhost:6006

# Option B: Docker
docker run -p 6006:6006 arizephoenix/phoenix:latest

# Then point a run at it:
Scroll --config configs/longmemeval/scroll_s.json --seed 1 \
    --tracing-url http://localhost:6006/v1/traces
```

Then open `http://localhost:6006` to see per-turn LLM spans, code-cell spans, and RLM sub-agent spans.

## Repo layout

```
src/Scroll/                    # framework
├── core/                      # abstractions (BaseEnvironment, ScrollAgent, …)
│   ├── _scroll_agent.py       ScrollAgent base class
│   ├── _codeact_agent.py      CodeAct substrate
│   ├── _rlm.py                dspy.RLM helper
│   ├── _environment.py        env ABC
│   └── _registry.py           env auto-discovery
├── tools/                     # REPL primitives (Memoryspace, LogHandle, sub_llm)
├── log.py                     ConversationLog (E)
├── benchmark.py               run loop (run_single, _run_session_loop)
├── cli.py                     CLI entry point
├── _tracing.py                OTel setup
└── benchmarks/                # plug-in benchmarks (each = one env)
    ├── longmemeval/
    │   ├── agents/agent.py   LongMemEvalAgent
    │   ├── env.py
    │   └── tasks/probes.py
    ├── vending/
    │   ├── agents/agent.py   VendingAgent
    │   ├── env.py
    │   └── auto_ingest.py
    └── beam/
        ├── agents/agent.py   BeamAgent
        ├── env.py
        ├── dataset.py
        └── tasks/probes.py

configs/<env>/                 one JSON per (model, variant)
docs/scroll.md                 design doc
scripts/
├── run_longmemeval.py         multi-QA orchestrator
├── run_parallel.py            Docker-parallel sweep
├── shard_lme_dataset.py       one-time dataset preprocessing
└── test_rlm.py                RLM wrapper smoke test
```

## Adding a new benchmark

The env contract is in [`src/Scroll/core/_environment.py`](src/Scroll/core/_environment.py). To add an env:

1. Create `src/Scroll/benchmarks/<env>/__init__.py` exporting `ENV_ID`, `ENV_CLS`, `DATASOURCE_CLS`, `parse_env_config`, and `create_agent`.
2. Implement `<env>/env.py:<Env>(BaseEnvironment)` with `visible_state` / `step_session` / `build_snapshot` (and optionally `today_logs` / `net_worth` / `is_terminal`).
3. Implement `<env>/datasource.py:<DataSource>(BaseDataSource)` (only `begin_session` is required; other channels have no-op defaults).
4. Implement `<env>/tasks/probes.py` with `PROBES`, `get_probes_for_session`, and (env-specific) scoring.
5. Write `<env>/agents/agent.py:<Env>Agent(ScrollAgent)` providing `_ensure_schema`, `_ingest_context` / `_ingest_outcomes`, and `session_prompt`.
6. Drop one config under `configs/<env>/scroll.json` with `"environment": "<env>"`.

The registry resolves `"environment": "<env>"` to `Scroll.benchmarks.<env>` automatically. No changes to `benchmark.py` or `core/` should be needed.

## Citation

```bibtex
@misc{scroll2026,
  title  = {{SCROLL}: Session-as-Context, Recoverable Off-context LLM Log},
  author = {<authors>},
  year   = {2026},
  note   = {Code: \url{<repo>}}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
