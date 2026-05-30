# Scroll

**S**ession-as-**C**ontext, **R**ecoverable **O**ff-context **L**LM **L**og.

Scroll is an architecture for long-horizon LLM agents that treats context as an *environment*, not a window. The agent's full history lives outside its prompt as an append-only event log `E` — durable across every session it ever runs — together with a derived memoryspace `W = build(E)` that indexes it for fast access. To use anything off-context, the agent doesn't wait for a retriever: it writes Python in a sandboxed REPL with `log` and `ms` bound, turning retrieval into arbitrary computation it authors on the spot.

This avoids the two compromises every long-running agent makes today. Compaction summarizes older turns and drops the originals once the window fills; memory systems extend reach but pre-commit at design time to *what gets extracted* and *what the retrieval pipeline can surface* — capping recall on any query nobody anticipated when the schema was drawn. Scroll keeps the full log as the single source of truth and lets the agent read it the way a programmer reads a file or queries a database.

Three benchmarks ship with the framework: **LongMemEval** (chat-memory probes), **Vending Machine** (long-horizon planning), and **BEAM** (long-context chat-memory at 100K–10M tokens).

See [`design/scroll.html`](design/scroll.html) for the design rationale and the (E, W, CodeAct) decomposition.

## Headline results

### LongMemEval `_s` split, 500 QA (incl. abstention twins)

| Agent model | Overall | k-update | multi-sess | s-asst | s-pref | s-user | temporal |
|---|---:|---:|---:|---:|---:|---:|---:|
| **scroll · deepseek-v4-pro** | **0.918** | 0.910 | 0.835 | 0.946 | 0.967 | 0.971 | **0.955** |
| scroll · qwen3.7-max | 0.908 | 0.936 | 0.812 | 0.982 | 0.967 | 0.957 | 0.917 |

SCROLL framework (E + W + CodeAct) + qwen3.7-max judge. Configs: [`scroll_qwen37max.json`](configs/longmemeval/scroll_qwen37max.json), [`scroll_deepseek.json`](configs/longmemeval/scroll_deepseek.json).

The same scaffold (prompts, retrieval primitives, REPL substrate) runs across both models — only the agent model name changes in the config. Deepseek-v4-pro wins on TR (+3.8pp) and multi-session (+2.3pp) at the cost of ~3× wall time (75s/QA vs 24s/QA); qwen3.7-max wins on KU (+2.6pp) and assistant (+3.6pp) at lower latency.

### Vending Machine, 180-day run, seed=1

Same agent model, same 180-day horizon, same 60-probe suite. Only difference: memory strategy.

| Agent | probe_score | A (factual recall) | B (multi-step reasoning) |
|---|---:|---:|---:|
| **scroll · qwen3.7-max** | **0.950** | **1.000** (32/32) | **0.893** (25/28) |
| baseline · qwen3.7-max | 0.683 | 0.719 (23/32) | 0.643 (18/28) |

The 60-question probe suite is phrased as realistic stakeholder asks — point-in-time recall, weekly/monthly trends, aggregations, supplier/inventory queries, top-K / abandoned-SKU / price-change detection. A* probes test factual recall (specific old values, non-headline fields, earliest-event lookup); B* probes test multi-step reasoning (paired-event correlation, full-history enumeration, trend comparisons). SCROLL routes off-context reads through `ms.sql_exec` over the persisted event log `E` (precise historical values); the baseline relies on a rolling LLM summary which smooths out fine-grained history.

Configs: [`vending/scroll.json`](configs/vending/scroll.json), [`vending/baseline.json`](configs/vending/baseline.json).

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

# Run the full 500-QA sweep in parallel (the 0.908 number above)
python scripts/run_longmemeval.py \
    --config configs/longmemeval/scroll_qwen37max.json \
    --seed 1 --max-parallel 8 --include-abstention

# Or use deepseek-v4-pro instead (0.918 — slower wall time)
python scripts/run_longmemeval.py \
    --config configs/longmemeval/scroll_deepseek.json \
    --seed 1 --max-parallel 8 --include-abstention

# Vending — 180-day single run (both policies)
Scroll --config configs/vending/scroll.json   --seed 1
Scroll --config configs/vending/baseline.json --seed 1
```

Outputs land under `output/<env>/<policy>_<seed>_<config-hash8>/`:

```
output/longmemeval/scroll_1_<hash>/
├── qa_<question_id>/             # one per LongMemEval QA
│   ├── conversation_log.jsonl    # the event log E
│   ├── probe_results.json        # judge scores + agent answers
│   └── ...
└── hypotheses.jsonl              # aggregate (LME-evaluator format)
```

## Configuration

Each run takes one JSON config file. Skeleton (matches the 0.908 LongMemEval config):

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
    "enable_distillation": false
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
design/scroll.html             design rationale (blog post)
scripts/
├── run_longmemeval.py         multi-QA orchestrator
├── run_vending.py             Docker-parallel sweep (Vending)
├── shard_lme_dataset.py       one-time dataset preprocessing
└── test_rlm.py                RLM wrapper smoke test
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
