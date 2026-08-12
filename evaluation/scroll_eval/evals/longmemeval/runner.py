"""Native (non-Harbor) runner for the LongMemEval memory benchmark.

For each generated instance task (``local-tasks/longmemeval/<qid>/``):
  1. Ingest ``sessions.json`` (the prior chat sessions) into a per-task **seed DB**,
     then copy it once into the chat's single **history DB**.
  2. Run the scroll agent in its own session against that DB to answer the ONE
     question: build a ``LoopContext`` pointed at it (with
     ``shared_run_ids=(SEED_RUN_ID,)`` so the agent retrieves the seeded prior
     conversation plus its own turns), and ``await agent.run(...)``. The probe's
     ``trajectory.json`` is written to its log dir.
  3. Grade the collected ``answers.json`` with the LongMemEval judge (subprocess,
     Harbor reward contract) → ``scores.json``.
  4. Aggregate ``summary.json`` + ``manifest.json`` (mirroring the beam runner so
     ``scroll-eval summary`` works).

This deliberately mirrors ``evals/beam/runner.py``; the difference is that a
LongMemEval task carries exactly one probe (its single QA), and the per-probe
message uses the LongMemEval question framing + per-qtype guidance.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from scroll_eval._tools_common import select_tools
from scroll_eval.evals.longmemeval.ingest import SEED_RUN_ID
from scroll_eval.harness.config import RunConfig
from scroll_eval.harness.runner import (
    _LOCAL_TASKS_ROOT,
    _list_all_tasks,
    _load_dotenv,
    _model_api_key_env,
    _slug,
    _uses_dashscope,
    _validate_keys,
)
from scroll_eval.runner import _build_agentscope_model, _trajectory_json
from scroll_eval.tracing import otel
from scroll_eval.types import LoopContext, TaskSpec, Trajectory

_DATASET = "longmemeval"
_DEFAULT_TOOLS = ["execute_python", "submit_answer"]  # no bash: LME needs no shell
_DEFAULT_CONCURRENCY = 4  # tasks each carry one probe; the shared cap bounds fan-out
_DEFAULT_JUDGE_WORKERS = 8


# The base user-turn nudge appended to every LongMemEval question (ported from the
# original Scroll LME prompts, with the ``rlm`` reference dropped since scroll_react
# does not bind it).
_BASE_POSTSCRIPT = (
    "Write Python cells with `execute_python` to query and combine evidence "
    "(`ms.search(scope=\"task\")`, `ms.sql_query`, `ms.expand`) — whatever you "
    "need to gather it. Commit by calling `submit_answer` with a plain-text answer "
    "once you have enough. If the search genuinely returns nothing, abstain "
    "explicitly: \"I don't have that information from our conversations\" — the "
    "judge scores that correct."
)

# Per-question-type nudges, layered on top of _BASE_POSTSCRIPT. Adapted from the
# original Scroll LME postscripts to scroll_context's memory API (FTS `ms.search`
# + `metadata.date` on `hist.conversation_history`; no `session_ts_iso`, no vector
# store). Keyed on the probe ``type``; abstention overrides the qtype nudge.
_QTYPE_POSTSCRIPT: dict[str, str] = {
    "temporal-reasoning": (
        "Time-sensitive question. Pattern: work out the date range the question "
        "implies first, then filter on `json_extract(metadata,'$.date')` (BETWEEN / "
        "ORDER BY) BEFORE reading content — don't keyword-scan the whole haystack. "
        "Off-by-one on day/week/month is not penalized."
    ),
    "knowledge-update": (
        "Recency-sensitive question. The user may have stated multiple values over "
        "time; the MOST RECENT statement is the current truth. Pattern: order the "
        "matching turns by `json_extract(metadata,'$.date')` (or `msg_index`) "
        "descending and take the latest value. Mentioning prior values is fine, but "
        "do not state an out-of-date value as current."
    ),
    "single-session-preference": (
        "Preference question. The literal subject may never have been discussed "
        "verbatim — that is the point. An exact keyword/SQL match will miss; let "
        "`ms.search(scope=\"task\")` broaden across your terms to surface related "
        "preferences (likes, dislikes, constraints, recurring interests) and ground "
        "the recommendation in those. \"I have no information about <topic>\" is the "
        "wrong frame here."
    ),
}

_ABSTENTION_POSTSCRIPT = (
    "This may be an unanswerable question — the user may never have stated the "
    "relevant information. After ≥2 distinct retrieval queries (different keywords "
    "AND different surfaces: `ms.search` and `ms.sql_query`) return nothing useful, "
    "abstain EXPLICITLY using the phrasing: \"I don't have that information from our "
    "conversations.\" The judge requires explicit refusal to score abstention "
    "correct. A similar-but-different entity is NOT a match — 'table tennis' ≠ "
    "'tennis', 'Sales Manager' ≠ 'Sales Engineer', 'Shinjuku' ≠ 'Harajuku'. Abstain "
    "rather than substitute the closest match."
)


def _instruction_for(probe: dict) -> str:
    """The full user message for a LongMemEval probe: framing + qtype nudge.

    Mirrors the original Scroll ``_build_probe`` question framing (the "As of
    <date> …" preamble + grounding note), then appends the base nudge and either
    the abstention or the per-qtype postscript.
    """
    qdate = probe.get("question_date") or "now"
    is_abstention = "_abs" in str(probe.get("id", ""))
    framing = (
        f"As of {qdate}, please answer the following based on everything you've "
        f"observed across the chat sessions so far.\n\n"
        f"Question: {probe['question']}\n\n"
        "Answer from what you find in the chat. Connecting two stated facts to reach "
        "a third is fine (e.g. 'user uses Cartwheel app' + 'Cartwheel = Target' → "
        "'redeemed at Target') — that's reading context, not fabricating. Only "
        "abstain when the chat truly contains nothing relevant; do NOT abstain just "
        "because the answer requires one inferential step from stated facts."
    )
    parts = [framing, _BASE_POSTSCRIPT]
    if is_abstention:
        parts.append(_ABSTENTION_POSTSCRIPT)
    elif probe.get("type") in _QTYPE_POSTSCRIPT:
        parts.append(_QTYPE_POSTSCRIPT[probe["type"]])
    return "\n\n".join(parts)


def _prepare_env(cfg: RunConfig) -> None:
    """Set the env the LLM clients read (mirrors harness/runner._build_env)."""
    _validate_keys(cfg)
    os.environ["OPENAI_BASE_URL"] = cfg.model.endpoint
    os.environ["SCROLL_MODEL"] = cfg.model.name
    # Prior sessions are seeded into each probe DB — surface them as an in-context
    # memory map. setdefault so an explicit override (SCROLL_SEED_INDEX=0) still wins.
    os.environ.setdefault("SCROLL_SEED_INDEX", "1")
    key_env = _model_api_key_env(cfg)
    if key_env and os.environ.get(key_env):
        os.environ.setdefault("OPENAI_API_KEY", os.environ[key_env])
        if _uses_dashscope(cfg):
            os.environ.setdefault("DASHSCOPE_API_KEY", os.environ[key_env])


def _resolve_tasks(cfg: RunConfig, tasks: list[str] | None) -> list[str]:
    if tasks:
        return tasks
    if cfg.tasks == "all":
        return _list_all_tasks(_DATASET)
    return list(cfg.tasks)


async def _answer_probe(
    *,
    agent_run,
    probe: dict,
    history_db: Path,
    probe_dir: Path,
    system_prompt: str,
    task_id: str,
    run_id: str,
    cfg: RunConfig,
    llm_openai,
    llm_agentscope,
    tracer,
    tools: list[dict],
) -> dict:
    """Run one probe against the chat's history DB; return its answer dict."""
    probe_dir.mkdir(parents=True, exist_ok=True)

    ctx = LoopContext(
        llm_openai=llm_openai,
        llm_agentscope=llm_agentscope,
        model_name=cfg.model.name,
        tracer=tracer,
        budget=cfg.budget,
        environment=None,
        tools=tools,
        run_id=run_id,
        history_db_path=str(history_db),
        history_max_tokens=cfg.memory.history_max_tokens,
        logs_dir=str(probe_dir),
        system_prompt=system_prompt,        # standing guidance lives here, once
        shared_run_ids=(SEED_RUN_ID,),      # seed tier shared; probe turns stay private
    )
    task = TaskSpec(task_id=task_id, instruction=_instruction_for(probe))

    with otel.task_run(
        tracer,
        task_id=task_id,
        agent=f"{cfg.agent.type}/{cfg.agent.id}",
        model=cfg.model.name,
        run_id=run_id,
    ):
        traj: Trajectory = await agent_run(task, ctx)

    (probe_dir / "trajectory.json").write_text(_trajectory_json(traj), encoding="utf-8")
    return {
        "id": probe["id"],
        "type": probe["type"],
        "question": probe["question"],
        "llm_response": traj.final_answer or "",
    }


async def _run_probes(probes, **kw) -> list[dict]:
    """Run a task's probes concurrently (bounded by ``concurrency``).

    A LongMemEval task has one probe, but keep the concurrent structure so a
    multi-probe task (or a future variant) still works; ``asyncio.gather``
    preserves input order so the judge aligns answers by type + index.
    """
    concurrency = max(1, int(kw.get("concurrency", _DEFAULT_CONCURRENCY)))
    sem = asyncio.Semaphore(concurrency)

    async def _one(i: int, probe: dict) -> dict:
        probe_dir = kw["task_out"] / "probes" / _slug(probe["type"]) / str(i)
        async with sem:
            try:
                ans = await _answer_probe(
                    agent_run=kw["agent_run"],
                    probe=probe,
                    history_db=kw["history_db"],
                    probe_dir=probe_dir,
                    system_prompt=kw["system_prompt"],
                    task_id=kw["task_id"],
                    run_id=f"{kw['label']}:{kw['name']}:{probe['id']}",
                    cfg=kw["cfg"],
                    llm_openai=kw["llm_openai"],
                    llm_agentscope=kw["llm_agentscope"],
                    tracer=kw["tracer"],
                    tools=kw["tools"],
                )
            except Exception as e:  # noqa: BLE001 - isolate one probe's failure
                print(f"      {probe['id']}: ERROR {e}", flush=True)
                return {"id": probe["id"], "type": probe["type"],
                        "question": probe["question"], "llm_response": ""}
        print(f"      {probe['id']}: {len(ans['llm_response'])} chars", flush=True)
        return ans

    return list(await asyncio.gather(*[_one(i, p) for i, p in enumerate(probes)]))


def _group_by_type(answers: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for a in answers:
        grouped.setdefault(a["type"], []).append(
            {"id": a["id"], "question": a["question"], "llm_response": a["llm_response"]}
        )
    return grouped


def _judge(task_dir: Path, task_out: Path, workers: int = _DEFAULT_JUDGE_WORKERS) -> dict:
    """Invoke the LongMemEval judge as a subprocess (Harbor reward contract)."""
    scores_path = task_out / "scores.json"
    cmd = [
        sys.executable, "-m", "scroll_eval.evals.longmemeval.judge",
        "--questions", str(task_dir / "tests" / "probing_questions.json"),
        "--answers", str(task_out / "answers.json"),
        "--out", str(scores_path),
        "--reward-file", str(task_out / "reward.txt"),
        "--workers", str(workers),
    ]
    subprocess.run(cmd, check=True, env=os.environ.copy())
    return json.loads(scores_path.read_text(encoding="utf-8"))


async def _run_task_async(
    cfg: RunConfig,
    name: str,
    task_out: Path,
    label: str,
    *,
    agent_run,
    llm_openai,
    llm_agentscope,
    tracer,
    tools: list[dict],
    system_prompt: str,
    index: int,
    total: int,
    concurrency: int = _DEFAULT_CONCURRENCY,
    judge_workers: int = _DEFAULT_JUDGE_WORKERS,
) -> dict:
    task_dir = _LOCAL_TASKS_ROOT / _DATASET / name
    if not (task_dir / "task.toml").exists():
        raise FileNotFoundError(f"not a longmemeval task dir: {task_dir}")
    task_out.mkdir(parents=True, exist_ok=True)
    task_id = f"{_DATASET}/{name}"

    print(f"[{index}/{total}] {name}: ingesting...", flush=True)
    from scroll_eval.evals.longmemeval.ingest import build_seed_db_for_task

    seed_index = os.environ.get("SCROLL_SEED_INDEX", "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    seed_db = task_out / "seed.db"
    n_turns = build_seed_db_for_task(task_dir, task_id, seed_db, seed_index=seed_index)
    history_db = task_out / "history.db"
    shutil.copyfile(seed_db, history_db)

    probes = json.loads((task_dir / "questions.json").read_text(encoding="utf-8"))
    print(f"[{index}/{total}] {name}: {n_turns} turns seeded; {len(probes)} probe(s)", flush=True)

    t0 = time.monotonic()
    answers = await _run_probes(
        probes,
        agent_run=agent_run,
        history_db=history_db,
        system_prompt=system_prompt,
        task_id=task_id,
        label=label,
        name=name,
        cfg=cfg,
        llm_openai=llm_openai,
        llm_agentscope=llm_agentscope,
        tracer=tracer,
        tools=tools,
        task_out=task_out,
        concurrency=concurrency,
    )

    (task_out / "answers.json").write_text(
        json.dumps(_group_by_type(answers), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    scores = _judge(task_dir, task_out, workers=judge_workers)
    elapsed = time.monotonic() - t0
    reward = scores["overall_reward"]
    print(f"[{index}/{total}] {name}: reward={reward:.4f} elapsed={elapsed:.0f}s", flush=True)
    return {
        "score": reward,
        "n_probes": scores["n_probes"],
        "per_type": {t: d["mean"] for t, d in scores["per_type"].items()},
    }


def _run_task(cfg: RunConfig, name: str, task_out: Path, label: str, **kw) -> dict:
    """Sync wrapper around :func:`_run_task_async` (single task / tests)."""
    return asyncio.run(_run_task_async(cfg, name, task_out, label, **kw))


async def _run_all(
    cfg: RunConfig,
    task_names: list[str],
    label: str,
    run_dir: Path,
    *,
    agent_run,
    llm_openai,
    llm_agentscope,
    tracer,
    tools: list[dict],
    system_prompt: str,
    concurrency: int,
    judge_workers: int,
) -> dict:
    """Run every task under one event loop so the async LLM client is reused."""
    summary: dict[str, dict] = {}
    total = len(task_names)
    for i, name in enumerate(task_names, start=1):
        try:
            summary[name] = await _run_task_async(
                cfg, name, run_dir / "tasks" / _slug(name), label,
                agent_run=agent_run, llm_openai=llm_openai, llm_agentscope=llm_agentscope,
                tracer=tracer, tools=tools, system_prompt=system_prompt,
                index=i, total=total, concurrency=concurrency, judge_workers=judge_workers,
            )
        except Exception as e:  # noqa: BLE001 - one bad task shouldn't kill the run
            print(f"[{i}/{total}] {name}: ERROR {e}", flush=True)
            summary[name] = {"error": str(e)}

    import gc

    gc.collect()
    await asyncio.sleep(0.25)
    return summary


def run(
    cfg: RunConfig,
    runs_root: Path = Path("runs"),
    tasks: list[str] | None = None,
    verbose: bool = False,
    concurrency: int = _DEFAULT_CONCURRENCY,
    judge_workers: int = _DEFAULT_JUDGE_WORKERS,
) -> Path:
    """Run the LongMemEval eval for ``cfg`` over the resolved tasks. Returns the run dir."""
    _load_dotenv()
    _prepare_env(cfg)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    label = f"{timestamp}__longmemeval_{_slug(cfg.agent.id)}__{_slug(cfg.model.name)}__{_slug(cfg.trace.phoenix_project)}"
    run_dir = runs_root / label
    (run_dir / "tasks").mkdir(parents=True, exist_ok=True)

    task_names = _resolve_tasks(cfg, tasks)
    if not task_names:
        raise ValueError(
            "no longmemeval tasks to run (empty `tasks` and none under "
            "local-tasks/longmemeval; generate them with scripts/gen_longmemeval_tasks.py)"
        )

    bare = cfg.model.name.split("/", 1)[-1] if "/" in cfg.model.name else cfg.model.name
    llm_openai = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or "",
        base_url=os.environ.get("OPENAI_BASE_URL"),
        max_retries=int(os.environ.get("SCROLL_LLM_MAX_RETRIES", "6")),
        timeout=float(os.environ.get("SCROLL_LLM_TIMEOUT_S", "120")),
    )
    llm_agentscope = _build_agentscope_model(
        bare, thinking=cfg.model.thinking, thinking_budget=cfg.model.thinking_budget
    )
    tracer = otel.init_for_phoenix(phoenix_project=cfg.trace.phoenix_project or None)
    tools = select_tools(cfg.tools if cfg.tools else _DEFAULT_TOOLS)

    agent_mod = importlib.import_module(f"scroll_eval.{cfg.agent.type}.{cfg.agent.id}.agent")
    if not hasattr(agent_mod, "run"):
        raise AttributeError(f"{agent_mod.__name__} must export async def run(task, ctx)")
    agent_run = agent_mod.run

    from scroll_eval.evals.longmemeval import prompts as lme_prompts

    system_prompt = lme_prompts.load("system")

    print(
        f"Running LongMemEval on {len(task_names)} task(s) with "
        f"{cfg.agent.type}/{cfg.agent.id} (task concurrency={concurrency})",
        flush=True,
    )
    summary = asyncio.run(
        _run_all(
            cfg, task_names, label, run_dir,
            agent_run=agent_run, llm_openai=llm_openai, llm_agentscope=llm_agentscope,
            tracer=tracer, tools=tools, system_prompt=system_prompt,
            concurrency=concurrency, judge_workers=judge_workers,
        )
    )

    scored = [e["score"] for e in summary.values() if "score" in e]
    manifest = {
        "benchmark": "longmemeval",
        "agent": {"type": cfg.agent.type, "id": cfg.agent.id},
        "model": {"endpoint": cfg.model.endpoint, "name": cfg.model.name},
        "judge": {"name": os.environ.get("SCROLL_JUDGE_MODEL") or cfg.model.name},
        "memory": {"history_max_tokens": cfg.memory.history_max_tokens},
        "tasks": task_names,
        "mean_reward": (sum(scored) / len(scored)) if scored else None,
        "timestamp_utc": timestamp,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"LongMemEval run complete: {run_dir}  mean_reward={manifest['mean_reward']}", flush=True)
    return run_dir
