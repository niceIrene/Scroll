"""Native (non-Harbor) runner for the BEAM memory benchmark.

For each migrated conversation task (``local-tasks/beam/<name>/``):
  1. Ingest ``chat.json`` into a per-task **seed DB** (prior sessions in memory),
     then copy it once into the chat's single **shared history DB**.
  2. For each probing question, run the scroll agent in its own session against
     that one shared DB: build a ``LoopContext`` pointed at it (with
     ``shared_run_ids=(SEED_RUN_ID,)`` so the agent only retrieves the seed tier
     plus its own turns — never a sibling probe's), and ``await agent.run(...)``
     so the agent answers from memory/retrieval. Each probe's
     ``trajectory.json`` is written to its log dir.
  3. Grade the collected ``answers.json`` with the BEAM judge (subprocess,
     Harbor reward contract) → ``scores.json``.
  4. Aggregate ``summary.json`` + ``manifest.json`` (mirroring the Harbor runner
     so ``scripts/analyze_run.py`` / ``scroll-eval summary`` work).

This deliberately bypasses Harbor: no sandbox, no ``harbor run``. The
conversation is the agent's long-term memory, not a file it reads.
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
from scroll_eval.evals.beam.ingest import SEED_RUN_ID
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

_DATASET = "beam"
_DEFAULT_TOOLS = ["execute_python", "submit_answer"]  # no bash: BEAM needs no shell
_DEFAULT_CONCURRENCY = 4  # probes run concurrently within a task, bounded by this
_DEFAULT_JUDGE_WORKERS = 8  # probes graded concurrently by the judge subprocess


# Per-qtype guidance appended to the probe question: a concise GUIDING PRINCIPLE
# for answering this kind of question well, plus its common failure mode — NOT a
# recipe, and deliberately free of any benchmark-specific detail (no schema,
# query, scoring, budget, or dataset hints). This is the only place the kind is
# conveyed (the system prompt does not enumerate the types) and the wording never
# names the `type` to the model. Keyed on the probe `type`; an unknown type just
# gets the raw question.
_QTYPE_POSTSCRIPT: dict[str, str] = {
    "information_extraction": (
        "Recall the ONE specific fact asked for and report it EXACTLY as stated — "
        "the precise value together with the attribute or purpose it was tied to. "
        "Don't approximate or generalise; if several similar values exist, give the "
        "one matching exactly what's asked."
    ),
    "instruction_following": (
        "The user earlier set a STANDING INSTRUCTION on how to answer this kind of "
        "request; that governing requirement lives in the prior conversation, not "
        "in this question. Find it first, then answer so both its form and its "
        "content comply — substance alone, ignoring the instruction, is not enough."
    ),
    "abstention": (
        "This may have no answer in the conversation. The question's qualifiers are "
        "binding: it asks for a specific thing under specific conditions. If, after "
        "genuinely searching, no turn states that thing under the exact conditions "
        "asked, the information isn't there — say so plainly. The same subject "
        "described under DIFFERENT conditions does NOT answer it, and a merely "
        "related topic is NOT the answer either. An explicit 'not enough "
        "information' is itself correct; never generalise from a near-miss or invent "
        "specifics to fill the gap."
    ),
    "contradiction_resolution": (
        "The conversation says conflicting things about this. A correct answer must "
        "both ACKNOWLEDGE the conflict and RESOLVE it — the user's own correction or "
        "their latest statement normally wins. Avoid a flat yes/no that hides the "
        "contradiction or commits to the superseded side."
    ),
    "event_ordering": (
        "List the events in the ORDER they happened. If a number is asked, give that "
        "many DISTINCT events spread across the whole period — different topics, not "
        "several facets of one thread, and not only the earliest. Both which events "
        "you choose and their order matter."
    ),
    "knowledge_update": (
        "This value CHANGED over time, so the answer is the most RECENT one. The "
        "first mention is likely outdated — gather the mentions, order them, and "
        "commit to the latest as current without calling it old. Don't report a "
        "superseded value or claim 'no change' when a later one exists."
    ),
    "multi_session_reasoning": (
        "The answer isn't in any single turn — gather the separate facts it depends "
        "on, then chain or compute them. Retrieve every required piece before "
        "reasoning; a partial answer from one place fails. For a count, identify "
        "each genuine instance and avoid both over-counting near-misses and "
        "under-counting distinct ones."
    ),
    "preference_following": (
        "The user established a PREFERENCE earlier — a chosen tool, version, style, "
        "or constraint — that your answer must respect, even if it isn't restated "
        "in this question. Surface that prior choice and ground your answer in it "
        "rather than giving generic advice. If you can't find an explicit "
        "preference statement, it may simply not be phrased as one: a preference is "
        "often carried implicitly by a concrete value the user gave in their own "
        "example (a number, a version, a tool). So rather than hunting indefinitely, "
        "check whether something you've already retrieved is the relevant choice for "
        "what's being asked — and if it clearly fits, ground your answer in it."
    ),
    "summarization": (
        "Give a faithful, concise synthesis where COVERAGE matters — span the main "
        "components, stages, and alternatives weighed, not just the easiest-to-"
        "recall thread. Outline the whole arc first, then add only the specifics "
        "worth quoting."
    ),
    "temporal_reasoning": (
        "This turns on timing. Resolve the referenced time precisely (a named "
        "deadline is a specific date, not a vague window) and anchor BOTH endpoints "
        "on the correct events before computing any gap. Distinguish when something "
        "was SAID from when it is DUE or happened; the usual error is anchoring on "
        "the wrong date, not the arithmetic."
    ),
}


def _prepare_env(cfg: RunConfig) -> None:
    """Set the env the LLM clients read (mirrors harness/runner._build_env)."""
    _validate_keys(cfg)
    os.environ["OPENAI_BASE_URL"] = cfg.model.endpoint
    os.environ["SCROLL_MODEL"] = cfg.model.name
    # BEAM seeds prior sessions into each probe DB — surface them to the agent as
    # an in-context memory map. setdefault so an explicit override (e.g. the
    # seed-index ablation, SCROLL_SEED_INDEX=0) still wins.
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
    """Run one probe against the chat's shared history DB; return its answer dict.

    Every probe of a chat reads/writes the SAME ``history_db`` (built once from
    the seed). Isolation is by retrieval scope, not a private file copy: the
    agent gets ``shared_run_ids=(SEED_RUN_ID,)``, so ``ms.search(scope='task')``
    returns the seeded prior conversation plus this probe's own turns and never a
    sibling probe's. The probe's writes go under its unique ``run_id`` /
    ``session_id``, so the shared DB stays correctly partitioned.
    """
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
        system_prompt=system_prompt,        # the repetitive guidance lives here, once
        shared_run_ids=(SEED_RUN_ID,),      # seed tier shared; probe turns stay private
    )
    # The per-probe user message is the question plus a short qtype EXPECTATION
    # (the bar a correct answer must clear, not a recipe). The bulk of standing
    # guidance still lives in the (cacheable, identical) system prompt above;
    # only this small, type-specific postscript varies per probe.
    instruction = probe["question"]
    postscript = _QTYPE_POSTSCRIPT.get(probe["type"])
    if postscript:
        instruction = f"{instruction}\n\n{postscript}"
    task = TaskSpec(task_id=task_id, instruction=instruction)

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

    Each probe is independent — its own seed-DB copy and log dir — so they run
    in parallel under a semaphore. ``asyncio.gather`` preserves input order, so
    the returned answers still align position-by-position with ``probes`` (the
    judge matches answers to rubrics by type + index). A single probe that
    raises is isolated: it yields an empty answer instead of failing the task.
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


def _judge(
    task_dir: Path,
    task_out: Path,
    workers: int = _DEFAULT_JUDGE_WORKERS,
) -> dict:
    """Invoke the BEAM judge as a subprocess (Harbor reward contract)."""
    scores_path = task_out / "scores.json"
    cmd = [
        sys.executable,
        "-m",
        "scroll_eval.evals.beam.judge",
        "--questions",
        str(task_dir / "tests" / "probing_questions.json"),
        "--answers",
        str(task_out / "answers.json"),
        "--out",
        str(scores_path),
        "--reward-file",
        str(task_out / "reward.txt"),
        "--workers",
        str(workers),
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
        raise FileNotFoundError(f"not a beam task dir: {task_dir}")
    task_out.mkdir(parents=True, exist_ok=True)
    task_id = f"{_DATASET}/{name}"

    print(f"[{index}/{total}] {name}: ingesting...", flush=True)
    from scroll_eval.evals.beam.ingest import build_seed_db_for_task

    # Build the pristine seed once, then stamp the chat's single shared history
    # DB from it. Every probe of this chat reads/writes that one file; isolation
    # is by retrieval scope (shared_run_ids=SEED_RUN_ID), not a per-probe copy.
    # The seed-index ablation (SCROLL_SEED_INDEX=0, set by `--no-index`) leaves the
    # `headline` column NULL so the index data is absent from the DB, not merely
    # hidden from the prompt — mirroring the agent's own `seed_on` read. By this
    # point `_prepare_env` has already applied its "1" default.
    seed_index = os.environ.get("SCROLL_SEED_INDEX", "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    seed_db = task_out / "seed.db"
    n_turns = build_seed_db_for_task(task_dir, task_id, seed_db, seed_index=seed_index)
    history_db = task_out / "history.db"
    shutil.copyfile(seed_db, history_db)

    probes = json.loads((task_dir / "questions.json").read_text(encoding="utf-8"))
    print(f"[{index}/{total}] {name}: {n_turns} turns seeded; {len(probes)} probes", flush=True)

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
    """Sync wrapper around :func:`_run_task_async` (single task / tests).

    The full ``run()`` drives all tasks under one event loop instead — do not
    call this per-task in a loop, or each call would open/close its own loop and
    orphan the shared async LLM client (``RuntimeError: Event loop is closed``).
    """
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

    # Drain async LLM-client finalizers while the loop is still alive, so their
    # aclose() doesn't fire after the loop closes ("Event loop is closed").
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
    """Run the BEAM eval for ``cfg`` over the resolved tasks. Returns the run dir."""
    _load_dotenv()
    _prepare_env(cfg)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    label = f"{timestamp}__beam_{_slug(cfg.agent.id)}__{_slug(cfg.model.name)}__{_slug(cfg.trace.phoenix_project)}"
    run_dir = runs_root / label
    (run_dir / "tasks").mkdir(parents=True, exist_ok=True)

    task_names = _resolve_tasks(cfg, tasks)
    if not task_names:
        raise ValueError("no beam tasks to run (empty `tasks` and none under local-tasks/beam)")

    # Shared model clients + tracer (built once; the agent reads ctx.llm_agentscope).
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

    # The standing BEAM guidance lives in ONE system prompt (deduplicated from
    # the per-probe user message); the agent receives it via ctx.system_prompt.
    from scroll_eval.evals.beam import prompts as beam_prompts

    system_prompt = beam_prompts.load("system")

    print(
        f"Running BEAM on {len(task_names)} task(s) with {cfg.agent.type}/{cfg.agent.id} "
        f"(probe concurrency={concurrency})",
        flush=True,
    )
    # One event loop for the whole run: the AgentScope/httpx async client is
    # created once and reused across all probes. Calling asyncio.run() per task
    # would close the loop under the still-open client and raise
    # "Event loop is closed" from its connection-pool finalizers.
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
        "benchmark": "beam",
        "agent": {"type": cfg.agent.type, "id": cfg.agent.id},
        "model": {"endpoint": cfg.model.endpoint, "name": cfg.model.name},
        # The grader: SCROLL_JUDGE_MODEL when set, else the agent model.
        "judge": {"name": os.environ.get("SCROLL_JUDGE_MODEL") or cfg.model.name},
        "memory": {"history_max_tokens": cfg.memory.history_max_tokens},
        "tasks": task_names,
        "mean_reward": (sum(scored) / len(scored)) if scored else None,
        "timestamp_utc": timestamp,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"BEAM run complete: {run_dir}  mean_reward={manifest['mean_reward']}", flush=True)
    return run_dir


def grade_run(
    run_dir: Path,
    *,
    judge_workers: int = _DEFAULT_JUDGE_WORKERS,
    verbose: bool = False,
) -> Path:
    """Re-grade an existing run dir in place, without re-running the agent.

    Reads each task's existing ``answers.json``, re-invokes the judge (honoring
    ``SCROLL_JUDGE_MODEL``), and rebuilds ``scores.json``/``reward.txt`` plus
    the run-level ``summary.json`` and ``manifest.json`` — recomputing
    ``mean_reward`` and stamping the judge model. Use it to re-score a run with a
    different/cheaper judge, or to recover a run whose grading died midway.
    """
    _load_dotenv()
    run_dir = Path(run_dir)
    tasks_root = run_dir / "tasks"
    if not tasks_root.is_dir():
        raise FileNotFoundError(f"not a run dir (no tasks/ under {run_dir})")

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    orig_model = (manifest.get("model") or {}).get("name") or ""
    orig_endpoint = (manifest.get("model") or {}).get("endpoint") or ""

    # Fall back to the run's original model/endpoint so an un-overridden re-grade
    # reproduces the first grading instead of failing on an unset judge model.
    if not os.environ.get("SCROLL_JUDGE_MODEL") and orig_model:
        os.environ["SCROLL_JUDGE_MODEL"] = orig_model
    if not os.environ.get("OPENAI_BASE_URL") and orig_endpoint:
        os.environ["OPENAI_BASE_URL"] = orig_endpoint
    judge_name = os.environ.get("SCROLL_JUDGE_MODEL") or orig_model

    # Prefer the manifest's recorded task list; else discover from the tree.
    names = manifest.get("tasks") or sorted(p.name for p in tasks_root.iterdir() if p.is_dir())
    print(f"Re-grading {run_dir} ({len(names)} task(s)) with judge={judge_name or '(unset)'}", flush=True)

    summary: dict[str, dict] = {}
    for i, name in enumerate(names, start=1):
        task_out = tasks_root / _slug(name)
        if not (task_out / "answers.json").exists():
            print(f"[{i}/{len(names)}] {name}: no answers.json, skipping", flush=True)
            continue
        task_dir = _LOCAL_TASKS_ROOT / _DATASET / name
        if not (task_dir / "tests" / "probing_questions.json").exists():
            print(f"[{i}/{len(names)}] {name}: rubrics not found under {task_dir}, skipping", flush=True)
            continue
        try:
            scores = _judge(task_dir, task_out, workers=judge_workers)
        except subprocess.CalledProcessError as e:  # one task's judge failing shouldn't sink the rest
            print(f"[{i}/{len(names)}] {name}: judge ERROR {e}", flush=True)
            summary[name] = {"error": str(e)}
            continue
        reward = scores["overall_reward"]
        print(f"[{i}/{len(names)}] {name}: reward={reward:.4f}", flush=True)
        summary[name] = {
            "score": reward,
            "n_probes": scores["n_probes"],
            "per_type": {t: d["mean"] for t, d in scores["per_type"].items()},
        }

    scored = [e["score"] for e in summary.values() if "score" in e]
    manifest.setdefault("benchmark", "beam")
    manifest["judge"] = {"name": judge_name}
    manifest["mean_reward"] = (sum(scored) / len(scored)) if scored else None
    manifest["regraded_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Re-grade complete: {run_dir}  mean_reward={manifest['mean_reward']}", flush=True)
    return run_dir
