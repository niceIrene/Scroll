from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from scroll_eval.harness.config import RunConfig


_LOCAL_TASKS_ROOT = Path("local-tasks")
_DOTENV_PATH = Path(".env.local")
_HARBOR_DATASET_TASK_GROUP = "__harbor_dataset__"


def _harbor_cmd() -> list[str]:
    """Command prefix used to invoke the Harbor CLI.

    Defaults to running Harbor through ``scroll_eval.harness._harbor_launcher``
    (under the uv-managed environment), which caps Harbor's hardcoded E2B
    sandbox timeout to ``E2B_SANDBOX_TIMEOUT`` before delegating to the real
    Harbor CLI — see that module for why. The launcher is a no-op for Docker.
    Works on any platform. Overridable via ``SCROLL_HARBOR_CMD`` (split on
    whitespace) for tests/CI or non-uv setups; an override bypasses the launcher
    (and thus the timeout cap).
    """
    override = os.environ.get("SCROLL_HARBOR_CMD")
    if override:
        return override.split()
    return ["uv", "run", "python", "-m", "scroll_eval.harness._harbor_launcher"]


def _load_dotenv(path: Path = _DOTENV_PATH) -> None:
    """Populate os.environ from a dotenv file. Shell-set vars take precedence."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in value)


def _list_all_tasks(dataset: str | None = None) -> list[str]:
    """Return locally-materialized task IDs from `local-tasks/`."""
    if not _LOCAL_TASKS_ROOT.exists():
        raise ValueError(f'local task root folder {str(_LOCAL_TASKS_ROOT)} does not exist.')

    if not dataset:
        raise ValueError('missing dataset name.')

    dataset_folder = Path(_LOCAL_TASKS_ROOT, dataset)
    if not dataset_folder.exists():
        raise ValueError(f'dataset folder {str(dataset_folder)} does not exist.')

    # TODO: count and show valid and invalid tasks
    return sorted(
        d.name
        for d in dataset_folder.iterdir()
        if d.is_dir() and (d / "task.toml").exists()
    )


def _dataset_ref(cfg: RunConfig) -> str:
    if cfg.dataset.type != "harbor":
        raise ValueError("dataset ref is only valid for harbor datasets")
    return f"{cfg.dataset.name}@{cfg.dataset.version}"


def _dataset_manifest_value(cfg: RunConfig) -> dict:
    return asdict(cfg.dataset)


def _dataset_task_flags(cfg: RunConfig, task_name: str) -> list[str]:
    if cfg.dataset.type == "local":
        return [
            "--path",
            str(Path(_LOCAL_TASKS_ROOT, cfg.dataset.name, task_name)),
        ]

    flags = ["--dataset", _dataset_ref(cfg)]
    if task_name != _HARBOR_DATASET_TASK_GROUP:
        flags.extend(["--include-task-name", task_name])
    if cfg.dataset.registry_url:
        flags.extend(["--registry-url", cfg.dataset.registry_url])
    if cfg.dataset.registry_path:
        flags.extend(["--registry-path", cfg.dataset.registry_path])
    if cfg.dataset.n_tasks is not None:
        flags.extend(["--n-tasks", str(cfg.dataset.n_tasks)])
    return flags


def _uses_dashscope(cfg: RunConfig) -> bool:
    return bool(cfg.model.endpoint and "dashscope" in cfg.model.endpoint.lower())


def _model_api_key_env(cfg: RunConfig) -> str | None:
    """Return the env var name that supplies the configured model key.

    Configs use OPENAI_API_KEY by default because most providers are
    OpenAI-compatible. For DashScope, allow DASHSCOPE_API_KEY as the fallback
    when OPENAI_API_KEY is absent so users do not need to duplicate the same
    credential under two names.
    """
    if os.environ.get(cfg.model.api_key_env):
        return cfg.model.api_key_env
    if (
        cfg.model.api_key_env == "OPENAI_API_KEY"
        and _uses_dashscope(cfg)
        and os.environ.get("DASHSCOPE_API_KEY")
    ):
        return "DASHSCOPE_API_KEY"
    return None


def _build_env(cfg: RunConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    key_env = _model_api_key_env(cfg)
    if key_env and key_env in env:
        if _uses_dashscope(cfg):
            env.setdefault("DASHSCOPE_API_KEY", env[key_env])
        if cfg.model.api_key_env in env:
            env.setdefault("OPENAI_API_KEY", env[cfg.model.api_key_env])
    return env


def _model_for_harbor(cfg: RunConfig) -> str:
    """Pick the litellm prefix for the configured model."""
    if _uses_dashscope(cfg):
        return f"dashscope/{cfg.model.name}"
    return f"openai/{cfg.model.name}"


def _invoke_harbor(
    cmd: list[str],
    task_output_dir: Path,
    env: dict[str, str] | None = None,
    verbose: bool = False,
) -> int:
    """Run Harbor and capture stdout/stderr. Returns exit code. Tests patch this.

    When verbose=True, stdout/stderr are also streamed live to the parent terminal
    in addition to being written to log files.
    """
    stdout_path = task_output_dir / "stdout.log"
    stderr_path = task_output_dir / "stderr.log"
    if verbose:
        with stdout_path.open("w", encoding="utf-8") as out:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                out.write(line)
            proc.wait()
        stderr_path.write_text("", encoding="utf-8")
    else:
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
            "w", encoding="utf-8"
        ) as err:
            proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, env=env)
    # Try to surface Harbor's result.json if it exists
    harbor_summary: dict = {"exit_code": proc.returncode}
    harbor_out = task_output_dir / "harbor-out"
    if harbor_out.exists():
        for result_json in harbor_out.rglob("result.json"):
            try:
                harbor_summary["result"] = json.loads(result_json.read_text())
                break
            except (OSError, json.JSONDecodeError):
                pass
        for lock_json in harbor_out.rglob("lock.json"):
            try:
                lock = json.loads(lock_json.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            (task_output_dir / "harbor-lock.json").write_text(
                json.dumps(lock, indent=2), encoding="utf-8"
            )
            resolved_tasks = _extract_resolved_tasks_from_lock(lock)
            if resolved_tasks:
                harbor_summary["resolved_tasks"] = resolved_tasks
            break
    (task_output_dir / "harbor.json").write_text(
        json.dumps(harbor_summary, indent=2), encoding="utf-8"
    )
    return proc.returncode


def _run_one(
    cfg: RunConfig,
    agent: str,
    task_name: str,
    task_output_dir: Path,
    verbose: bool = False,
) -> int:
    """Real production invocation. Tests monkeypatch `_invoke_harbor`."""
    agent_flags: list[str]
    if agent == "scroll-eval":
        agent_flags = [
            "--agent-import-path",
            "scroll_eval.runner:ScrollEvalAgent",
        ]
    else:
        agent_flags = ["--agent", agent]

    cmd = [
        *_harbor_cmd(),
        "run",
        *_dataset_task_flags(cfg, task_name),
        *agent_flags,
        "--model",
        _model_for_harbor(cfg),
        "--env",
        cfg.sandbox.type,
        "--env-file",
        ".env.local",
        "--jobs-dir",
        str(Path(task_output_dir, "harbor-out")),
        "--yes",
        "--no-force-build",
    ]
    # Raise the verifier (test) timeout for slow graders, if configured.
    if cfg.verifier.timeout_multiplier is not None:
        cmd += ["--verifier-timeout-multiplier", str(cfg.verifier.timeout_multiplier)]
    if cfg.verifier.timeout_sec is not None:
        cmd += ["--verifier-timeout-sec", str(cfg.verifier.timeout_sec)]
    env = _build_env(cfg)
    env["SCROLL_MODEL"] = cfg.model.name
    env["OPENAI_BASE_URL"] = cfg.model.endpoint
    if cfg.model.thinking is not None:
        env["SCROLL_ENABLE_THINKING"] = "true" if cfg.model.thinking else "false"
    if cfg.model.thinking_budget is not None:
        env["SCROLL_THINKING_BUDGET"] = str(cfg.model.thinking_budget)
    env["SCROLL_MAX_TOKENS"] = str(cfg.budget.max_tokens)
    env["SCROLL_WALL_TIME_S"] = str(cfg.budget.wall_time_s)
    if agent == "scroll-eval":
        env["SCROLL_AGENT_TYPE"] = cfg.agent.type
        env["SCROLL_AGENT_ID"] = cfg.agent.id
    env["SCROLL_TASK_ID"] = task_name
    env["SCROLL_PHOENIX_PROJECT"] = cfg.trace.phoenix_project
    # A concrete, run-stable id (the run_dir label) so every session of this
    # run shares a run_id and session_id = f"{run_id}:{task_id}" is unique.
    run_id = (
        cfg.trace.run_id
        if cfg.trace.run_id and cfg.trace.run_id != "auto"
        else task_output_dir.parent.parent.name
    )
    env["SCROLL_RUN_ID"] = run_id
    env["SCROLL_MEMORY_DB"] = cfg.memory.db_path
    env["SCROLL_HISTORY_MAX_TOKENS"] = str(cfg.memory.history_max_tokens)
    # setdefault: a shell-set SCROLL_SUMMARY_CHUNK_TOKENS keeps precedence over
    # the config knob, matching the in-process agent's env-first resolution.
    if cfg.memory.summary_chunk_tokens is not None:
        env.setdefault(
            "SCROLL_SUMMARY_CHUNK_TOKENS", str(cfg.memory.summary_chunk_tokens)
        )
    if cfg.tools is not None:
        env["SCROLL_TOOLS"] = ",".join(cfg.tools)
    return _invoke_harbor(cmd, task_output_dir, env=env, verbose=verbose)


def _resolve_tasks(cfg: RunConfig) -> list[str]:
    if cfg.tasks == "all":
        if cfg.dataset.type == "harbor":
            tasks = [_HARBOR_DATASET_TASK_GROUP]
        else:
            tasks = _list_all_tasks(cfg.dataset.name)
    else:
        tasks = list(cfg.tasks)

    if not tasks:
        raise ValueError(f'missing task setting in cfg: {json.dumps(asdict(cfg))}')

    return tasks


def _extract_resolved_tasks_from_lock(lock: dict) -> list[dict]:
    resolved: list[dict] = []
    trials = lock.get("trials") if isinstance(lock, dict) else None
    if not isinstance(trials, list):
        return resolved
    seen: set[tuple[str, str]] = set()
    for trial in trials:
        task = trial.get("task") if isinstance(trial, dict) else None
        if not isinstance(task, dict):
            continue
        name = task.get("name")
        digest = task.get("digest")
        if not isinstance(name, str) or not isinstance(digest, str):
            continue
        key = (name, digest)
        if key in seen:
            continue
        seen.add(key)
        entry = {
            "name": name,
            "digest": digest,
            "type": task.get("type"),
            "source": task.get("source"),
        }
        if task.get("path") is not None:
            entry["path"] = str(task["path"])
        if task.get("git_url") is not None:
            entry["git_url"] = task["git_url"]
        if task.get("git_commit_id") is not None:
            entry["git_commit_id"] = task["git_commit_id"]
        resolved.append(entry)
    return resolved


def _score_from_rewards(rewards: dict | None) -> float | None:
    if not isinstance(rewards, dict):
        return None
    for preferred in ("reward", "score", "overall_score"):
        value = rewards.get(preferred)
        if isinstance(value, (int, float)):
            return float(value)
    numeric = [float(value) for value in rewards.values() if isinstance(value, (int, float))]
    if len(numeric) == 1:
        return numeric[0]
    return None


def _extract_trial_summaries(harbor_json_path: Path) -> dict[str, dict]:
    if not harbor_json_path.exists():
        return {}
    try:
        data = json.loads(harbor_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    result = data.get("result", {}) if isinstance(data, dict) else {}
    trial_results = result.get("trial_results", {}) if isinstance(result, dict) else {}
    if not isinstance(trial_results, list):
        return {}

    resolved_by_name = {
        task["name"]: task
        for task in data.get("resolved_tasks", [])
        if isinstance(task, dict) and isinstance(task.get("name"), str)
    }
    summaries: dict[str, dict] = {}
    for trial in trial_results:
        if not isinstance(trial, dict):
            continue
        task_name = trial.get("task_name")
        if not isinstance(task_name, str) or not task_name:
            task_id = trial.get("task_id")
            if isinstance(task_id, dict):
                task_name = task_id.get("name")
        if not isinstance(task_name, str) or not task_name:
            continue
        exception_info = trial.get("exception_info")
        entry: dict = {"exit_code": 1 if exception_info else 0}
        verifier_result = trial.get("verifier_result")
        rewards = (
            verifier_result.get("rewards")
            if isinstance(verifier_result, dict)
            else None
        )
        score = _score_from_rewards(rewards)
        if score is not None:
            entry["score"] = score
        if rewards:
            entry["rewards"] = rewards
        if task_name in resolved_by_name:
            entry["resolved_task"] = resolved_by_name[task_name]
        summaries[task_name] = entry
    return summaries


def _collect_resolved_tasks(run_dir: Path) -> list[dict]:
    resolved: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for harbor_json in sorted((run_dir / "tasks").glob("*/harbor.json")):
        try:
            data = json.loads(harbor_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for task in data.get("resolved_tasks", []):
            if not isinstance(task, dict):
                continue
            name = task.get("name")
            digest = task.get("digest")
            if not isinstance(name, str) or not isinstance(digest, str):
                continue
            key = (name, digest)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(task)
    return resolved


def _extract_score(harbor_json_path: Path) -> float | None:
    """Pull the mean reward out of a harbor.json blob.

    Supports two shapes:
    - Test/fake shape: ``{"score": <float>, ...}`` at the top level.
    - Real Harbor shape: ``result.stats.evals.<key>.metrics[0].mean``.
    """
    if not harbor_json_path.exists():
        return None
    try:
        data = json.loads(harbor_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and isinstance(data.get("score"), (int, float)):
        return float(data["score"])
    evals = (
        data.get("result", {}).get("stats", {}).get("evals", {})
        if isinstance(data, dict)
        else {}
    )
    for eval_summary in evals.values():
        metrics = eval_summary.get("metrics") or []
        if metrics and isinstance(metrics[0], dict) and "mean" in metrics[0]:
            try:
                return float(metrics[0]["mean"])
            except (TypeError, ValueError):
                continue
    return None


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _effective_agent_label(agent: str, cfg: RunConfig) -> str:
    """For our custom agent the (type, id) drives the recorded label.
    For Harbor built-ins (terminus-2, oracle, ...) Harbor uses its own
    internal loop — record 'default'.
    """
    if agent == "scroll-eval":
        return f"{cfg.agent.type}_{cfg.agent.id}"
    return "default"


def _agent_loop_slug(agent: str, cfg: RunConfig) -> str:
    return f"{_slug(agent)}_{_slug(_effective_agent_label(agent, cfg))}"


def _validate_keys(cfg: RunConfig) -> str | None:
    """Raise upon missing critical keys in configuration.
    TODO: validate all required fields in cfg.
    """
    if _model_api_key_env(cfg) is None:
        raise ValueError(f"missing model api key in cfg: {json.dumps(asdict(cfg))}")
    if cfg.sandbox.type == "e2b" and not os.environ.get("E2B_API_KEY"):
        raise ValueError(f"missing e2b api key in cfg: {json.dumps(asdict(cfg))}")


def _run_task(
    cfg: RunConfig,
    agent: str,
    task_name: str,
    task_output_dir: Path,
    index: int,
    n: int,
    *,
    verbose: bool,
) -> tuple[str, dict]:
    """Run a single task (one ``harbor run``) and return its summary entry.

    Safe to call concurrently from a thread pool: each task owns its own
    ``task_output_dir``/``harbor-out``, so there is no shared mutable state.
    """
    task_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{index}/{n}] {task_name}: starting...", flush=True)
    t0 = time.monotonic()
    rc = _run_one(cfg, agent, task_name, task_output_dir, verbose=verbose)
    elapsed = time.monotonic() - t0
    entry: dict = {"exit_code": rc}
    score = _extract_score(task_output_dir / "harbor.json")
    if score is not None:
        entry["score"] = score
    score_str = f" reward={score}" if score is not None else ""
    print(
        f"[{index}/{n}] {task_name}: exit={rc}{score_str} elapsed={elapsed:.0f}s",
        flush=True,
    )
    return task_name, entry


def run(
    cfg: RunConfig,
    agent: str,
    runs_root: Path,
    verbose: bool = False,
) -> Path:
    _load_dotenv()
    _validate_keys(cfg)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    effective_label = _effective_agent_label(agent, cfg)
    label = (
        f"{timestamp}__{_agent_loop_slug(agent, cfg)}"
        f"__{_slug(cfg.model.name)}__{_slug(cfg.trace.phoenix_project)}"
    )
    run_dir = runs_root / label
    (run_dir / "tasks").mkdir(parents=True, exist_ok=True)

    tasks = _resolve_tasks(cfg)
    n = len(tasks)

    # might crash the sandbox server if n is too large.
    # TODO: limit parallelism
    workers = max(1, min(cfg.sandbox.parallelism, n))
    print(
        f"Running {n} task(s) on '{cfg.sandbox.type}' "
        f"with parallelism={workers}",
        flush=True,
    )

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_task,
                cfg,
                agent,
                task_name,
                run_dir / "tasks" / _slug(task_name),
                i,
                n,
                verbose=verbose,
            ): task_name
            for i, task_name in enumerate(tasks, start=1)
        }
        for fut in as_completed(futures):
            task_name, entry = fut.result()
            results[task_name] = entry

    # Preserve the configured task order for local runs. Harbor dataset-wide
    # runs only know concrete task names after Harbor resolves the dataset, so
    # derive those names from Harbor's result.json when available.
    if cfg.dataset.type == "harbor":
        per_task_summary = {}
        for task_name in tasks:
            task_dir = run_dir / "tasks" / _slug(task_name)
            trial_summaries = _extract_trial_summaries(task_dir / "harbor.json")
            if trial_summaries:
                per_task_summary.update(trial_summaries)
            elif task_name in results:
                per_task_summary[task_name] = results[task_name]
    else:
        per_task_summary = {
            task_name: results[task_name] for task_name in tasks if task_name in results
        }

    resolved_tasks = _collect_resolved_tasks(run_dir)

    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "agent": agent,
                "agent_label": effective_label,
                "config_agent": {"type": cfg.agent.type, "id": cfg.agent.id},
                "model": asdict(cfg.model),
                "dataset": _dataset_manifest_value(cfg),
                "sandbox": asdict(cfg.sandbox),
                "tasks": list(per_task_summary) if cfg.dataset.type == "harbor" else tasks,
                "resolved_tasks": resolved_tasks,
                "git_sha": _git_sha(),
                "timestamp_utc": timestamp,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(per_task_summary, indent=2), encoding="utf-8"
    )
    return run_dir
