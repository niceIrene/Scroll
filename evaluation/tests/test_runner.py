import threading
from pathlib import Path

from scroll_eval.harness import runner
from scroll_eval.harness.config import (
    AgentSpec,
    BudgetSpec,
    DatasetSpec,
    ModelSpec,
    RunConfig,
    SandboxSpec,
    TraceSpec,
)


def _cfg(
    tasks,
    *,
    agent_type="base_agents",
    agent_id="base_agent_A",
    sandbox_type="docker",
    parallelism=1,
    dataset=None,
):
    return RunConfig(
        agent=AgentSpec(type=agent_type, id=agent_id),
        model=ModelSpec(endpoint="http://x/v1", name="m", api_key_env="K"),
        dataset=dataset or DatasetSpec(name="terminal-bench", type="local"),
        tasks=tasks,
        budget=BudgetSpec(),
        trace=TraceSpec(phoenix_project="t"),
        sandbox=SandboxSpec(type=sandbox_type, parallelism=parallelism),
    )


def test_run_creates_manifest_and_per_task_dirs(tmp_path: Path, monkeypatch) -> None:
    # Fake `harbor run` that just writes a known harbor.json into the per-task dir.
    def fake_invoke(cmd, task_dir, env=None, verbose=False):
        (task_dir / "harbor.json").write_text('{"score": 1.0, "exit_code": 0}')
        (task_dir / "stdout.log").write_text("ok\n")
        return 0

    monkeypatch.setenv("K", "fake-key")
    monkeypatch.setattr(runner, "_invoke_harbor", fake_invoke)
    cfg = _cfg(["hello-world", "fix-permissions"])

    run_dir = runner.run(cfg, agent="terminus-2", runs_root=tmp_path)

    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "tasks" / "hello-world" / "harbor.json").exists()
    assert (run_dir / "tasks" / "fix-permissions" / "harbor.json").exists()
    summary = (run_dir / "summary.json").read_text()
    assert "hello-world" in summary


def test_run_handles_all_tasks_literal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("K", "fake-key")
    monkeypatch.setattr(runner, "_invoke_harbor", lambda c, d, env=None, verbose=False: 0)
    monkeypatch.setattr(
        runner, "_list_all_tasks", lambda *args, **kwargs: ["a", "b", "c"]
    )
    cfg = _cfg("all")

    run_dir = runner.run(cfg, agent="terminus-2", runs_root=tmp_path)
    task_dirs = sorted(p.name for p in (run_dir / "tasks").iterdir())
    assert task_dirs == ["a", "b", "c"]


def test_run_dir_label_for_ours_includes_agent_type_and_id(tmp_path: Path, monkeypatch):
    """For --agent scroll-eval, the run-dir label includes 'scroll-eval_base_agents_base_agent_A'."""
    monkeypatch.setattr(runner, "_invoke_harbor", lambda c, d, env=None, verbose=False: 0)
    monkeypatch.setenv("K", "k")
    cfg = _cfg(["t"])
    run_dir = runner.run(cfg, agent="scroll-eval", runs_root=tmp_path)
    import json as _json
    manifest = _json.loads((run_dir / "manifest.json").read_text())
    assert "scroll-eval_base_agents_base_agent_A" in run_dir.name
    assert manifest["agent_label"] == "base_agents_base_agent_A"
    assert manifest["config_agent"] == {"type": "base_agents", "id": "base_agent_A"}


def test_run_dir_label_for_builtin_agent_uses_default(tmp_path: Path, monkeypatch):
    """For --agent terminus-2, run-dir is 'terminus-2_default'; manifest records loop='default'."""
    monkeypatch.setattr(runner, "_invoke_harbor", lambda c, d, env=None, verbose=False: 0)
    monkeypatch.setenv("K", "k")
    cfg = _cfg(["t"])
    run_dir = runner.run(cfg, agent="terminus-2", runs_root=tmp_path)
    import json as _json
    manifest = _json.loads((run_dir / "manifest.json").read_text())
    assert "terminus-2_default" in run_dir.name
    assert "base_agent" not in run_dir.name
    assert manifest["agent_label"] == "default"
    assert manifest["config_agent"] == {"type": "base_agents", "id": "base_agent_A"}


def test_run_propagates_agent_type_and_id_to_env(tmp_path: Path, monkeypatch):
    captured: dict = {}
    def fake_invoke(cmd, task_dir, env=None, verbose=False):
        captured["env"] = env or {}
        captured["cmd"] = cmd
        (task_dir / "harbor.json").write_text('{"exit_code": 0}')
        return 0
    monkeypatch.setattr(runner, "_invoke_harbor", fake_invoke)
    monkeypatch.setenv("K", "fake-key")
    cfg = _cfg(["t1"], agent_type="base_agents", agent_id="scroll_react")
    runner.run(cfg, agent="scroll-eval", runs_root=tmp_path)
    assert captured["env"]["SCROLL_AGENT_TYPE"] == "base_agents"
    assert captured["env"]["SCROLL_AGENT_ID"] == "scroll_react"
    assert "scroll_eval.runner:ScrollEvalAgent" in captured["cmd"]


def test_verifier_timeout_flags_passed_to_harbor(tmp_path: Path, monkeypatch):
    from dataclasses import replace
    from scroll_eval.harness.config import VerifierSpec

    captured: dict = {}
    def fake_invoke(cmd, task_dir, env=None, verbose=False):
        captured["cmd"] = cmd
        (task_dir / "harbor.json").write_text('{"exit_code": 0}')
        return 0
    monkeypatch.setattr(runner, "_invoke_harbor", fake_invoke)
    monkeypatch.setenv("K", "k")
    cfg = replace(_cfg(["t1"]), verifier=VerifierSpec(timeout_multiplier=3, timeout_sec=900))
    runner.run(cfg, agent="scroll-eval", runs_root=tmp_path)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--verifier-timeout-multiplier") + 1] == "3"
    assert cmd[cmd.index("--verifier-timeout-sec") + 1] == "900"


def test_no_verifier_flags_by_default(tmp_path: Path, monkeypatch):
    captured: dict = {}
    def fake_invoke(cmd, task_dir, env=None, verbose=False):
        captured["cmd"] = cmd
        (task_dir / "harbor.json").write_text('{"exit_code": 0}')
        return 0
    monkeypatch.setattr(runner, "_invoke_harbor", fake_invoke)
    monkeypatch.setenv("K", "k")
    runner.run(_cfg(["t1"]), agent="scroll-eval", runs_root=tmp_path)
    assert "--verifier-timeout-multiplier" not in captured["cmd"]
    assert "--verifier-timeout-sec" not in captured["cmd"]


def test_terminus2_does_not_set_agent_env(tmp_path: Path, monkeypatch):
    """Built-in agents go through Harbor's --agent path; no SCROLL_AGENT_* env."""
    captured: dict = {}
    def fake_invoke(cmd, task_dir, env=None, verbose=False):
        captured["env"] = env or {}
        captured["cmd"] = cmd
        return 0
    monkeypatch.setattr(runner, "_invoke_harbor", fake_invoke)
    monkeypatch.setenv("K", "k")
    runner.run(_cfg(["t1"]), agent="terminus-2", runs_root=tmp_path)
    assert "SCROLL_AGENT_TYPE" not in captured["env"]
    assert "--agent" in captured["cmd"] and "terminus-2" in captured["cmd"]


def test_run_loads_dotenv_when_present(tmp_path: Path, monkeypatch) -> None:
    """run() should populate os.environ from .env.local before checking api_key_env."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.local").write_text('K=loaded-from-dotenv\n')
    monkeypatch.delenv("K", raising=False)
    monkeypatch.setattr(runner, "_invoke_harbor", lambda c, d, env=None, verbose=False: 0)

    run_dir = runner.run(_cfg(["t1"]), agent="scroll-eval", runs_root=tmp_path / "out")
    import json
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["t1"] == {"exit_code": 0}  # not "config_error"


def test_run_uses_dashscope_key_when_openai_key_missing(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_invoke(cmd, task_dir, env=None, verbose=False):
        captured["env"] = env or {}
        (task_dir / "harbor.json").write_text('{"exit_code": 0}')
        return 0

    monkeypatch.setattr(runner, "_invoke_harbor", fake_invoke)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
    cfg = RunConfig(
        agent=AgentSpec(type="base_agents", id="base_agent_A"),
        model=ModelSpec(
            endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
            name="qwen-plus",
            api_key_env="OPENAI_API_KEY",
        ),
        dataset=DatasetSpec(name="terminal-bench", type="local"),
        tasks=["t1"],
        budget=BudgetSpec(),
        trace=TraceSpec(phoenix_project="t"),
        sandbox=SandboxSpec(),
    )

    run_dir = runner.run(cfg, agent="scroll-eval", runs_root=tmp_path)

    import json

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["t1"] == {"exit_code": 0}
    assert captured["env"]["DASHSCOPE_API_KEY"] == "dashscope-key"
    assert captured["env"]["OPENAI_BASE_URL"] == cfg.model.endpoint


def test_run_does_not_use_dashscope_key_for_non_dashscope_endpoint(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
    monkeypatch.setattr(runner, "_invoke_harbor", lambda c, d, env=None, verbose=False: 0)
    cfg = RunConfig(
        agent=AgentSpec(type="base_agents", id="base_agent_A"),
        model=ModelSpec(
            endpoint="https://api.openai.com/v1",
            name="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
        ),
        dataset=DatasetSpec(name="terminal-bench", type="local"),
        tasks=["t1"],
        budget=BudgetSpec(),
        trace=TraceSpec(phoenix_project="t"),
        sandbox=SandboxSpec(),
    )

    try:
        runner.run(cfg, agent="scroll-eval", runs_root=tmp_path)
    except ValueError as exc:
        assert "missing model api key" in str(exc)
        return
    raise AssertionError("expected missing key validation error")


def test_run_forwards_verbose_to_invoke(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_invoke(cmd, task_dir, env=None, verbose=False):
        captured["verbose"] = verbose
        (task_dir / "harbor.json").write_text('{"exit_code": 0}')
        return 0

    monkeypatch.setattr(runner, "_invoke_harbor", fake_invoke)
    monkeypatch.setenv("K", "fake-key")

    runner.run(_cfg(["t1"]), agent="scroll-eval", runs_root=tmp_path, verbose=True)
    assert captured["verbose"] is True


def test_run_dotenv_does_not_clobber_shell_env(tmp_path: Path, monkeypatch) -> None:
    """Shell-set env vars take precedence over .env.local values."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.local").write_text('K=from-file\n')
    monkeypatch.setenv("K", "from-shell")
    monkeypatch.setattr(runner, "_invoke_harbor", lambda c, d, env=None, verbose=False: 0)

    runner.run(_cfg(["t1"]), agent="scroll-eval", runs_root=tmp_path / "out")
    import os
    assert os.environ["K"] == "from-shell"


# --- sandbox backend (--env) -------------------------------------------------


def test_run_passes_docker_env_by_default(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_invoke(cmd, task_dir, env=None, verbose=False):
        captured["cmd"] = cmd
        (task_dir / "harbor.json").write_text('{"exit_code": 0}')
        return 0

    monkeypatch.setattr(runner, "_invoke_harbor", fake_invoke)
    monkeypatch.setenv("K", "k")
    runner.run(_cfg(["t1"]), agent="terminus-2", runs_root=tmp_path)
    cmd = captured["cmd"]
    assert "--env" in cmd
    assert cmd[cmd.index("--env") + 1] == "docker"


def test_local_dataset_uses_path_flag(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_invoke(cmd, task_dir, env=None, verbose=False):
        captured["cmd"] = cmd
        (task_dir / "harbor.json").write_text('{"exit_code": 0}')
        return 0

    monkeypatch.setattr(runner, "_invoke_harbor", fake_invoke)
    monkeypatch.setenv("K", "k")

    runner.run(_cfg(["t1"]), agent="terminus-2", runs_root=tmp_path)
    cmd = captured["cmd"]
    assert "--path" in cmd
    assert cmd[cmd.index("--path") + 1] == "local-tasks/terminal-bench/t1"
    assert "--dataset" not in cmd


def test_harbor_dataset_uses_dataset_flag_and_task_filter(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict = {}
    lock = {
        "trials": [
            {
                "task": {
                    "name": "2000_easy_01_buy_only_baseline",
                    "type": "package",
                    "digest": "sha256:taskdigest",
                    "source": "gorilla/bfcl",
                }
            }
        ]
    }

    def fake_invoke(cmd, task_dir, env=None, verbose=False):
        captured["cmd"] = cmd
        (task_dir / "harbor-lock.json").write_text(__import__("json").dumps(lock))
        (task_dir / "harbor.json").write_text(
            __import__("json").dumps(
                {
                    "exit_code": 0,
                    "result": {"trial_results": [], "stats": {"evals": {}}},
                    "resolved_tasks": runner._extract_resolved_tasks_from_lock(lock),
                }
            )
        )
        return 0

    monkeypatch.setattr(runner, "_invoke_harbor", fake_invoke)
    monkeypatch.setenv("K", "k")
    cfg = _cfg(
        ["2000_easy_01_buy_only_baseline"],
        dataset=DatasetSpec(
            type="harbor",
            name="gorilla/bfcl",
            version="sha256:datasetdigest",
        ),
    )

    run_dir = runner.run(cfg, agent="terminus-2", runs_root=tmp_path)

    cmd = captured["cmd"]
    assert "--dataset" in cmd
    assert cmd[cmd.index("--dataset") + 1] == "gorilla/bfcl@sha256:datasetdigest"
    assert "--include-task-name" in cmd
    assert cmd[cmd.index("--include-task-name") + 1] == "2000_easy_01_buy_only_baseline"
    assert "--path" not in cmd

    import json

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["dataset"] == {
        "name": "gorilla/bfcl",
        "type": "harbor",
        "version": "sha256:datasetdigest",
        "registry_url": None,
        "registry_path": None,
        "n_tasks": None,
    }
    assert manifest["resolved_tasks"] == [
        {
            "name": "2000_easy_01_buy_only_baseline",
            "digest": "sha256:taskdigest",
            "type": "package",
            "source": "gorilla/bfcl",
        }
    ]


def test_harbor_dataset_all_runs_once_and_expands_summary_from_result(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict = {}

    def fake_invoke(cmd, task_dir, env=None, verbose=False):
        captured["cmd"] = cmd
        lock = {
            "trials": [
                {
                    "task": {
                        "name": "task-a",
                        "type": "package",
                        "digest": "sha256:a",
                        "source": "gorilla/bfcl",
                    }
                },
                {
                    "task": {
                        "name": "task-b",
                        "type": "package",
                        "digest": "sha256:b",
                        "source": "gorilla/bfcl",
                    }
                },
            ]
        }
        result = {
            "trial_results": [
                {
                    "task_name": "task-a",
                    "exception_info": None,
                    "verifier_result": {"rewards": {"reward": 1.0}},
                },
                {
                    "task_name": "task-b",
                    "exception_info": {"exception_type": "Error"},
                    "verifier_result": {"rewards": {"reward": 0.0}},
                },
            ],
            "stats": {"evals": {}},
        }
        (task_dir / "harbor-lock.json").write_text(__import__("json").dumps(lock))
        (task_dir / "harbor.json").write_text(
            __import__("json").dumps(
                {
                    "exit_code": 0,
                    "result": result,
                    "resolved_tasks": runner._extract_resolved_tasks_from_lock(lock),
                }
            )
        )
        return 0

    monkeypatch.setattr(runner, "_invoke_harbor", fake_invoke)
    monkeypatch.setenv("K", "k")
    cfg = _cfg(
        "all",
        dataset=DatasetSpec(
            type="harbor",
            name="gorilla/bfcl",
            version="sha256:datasetdigest",
        ),
    )

    run_dir = runner.run(cfg, agent="terminus-2", runs_root=tmp_path)

    cmd = captured["cmd"]
    assert cmd[cmd.index("--dataset") + 1] == "gorilla/bfcl@sha256:datasetdigest"
    assert "--include-task-name" not in cmd

    import json

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["task-a"]["score"] == 1.0
    assert summary["task-a"]["resolved_task"]["digest"] == "sha256:a"
    assert summary["task-b"]["exit_code"] == 1

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["tasks"] == ["task-a", "task-b"]
    assert {task["digest"] for task in manifest["resolved_tasks"]} == {
        "sha256:a",
        "sha256:b",
    }


def test_run_passes_e2b_env_when_configured(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_invoke(cmd, task_dir, env=None, verbose=False):
        captured["cmd"] = cmd
        (task_dir / "harbor.json").write_text('{"exit_code": 0}')
        return 0

    monkeypatch.setattr(runner, "_invoke_harbor", fake_invoke)
    monkeypatch.setenv("K", "k")
    monkeypatch.setenv("E2B_API_KEY", "e2b-key")
    runner.run(_cfg(["t1"], sandbox_type="e2b"), agent="terminus-2", runs_root=tmp_path)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--env") + 1] == "e2b"


def test_e2b_without_api_key_is_config_error(tmp_path: Path, monkeypatch) -> None:
    """sandbox.type=e2b without E2B_API_KEY fails fast per task, never invoking harbor."""
    called = {"n": 0}

    def fake_invoke(cmd, task_dir, env=None, verbose=False):
        called["n"] += 1
        return 0

    monkeypatch.setattr(runner, "_invoke_harbor", fake_invoke)
    monkeypatch.setattr(runner, "_load_dotenv", lambda: None)
    monkeypatch.setenv("K", "k")
    monkeypatch.delenv("E2B_API_KEY", raising=False)

    try:
        runner.run(_cfg(["t1"], sandbox_type="e2b"), agent="terminus-2", runs_root=tmp_path)
        raise ValueError('expected exception did not raise')
    except Exception as e:
        assert 'missing e2b api key in cfg' in str(e)


# --- parallelism -------------------------------------------------------------


def test_parallelism_runs_tasks_concurrently(tmp_path: Path, monkeypatch) -> None:
    """With parallelism=3, three tasks should be in `_invoke_harbor` at once."""
    tasks = ["a", "b", "c"]
    barrier = threading.Barrier(len(tasks), timeout=5)
    reached = []

    def fake_invoke(cmd, task_dir, env=None, verbose=False):
        reached.append(task_dir.name)
        barrier.wait()  # blocks until all three workers arrive — proves concurrency
        (task_dir / "harbor.json").write_text('{"exit_code": 0}')
        return 0

    monkeypatch.setattr(runner, "_invoke_harbor", fake_invoke)
    monkeypatch.setenv("K", "k")

    run_dir = runner.run(
        _cfg(tasks, parallelism=3), agent="terminus-2", runs_root=tmp_path
    )
    import json

    summary = json.loads((run_dir / "summary.json").read_text())
    # Barrier didn't time out → all ran concurrently; order preserved in summary.
    assert list(summary.keys()) == tasks
    assert sorted(reached) == tasks


def test_parallelism_caps_workers_at_task_count(tmp_path: Path, monkeypatch) -> None:
    """parallelism larger than the task count must not error (workers clamped)."""
    monkeypatch.setattr(
        runner, "_invoke_harbor", lambda c, d, env=None, verbose=False: 0
    )
    monkeypatch.setenv("K", "k")
    run_dir = runner.run(
        _cfg(["only"], parallelism=8), agent="terminus-2", runs_root=tmp_path
    )
    assert (run_dir / "tasks" / "only").exists()


# --- harbor command resolution ----------------------------------------------


def test_harbor_cmd_default_uses_launcher(monkeypatch) -> None:
    monkeypatch.delenv("SCROLL_HARBOR_CMD", raising=False)
    assert runner._harbor_cmd() == [
        "uv",
        "run",
        "python",
        "-m",
        "scroll_eval.harness._harbor_launcher",
    ]


def test_harbor_cmd_override(monkeypatch) -> None:
    monkeypatch.setenv("SCROLL_HARBOR_CMD", "/opt/harbor/bin/harbor")
    assert runner._harbor_cmd() == ["/opt/harbor/bin/harbor"]
