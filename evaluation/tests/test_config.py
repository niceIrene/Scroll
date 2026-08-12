from pathlib import Path

import pytest

from scroll_eval.harness import config


def test_loads_smoke_config(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        """
agent:
  type: base_agents
  id: base_agent_A
model:
  endpoint: http://x/v1
  name: m
  api_key_env: K
tasks: [t1, t2]
budget: { max_tokens: 100, wall_time_s: 60 }
trace: { run_id: auto, phoenix_project: smoke }
""".strip()
    )
    cfg = config.load(p)
    assert cfg.agent == config.AgentSpec(type="base_agents", id="base_agent_A")
    assert cfg.model.endpoint == "http://x/v1"
    assert cfg.dataset == config.DatasetSpec(name="", type="local")
    assert cfg.tasks == ["t1", "t2"]
    assert cfg.budget.max_tokens == 100


def test_tasks_can_be_literal_all(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        """
agent: { type: base_agents, id: base_agent_A }
model: { endpoint: x, name: m, api_key_env: K }
tasks: all
""".strip()
    )
    cfg = config.load(p)
    assert cfg.tasks == "all"


def test_override_tasks_replaces_config_value() -> None:
    cfg = config.RunConfig(
        agent=config.AgentSpec(type="base_agents", id="base_agent_A"),
        model=config.ModelSpec(endpoint="x", name="m", api_key_env="K"),
        tasks=["a", "b"],
    )
    overridden = config.with_tasks(cfg, ["only"])
    assert overridden.tasks == ["only"]
    assert cfg.tasks == ["a", "b"]  # original unchanged


def test_override_with_all_tasks_sets_literal() -> None:
    cfg = config.RunConfig(
        agent=config.AgentSpec(type="base_agents", id="base_agent_A"),
        model=config.ModelSpec(endpoint="x", name="m", api_key_env="K"),
        tasks=["a"],
    )
    overridden = config.with_all_tasks(cfg)
    assert overridden.tasks == "all"


def test_rejects_missing_required_field(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("agent: {type: base_agents, id: base_agent_A}\n")
    with pytest.raises(config.ConfigError):
        config.load(p)


def test_load_parses_agent_type_and_id(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "agent:\n"
        "  type: base_agents\n"
        "  id: scroll_react\n"
        "model:\n"
        "  endpoint: http://x\n"
        "  name: m\n"
        "  api_key_env: K\n"
        "tasks: [t1]\n",
        encoding="utf-8",
    )
    cfg = config.load(p)
    assert cfg.agent.type == "base_agents"
    assert cfg.agent.id == "scroll_react"


def _minimal(extra: str = "") -> str:
    return (
        "agent: { type: base_agents, id: base_agent_A }\n"
        "model: { endpoint: x, name: m, api_key_env: K }\n"
        "tasks: [t1]\n" + extra
    )


def test_sandbox_defaults_to_docker_serial(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(_minimal())
    cfg = config.load(p)
    assert cfg.sandbox == config.SandboxSpec(type="docker", parallelism=1)


def test_sandbox_block_is_parsed(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(_minimal("sandbox: { type: e2b, parallelism: 4 }\n"))
    cfg = config.load(p)
    assert cfg.sandbox.type == "e2b"
    assert cfg.sandbox.parallelism == 4


def test_sandbox_rejects_unknown_type(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(_minimal("sandbox: { type: kubernetes }\n"))
    with pytest.raises(config.ConfigError):
        config.load(p)


def test_sandbox_rejects_non_positive_parallelism(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(_minimal("sandbox: { parallelism: 0 }\n"))
    with pytest.raises(config.ConfigError):
        config.load(p)


def test_sandbox_rejects_unknown_field(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(_minimal("sandbox: { type: docker, workers: 3 }\n"))
    with pytest.raises(config.ConfigError):
        config.load(p)


def test_string_dataset_is_local_dataset_spec(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(_minimal("dataset: terminal-bench-2.1\n"))
    cfg = config.load(p)
    assert cfg.dataset == config.DatasetSpec(name="terminal-bench-2.1", type="local")


def test_harbor_dataset_requires_pinned_version(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        _minimal(
            """
dataset:
  type: harbor
  name: gorilla/bfcl
""".lstrip()
        )
    )
    with pytest.raises(config.ConfigError, match="pinned"):
        config.load(p)


def test_harbor_dataset_rejects_latest(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        _minimal(
            """
dataset:
  type: harbor
  name: gorilla/bfcl
  version: latest
""".lstrip()
        )
    )
    with pytest.raises(config.ConfigError, match="non-'latest'"):
        config.load(p)


def test_harbor_dataset_parses_pinned_version(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        _minimal(
            """
dataset:
  type: harbor
  name: gorilla/bfcl
  version: sha256:abc
  n_tasks: 3
""".lstrip()
        )
    )
    cfg = config.load(p)
    assert cfg.dataset == config.DatasetSpec(
        type="harbor",
        name="gorilla/bfcl",
        version="sha256:abc",
        n_tasks=3,
    )


def test_load_rejects_legacy_top_level_loop(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "loop: qwenpaw\n"
        "model: {endpoint: x, name: m, api_key_env: K}\n",
        encoding="utf-8",
    )
    try:
        config.load(p)
    except config.ConfigError as exc:
        assert "loop" in str(exc).lower()
        return
    raise AssertionError("expected ConfigError for legacy `loop:` key")
