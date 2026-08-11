"""Sanity tests for every config in `configs/`.

These guard against three classes of breakage that the loader alone won't catch:
- Configs that reference tasks no longer in local-tasks/.
- Configs whose `${VAR}` placeholders don't resolve because the env var
  isn't set at test time.
- Configs that fail to parse for any reason.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scroll_eval._tools_common import TOOLS
from scroll_eval.harness import config as cfg_mod


_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
_LOCAL_TASKS = Path(__file__).parent.parent.parent / "local-tasks"


def _config_paths() -> list[Path]:
    return sorted(_CONFIGS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", _config_paths(), ids=lambda p: p.name)
def test_config_loads_and_tasks_exist(config_path, monkeypatch):
    # Ensure the env vars that configs reference are present so ${VAR}
    # expansion produces values (not literals) for the placeholder check below.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "test-model")
    monkeypatch.setenv("BFCL_VERSION", "sha256:test-dataset-digest")

    cfg = cfg_mod.load(config_path)

    # Tasks: every concrete local task name must exist on disk. Harbor datasets
    # are resolved by Harbor at run time and must instead be pinned.
    if cfg.dataset.type == "local" and cfg.tasks != "all":
        for task in cfg.tasks:
            assert (_LOCAL_TASKS / cfg.dataset.name / task).is_dir(), (
                f"{config_path.name} references missing task '{task}' "
                f"(no folder at local-tasks/{cfg.dataset.name}/{task}/)"
            )
    if cfg.dataset.type == "harbor":
        assert cfg.dataset.version and cfg.dataset.version != "latest"

    # Model fields: no unresolved ${...} placeholders post-expansion.
    for field_name in ("endpoint", "name", "api_key_env"):
        value = getattr(cfg.model, field_name)
        assert "${" not in value, (
            f"{config_path.name}: model.{field_name} still contains a "
            f"${{...}} placeholder after expansion: {value!r}. "
            f"Did you set the env var?"
        )

    # tools: when set, every name must be a registered tool.
    if cfg.tools is not None:
        for name in cfg.tools:
            assert name in TOOLS, (
                f"{config_path.name}: unknown tool {name!r}; "
                f"registered tools are {sorted(TOOLS)}"
            )


def test_tools_field_rejects_unknown_name(tmp_path):
    """Misconfigured tools: should fail at load, not at first model call."""
    cfg_text = """\
agent: { type: base_agents, id: base_agent_A }
model:
  endpoint: https://example.test/v1
  name: test-model
  api_key_env: OPENAI_API_KEY
dataset: terminal-bench-2.1
tasks: [hello-world]
tools: [bash, not_a_real_tool]
"""
    path = tmp_path / "bad.yaml"
    path.write_text(cfg_text, encoding="utf-8")
    with pytest.raises(cfg_mod.ConfigError):
        cfg_mod.load(path)


def test_tools_field_omitted_keeps_default(tmp_path):
    """When tools: is absent, cfg.tools is None (legacy behavior)."""
    cfg_text = """\
agent: { type: base_agents, id: base_agent_A }
model:
  endpoint: https://example.test/v1
  name: test-model
  api_key_env: OPENAI_API_KEY
dataset: terminal-bench-2.1
tasks: [hello-world]
"""
    path = tmp_path / "ok.yaml"
    path.write_text(cfg_text, encoding="utf-8")
    cfg = cfg_mod.load(path)
    assert cfg.tools is None
