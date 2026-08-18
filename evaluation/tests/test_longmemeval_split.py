"""The config's dataset.name selects the LongMemEval split's task directory."""
from __future__ import annotations

from pathlib import Path

from scroll_eval.harness import config as cfg_mod
from scroll_eval.evals.longmemeval.runner import _DEFAULT_DATASET, _dataset

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_s_config_resolves_default_dataset(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "mock")
    cfg = cfg_mod.load(_CONFIGS / "longmemeval.yaml")
    assert _dataset(cfg) == "longmemeval" == _DEFAULT_DATASET


def test_m_config_resolves_its_own_task_dir(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "mock")
    cfg = cfg_mod.load(_CONFIGS / "longmemeval-m.yaml")
    assert _dataset(cfg) == "longmemeval-m"
    # Everything but the split and trace project matches the s config — the
    # split is the only experimental variable between the two files.
    s = cfg_mod.load(_CONFIGS / "longmemeval.yaml")
    assert cfg.agent == s.agent
    assert cfg.memory.history_max_tokens == s.memory.history_max_tokens
    assert cfg.tools == s.tools
