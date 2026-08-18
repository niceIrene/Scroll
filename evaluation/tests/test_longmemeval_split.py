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


def test_run_all_fans_tasks_out_concurrently(monkeypatch, tmp_path):
    """--concurrency bounds TASKS (each LME task has one probe): 4 tasks at
    concurrency 4 must overlap, not run back to back."""
    import asyncio

    from scroll_eval.evals.longmemeval import runner as lme_runner

    running = {"now": 0, "peak": 0}

    async def fake_task(cfg, name, task_out, label, **kw):
        running["now"] += 1
        running["peak"] = max(running["peak"], running["now"])
        await asyncio.sleep(0.05)
        running["now"] -= 1
        return {"score": 1.0}

    monkeypatch.setattr(lme_runner, "_run_task_async", fake_task)
    summary = asyncio.run(
        lme_runner._run_all(
            cfg=None, task_names=["a", "b", "c", "d"], label="l", run_dir=tmp_path,
            agent_run=None, llm_openai=None, llm_agentscope=None, tracer=None,
            tools=[], system_prompt="", concurrency=4, judge_workers=1,
        )
    )
    assert list(summary) == ["a", "b", "c", "d"]      # order preserved
    # All four tasks in flight at once — the definitive overlap proof (wall
    # time is not asserted: _run_all ends with a fixed GC-settle sleep).
    assert running["peak"] == 4

    # And the semaphore still bounds the fan-out.
    running["now"] = running["peak"] = 0
    asyncio.run(
        lme_runner._run_all(
            cfg=None, task_names=["a", "b", "c", "d"], label="l", run_dir=tmp_path,
            agent_run=None, llm_openai=None, llm_agentscope=None, tracer=None,
            tools=[], system_prompt="", concurrency=2, judge_workers=1,
        )
    )
    assert running["peak"] == 2
