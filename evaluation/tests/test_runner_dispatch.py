from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Imported for its side effect BEFORE any patch.dict("sys.modules", ...) block:
# the dispatch path imports agentscope.model (and transitively mcp/pydantic
# submodules) lazily. patch.dict restores sys.modules to its entry snapshot on
# exit, so any module first imported INSIDE the block would be evicted while
# still referenced elsewhere — a later fresh import then crashes (observed as
# pydantic KeyError: 'pydantic.root_model'). Pre-importing keeps the block from
# loading anything new.
import agentscope.model  # noqa: F401

from scroll_eval.runner import ScrollEvalAgent


def test_dispatch_resolves_agent_module_by_type_and_id(tmp_path):
    """ScrollEvalAgent.run imports scroll_eval.<type>.<id>.agent and calls its run()."""
    called = {}

    async def fake_run(task, ctx):
        called["task_id"] = task.task_id
        called["model_name"] = ctx.model_name
        called["env"] = ctx.environment
        from scroll_eval.types import Trajectory, TerminationReason
        return Trajectory(
            task_id=task.task_id,
            steps=[],
            final_answer="ok",
            terminated=TerminationReason.SUCCESS,
            metrics={"tokens_in": 0, "tokens_out": 0, "wall_time_s": 0.0, "step_count": 0},
        )

    fake_mod = SimpleNamespace(run=fake_run)

    agent = ScrollEvalAgent(
        logs_dir=tmp_path,
        model_name="qwen3.7-max",
    )
    env = SimpleNamespace(task_id="hello")

    with patch.dict(
        "sys.modules",
        {"scroll_eval.base_agents.base_agent_A.agent": fake_mod},
    ), patch.dict(
        "os.environ",
        {"SCROLL_AGENT_TYPE": "base_agents", "SCROLL_AGENT_ID": "base_agent_A"},
    ):
        ctx = SimpleNamespace(
            n_input_tokens=0, n_output_tokens=0, cost_usd=0.0, metadata={}
        )
        asyncio.run(agent.run("solve this", env, ctx))

    assert called["task_id"] == "hello"
    assert called["model_name"] == "qwen3.7-max"
    assert ctx.metadata["family"] == "base_agents"
    assert ctx.metadata["agent_id"] == "base_agent_A"


def test_dispatch_rejects_unknown_combination(tmp_path):
    agent = ScrollEvalAgent(logs_dir=tmp_path, model_name="m")
    env = SimpleNamespace(task_id="t")
    ctx = SimpleNamespace(n_input_tokens=0, n_output_tokens=0, cost_usd=0.0, metadata={})
    with patch.dict(
        "os.environ",
        {"SCROLL_AGENT_TYPE": "nope", "SCROLL_AGENT_ID": "nada"},
    ):
        with pytest.raises(ModuleNotFoundError):
            asyncio.run(agent.run("x", env, ctx))
