from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

from opentelemetry import trace

from scroll_eval.base_agents.base_agent_A.agent import run as base_run
from scroll_eval.types import LoopContext, TaskSpec, TerminationReason


def _ctx(model, environment=None):
    return LoopContext(
        llm_openai=None,
        llm_agentscope=model,
        model_name="mock",
        tracer=trace.get_tracer("test"),
        budget=None,
        environment=environment,
    )


def _fake_tool_call_block(name: str, input_dict: dict) -> SimpleNamespace:
    """Mimics agentscope.message.ToolCallBlock.

    The loop reads .type, .name, .id, and .input (raw JSON string).
    """
    return SimpleNamespace(
        type="tool_call",
        id=uuid.uuid4().hex,
        name=name,
        input=json.dumps(input_dict),
    )


def _fake_text_block(text: str) -> SimpleNamespace:
    """Mimics agentscope.message.TextBlock."""
    return SimpleNamespace(type="text", text=text)


def _fake_chat_response(
    *,
    content_blocks=None,
    usage_in=5,
    usage_out=5,
):
    """Mimics agentscope.model._model_response.ChatResponse.

    The base_agent loop reads .content (list of block-like objects) and
    .usage with .input_tokens / .output_tokens.
    """
    return SimpleNamespace(
        content=content_blocks or [],
        usage=SimpleNamespace(input_tokens=usage_in, output_tokens=usage_out),
    )


def test_base_agent_succeeds_on_submit_answer():
    model = AsyncMock()
    model.return_value = _fake_chat_response(
        content_blocks=[
            _fake_text_block("trivial"),
            _fake_tool_call_block("submit_answer", {"answer": "42"}),
        ],
    )
    traj = asyncio.run(
        base_run(TaskSpec("t", "what is the answer?"), _ctx(model))
    )
    assert traj.terminated == TerminationReason.SUCCESS
    assert traj.final_answer == "42"
    assert len(traj.steps) == 1


def test_base_agent_runs_bash_then_submits():
    env = AsyncMock()
    env.exec = AsyncMock(
        return_value=SimpleNamespace(stdout="hello\n", stderr="", return_code=0)
    )

    model = AsyncMock()
    model.side_effect = [
        _fake_chat_response(
            content_blocks=[
                _fake_text_block("let me check"),
                _fake_tool_call_block("bash", {"command": "echo hello"}),
            ],
        ),
        _fake_chat_response(
            content_blocks=[
                _fake_text_block("done"),
                _fake_tool_call_block("submit_answer", {"answer": "found hello"}),
            ],
        ),
    ]

    traj = asyncio.run(
        base_run(TaskSpec("t", "echo hello"), _ctx(model, environment=env))
    )
    assert traj.terminated == TerminationReason.SUCCESS
    assert traj.final_answer == "found hello"
    assert len(traj.steps) == 2
    assert env.exec.await_count == 1
