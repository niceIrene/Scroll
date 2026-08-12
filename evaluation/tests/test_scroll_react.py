"""Integration tests for scroll_react.

Mirrors tests/test_base_agent_A.py: scripted AgentScope ChatResponses fed
through the agent loop, with assertions on Trajectory shape, the three
tool branches, and runtime/log integration.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

from opentelemetry import trace

from scroll_eval.base_agents.scroll_react.agent import run as scroll_run
from scroll_eval.types import LoopContext, TaskSpec, TerminationReason


def _ctx(model, environment=None, tools=None, db_path=None, history_max_tokens=None,
         logs_dir=None):
    return LoopContext(
        llm_openai=None,
        llm_agentscope=model,
        model_name="mock",
        tracer=trace.get_tracer("test"),
        budget=None,
        environment=environment,
        tools=tools,
        run_id="r",
        history_db_path=db_path,
        history_max_tokens=history_max_tokens,
        logs_dir=logs_dir,
    )


def _fake_tool_call(name: str, input_dict: dict) -> SimpleNamespace:
    return SimpleNamespace(
        type="tool_call",
        id=uuid.uuid4().hex,
        name=name,
        input=json.dumps(input_dict),
    )


def _fake_text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _fake_response(*, blocks=None, usage_in=5, usage_out=5):
    return SimpleNamespace(
        content=blocks or [],
        usage=SimpleNamespace(input_tokens=usage_in, output_tokens=usage_out),
    )


def test_ctx_system_prompt_is_appended_not_replacing(tmp_path):
    """ctx.system_prompt augments the agent's capability prompt, not replaces it.

    The composed system message must contain BOTH the bundled system.md
    (capability layer) and the eval-supplied addendum, in that order.
    """
    from scroll_eval.base_agents.scroll_react import prompts as agent_prompts
    base = agent_prompts.load("system")

    model = AsyncMock()
    model.return_value = _fake_response(
        blocks=[_fake_tool_call("submit_answer", {"answer": "ok"})]
    )
    ctx = _ctx(model, db_path=str(tmp_path / "memory.db"))
    ctx.system_prompt = "TASK-SPECIFIC-ADDENDUM-SENTINEL"
    asyncio.run(scroll_run(TaskSpec("t", "q?"), ctx))

    raw = model.call_args[0][0][0].content
    system_content = raw if isinstance(raw, str) else "".join(
        getattr(b, "text", "") for b in raw
    )
    assert base in system_content                          # capability layer kept
    assert "TASK-SPECIFIC-ADDENDUM-SENTINEL" in system_content  # task layer added
    assert system_content.index(base) < system_content.index("TASK-SPECIFIC-ADDENDUM-SENTINEL")


def test_scroll_agent_terminates_on_submit_answer(tmp_path):
    model = AsyncMock()
    model.return_value = _fake_response(
        blocks=[
            _fake_text("trivial"),
            _fake_tool_call("submit_answer", {"answer": "42"}),
        ],
    )
    traj = asyncio.run(scroll_run(
        TaskSpec("t", "answer?"), _ctx(model, db_path=str(tmp_path / "history.db"))
    ))
    assert traj.terminated == TerminationReason.SUCCESS
    assert traj.final_answer == "42"
    assert len(traj.steps) == 1


def test_scroll_agent_runs_bash_then_submits(tmp_path):
    env = AsyncMock()
    env.exec = AsyncMock(
        return_value=SimpleNamespace(stdout="hello\n", stderr="", return_code=0)
    )
    model = AsyncMock()
    model.side_effect = [
        _fake_response(blocks=[
            _fake_text("let me check"),
            _fake_tool_call("bash", {"command": "echo hello"}),
        ]),
        _fake_response(blocks=[
            _fake_text("done"),
            _fake_tool_call("submit_answer", {"answer": "found"}),
        ]),
    ]
    traj = asyncio.run(scroll_run(
        TaskSpec("t", "echo hello"),
        _ctx(model, environment=env, db_path=str(tmp_path / "history.db")),
    ))
    assert traj.terminated == TerminationReason.SUCCESS
    assert traj.final_answer == "found"
    assert len(traj.steps) == 2
    assert env.exec.await_count == 1


def test_scroll_agent_runs_execute_python_then_submits(tmp_path):
    """execute_python should run in the runtime and feed back stdout."""
    model = AsyncMock()
    model.side_effect = [
        _fake_response(blocks=[
            _fake_text("computing"),
            _fake_tool_call("execute_python", {"source": "print(6 * 7)"}),
        ]),
        _fake_response(blocks=[
            _fake_text("done"),
            _fake_tool_call("submit_answer", {"answer": "42"}),
        ]),
    ]
    traj = asyncio.run(scroll_run(
        TaskSpec("t", "compute 6*7"), _ctx(model, db_path=str(tmp_path / "history.db"))
    ))
    assert traj.terminated == TerminationReason.SUCCESS
    assert traj.final_answer == "42"
    assert len(traj.steps) == 2
    # The execute_python step's observation must include stdout of '42'.
    exec_step = traj.steps[0]
    assert exec_step.action["tool"] == "execute_python"
    assert "42" in exec_step.observation


def test_scroll_agent_inline_submit_does_not_terminate(tmp_path):
    """submit_answer(...) inside execute_python is a plain error observation,
    not a terminator — the answer must come via the submit_answer tool."""
    model = AsyncMock()
    model.side_effect = [
        _fake_response(blocks=[
            _fake_text("submitting inline"),
            _fake_tool_call("execute_python", {"source": "submit_answer('inline-result')"}),
        ]),
        _fake_response(blocks=[
            _fake_tool_call("submit_answer", {"answer": "tool-result"}),
        ]),
    ]
    traj = asyncio.run(scroll_run(
        TaskSpec("t", "submit"), _ctx(model, db_path=str(tmp_path / "history.db"))
    ))
    assert traj.terminated == TerminationReason.SUCCESS
    assert traj.final_answer == "tool-result"
    assert len(traj.steps) == 2
    assert "error" in traj.steps[0].observation


def test_scroll_agent_history_is_visible_to_namespace(tmp_path):
    """After one model turn, this session's history is queryable via hist."""
    model = AsyncMock()
    model.side_effect = [
        _fake_response(blocks=[
            _fake_text("first turn"),
            _fake_tool_call("execute_python", {"source": (
                "rows = ms.sql_query("
                "'SELECT seq FROM hist.conversation_history WHERE session_id=?', "
                "(ms.session_id,))\n"
                "print(len(rows))"
            )}),
        ]),
        _fake_response(blocks=[
            _fake_tool_call("submit_answer", {"answer": "done"}),
        ]),
    ]
    traj = asyncio.run(scroll_run(
        TaskSpec("t", "introspect"), _ctx(model, db_path=str(tmp_path / "history.db"))
    ))
    assert traj.terminated == TerminationReason.SUCCESS
    # The session records the task instruction plus this turn's model_turn → 2.
    assert "2" in traj.steps[0].observation


def test_scroll_agent_respects_ctx_tools_override(tmp_path):
    """If ctx.tools is set, that schema is what's passed to the model."""
    from scroll_eval._tools_common import select_tools

    received: list = []

    async def fake_model(messages, tools=None):
        received.append(tools)
        return _fake_response(blocks=[
            _fake_tool_call("submit_answer", {"answer": "ok"}),
        ])

    override = select_tools(["execute_python", "submit_answer"])
    traj = asyncio.run(
        scroll_run(
            TaskSpec("t", "no bash"),
            _ctx(fake_model, tools=override, db_path=str(tmp_path / "history.db")),
        )
    )
    assert traj.terminated == TerminationReason.SUCCESS
    assert received[0] == override
    names = [t["function"]["name"] for t in received[0]]
    assert names == ["execute_python", "submit_answer"]


def test_command_timeout_clamping_and_default():
    from scroll_eval.base_agents.scroll_react.agent import _command_timeout
    assert _command_timeout({}, None) == 120                 # default
    assert _command_timeout({"timeout": 400}, None) == 400    # honored
    assert _command_timeout({"timeout": 10000}, None) == 600  # hard cap
    assert _command_timeout({"timeout": 1}, None) == 5        # floor (never ~1s)
    assert _command_timeout({"timeout": "bad"}, None) == 120  # bad input -> default
    assert _command_timeout({"timeout": 300}, 50) == 50       # capped to remaining
    assert _command_timeout({"timeout": 300}, 2) == 5         # floored, never 1s


def test_scroll_tools_advertise_bash_timeout():
    from scroll_eval.base_agents.scroll_react.agent import _scroll_tools
    from scroll_eval._tools_common import TOOLS
    tools = _scroll_tools()
    bash = next(t for t in tools if t["function"]["name"] == "bash")
    assert "timeout" in bash["function"]["parameters"]["properties"]
    # The shared canonical schema must stay untouched (fairness invariant).
    assert "timeout" not in TOOLS["bash"]["function"]["parameters"]["properties"]


def test_scroll_agent_honors_model_set_bash_timeout(tmp_path):
    captured: dict = {}

    async def _exec(command, timeout_sec=60):
        captured["timeout_sec"] = timeout_sec
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    env = SimpleNamespace(exec=_exec)
    model = AsyncMock()
    model.side_effect = [
        _fake_response(blocks=[_fake_tool_call("bash", {"command": "sleep 1", "timeout": 400})]),
        _fake_response(blocks=[_fake_tool_call("submit_answer", {"answer": "ok"})]),
    ]
    asyncio.run(scroll_run(
        TaskSpec("t", "x"),
        _ctx(model, environment=env, db_path=str(tmp_path / "history.db")),
    ))
    assert captured["timeout_sec"] == 400  # no wall budget -> honored within cap


def test_scroll_agent_stops_on_token_budget(tmp_path):
    """The loop budget gate breaks with BUDGET before MAX_STEPS is reached."""
    model = AsyncMock()
    # Never submits; would otherwise run to MAX_STEPS.
    model.return_value = _fake_response(
        blocks=[_fake_tool_call("execute_python", {"source": "print(1)"})],
        usage_in=5, usage_out=5,
    )
    ctx = _ctx(model, db_path=str(tmp_path / "history.db"))
    ctx.budget = SimpleNamespace(wall_time_s=None, max_tokens=5)
    traj = asyncio.run(scroll_run(TaskSpec("t", "x"), ctx))
    assert traj.terminated == TerminationReason.BUDGET
    # Turn 0 runs (0 tokens spent at the gate), spends 10 tokens; turn 1's gate
    # fires (10 >= 5) before the model is called again.
    assert len(traj.steps) == 1


def test_scroll_agent_reserves_last_step_for_submit(tmp_path):
    """The final step (MAX_STEPS-1) is reserved: tools narrow to submit_answer."""
    from scroll_eval.base_agents.scroll_react.agent import MAX_STEPS

    search = _fake_response(
        blocks=[_fake_tool_call("execute_python", {"source": "print(1)"})],
        usage_in=1, usage_out=1,
    )
    submit = _fake_response(
        blocks=[_fake_tool_call("submit_answer", {"answer": "reserved-42"})],
        usage_in=1, usage_out=1,
    )
    model = AsyncMock()
    # Never submits on its own — only the reserved last turn yields an answer.
    model.side_effect = [search] * (MAX_STEPS - 1) + [submit]
    traj = asyncio.run(scroll_run(
        TaskSpec("t", "x"), _ctx(model, db_path=str(tmp_path / "history.db"))
    ))
    assert traj.final_answer == "reserved-42"
    assert traj.terminated == TerminationReason.SUCCESS
    assert traj.metrics.get("forced_final_answer") is True
    # Exactly MAX_STEPS calls — no turn beyond budget — and the last was submit-only.
    assert model.call_count == MAX_STEPS
    last_tools = model.call_args_list[-1].kwargs["tools"]
    assert [t["function"]["name"] for t in last_tools] == ["submit_answer"]


def test_scroll_agent_reserved_turn_salvages_text_when_still_no_submit(tmp_path):
    """If the model ignores submit-only on the reserved turn, salvage its text."""
    from scroll_eval.base_agents.scroll_react.agent import MAX_STEPS

    search = _fake_response(
        blocks=[_fake_tool_call("execute_python", {"source": "print(1)"})],
        usage_in=1, usage_out=1,
    )
    text_only = _fake_response(blocks=[_fake_text("my best estimate is 7")], usage_in=1, usage_out=1)
    model = AsyncMock()
    model.side_effect = [search] * (MAX_STEPS - 1) + [text_only]
    traj = asyncio.run(scroll_run(
        TaskSpec("t", "x"), _ctx(model, db_path=str(tmp_path / "history.db"))
    ))
    assert traj.final_answer == "my best estimate is 7"
    assert traj.metrics.get("forced_final_answer") is True


def test_scroll_agent_reserved_turn_salvages_reasoning_when_text_empty(tmp_path):
    """Thinking mode: if the reserved turn yields only a thinking block (no submit,
    no visible text), salvage the reasoning so the answer isn't empty."""
    from scroll_eval.base_agents.scroll_react.agent import MAX_STEPS

    search = _fake_response(
        blocks=[_fake_tool_call("execute_python", {"source": "print(1)"})],
        usage_in=1, usage_out=1,
    )
    thinking_only = _fake_response(
        blocks=[SimpleNamespace(type="thinking", thinking="... so the total is 4800")],
        usage_in=1, usage_out=1,
    )
    model = AsyncMock()
    model.side_effect = [search] * (MAX_STEPS - 1) + [thinking_only]
    traj = asyncio.run(scroll_run(
        TaskSpec("t", "x"), _ctx(model, db_path=str(tmp_path / "history.db"))
    ))
    assert traj.final_answer == "... so the total is 4800"   # reasoning salvaged
    assert traj.metrics.get("forced_final_answer") is True


def test_scroll_agent_reserve_can_be_disabled(tmp_path, monkeypatch):
    """SCROLL_FORCE_FINAL_ANSWER=0 restores plain run-to-cap-and-stop-empty."""
    from scroll_eval.base_agents.scroll_react.agent import MAX_STEPS

    monkeypatch.setenv("SCROLL_FORCE_FINAL_ANSWER", "0")
    model = AsyncMock()
    model.return_value = _fake_response(
        blocks=[_fake_tool_call("execute_python", {"source": "print(1)"})],
        usage_in=1, usage_out=1,
    )
    traj = asyncio.run(scroll_run(
        TaskSpec("t", "x"), _ctx(model, db_path=str(tmp_path / "history.db"))
    ))
    assert traj.final_answer is None
    assert traj.terminated == TerminationReason.BUDGET
    assert "forced_final_answer" not in traj.metrics
    assert model.call_count == MAX_STEPS  # ran every step on search, none reserved


def _mgr(tmp_path, **kw):
    from scroll_context import ScrollContextManager
    kw.setdefault("history_db_path", str(tmp_path / "digest.db"))
    kw.setdefault("session_id", "r:t")
    kw.setdefault("history_max_tokens", 0)
    kw.setdefault("repl_name", "execute_python")
    return ScrollContextManager(**kw)


def test_digest_message_includes_reflection_prompt(tmp_path):
    """The per-turn working-notes message carries the self-check nudge."""
    mgr = _mgr(tmp_path)
    content = mgr.digest_message()["content"]
    assert "[working memory]" in content
    assert "vars: (empty)" in content             # the runtime digest is included
    assert "judge in one line" in content          # the reflection nudge
    assert "change approach" in content
    # No eviction yet -> no retrieval guidance (avoid generic boilerplate).
    assert "no longer in this prompt" not in content
    mgr.close()


def test_digest_message_surfaces_evicted_history_search(tmp_path):
    """Once history is evicted, the digest surfaces a content-search retrieval cue."""
    mgr = _mgr(tmp_path)
    mgr.totals["evicted_msgs"] = 7
    content = mgr.digest_message()["content"]
    assert "7 earlier turn(s) are no longer in this prompt" in content
    assert "ms.search(" in content and "ms.expand(" in content  # recall cue
    assert "judge in one line" in content           # reflection nudge still present
    mgr.close()


def test_scroll_agent_dumps_call_messages_jsonl(tmp_path):
    """Each turn's exact messages are streamed to call_messages.jsonl in logs_dir."""
    import json as _json

    logs = tmp_path / "logs"
    logs.mkdir()
    model = AsyncMock()
    model.side_effect = [
        _fake_response(blocks=[
            _fake_text("looking"),
            _fake_tool_call("bash", {"command": "ls"}),
        ]),
        _fake_response(blocks=[_fake_tool_call("submit_answer", {"answer": "ok"})]),
    ]
    env = AsyncMock()
    env.exec = AsyncMock(
        return_value=SimpleNamespace(stdout="x\n", stderr="", return_code=0)
    )
    asyncio.run(scroll_run(
        TaskSpec("tD", "do it"),
        _ctx(model, environment=env, db_path=str(tmp_path / "history.db"),
             logs_dir=str(logs)),
    ))
    dump = logs / "call_messages.jsonl"
    assert dump.exists()
    lines = [_json.loads(ln) for ln in dump.read_text().splitlines()]
    # First line is the constant system prompt, written once.
    assert lines[0]["system"]["role"] == "system"
    # Remaining lines are per-turn, and never repeat the system message.
    turns = lines[1:]
    assert [l["step"] for l in turns] == [0, 1]
    for t in turns:
        assert all(m["role"] != "system" for m in t["messages"])
    # Turn 0 starts with the task; turn 1 adds the assistant turn (+ digest).
    assert "do it" in str(turns[0]["messages"][0]["content"])
    assert len(turns[1]["messages"]) > len(turns[0]["messages"])


def test_scroll_agent_persists_to_conversation_history(tmp_path):
    """Every turn is write-through-persisted into the durable store."""
    import sqlite3

    db = tmp_path / "history.db"
    model = AsyncMock()
    model.return_value = _fake_response(blocks=[
        _fake_text("hi"),
        _fake_tool_call("submit_answer", {"answer": "x"}),
    ])
    asyncio.run(scroll_run(TaskSpec("tZ", "do it"), _ctx(model, db_path=str(db))))

    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT kind, content FROM conversation_history WHERE task_id='tZ' ORDER BY seq"
    ).fetchall()
    conn.close()
    kinds = [r[0] for r in rows]
    assert "task" in kinds          # the instruction was recorded
    assert "model_turn" in kinds    # the assistant turn was recorded
    # task instruction content is retrievable across sessions
    assert any(c == "do it" for _, c in rows)


def test_scroll_agent_second_session_can_read_first(tmp_path):
    """A later session retrieves a prior session's history from the shared DB."""
    db = tmp_path / "history.db"

    first = AsyncMock()
    first.return_value = _fake_response(blocks=[
        _fake_text("session one note"),
        _fake_tool_call("submit_answer", {"answer": "1"}),
    ])
    asyncio.run(scroll_run(TaskSpec("shared", "first task"), _ctx(first, db_path=str(db))))

    # Second session (different run_id) reads the first session's rows via the
    # read-only hist attach and reports how many it found.
    second = AsyncMock()
    second.side_effect = [
        _fake_response(blocks=[
            _fake_text("checking history"),
            _fake_tool_call("execute_python", {"source": (
                "rows = ms.sql_query("
                "\"SELECT content FROM hist.conversation_history "
                "WHERE task_id='shared' AND content='session one note'\")\n"
                "print('HITS', len(rows))"
            )}),
        ]),
        _fake_response(blocks=[_fake_tool_call("submit_answer", {"answer": "done"})]),
    ]
    ctx2 = _ctx(second, db_path=str(db))
    ctx2.run_id = "r2"
    traj = asyncio.run(scroll_run(TaskSpec("shared", "second task"), ctx2))
    assert traj.terminated == TerminationReason.SUCCESS
    assert "HITS 1" in traj.steps[0].observation


def test_resolve_response_passthrough_on_chatresponse():
    """A non-stream ChatResponse must pass through untouched.

    Regression: ChatResponse subclasses dict and its __getattr__ raises KeyError
    (not AttributeError) for missing keys, so a hasattr('__aiter__') probe leaked
    `KeyError('__aiter__')`. Detection must not touch instance attributes.
    """
    from agentscope.model._model_response import ChatResponse
    from scroll_eval.base_agents.scroll_react.agent import _resolve_response

    cr = ChatResponse(content=["x"], is_last=True)
    out = asyncio.run(_resolve_response(cr))  # must not raise
    assert out is cr


def test_resolve_response_collapses_stream_to_final_chunk():
    """A streamed async generator collapses to its terminal (is_last) chunk."""
    from agentscope.model._model_response import ChatResponse
    from scroll_eval.base_agents.scroll_react.agent import _resolve_response

    async def _gen():
        yield ChatResponse(content=["delta"], is_last=False)
        yield ChatResponse(content=["FINAL"], is_last=True)

    final = asyncio.run(_resolve_response(_gen()))
    assert final.is_last is True
    assert final.content == ["FINAL"]


def test_scroll_agent_retries_transient_model_error(tmp_path, monkeypatch):
    """A transient model failure is retried, then the run completes normally."""
    import scroll_eval.base_agents.scroll_react.agent as agent_mod

    async def _no_sleep(_):  # don't actually back off in tests
        return None

    monkeypatch.setattr(agent_mod.asyncio, "sleep", _no_sleep)

    ok = _fake_response(blocks=[_fake_tool_call("submit_answer", {"answer": "42"})])
    model = AsyncMock()
    # First call raises (e.g. a 429 burst), second call succeeds.
    model.side_effect = [RuntimeError("429 rate limit"), ok]

    traj = asyncio.run(scroll_run(
        TaskSpec("t", "answer?"), _ctx(model, db_path=str(tmp_path / "memory.db"))
    ))
    assert traj.terminated == TerminationReason.SUCCESS
    assert traj.final_answer == "42"
    assert model.call_count == 2


def test_scroll_agent_terminal_model_error_yields_error_trajectory(tmp_path, monkeypatch):
    """When retries are exhausted, the loop ends cleanly as ERROR (no exception)."""
    import scroll_eval.base_agents.scroll_react.agent as agent_mod

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(agent_mod.asyncio, "sleep", _no_sleep)

    model = AsyncMock()
    model.side_effect = RuntimeError("503 service unavailable")

    traj = asyncio.run(scroll_run(
        TaskSpec("t", "answer?"), _ctx(model, db_path=str(tmp_path / "memory.db"))
    ))
    # A trajectory is still produced — distinguishable from a budget stop.
    assert traj.terminated == TerminationReason.ERROR
    assert traj.final_answer is None
    assert "503 service unavailable" in traj.metrics.get("error", "")
    # Every attempt was made before giving up.
    assert model.call_count == agent_mod.LLM_MAX_ATTEMPTS


def test_strip_headline_schema_removes_column_for_index_off():
    """The index-OFF path must not advertise the `headline` column.

    Guards the coupling between `core.md`'s schema docs and
    `strip_headline_schema`: if a prompt edit changes the schema wording so the
    strip fragments no longer match, this fails loudly instead of silently
    re-leaking the index into a `--no-index` run.
    """
    from scroll_context import protocol_prompt, strip_headline_schema

    base = protocol_prompt("execute_python", index=True)
    # Sanity: the index-ON prompt DOES advertise the column (else the strip is moot).
    assert "headline" in base.lower()

    off = protocol_prompt("execute_python", index=False)
    assert "headline" not in off.lower(), "headline column still advertised"
    # The strip itself must keep matching core.md's wording.
    core_on = protocol_prompt("execute_python", index=True)
    assert strip_headline_schema(core_on) != core_on, (
        "strip fragments no longer match core.md — update them"
    )
