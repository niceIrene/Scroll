"""scroll_tools: JSON retrieval dispatch, formatting, and the full loop."""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

from opentelemetry import trace

from scroll_context import ScrollContextManager
from scroll_eval._tools_common import OPENAI_TOOLS_SCHEMA, select_tools
from scroll_eval.base_agents.scroll_tools.agent import (
    _dispatch_expand,
    _dispatch_search,
    run as tools_run,
)
from scroll_eval.evals.beam.ingest import SEED_RUN_ID, build_seed_db
from scroll_eval.types import LoopContext, TaskSpec, TerminationReason


def _chat() -> list[dict]:
    return [
        {"batch_number": 1, "turns": [[
            {"role": "user", "content": "Let's build a Flask budget tracker.",
             "id": 0, "time_anchor": "March-15-2024"},
            {"role": "assistant", "content": "Great — start with a transactions table.",
             "id": 1},
        ]]},
        {"batch_number": 2, "turns": [[
            {"role": "user", "content": "Add a category column to transactions.",
             "id": 0, "time_anchor": "April-01-2024"},
        ]]},
    ]


def _seeded_ms(tmp_path, task_id="beam/t1"):
    db = tmp_path / "hist.db"
    build_seed_db(_chat(), task_id, db)
    mgr = ScrollContextManager(
        history_db_path=db, session_id=f"r:{task_id}", run_id="r", task_id=task_id,
        history_max_tokens=0, shared_run_ids=(SEED_RUN_ID,),
    )
    return mgr, mgr.runtime.memoryspace


# --- dispatch/formatting -------------------------------------------------------


def test_search_formats_one_line_per_hit(tmp_path):
    mgr, ms = _seeded_ms(tmp_path)
    out = _dispatch_search(ms, {"query": "Flask budget tracker"})
    assert "hits" in out.splitlines()[0]
    assert "seq=" in out and "S1" in out and "2024-03-15" in out
    assert "conversation/user" in out
    mgr.close()


def test_search_k_saturation_note_and_zero_hits(tmp_path):
    mgr, ms = _seeded_ms(tmp_path)
    saturated = _dispatch_search(ms, {"query": "transactions OR Flask OR category", "k": 1})
    assert "filled k=1" in saturated
    empty = _dispatch_search(ms, {"query": "zeppelin quantum walrus"})
    assert empty.startswith("0 hits")
    mgr.close()


def test_search_seq_range_bounds_and_bad_args_are_observations(tmp_path):
    mgr, ms = _seeded_ms(tmp_path)
    all_hits = _dispatch_search(ms, {"query": "transactions"})
    assert "S1" in all_hits and "S2" in all_hits
    # Bound to session 2's seq (seq 3 is the third seeded row).
    bounded = _dispatch_search(ms, {"query": "transactions", "seq_range": [3, 3]})
    assert "S2" in bounded and "S1" not in bounded
    # Malformed args degrade to notes/errors, never exceptions.
    bad_range = _dispatch_search(ms, {"query": "transactions", "seq_range": [3]})
    assert "[bad argument] seq_range" in bad_range
    bad_k = _dispatch_search(ms, {"query": "transactions", "k": "ten"})
    assert "hits" in bad_k                       # fell back to default k
    missing_query = _dispatch_search(ms, {})
    assert missing_query.startswith("error:")
    broken_fts = _dispatch_search(ms, {"query": 'flask AND ("'})
    assert isinstance(broken_fts, str) and broken_fts  # sanitized internally
    mgr.close()


def test_expand_full_content_and_missing_seq_note(tmp_path):
    mgr, ms = _seeded_ms(tmp_path)
    out = _dispatch_expand(ms, {"seqs": [1, 999]})
    assert "--- seq=1" in out
    assert "Flask budget tracker" in out          # full content, not a snippet
    assert "1 of 2 requested seqs found" in out
    assert _dispatch_expand(ms, {"seqs": []}).startswith("error:")
    assert _dispatch_expand(ms, {"seqs": ["abc"]}).startswith("error:")
    mgr.close()


def test_tool_registry_serves_new_tools_and_legacy_surface_unchanged():
    tools = select_tools(["search_history", "expand_turns", "submit_answer"])
    assert [t["function"]["name"] for t in tools] == [
        "search_history", "expand_turns", "submit_answer",
    ]
    assert [t["function"]["name"] for t in OPENAI_TOOLS_SCHEMA] == ["bash", "submit_answer"]


# --- full loop -----------------------------------------------------------------


def _fake_tool_call(name, input_dict):
    return SimpleNamespace(type="tool_call", id=uuid.uuid4().hex, name=name,
                           input=json.dumps(input_dict))


def _fake_response(blocks):
    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(input_tokens=5, output_tokens=5),
    )


def test_loop_search_expand_submit(tmp_path):
    db = tmp_path / "hist.db"
    build_seed_db(_chat(), "beam/t1", db)
    responses = [
        _fake_response([_fake_tool_call("search_history", {"query": "Flask budget"})]),
        _fake_response([_fake_tool_call("expand_turns", {"seqs": [1]})]),
        _fake_response([_fake_tool_call("submit_answer", {"answer": "a Flask budget tracker"})]),
    ]
    model = AsyncMock(side_effect=responses)
    ctx = LoopContext(
        llm_openai=None, llm_agentscope=model, model_name="mock",
        tracer=trace.get_tracer("test"), budget=None, environment=None,
        tools=None, run_id="r", history_db_path=str(db),
        history_max_tokens=100_000, logs_dir=str(tmp_path / "logs"),
        system_prompt="BEAM-ADDENDUM", shared_run_ids=(SEED_RUN_ID,),
    )

    traj = asyncio.run(tools_run(TaskSpec("beam/t1", "What app did we build?"), ctx))

    assert traj.terminated == TerminationReason.SUCCESS
    assert traj.final_answer == "a Flask budget tracker"
    assert [s.action["tool"] for s in traj.steps] == [
        "search_history", "expand_turns", "submit_answer",
    ]
    assert "seq=" in traj.steps[0].observation          # formatted hit lines
    assert "Flask budget tracker" in traj.steps[1].observation  # full content
    assert traj.metrics["ms_ops"].get("hist_fts", 0) >= 1       # counters survive
    # The retrieval surface is tools-only: no execute_python schema was offered.
    offered = model.call_args_list[0].kwargs.get("tools") or model.call_args_list[0].args[1]
    names = [t["function"]["name"] for t in offered]
    assert "execute_python" not in names
    assert {"search_history", "expand_turns", "submit_answer"} <= set(names)
