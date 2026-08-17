"""longctx_baseline: transcript reconstruction, recency truncation, single-shot QA."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from opentelemetry import trace

from scroll_eval.base_agents.longctx_baseline.agent import (
    _fit_transcript,
    _load_seed_turns,
    run as baseline_run,
)
from scroll_eval.evals.beam.ingest import build_seed_db
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


def _ctx(model, db_path, history_max_tokens=None, system_prompt=None, logs_dir=None):
    return LoopContext(
        llm_openai=None,
        llm_agentscope=model,
        model_name="mock",
        tracer=trace.get_tracer("test"),
        budget=None,
        environment=None,
        tools=None,
        run_id="r",
        history_db_path=db_path,
        history_max_tokens=history_max_tokens,
        logs_dir=logs_dir,
        system_prompt=system_prompt,
    )


def _fake_response(text="the answer is 42"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
    )


def test_load_seed_turns_ordered_with_session_prefixes(tmp_path):
    db = tmp_path / "seed.db"
    build_seed_db(_chat(), "beam/t1", db)
    turns = _load_seed_turns(str(db))
    assert len(turns) == 3
    assert turns[0].startswith("[Session 1 | 2024-03-15]")
    assert turns[2].startswith("[Session 2 | 2024-04-01]")
    assert "Flask budget tracker" in turns[0]
    assert "category column" in turns[2]


def test_fit_transcript_no_drop_under_budget():
    turns = ["turn one", "turn two", "turn three"]
    text, dropped = _fit_transcript(list(turns), budget_chars=10_000)
    assert dropped == 0
    assert "[NOTE:" not in text
    assert text == "turn one\nturn two\nturn three"


def test_fit_transcript_drops_from_head_and_notices():
    turns = [f"turn number {i} with some padding text" for i in range(10)]
    budget = sum(len(t) + 1 for t in turns[5:])  # room for the last 5 only
    text, dropped = _fit_transcript(list(turns), budget_chars=budget)
    assert dropped == 5
    assert "earliest 5 turns are omitted" in text
    assert "turn number 9" in text        # most recent always survives
    assert "turn number 0" not in text    # oldest dropped
    # Recency: everything kept is a contiguous tail.
    assert text.index("turn number 5") < text.index("turn number 9")


def test_fit_transcript_degenerate_tiny_budget_keeps_tail():
    turns = ["x" * 50, "y" * 500]
    text, dropped = _fit_transcript(list(turns), budget_chars=100)
    assert dropped == 1
    assert text.endswith("y" * 100)  # tail slice of the last turn, never empty


def test_single_shot_run_shape_and_metrics_stub(tmp_path):
    db = tmp_path / "seed.db"
    build_seed_db(_chat(), "beam/t1", db)
    model = AsyncMock(return_value=_fake_response("They built a Flask budget tracker."))
    ctx = _ctx(model, str(db), history_max_tokens=100_000,
               system_prompt="ADDENDUM-SENTINEL", logs_dir=str(tmp_path / "logs"))

    traj = asyncio.run(baseline_run(TaskSpec("beam/t1", "What app did we build?"), ctx))

    assert model.await_count == 1                       # single shot
    assert model.call_args.kwargs.get("tools") is None  # no tool surface at all
    assert traj.final_answer == "They built a Flask budget tracker."
    assert traj.terminated == TerminationReason.SUCCESS
    assert len(traj.steps) == 1 and traj.steps[0].action is None
    # The model saw the transcript and the eval addendum.
    sent = model.call_args[0][0]
    all_text = "".join(
        b if isinstance((b := getattr(m, "content", "")), str)
        else "".join(getattr(x, "text", "") for x in b)
        for m in sent
    )
    assert "Flask budget tracker" in all_text
    assert "ADDENDUM-SENTINEL" in all_text
    # Metrics stubs keep run-analysis tooling working on baseline runs.
    assert traj.metrics["ms_ops"] == {}
    assert traj.metrics["eviction"]["turns"] == 0
    assert traj.metrics["step_count"] == 1
    assert traj.metrics["truncated"] is False
    assert traj.metrics["transcript_turns"] == 3


def test_run_truncates_when_budget_is_tiny(tmp_path, monkeypatch):
    db = tmp_path / "seed.db"
    # Long enough that the transcript exceeds the 1000-char budget floor.
    chat = [
        {"batch_number": n, "turns": [[
            {"role": "user", "content": f"session {n} filler " + "lorem ipsum " * 40,
             "id": 0, "time_anchor": "March-15-2024"},
        ]]}
        for n in range(1, 6)
    ]
    build_seed_db(chat, "beam/t1", db)
    model = AsyncMock(return_value=_fake_response())
    # 250 tokens ≈ 900 chars < the ~2500-char transcript — only the tail fits.
    monkeypatch.setenv("SCROLL_LONGCTX_MAX_TOKENS", "250")
    ctx = _ctx(model, str(db), history_max_tokens=100_000)

    traj = asyncio.run(baseline_run(TaskSpec("beam/t1", "q?"), ctx))

    assert traj.metrics["truncated"] is True
    assert traj.metrics["transcript_dropped_turns"] >= 1
    sent = model.call_args[0][0]
    all_text = "".join(
        b if isinstance((b := getattr(m, "content", "")), str)
        else "".join(getattr(x, "text", "") for x in b)
        for m in sent
    )
    assert "omitted" in all_text  # truncation notice reached the model
