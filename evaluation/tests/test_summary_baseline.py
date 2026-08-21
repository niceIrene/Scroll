"""summary_baseline: chunked rolling summary (QwenPaw en template), summary-only QA."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from opentelemetry import trace

from scroll_eval.base_agents.summary_baseline.agent import (
    _DEFAULT_CHUNK_TOKENS,
    _build_summary_prompt,
    _chunk_rows,
    _chunk_tokens,
    _clean_summary,
    _load_seed_rows,
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


def _ctx(
    model,
    db_path,
    history_max_tokens=None,
    system_prompt=None,
    logs_dir=None,
    summary_chunk_tokens=None,
):
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
        summary_chunk_tokens=summary_chunk_tokens,
        logs_dir=logs_dir,
        system_prompt=system_prompt,
    )


def _fake_response(text):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
    )


_SUMMARY_MD = """## Active Task
Build a Flask budget tracker
Status: in_progress

## Current State
- transactions table designed

## Constraints
- (none)

## Decisions
- (none)

## Open Work
- add category column
"""


def test_load_seed_rows_keeps_seq_order(tmp_path):
    db = tmp_path / "seed.db"
    build_seed_db(_chat(), "beam/t1", db)
    rows = _load_seed_rows(str(db))
    assert len(rows) == 3
    seqs = [seq for seq, _ in rows]
    assert seqs == sorted(seqs)
    assert "Flask budget tracker" in rows[0][1]
    assert "category column" in rows[2][1]


def test_chunk_rows_packs_whole_turns():
    rows = [(i, f"turn {i} " + "x" * 20) for i in range(6)]
    per_turn = len(rows[0][1]) + 1
    chunks = _chunk_rows(rows, chunk_chars=per_turn * 2)
    assert [len(c) for c in chunks] == [2, 2, 2]
    # Turns are never split and order is preserved.
    assert [seq for c in chunks for seq, _ in c] == list(range(6))


def test_chunk_rows_oversize_turn_gets_own_chunk():
    rows = [(1, "small"), (2, "y" * 500), (3, "small too")]
    chunks = _chunk_rows(rows, chunk_chars=100)
    assert [[seq for seq, _ in c] for c in chunks] == [[1], [2], [3]]


def test_build_summary_prompt_initial_vs_update():
    initial = _build_summary_prompt("initial", None, "archived text", (1, 5))
    assert "Create the first continuation summary" in initial
    assert "## Active Task" in initial and "## Open Work" in initial
    assert "1–5" in initial                      # covered seq range
    assert "(none)" in initial                   # no previous summary
    assert "archived text" in initial
    assert "never exceed 4000 tokens" in initial

    update = _build_summary_prompt("update", "PREV-SENTINEL", "new chunk", (1, 9))
    assert "Update the previous continuation summary" in update
    assert "PREV-SENTINEL" in update
    assert "new chunk" in update
    assert "1–9" in update


def test_clean_summary_strips_fence_and_source_links():
    fenced = "```markdown\n## Active Task\nthing [seq:3-7] done [file:a.py]\n```"
    assert _clean_summary(fenced) == "## Active Task\nthing  done"


def test_chunk_tokens_precedence_env_then_ctx_then_default(monkeypatch):
    monkeypatch.delenv("SCROLL_SUMMARY_CHUNK_TOKENS", raising=False)
    assert _chunk_tokens(None) == _DEFAULT_CHUNK_TOKENS
    ctx = _ctx(None, None, summary_chunk_tokens=450_000)
    assert _chunk_tokens(ctx) == 450_000
    monkeypatch.setenv("SCROLL_SUMMARY_CHUNK_TOKENS", "1234")
    assert _chunk_tokens(ctx) == 1234  # env var wins over the config knob


def test_run_uses_ctx_summary_chunk_tokens(tmp_path, monkeypatch):
    """The config knob shapes chunking end-to-end when the env var is unset."""
    monkeypatch.delenv("SCROLL_SUMMARY_CHUNK_TOKENS", raising=False)
    db = tmp_path / "seed.db"
    build_seed_db(_chat(), "beam/t1", db)
    model = AsyncMock(side_effect=[_fake_response(_SUMMARY_MD), _fake_response("42")])
    # A huge chunk budget folds the whole seed in ONE call: 1 fold + 1 QA.
    ctx = _ctx(model, str(db), summary_chunk_tokens=1_000_000)
    traj = asyncio.run(baseline_run(TaskSpec(task_id="t", instruction="q?"), ctx))
    assert traj.metrics["chunk_tokens"] == 1_000_000
    assert traj.metrics["summary_chunks"] == 1
    assert traj.final_answer == "42"


def test_run_rolls_summary_then_answers_from_it(tmp_path, monkeypatch):
    db = tmp_path / "seed.db"
    build_seed_db(_chat(), "beam/t1", db)
    # Tiny chunk budget → each of the 3 turns becomes its own chunk.
    monkeypatch.setenv("SCROLL_SUMMARY_CHUNK_TOKENS", "1")
    model = AsyncMock(side_effect=[
        _fake_response(_SUMMARY_MD),
        _fake_response(_SUMMARY_MD.replace("in_progress", "blocked")),
        _fake_response(_SUMMARY_MD),
        _fake_response("They built a Flask budget tracker."),
    ])
    ctx = _ctx(model, str(db), system_prompt="ADDENDUM-SENTINEL",
               logs_dir=str(tmp_path / "logs"))

    traj = asyncio.run(baseline_run(TaskSpec("beam/t1", "What app did we build?"), ctx))

    assert model.await_count == 4                       # 3 summary updates + QA
    assert all(c.kwargs.get("tools") is None for c in model.call_args_list)
    assert traj.final_answer == "They built a Flask budget tracker."
    assert traj.terminated == TerminationReason.SUCCESS
    assert len(traj.steps) == 4 and all(s.action is None for s in traj.steps)

    def _text(call):
        return "".join(
            b if isinstance((b := getattr(m, "content", "")), str)
            else "".join(getattr(x, "text", "") for x in b)
            for m in call[0][0]
        )

    assert "Create the first continuation summary" in _text(model.call_args_list[0])
    # Update calls carry the previous summary forward as the baseline.
    second = _text(model.call_args_list[1])
    assert "Update the previous continuation summary" in second
    assert "transactions table designed" in second
    # The QA call sees the FINAL summary and the eval addendum — no transcript.
    final = _text(model.call_args_list[3])
    assert "## Active Task" in final
    assert "ADDENDUM-SENTINEL" in final
    assert "What app did we build?" in final
    assert "Let's build a Flask budget tracker." not in final

    # Metrics stubs keep run-analysis tooling working on baseline runs.
    assert traj.metrics["ms_ops"] == {}
    assert traj.metrics["eviction"]["turns"] == 0
    assert traj.metrics["step_count"] == 4
    assert traj.metrics["summary_updates"] == 3
    assert traj.metrics["summary_chunks"] == 3
    assert traj.metrics["summary_failed_updates"] == 0
    assert traj.metrics["summary_cached"] is False
    assert traj.metrics["transcript_turns"] == 3
    assert traj.metrics["tokens_in"] == 28 and traj.metrics["tokens_out"] == 12


def test_run_failed_update_keeps_last_good_summary(tmp_path, monkeypatch):
    db = tmp_path / "seed.db"
    build_seed_db(_chat(), "beam/t1", db)
    monkeypatch.setenv("SCROLL_SUMMARY_CHUNK_TOKENS", "1")
    model = AsyncMock(side_effect=[
        _fake_response(_SUMMARY_MD),
        _fake_response(""),               # failed update: empty response
        _fake_response(""),               # failed update: empty response
        _fake_response("answer"),
    ])
    ctx = _ctx(model, str(db))

    traj = asyncio.run(baseline_run(TaskSpec("beam/t1", "q?"), ctx))

    assert traj.metrics["summary_failed_updates"] == 2
    # The QA call still sees the last good summary, not an empty one.
    qa_messages = model.call_args_list[3][0][0]
    qa_text = "".join(
        b if isinstance((b := getattr(m, "content", "")), str)
        else "".join(getattr(x, "text", "") for x in b)
        for m in qa_messages
    )
    assert "transactions table designed" in qa_text
    assert traj.final_answer == "answer"


def test_run_no_seed_rows_answers_from_empty_summary(tmp_path):
    model = AsyncMock(return_value=_fake_response("no idea"))
    ctx = _ctx(model, None)

    traj = asyncio.run(baseline_run(TaskSpec("beam/t1", "q?"), ctx))

    assert model.await_count == 1          # no summarization calls, QA only
    assert traj.metrics["summary_updates"] == 0
    assert traj.final_answer == "no idea"


def _qa_text(call) -> str:
    return "".join(
        b if isinstance((b := getattr(m, "content", "")), str)
        else "".join(getattr(x, "text", "") for x in b)
        for m in call[0][0]
    )


def test_run_reuses_cached_summary_across_probes(tmp_path, monkeypatch):
    db = tmp_path / "seed.db"
    build_seed_db(_chat(), "beam/t1", db)
    monkeypatch.setenv("SCROLL_SUMMARY_CHUNK_TOKENS", "1")
    first = AsyncMock(side_effect=[
        _fake_response(_SUMMARY_MD),
        _fake_response(_SUMMARY_MD),
        _fake_response(_SUMMARY_MD),
        _fake_response("answer one"),
    ])
    asyncio.run(baseline_run(TaskSpec("beam/t1", "q1?"), _ctx(first, str(db))))
    assert (tmp_path / "seed.db.summary.json").exists()

    # A sibling probe against the same seed DB skips phase 1 entirely.
    second = AsyncMock(return_value=_fake_response("answer two"))
    traj = asyncio.run(baseline_run(TaskSpec("beam/t1", "q2?"), _ctx(second, str(db))))

    assert second.await_count == 1                 # QA only
    assert traj.metrics["summary_cached"] is True
    assert traj.metrics["summary_updates"] == 0
    assert traj.final_answer == "answer two"
    # The QA call sees the CACHED summary.
    assert "transactions table designed" in _qa_text(second.call_args_list[0])


def test_run_cache_invalidated_by_chunk_tokens_change(tmp_path, monkeypatch):
    db = tmp_path / "seed.db"
    build_seed_db(_chat(), "beam/t1", db)
    monkeypatch.setenv("SCROLL_SUMMARY_CHUNK_TOKENS", "1")
    first = AsyncMock(side_effect=[_fake_response(_SUMMARY_MD)] * 3
                      + [_fake_response("a1")])
    asyncio.run(baseline_run(TaskSpec("beam/t1", "q1?"), _ctx(first, str(db))))

    # A different chunk budget must not reuse the stale summary.
    monkeypatch.setenv("SCROLL_SUMMARY_CHUNK_TOKENS", "2")
    second = AsyncMock(side_effect=[_fake_response(_SUMMARY_MD)] * 3
                       + [_fake_response("a2")])
    traj = asyncio.run(baseline_run(TaskSpec("beam/t1", "q2?"), _ctx(second, str(db))))

    assert traj.metrics["summary_cached"] is False
    assert traj.metrics["summary_updates"] >= 1


def test_run_deadline_stops_summarization_marks_budget(tmp_path, monkeypatch):
    db = tmp_path / "seed.db"
    build_seed_db(_chat(), "beam/t1", db)
    monkeypatch.setenv("SCROLL_SUMMARY_CHUNK_TOKENS", "1")
    model = AsyncMock(return_value=_fake_response("partial answer"))
    ctx = _ctx(model, str(db))
    ctx.budget = SimpleNamespace(wall_time_s=0)   # deadline already passed

    traj = asyncio.run(baseline_run(TaskSpec("beam/t1", "q?"), ctx))

    assert model.await_count == 1                  # QA only, no chunk calls
    assert traj.terminated == TerminationReason.BUDGET
    assert traj.metrics["summary_updates"] == 0
    assert traj.final_answer == "partial answer"
    # A budget-cut (partial) summary is never cached.
    assert not (tmp_path / "seed.db.summary.json").exists()


def test_concurrent_probes_summarize_once(tmp_path, monkeypatch):
    db = tmp_path / "seed.db"
    build_seed_db(_chat(), "beam/t1", db)
    monkeypatch.setenv("SCROLL_SUMMARY_CHUNK_TOKENS", "1")
    # 3 chunk calls total (first probe) + 2 QA calls; without the per-DB lock
    # the second probe would add 3 more chunk calls.
    model = AsyncMock(side_effect=[_fake_response(_SUMMARY_MD)] * 3
                      + [_fake_response("done")] * 2)

    async def _two_probes():
        return await asyncio.gather(
            baseline_run(TaskSpec("beam/t1", "q1?"), _ctx(model, str(db))),
            baseline_run(TaskSpec("beam/t1", "q2?"), _ctx(model, str(db))),
        )

    t1, t2 = asyncio.run(_two_probes())

    assert model.await_count == 5
    assert {t1.metrics["summary_cached"], t2.metrics["summary_cached"]} == {True, False}
    assert t1.final_answer == "done" and t2.final_answer == "done"
