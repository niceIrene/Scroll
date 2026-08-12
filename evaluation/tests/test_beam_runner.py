"""Native runner: helpers + a full _run_task with a stubbed agent and judge."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from scroll_eval.evals.beam import runner as beam_runner
from scroll_eval.harness import config as cfg_mod
from scroll_eval.types import TerminationReason, Trajectory


def test_group_by_type_preserves_order() -> None:
    answers = [
        {"id": "abstention-0", "type": "abstention", "question": "a", "llm_response": "x"},
        {"id": "abstention-1", "type": "abstention", "question": "b", "llm_response": "y"},
        {"id": "summarization-0", "type": "summarization", "question": "c", "llm_response": "z"},
    ]
    grouped = beam_runner._group_by_type(answers)
    assert list(grouped) == ["abstention", "summarization"]
    assert [a["id"] for a in grouped["abstention"]] == ["abstention-0", "abstention-1"]
    assert grouped["abstention"][0].keys() == {"id", "question", "llm_response"}


def _fake_run_dir(tmp_path: Path, monkeypatch, *, tasks=("T1", "T2"), answered=None):
    """A self-contained run dir + fake local-tasks root (rubric stubs)."""
    answered = set(tasks if answered is None else answered)
    local = tmp_path / "local"
    for name in tasks:
        (local / "beam" / name / "tests").mkdir(parents=True)
        (local / "beam" / name / "tests" / "probing_questions.json").write_text("{}")
    monkeypatch.setattr(beam_runner, "_LOCAL_TASKS_ROOT", local)

    run_dir = tmp_path / "run"
    for name in tasks:
        (run_dir / "tasks" / name).mkdir(parents=True)
        if name in answered:
            (run_dir / "tasks" / name / "answers.json").write_text("{}")
    (run_dir / "manifest.json").write_text(json.dumps({
        "benchmark": "beam", "agent": {"type": "base_agents", "id": "scroll_react"},
        "model": {"endpoint": "https://x/v1", "name": "big-model"},
        "tasks": list(tasks), "mean_reward": 0.9, "timestamp_utc": "t0",
    }))
    return run_dir


def test_grade_run_rebuilds_summary_and_manifest(tmp_path: Path, monkeypatch) -> None:
    run_dir = _fake_run_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(
        beam_runner, "_judge",
        lambda task_dir, task_out, workers=8: {
            "overall_reward": 0.4 if task_dir.name == "T1" else 0.8,
            "n_probes": 20, "per_type": {"abstention": {"mean": 0.6}},
        },
    )
    monkeypatch.setenv("SCROLL_JUDGE_MODEL", "small-judge")

    out = beam_runner.grade_run(run_dir, judge_workers=4)
    assert out == run_dir

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["T1"]["score"] == 0.4 and summary["T2"]["score"] == 0.8

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["mean_reward"] == pytest.approx(0.6)        # recomputed
    assert manifest["judge"] == {"name": "small-judge"}          # stamped
    assert "regraded_utc" in manifest
    assert manifest["agent"]["id"] == "scroll_react"           # original fields kept


def test_grade_run_skips_tasks_without_answers(tmp_path: Path, monkeypatch) -> None:
    # T2 has no answers.json -> skipped; mean is over graded tasks only.
    run_dir = _fake_run_dir(tmp_path, monkeypatch, tasks=("T1", "T2"), answered={"T1"})
    monkeypatch.setattr(
        beam_runner, "_judge",
        lambda task_dir, task_out, workers=8: {
            "overall_reward": 0.5, "n_probes": 20, "per_type": {"abstention": {"mean": 0.5}},
        },
    )
    monkeypatch.delenv("SCROLL_JUDGE_MODEL", raising=False)

    beam_runner.grade_run(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert set(summary) == {"T1"}                                # T2 skipped
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["mean_reward"] == pytest.approx(0.5)
    # No judge override -> falls back to the run's original model.
    assert manifest["judge"] == {"name": "big-model"}


def _cfg(monkeypatch) -> object:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "test-model")
    return cfg_mod.load(Path("configs/beam.yaml"))


def test_run_task_end_to_end_with_stubs(tmp_path: Path, monkeypatch) -> None:
    """ingest -> per-probe loop -> grouped answers -> (stub) judge, on real 100K-1."""
    if not Path("local-tasks/beam/100K-1/task.toml").exists():
        pytest.skip("beam 100K not migrated; run scripts/migrate_beam.py")

    cfg = _cfg(monkeypatch)
    seen: dict = {}

    # Stub agent: instant Trajectory; record what the runner handed it.
    async def fake_agent_run(task, ctx):
        seen["system_prompt"] = ctx.system_prompt
        seen["instruction"] = task.instruction
        return Trajectory(
            task_id=task.task_id, steps=[],
            final_answer=f"answer::{task.task_id}", terminated=TerminationReason.SUCCESS,
            metrics={"tokens_in": 1, "tokens_out": 1},
        )

    # Neutralize tracing + the judge subprocess.
    @contextmanager
    def fake_task_run(*a, **k):
        yield None

    monkeypatch.setattr(beam_runner.otel, "task_run", fake_task_run)
    monkeypatch.setattr(
        beam_runner, "_judge",
        lambda task_dir, task_out, workers=8: {
            "overall_reward": 0.5, "n_probes": 20,
            "per_type": {"abstention": {"mean": 0.5}},
        },
    )

    task_out = tmp_path / "task"
    entry = beam_runner._run_task(
        cfg, "100K-1", task_out, "lbl",
        agent_run=fake_agent_run, llm_openai=None, llm_agentscope=None,
        tracer=None, tools=[], system_prompt="SYS-GUIDANCE", index=1, total=1,
    )

    assert entry["score"] == 0.5
    assert entry["n_probes"] == 20

    # Guidance went into the system prompt; the user message is just the question.
    assert seen["system_prompt"] == "SYS-GUIDANCE"
    assert "SYS-GUIDANCE" not in seen["instruction"]
    assert seen["instruction"]  # non-empty question text

    # Answers were grouped by type and written for the judge.
    answers = json.loads((task_out / "answers.json").read_text())
    assert sum(len(v) for v in answers.values()) == 20
    assert answers["abstention"][0]["llm_response"] == "answer::beam/100K-1"

    # Each probe wrote a trajectory.json under its log dir.
    trajs = list((task_out / "probes").rglob("trajectory.json"))
    assert len(trajs) == 20

    # The seed DB was materialized.
    assert (task_out / "seed.db").exists()
