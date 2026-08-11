"""Native runner: instruction framing + a full _run_task with stubbed agent/judge."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from scroll_eval.evals.longmemeval import runner as lme_runner
from scroll_eval.harness import config as cfg_mod
from scroll_eval.types import TerminationReason, Trajectory


def test_group_by_type_preserves_order() -> None:
    answers = [
        {"id": "qa-1", "type": "multi-session", "question": "a", "llm_response": "x"},
    ]
    grouped = lme_runner._group_by_type(answers)
    assert list(grouped) == ["multi-session"]
    assert grouped["multi-session"][0].keys() == {"id", "question", "llm_response"}


def test_instruction_carries_framing_and_qtype_nudge() -> None:
    probe = {"id": "qa-1", "type": "temporal-reasoning", "question": "When did X?",
             "question_date": "2024/06/01 (Sat) 10:00"}
    instr = lme_runner._instruction_for(probe)
    assert "As of 2024/06/01 (Sat) 10:00" in instr        # framing dated
    assert "When did X?" in instr
    assert "Time-sensitive question" in instr             # qtype postscript
    assert "submit_answer" in instr                       # base nudge


def test_instruction_abstention_overrides_qtype() -> None:
    probe = {"id": "qa-1_abs", "type": "single-session-preference",
             "question": "Recommend a hike.", "question_date": "2024/06/01"}
    instr = lme_runner._instruction_for(probe)
    assert "unanswerable" in instr.lower()                # abstention nudge chosen
    assert "Preference question" not in instr             # qtype nudge suppressed


def _make_task(local: Path) -> None:
    d = local / "longmemeval" / "qa-1"
    (d / "tests").mkdir(parents=True)
    (d / "task.toml").write_text(
        '[task]\nname = "longmemeval/qa-1"\n[metadata]\nrunner = "native"\n'
        'benchmark = "longmemeval"\nquestion_type = "multi-session"\n',
        encoding="utf-8",
    )
    (d / "sessions.json").write_text(json.dumps({
        "question_date": "2024/06/01 (Sat) 10:00",
        "sessions": [
            {"session_id": "s0", "date": "2024/03/15 (Fri) 09:30", "turns": [
                {"role": "user", "content": "I moved to Paris in March."},
                {"role": "assistant", "content": "Noted — Paris since March."},
            ]},
        ],
    }), encoding="utf-8")
    (d / "questions.json").write_text(json.dumps([
        {"id": "qa-1", "type": "multi-session", "question": "Where do I live?",
         "question_date": "2024/06/01 (Sat) 10:00"},
    ]), encoding="utf-8")
    (d / "tests" / "probing_questions.json").write_text(json.dumps({
        "multi-session": [{"id": "qa-1", "question": "Where do I live?", "answer": "Paris",
                           "question_type": "multi-session", "is_abstention": False}],
    }), encoding="utf-8")


def _cfg(monkeypatch) -> object:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "test-model")
    return cfg_mod.load(Path("configs/longmemeval.yaml"))


def test_run_task_end_to_end_with_stubs(tmp_path: Path, monkeypatch) -> None:
    """ingest -> single-probe run -> grouped answers -> (stub) judge, on a synthetic task."""
    local = tmp_path / "local"
    _make_task(local)
    monkeypatch.setattr(lme_runner, "_LOCAL_TASKS_ROOT", local)

    cfg = _cfg(monkeypatch)
    seen: dict = {}

    async def fake_agent_run(task, ctx):
        seen["system_prompt"] = ctx.system_prompt
        seen["instruction"] = task.instruction
        seen["shared_run_ids"] = ctx.shared_run_ids
        return Trajectory(
            task_id=task.task_id, steps=[],
            final_answer="You live in Paris.", terminated=TerminationReason.SUCCESS,
            metrics={"tokens_in": 1, "tokens_out": 1},
        )

    @contextmanager
    def fake_task_run(*a, **k):
        yield None

    monkeypatch.setattr(lme_runner.otel, "task_run", fake_task_run)
    monkeypatch.setattr(
        lme_runner, "_judge",
        lambda task_dir, task_out, workers=8: {
            "overall_reward": 1.0, "n_probes": 1,
            "per_type": {"multi-session": {"mean": 1.0}},
        },
    )

    task_out = tmp_path / "task"
    entry = lme_runner._run_task(
        cfg, "qa-1", task_out, "lbl",
        agent_run=fake_agent_run, llm_openai=None, llm_agentscope=None,
        tracer=None, tools=[], system_prompt="SYS-GUIDANCE", index=1, total=1,
    )

    assert entry["score"] == 1.0
    assert entry["n_probes"] == 1

    # Guidance rode in the system prompt; the user message is the framed question.
    assert seen["system_prompt"] == "SYS-GUIDANCE"
    assert "SYS-GUIDANCE" not in seen["instruction"]
    assert "Where do I live?" in seen["instruction"]
    assert seen["shared_run_ids"] == (lme_runner.SEED_RUN_ID,)

    # Answers grouped by type + written for the judge.
    answers = json.loads((task_out / "answers.json").read_text())
    assert answers["multi-session"][0]["llm_response"] == "You live in Paris."

    # The probe wrote a trajectory.json and the seed DB was materialized.
    assert list((task_out / "probes").rglob("trajectory.json"))
    assert (task_out / "seed.db").exists()
