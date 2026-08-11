from __future__ import annotations

import json
from pathlib import Path

from scroll_eval.harness import summarize


def _make_run(
    root: Path,
    name: str,
    *,
    agent: str = "scroll-eval",
    agent_label: str = "base_agents_base_agent_A",
    model: str = "m",
    tasks: dict[str, dict] | None = None,
) -> Path:
    run_dir = root / name
    (run_dir / "tasks").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"agent": agent, "agent_label": agent_label, "model": {"name": model}})
    )
    summary = {}
    for tid, payload in (tasks or {}).items():
        td = run_dir / "tasks" / tid
        td.mkdir()
        harbor_summary = {"exit_code": payload.get("exit_code", 0)}
        if "score" in payload:
            harbor_summary["score"] = payload["score"]
        (td / "harbor.json").write_text(json.dumps(harbor_summary))
        summary[tid] = {
            "exit_code": payload.get("exit_code", 0),
            **({"score": payload["score"]} if "score" in payload else {}),
        }
        # optional trajectory.json under harbor-out/job/trial/agent/
        if "trajectory" in payload:
            agent_dir = td / "harbor-out" / "j" / "t" / "agent"
            agent_dir.mkdir(parents=True)
            (agent_dir / "trajectory.json").write_text(json.dumps(payload["trajectory"]))
    (run_dir / "summary.json").write_text(json.dumps(summary))
    return run_dir


def test_summarize_basic_aggregates(tmp_path: Path) -> None:
    run = _make_run(
        tmp_path,
        "R",
        tasks={
            "a": {"exit_code": 0, "score": 1.0},
            "b": {"exit_code": 0, "score": 0.0},
            "c": {"exit_code": 1},  # errored - no score
            "d": {"exit_code": 0, "score": 1.0},
        },
    )
    s = summarize.summarize(run)
    assert s.total == 4
    assert s.passed == 2
    assert s.failed == 1
    assert s.errored == 1
    assert s.pass_rate == 0.5
    assert s.agent == "scroll-eval"
    assert s.agent_label == "base_agents_base_agent_A"


def test_summarize_pulls_trajectory_metrics(tmp_path: Path) -> None:
    run = _make_run(
        tmp_path,
        "R",
        tasks={
            "task-x": {
                "exit_code": 0,
                "score": 1.0,
                "trajectory": {
                    "metrics": {
                        "tokens_in": 1234,
                        "tokens_out": 567,
                        "step_count": 7,
                        "wall_time_s": 89.5,
                    }
                },
            },
        },
    )
    s = summarize.summarize(run)
    row = s.rows[0]
    assert row.tokens_in == 1234
    assert row.tokens_out == 567
    assert row.steps == 7
    assert row.wall_time_s == 89.5
    assert s.total_tokens_in == 1234
    assert s.total_tokens_out == 567


def test_render_table_includes_header_and_rows(tmp_path: Path) -> None:
    run = _make_run(
        tmp_path,
        "RUN1",
        tasks={"hello-world": {"exit_code": 0, "score": 1.0}},
    )
    out = summarize.render_table(summarize.summarize(run))
    assert "RUN1" in out
    assert "hello-world" in out
    assert "Pass rate:" in out


def test_render_table_filter_failed_only(tmp_path: Path) -> None:
    run = _make_run(
        tmp_path,
        "R",
        tasks={
            "pass1": {"exit_code": 0, "score": 1.0},
            "pass2": {"exit_code": 0, "score": 1.0},
            "fail1": {"exit_code": 0, "score": 0.0},
        },
    )
    out = summarize.render_table(summarize.summarize(run), show_only="failed")
    assert "fail1" in out
    # passed tasks should not appear in the per-task rows
    rows_part = out.split("------")[-1]
    assert "pass1" not in rows_part
    assert "pass2" not in rows_part


def test_render_markdown_writes_table_format(tmp_path: Path) -> None:
    run = _make_run(
        tmp_path,
        "RUN2",
        tasks={"t": {"exit_code": 0, "score": 1.0}},
    )
    md = summarize.render_markdown(summarize.summarize(run))
    assert md.startswith("# Summary: RUN2")
    assert "| task |" in md
    assert "| t | 0 | 1.00 |" in md
