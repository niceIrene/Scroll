from __future__ import annotations

import json
from pathlib import Path

from scroll_eval.harness import compare


def _make_run(root: Path, name: str, tasks: dict[str, dict]) -> Path:
    run_dir = root / name
    (run_dir / "tasks").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"agent": name}))
    summary = {}
    for tid, payload in tasks.items():
        td = run_dir / "tasks" / tid
        td.mkdir()
        (td / "harbor.json").write_text(json.dumps(payload))
        summary[tid] = payload
    (run_dir / "summary.json").write_text(json.dumps(summary))
    return run_dir


def test_compare_aligns_by_task_id(tmp_path: Path) -> None:
    a = _make_run(tmp_path, "A", {"t1": {"score": 1.0}, "t2": {"score": 0.0}})
    b = _make_run(tmp_path, "B", {"t1": {"score": 1.0}, "t2": {"score": 1.0}})

    report = compare.compare(a, b)

    rows = {r.task_id: r for r in report.rows}
    assert rows["t1"].a_score == 1.0 and rows["t1"].b_score == 1.0
    assert rows["t2"].a_score == 0.0 and rows["t2"].b_score == 1.0


def test_compare_handles_missing_task_in_one_run(tmp_path: Path) -> None:
    a = _make_run(tmp_path, "A", {"t1": {"score": 1.0}})
    b = _make_run(tmp_path, "B", {"t1": {"score": 1.0}, "t3": {"score": 0.5}})

    report = compare.compare(a, b)
    rows = {r.task_id: r for r in report.rows}
    assert rows["t3"].a_score is None
    assert rows["t3"].b_score == 0.5


def test_render_markdown_contains_both_run_names(tmp_path: Path) -> None:
    a = _make_run(tmp_path, "A", {"t1": {"score": 1.0}})
    b = _make_run(tmp_path, "B", {"t1": {"score": 0.0}})
    report = compare.compare(a, b)
    md = compare.render_markdown(report)
    assert "A" in md and "B" in md and "t1" in md
