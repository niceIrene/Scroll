from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Row:
    task_id: str
    a_score: float | None
    b_score: float | None
    a_exit: int | None
    b_exit: int | None


@dataclass(frozen=True)
class Report:
    a_name: str
    b_name: str
    rows: list[Row]


def _load_summary(run_dir: Path) -> dict[str, dict]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def compare(a_dir: Path, b_dir: Path) -> Report:
    a = _load_summary(a_dir)
    b = _load_summary(b_dir)
    task_ids = sorted(set(a) | set(b))
    rows = [
        Row(
            task_id=tid,
            a_score=a.get(tid, {}).get("score"),
            b_score=b.get(tid, {}).get("score"),
            a_exit=a.get(tid, {}).get("exit_code"),
            b_exit=b.get(tid, {}).get("exit_code"),
        )
        for tid in task_ids
    ]
    return Report(a_name=a_dir.name, b_name=b_dir.name, rows=rows)


def render_table(report: Report) -> str:
    header = f"{'task':<30} {report.a_name:>20} {report.b_name:>20}"
    lines = [header, "-" * len(header)]
    for r in report.rows:
        lines.append(
            f"{r.task_id:<30} "
            f"{_fmt_score(r.a_score):>20} "
            f"{_fmt_score(r.b_score):>20}"
        )
    return "\n".join(lines)


def _fmt_score(score: float | None) -> str:
    return "—" if score is None else f"{score:.2f}"


def render_markdown(report: Report) -> str:
    lines = [
        f"# Compare: {report.a_name} vs {report.b_name}",
        "",
        f"| task | {report.a_name} | {report.b_name} |",
        "|------|----------------|------------------|",
    ]
    for r in report.rows:
        lines.append(
            f"| {r.task_id} | {_fmt_score(r.a_score)} | {_fmt_score(r.b_score)} |"
        )
    return "\n".join(lines)
