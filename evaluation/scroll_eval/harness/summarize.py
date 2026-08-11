from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskRow:
    task_id: str
    exit_code: int | None
    score: float | None
    steps: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    wall_time_s: float | None = None


@dataclass(frozen=True)
class Summary:
    run_dir: Path
    agent: str
    agent_label: str | None
    model_name: str
    rows: list[TaskRow]

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.rows if r.score is not None and r.score >= 1.0)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.rows if r.score is not None and r.score < 1.0)

    @property
    def errored(self) -> int:
        return sum(1 for r in self.rows if r.score is None)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def total_tokens_in(self) -> int:
        return sum(r.tokens_in or 0 for r in self.rows)

    @property
    def total_tokens_out(self) -> int:
        return sum(r.tokens_out or 0 for r in self.rows)

    @property
    def total_wall_time_s(self) -> float:
        return sum(r.wall_time_s or 0.0 for r in self.rows)


def _load_trajectory(task_dir: Path) -> dict | None:
    """Find trajectory.json under task_dir/harbor-out/<job>/<trial>/agent/."""
    matches = list(task_dir.rglob("trajectory.json"))
    if not matches:
        return None
    try:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_harbor(task_dir: Path) -> dict | None:
    p = task_dir / "harbor.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _extract_score(harbor: dict | None) -> float | None:
    """Pull the mean reward from harbor.json (test or real shape)."""
    if not harbor:
        return None
    if isinstance(harbor.get("score"), (int, float)):
        return float(harbor["score"])
    evals = (harbor.get("result", {}) or {}).get("stats", {}).get("evals", {}) or {}
    for s in evals.values():
        metrics = s.get("metrics") or []
        if metrics and isinstance(metrics[0], dict) and "mean" in metrics[0]:
            try:
                return float(metrics[0]["mean"])
            except (TypeError, ValueError):
                continue
    return None


def summarize(run_dir: Path) -> Summary:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary_data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    rows: list[TaskRow] = []
    for task_id, entry in summary_data.items():
        task_dir = run_dir / "tasks" / task_id
        harbor = _load_harbor(task_dir)
        traj = _load_trajectory(task_dir)
        score = entry.get("score")
        if score is None:
            score = _extract_score(harbor)
        metrics = (traj or {}).get("metrics", {}) if isinstance(traj, dict) else {}
        rows.append(
            TaskRow(
                task_id=task_id,
                exit_code=entry.get("exit_code"),
                score=float(score) if isinstance(score, (int, float)) else None,
                steps=metrics.get("step_count"),
                tokens_in=metrics.get("tokens_in"),
                tokens_out=metrics.get("tokens_out"),
                wall_time_s=metrics.get("wall_time_s"),
            )
        )

    rows.sort(key=lambda r: r.task_id)
    return Summary(
        run_dir=run_dir,
        agent=manifest.get("agent", "?"),
        agent_label=manifest.get("agent_label") or manifest.get("loop"),
        model_name=(manifest.get("model") or {}).get("name", "?"),
        rows=rows,
    )


def _fmt_score(score: float | None) -> str:
    return "—" if score is None else f"{score:.2f}"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def render_header(s: Summary) -> str:
    return "\n".join(
        [
            f"Run:    {s.run_dir.name}",
            f"Agent:  {s.agent}    Agent: {s.agent_label or '-'}    Model: {s.model_name}",
            f"Tasks:  {s.total}    Passed: {s.passed}    Failed: {s.failed}    "
            f"Errored: {s.errored}    Pass rate: {s.pass_rate:.1%}",
            f"Tokens: in={s.total_tokens_in:,}  out={s.total_tokens_out:,}    "
            f"Total wall-time: {_fmt_duration(s.total_wall_time_s)}",
        ]
    )


def render_table(s: Summary, *, show_only: str | None = None) -> str:
    """Render a per-task table.

    `show_only`: "passed" | "failed" | "errored" | None.
    """
    header = render_header(s)
    cols = f"{'task':<36} {'exit':>5} {'score':>6} {'steps':>6} {'tok_in':>8} {'tok_out':>8} {'wall':>7}"
    sep = "-" * len(cols)
    lines = [header, "", cols, sep]
    for r in s.rows:
        if show_only == "passed" and not (r.score is not None and r.score >= 1.0):
            continue
        if show_only == "failed" and not (r.score is not None and r.score < 1.0):
            continue
        if show_only == "errored" and r.score is not None:
            continue
        task = r.task_id if len(r.task_id) <= 36 else r.task_id[:33] + "..."
        exit_s = "—" if r.exit_code is None else str(r.exit_code)
        steps_s = "—" if r.steps is None else str(r.steps)
        ti = "—" if r.tokens_in is None else f"{r.tokens_in:,}"
        to = "—" if r.tokens_out is None else f"{r.tokens_out:,}"
        wt = "—" if r.wall_time_s is None else _fmt_duration(r.wall_time_s)
        lines.append(
            f"{task:<36} {exit_s:>5} {_fmt_score(r.score):>6} {steps_s:>6} {ti:>8} {to:>8} {wt:>7}"
        )
    return "\n".join(lines)


def render_markdown(s: Summary) -> str:
    lines = [
        f"# Summary: {s.run_dir.name}",
        "",
        f"- Agent: `{s.agent}`",
        f"- Agent: `{s.agent_label or '-'}`",
        f"- Model: `{s.model_name}`",
        f"- Total: {s.total}  /  Passed: {s.passed}  /  Failed: {s.failed}  /  "
        f"Errored: {s.errored}",
        f"- Pass rate: **{s.pass_rate:.1%}**",
        f"- Total tokens: {s.total_tokens_in:,} in / {s.total_tokens_out:,} out",
        f"- Total wall-time: {_fmt_duration(s.total_wall_time_s)}",
        "",
        "| task | exit | score | steps | tok_in | tok_out | wall |",
        "|------|-----:|------:|------:|-------:|--------:|-----:|",
    ]
    for r in s.rows:
        exit_s = "—" if r.exit_code is None else str(r.exit_code)
        steps_s = "—" if r.steps is None else str(r.steps)
        ti = "—" if r.tokens_in is None else f"{r.tokens_in:,}"
        to = "—" if r.tokens_out is None else f"{r.tokens_out:,}"
        wt = "—" if r.wall_time_s is None else _fmt_duration(r.wall_time_s)
        lines.append(
            f"| {r.task_id} | {exit_s} | {_fmt_score(r.score)} | {steps_s} | {ti} | {to} | {wt} |"
        )
    return "\n".join(lines)
