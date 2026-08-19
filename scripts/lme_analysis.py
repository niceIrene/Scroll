#!/usr/bin/env python3
"""Analyses over a LongMemEval run directory.

A LongMemEval run lays out one directory per task (one haystack chat + its
single probing question)::

    <run>/tasks/<task>/scores.json
    <run>/tasks/<task>/probes/<qtype>/<i>/trajectory.json

``scores.json`` carries the judge verdict (``per_type`` mean + ``overall_reward``)
and each ``trajectory.json`` carries ``metrics`` (tokens_in, tokens_out,
step_count, wall_time_s) plus the step-by-step actions. Tasks whose id ends in
``_abs`` are the *abstention* variants: the answer is not in the haystack and the
agent is graded on refusing, so they are also reported as their own split.

Subcommands:
    scores    Mean judge score per question type over the tasks graded so far
              (reads per-task tasks/*/scores.json, so it works mid-run, before
              the final summary.json is written). OVERALL is the micro mean over
              tasks — the same number the runner writes as manifest
              ``mean_reward``. Also splits answerable vs ``_abs`` abstention
              tasks. ``--task <id>`` instead drills into one task: its probe
              question, judge score, and answer.
    cost      Per-task input/output token cost, step count and wall time — one
              row per task, grouped by question type with a per-type ``ALL``
              subtotal and a run TOTAL. ``--avg`` collapses to a per-type mean
              per task across the run.
    overview  One-screen digest of the run: manifest header, score per question
              type, abstention split, and cost/latency totals + means. This is
              the "what did this run get, and what did it cost" view.

Note on latency: ``wall_time_s`` is per-task agent wall time (judging excluded),
and tasks run concurrently, so the TOTAL wall column is machine-seconds of agent
work, not the run's wall-clock duration. The percentiles are the per-task ones.

Usage::

    python3 scripts/lme_analysis.py overview runs/<run-dir>
    python3 scripts/lme_analysis.py scores runs/<run-dir>
    python3 scripts/lme_analysis.py scores runs/<run-dir> --csv > scores.csv
    python3 scripts/lme_analysis.py scores runs/<run-dir> --task 5e1b23de
    python3 scripts/lme_analysis.py cost runs/<run-dir>
    python3 scripts/lme_analysis.py cost runs/<run-dir> --avg
    python3 scripts/lme_analysis.py cost runs/<run-dir> --avg --csv > cost.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# LongMemEval's own presentation order for the question types; anything not in
# this list (a custom split) sorts after it alphabetically.
_QTYPE_ORDER = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)


def _qtype_sort_key(qtype: str):
    try:
        return (0, _QTYPE_ORDER.index(qtype), qtype)
    except ValueError:
        return (1, 0, qtype)


def _is_abstention(task: str) -> bool:
    """``_abs`` tasks ask something the haystack does not answer (refusal test)."""
    return task.endswith("_abs")


def _tasks_root(run_dir: Path) -> Path:
    root = run_dir / "tasks"
    if not root.is_dir():
        raise SystemExit(f"no tasks/ under {run_dir} (not a run dir?)")
    return root


def _read_manifest(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


# --- collection -------------------------------------------------------------

def _iter_task_trajectories(run_dir: Path):
    """Yield ``(task, qtype, trajectory_dict_or_None)`` for every probe.

    A LongMemEval task carries a single probe, so this is effectively one row
    per task; the loop still walks ``probes/<qtype>/<i>/`` so a multi-probe task
    (should the generator ever emit one) is not silently dropped. Probes whose
    trajectory is missing/unreadable yield ``None`` for the caller to count.
    """
    for traj_path in sorted(_tasks_root(run_dir).glob("*/probes/*/*/trajectory.json")):
        # .../tasks/<task>/probes/<qtype>/<i>/trajectory.json
        qtype = traj_path.parent.parent.name
        task = traj_path.parent.parent.parent.parent.name
        try:
            yield task, qtype, json.loads(traj_path.read_text())
        except (OSError, json.JSONDecodeError):
            yield task, qtype, None


def _iter_task_scores(run_dir: Path):
    """Yield ``(task, scores_dict_or_None)`` from every ``tasks/<task>/scores.json``.

    Each task writes its own scores.json the moment it is graded, so reading
    these gives a cumulative score even while the run is still in flight — we
    don't wait for the final ``summary.json``, which lands only once every task
    has finished.
    """
    for path in sorted(_tasks_root(run_dir).glob("*/scores.json")):
        try:
            yield path.parent.name, json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            yield path.parent.name, None


def _collect_scores(run_dir: Path):
    """Per-task judge score, keyed by question type.

    Returns ``(rows, tasks, skipped)`` where ``rows[qtype]`` is
    ``{"sum", "n", "abs_sum", "abs_n"}`` — the ``abs_*`` pair being the subset of
    that type's tasks that are abstention (``_abs``) variants, which are also
    counted in ``sum``/``n``. Since a task holds one probe, a task's per_type
    mean *is* its score, so summing means and dividing by ``n`` yields the micro
    (per-question) accuracy LongMemEval reports.
    """
    rows: dict[str, dict[str, float]] = defaultdict(
        lambda: {"sum": 0.0, "n": 0, "abs_sum": 0.0, "abs_n": 0}
    )
    tasks = skipped = 0
    for task, scores in _iter_task_scores(run_dir):
        per_type = scores.get("per_type") if isinstance(scores, dict) else None
        if not isinstance(per_type, dict):
            skipped += 1
            continue
        tasks += 1
        for qtype, d in per_type.items():
            mean = d.get("mean") if isinstance(d, dict) else None
            if mean is None:
                continue
            r = rows[qtype]
            r["sum"] += float(mean)
            r["n"] += 1
            if _is_abstention(task):
                r["abs_sum"] += float(mean)
                r["abs_n"] += 1
    if not rows and not skipped:
        raise SystemExit(
            f"no tasks/*/scores.json under {run_dir} (no task finished/graded yet?)"
        )
    return rows, tasks, skipped


# Per-task cost columns, read from the top level of the trajectory ``metrics``
# block. Each entry is (column_name, metrics_key):
#   tokens_in  - cumulative prompt tokens the model was sent across all steps
#   tokens_out - cumulative completion tokens it generated
#   steps      - agent steps taken before the final answer (metrics.step_count)
#   wall_s     - agent wall time for the task in seconds (judging excluded)
_COST_METRICS = (
    ("tokens_in", "tokens_in"),
    ("tokens_out", "tokens_out"),
    ("steps", "step_count"),
    ("wall_s", "wall_time_s"),
)
_COST_FIELDS = tuple(name for name, _ in _COST_METRICS)
# wall_s is a float (seconds); the rest are integer counters.
_FLOAT_FIELDS = frozenset({"wall_s"})


def _collect_cost(run_dir: Path):
    """One cost row per task. Returns ``(rows, skipped)``.

    Each row is ``{task, qtype, tokens_in, tokens_out, steps, wall_s}``; a probe
    missing a metrics key contributes 0 for it rather than being dropped.
    """
    rows: list[dict] = []
    skipped = 0
    for task, qtype, traj in _iter_task_trajectories(run_dir):
        metrics = (traj or {}).get("metrics") if traj else None
        if not isinstance(metrics, dict):
            skipped += 1
            continue
        row = {"task": task, "qtype": qtype}
        for col, mkey in _COST_METRICS:
            v = metrics.get(mkey, 0)
            v = v if isinstance(v, (int, float)) else 0
            row[col] = float(v) if col in _FLOAT_FIELDS else int(v)
        rows.append(row)
    return rows, skipped


def _fmt(field: str, value) -> str:
    """Render a cost cell: seconds get 2dp, counters stay integers."""
    if field in _FLOAT_FIELDS:
        return f"{value:.2f}"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def _percentiles(values: list[float], pcts=(50, 95)) -> dict[int, float]:
    """Nearest-rank percentiles over ``values`` (empty -> zeros)."""
    if not values:
        return {p: 0.0 for p in pcts}
    ordered = sorted(values)
    out = {}
    for p in pcts:
        rank = max(1, min(len(ordered), -(-p * len(ordered) // 100)))
        out[p] = ordered[rank - 1]
    return out


# --- scores -----------------------------------------------------------------

def _score_table_rows(rows: dict) -> list[tuple]:
    """``(qtype, tasks, mean)`` per type, then MACRO and OVERALL summary rows.

    OVERALL is the micro mean over every graded task (what the runner records as
    ``mean_reward``); MACRO is the unweighted mean of the per-type means, which
    is the number to quote when the type mix is uneven.
    """
    out = []
    total_sum = total_n = 0.0
    means = []
    for qtype in sorted(rows, key=_qtype_sort_key):
        r = rows[qtype]
        n = int(r["n"])
        mean = r["sum"] / n if n else 0.0
        out.append((qtype, n, mean))
        means.append(mean)
        total_sum += r["sum"]
        total_n += n
    out.append(("MACRO (mean of types)", "", sum(means) / len(means) if means else 0.0))
    out.append(("OVERALL", int(total_n), total_sum / total_n if total_n else 0.0))
    return out


def _abstention_split(rows: dict) -> list[tuple]:
    """``(label, tasks, mean)`` for the answerable vs abstention halves."""
    abs_sum = sum(r["abs_sum"] for r in rows.values())
    abs_n = sum(r["abs_n"] for r in rows.values())
    all_sum = sum(r["sum"] for r in rows.values())
    all_n = sum(r["n"] for r in rows.values())
    ans_sum, ans_n = all_sum - abs_sum, all_n - abs_n
    return [
        ("answerable", int(ans_n), ans_sum / ans_n if ans_n else 0.0),
        ("abstention", int(abs_n), abs_sum / abs_n if abs_n else 0.0),
    ]


def _print_score_table(table: list[tuple], count_col: str,
                       label_col: str = "question type") -> None:
    label_w = max(max(len(str(r[0])) for r in table), len(label_col))
    print(f"{label_col.ljust(label_w)}  {count_col:>5}  {'mean_score':>10}")
    print(f"{'-' * label_w}  {'-' * 5}  {'-' * 10}")
    for label, count, mean in table:
        if label.startswith("MACRO"):
            print(f"{'-' * label_w}  {'-' * 5}  {'-' * 10}")
        print(f"{str(label).ljust(label_w)}  {str(count):>5}  {mean:>10.4f}")


def _cmd_scores_one_task(run_dir: Path, task: str, as_csv: bool) -> None:
    """Drill into one task: its probe question, judge score, and answer."""
    tasks_root = _tasks_root(run_dir)
    path = tasks_root / task / "scores.json"
    if not path.is_file():
        raise SystemExit(
            f"no scores.json for task {task!r} under {run_dir} "
            f"({len(list(tasks_root.glob('*/scores.json')))} task(s) graded so far)"
        )
    try:
        scores = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"could not read {path}: {e}")
    per_type = scores.get("per_type") if isinstance(scores, dict) else None
    if not isinstance(per_type, dict):
        raise SystemExit(f"no per_type block in {path}")

    # answers.json holds the agent's response text, grouped by question type.
    try:
        answers = json.loads((tasks_root / task / "answers.json").read_text())
    except (OSError, json.JSONDecodeError):
        answers = {}
    responses = {
        a.get("id"): a.get("llm_response", "")
        for group in answers.values() if isinstance(group, list)
        for a in group if isinstance(a, dict)
    }

    probes = []
    for qtype, d in sorted(per_type.items(), key=lambda kv: _qtype_sort_key(kv[0])):
        for p in (d.get("probes") or []) if isinstance(d, dict) else []:
            probes.append({
                "qtype": qtype,
                "id": p.get("id", ""),
                "score": float(p.get("primary", 0.0)),
                "question": p.get("question", ""),
                "answer": responses.get(p.get("id"), ""),
            })
    if as_csv:
        w = csv.writer(sys.stdout)
        w.writerow(["task", "qtype", "probe", "score", "question", "answer"])
        for p in probes:
            w.writerow([task, p["qtype"], p["id"], f"{p['score']:.4f}",
                        p["question"], p["answer"]])
        return

    print(f"task     : {task}{'  (abstention)' if _is_abstention(task) else ''}")
    print(f"reward   : {float(scores.get('overall_reward', 0.0)):.4f}"
          f"  ({scores.get('n_probes', len(probes))} probe(s))")
    for p in probes:
        print()
        print(f"  type     : {p['qtype']}")
        print(f"  score    : {p['score']:.4f}")
        print(f"  question : {p['question']}")
        print(f"  answer   : {p['answer']}")


def cmd_scores(args: argparse.Namespace) -> None:
    if args.task:
        _cmd_scores_one_task(args.run_dir, args.task, args.csv)
        return

    rows, tasks, skipped = _collect_scores(args.run_dir)
    if not rows:
        raise SystemExit(
            f"no graded per-type means in tasks/*/scores.json under {args.run_dir}"
        )
    table = _score_table_rows(rows)
    split = _abstention_split(rows)

    if args.csv:
        w = csv.writer(sys.stdout)
        w.writerow(["question_type", "tasks", "mean_score"])
        for label, count, mean in table:
            w.writerow([label, count, f"{mean:.4f}"])
        for label, count, mean in split:
            w.writerow([label, count, f"{mean:.4f}"])
        return

    _print_score_table(table, "tasks")
    if split[1][1]:  # only worth showing when the run has _abs tasks
        print()
        _print_score_table(split, "tasks", label_col="split")
    note = f"({tasks} task(s) graded" + (f", {skipped} skipped" if skipped else "") + ")"
    print(f"\n{note}")


# --- cost -------------------------------------------------------------------

def _cost_sort_key(row: dict):
    """Order rows by question type (LongMemEval's order), then task id."""
    return (_qtype_sort_key(row["qtype"]), row["task"])


def _print_cost_table(rows: list, note: str | None) -> None:
    """One row per task, grouped by question type with ``ALL`` subtotals."""
    rows = sorted(rows, key=_cost_sort_key)
    header = ["qtype", "task", *_COST_FIELDS]
    cells = [[r["qtype"], r["task"], *[_fmt(f, r[f]) for f in _COST_FIELDS]] for r in rows]
    widths = [len(h) for h in header]
    for row_cells in cells:
        for i, c in enumerate(row_cells):
            widths[i] = max(widths[i], len(c))
    # Subtotal rows carry sums, which can be wider than any single task's cell.
    grand = {f: 0 for f in _COST_FIELDS}
    for r in rows:
        for f in _COST_FIELDS:
            grand[f] += r[f]
    for i, f in enumerate(_COST_FIELDS):
        widths[2 + i] = max(widths[2 + i], len(_fmt(f, grand[f])))
    widths[1] = max(widths[1], len(f"ALL ({len(rows)})"))

    def line(cs) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cs))

    print(line(header))
    print(line(["-" * w for w in widths]))

    current = None
    sub = {f: 0 for f in _COST_FIELDS}
    sub_tasks = 0

    def flush() -> None:
        # On ALL/TOTAL rows the ``task`` column holds the count of tasks summed.
        if current is not None:
            print(line([current, f"ALL ({sub_tasks})",
                        *[_fmt(f, sub[f]) for f in _COST_FIELDS]]))
            print()

    for r in rows:
        if r["qtype"] != current:
            flush()
            current = r["qtype"]
            sub = {f: 0 for f in _COST_FIELDS}
            sub_tasks = 0
        print(line([r["qtype"], r["task"], *[_fmt(f, r[f]) for f in _COST_FIELDS]]))
        for f in _COST_FIELDS:
            sub[f] += r[f]
        sub_tasks += 1
    flush()
    print(line(["TOTAL", f"ALL ({len(rows)})",
                *[_fmt(f, grand[f]) for f in _COST_FIELDS]]))
    if note:
        print(f"\n{note}")
    print("(wall_s is per-task agent wall time; tasks run concurrently, so the "
          "TOTAL is machine-seconds, not run wall-clock)")


def _average_by_qtype(rows: list):
    """Per-question-type mean per task, plus an OVERALL mean over all tasks.

    Returns a list of ``(qtype, tasks, {field: mean})``.
    """
    sums: dict[str, dict[str, float]] = defaultdict(lambda: {f: 0.0 for f in _COST_FIELDS})
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["qtype"]] += 1
        for f in _COST_FIELDS:
            sums[r["qtype"]][f] += r[f]
    out = []
    for qtype in sorted(sums, key=_qtype_sort_key):
        n = counts[qtype]
        out.append((qtype, n, {f: sums[qtype][f] / n for f in _COST_FIELDS}))
    total = len(rows)
    overall = {
        f: (sum(s[f] for s in sums.values()) / total if total else 0.0)
        for f in _COST_FIELDS
    }
    out.append(("OVERALL", total, overall))
    return out


def _print_cost_avg_table(avg_rows: list, note: str | None) -> None:
    label_w = max(len("qtype"), max(len(r[0]) for r in avg_rows))
    num_w = {f: max(len(f), 10) for f in _COST_FIELDS}
    print("  ".join(["qtype".ljust(label_w), "tasks".rjust(5),
                     *[f.rjust(num_w[f]) for f in _COST_FIELDS]]))
    print("  ".join(["-" * label_w, "-" * 5, *["-" * num_w[f] for f in _COST_FIELDS]]))
    for qtype, n, means in avg_rows:
        if qtype == "OVERALL":
            print("  ".join(["-" * label_w, "-" * 5,
                             *["-" * num_w[f] for f in _COST_FIELDS]]))
        print("  ".join([qtype.ljust(label_w), str(n).rjust(5),
                         *[f"{means[f]:.2f}".rjust(num_w[f]) for f in _COST_FIELDS]]))
    print("\n(values are the mean per task)")
    if note:
        print(note)


def cmd_cost(args: argparse.Namespace) -> None:
    rows, skipped = _collect_cost(args.run_dir)
    if not rows:
        raise SystemExit(f"no task trajectories with metrics found under {args.run_dir}")
    note = (f"(skipped {skipped} task(s): missing trajectory.json or metrics)"
            if skipped else None)

    if args.avg:
        avg_rows = _average_by_qtype(rows)
        if args.csv:
            w = csv.writer(sys.stdout)
            w.writerow(["qtype", "tasks", *_COST_FIELDS])
            for qtype, n, means in avg_rows:
                w.writerow([qtype, n, *[f"{means[f]:.4f}" for f in _COST_FIELDS]])
            return
        _print_cost_avg_table(avg_rows, note)
        return

    if args.csv:
        w = csv.writer(sys.stdout)
        w.writerow(["qtype", "task", *_COST_FIELDS])
        for r in sorted(rows, key=_cost_sort_key):
            w.writerow([r["qtype"], r["task"], *[_fmt(f, r[f]) for f in _COST_FIELDS]])
        return
    _print_cost_table(rows, note)


# --- overview ---------------------------------------------------------------

def cmd_overview(args: argparse.Namespace) -> None:
    """Score + cost digest: the "what did this run get, and what did it cost" view."""
    manifest = _read_manifest(args.run_dir)
    score_rows, graded, score_skipped = _collect_scores(args.run_dir)
    cost_rows, cost_skipped = _collect_cost(args.run_dir)

    print(f"run      : {args.run_dir}")
    if manifest:
        model = manifest.get("model") or {}
        agent = manifest.get("agent") or {}
        judge = manifest.get("judge") or {}
        print(f"agent    : {agent.get('type', '?')}/{agent.get('id', '?')}")
        print(f"model    : {model.get('name', '?')}")
        print(f"judge    : {judge.get('name', '?')}")
        print(f"dataset  : {manifest.get('dataset', '?')}"
              f"  ({len(manifest.get('tasks') or [])} task(s) planned)")
        if manifest.get("timestamp_utc"):
            print(f"started  : {manifest['timestamp_utc']} UTC")

    print()
    print("SCORE")
    _print_score_table(_score_table_rows(score_rows), "tasks")
    split = _abstention_split(score_rows)
    if split[1][1]:
        print()
        _print_score_table(split, "tasks", label_col="split")
    print(f"\n({graded} task(s) graded"
          + (f", {score_skipped} skipped" if score_skipped else "") + ")")

    if not cost_rows:
        print("\nCOST\n(no task trajectories with metrics found)")
        return

    n = len(cost_rows)
    totals = {f: sum(r[f] for r in cost_rows) for f in _COST_FIELDS}  # ints stay ints
    walls = [r["wall_s"] for r in cost_rows]
    pcts = _percentiles(walls)
    print()
    print("COST")
    print(f"{'metric':<14}{'total':>16}{'mean/task':>14}")
    print(f"{'-' * 14}{'-' * 16:>16}{'-' * 14:>14}")
    for f in _COST_FIELDS:
        print(f"{f:<14}{_fmt(f, totals[f]):>16}{totals[f] / n:>14.2f}")
    print()
    print(f"latency      : p50 {pcts[50]:.2f}s  p95 {pcts[95]:.2f}s  "
          f"max {max(walls):.2f}s  (per task)")
    print(f"tokens       : {totals['tokens_in'] / n:,.0f} in / "
          f"{totals['tokens_out'] / n:,.0f} out per task")
    print(f"tasks costed : {n}"
          + (f"  ({cost_skipped} skipped: missing trajectory.json or metrics)"
             if cost_skipped else ""))
    print("\n(wall_s totals are machine-seconds of agent work — tasks run "
          "concurrently, so this is not the run's wall-clock duration)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p_ov = sub.add_parser(
        "overview", help="Score + cost digest for the run (header, scores, cost, latency)."
    )
    p_ov.add_argument("run_dir", type=Path,
                      help="The LongMemEval run directory (contains tasks/).")
    p_ov.set_defaults(func=cmd_overview)

    p_sc = sub.add_parser(
        "scores", help="Mean judge score per question type over graded tasks."
    )
    p_sc.add_argument("run_dir", type=Path,
                      help="The LongMemEval run directory (contains tasks/).")
    p_sc.add_argument("--csv", action="store_true",
                      help="Emit CSV to stdout instead of a table.")
    p_sc.add_argument("--task", metavar="TASK",
                      help="Drill into one task (e.g. 5e1b23de): its question, judge "
                           "score and answer. Default aggregates the whole run.")
    p_sc.set_defaults(func=cmd_scores)

    p_cost = sub.add_parser(
        "cost", help="Per-task input/output tokens, step count and wall time."
    )
    p_cost.add_argument("run_dir", type=Path,
                        help="The LongMemEval run directory (contains tasks/).")
    p_cost.add_argument("--csv", action="store_true",
                        help="Emit CSV to stdout instead of a table.")
    p_cost.add_argument("--avg", action="store_true",
                        help="Collapse tasks: show the per-question-type mean per task.")
    p_cost.set_defaults(func=cmd_cost)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Piped into head/less and the reader went away: point stdout at
        # /dev/null so the interpreter's final flush can't re-raise on exit.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)
