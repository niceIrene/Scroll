#!/usr/bin/env python3
"""Dump per-probe raw cost from a BEAM or LongMemEval run as a Python literal.

Both runners lay probes out the same way::

    <run>/tasks/<task>/probes/<category>/<i>/trajectory.json

so this works on either kind of run. It prints the trajectory ``metrics``
untouched (no summing or averaging) in a form you can paste straight into a
notebook:

    cost = [
        # (task, category, probe, tokens_in, tokens_out, turns)
        ("001be529", "single-session-user", 0, 14700, 391, 3),
        ...
    ]

``--in-tokens`` / ``--out-tokens`` / ``--turns`` instead print a single bare
1-d list of just that metric, one entry per probe (no labels, no categories) —
paste it straight into a notebook. All modes emit probes in the same (sorted
path) order, so separately-dumped lists align index-for-index. ``--lists``
emits all of them at once as parallel named lists. Probes with a missing or
unreadable trajectory/metrics are skipped (noted on stderr in bare-list modes).

Usage::

    python3 scripts/dump_run_cost.py runs/<run-dir>
    python3 scripts/dump_run_cost.py runs/<run-dir> --out-tokens
    python3 scripts/dump_run_cost.py runs/<run-dir> --in-tokens
    python3 scripts/dump_run_cost.py runs/<run-dir> --turns
    python3 scripts/dump_run_cost.py runs/<run-dir> --lists
    python3 scripts/dump_run_cost.py runs/<run-dir> > cost.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def collect(run_dir: Path) -> tuple[list[tuple], int]:
    """One (task, category, probe, tokens_in, tokens_out, turns) row per probe."""
    tasks_root = run_dir / "tasks"
    if not tasks_root.is_dir():
        raise SystemExit(f"no tasks/ under {run_dir} (not a run dir?)")
    rows: list[tuple] = []
    skipped = 0
    for traj_path in sorted(tasks_root.glob("*/probes/*/*/trajectory.json")):
        # .../tasks/<task>/probes/<category>/<i>/trajectory.json
        probe = traj_path.parent.name
        category = traj_path.parent.parent.name
        task = traj_path.parent.parent.parent.parent.name
        try:
            metrics = json.loads(traj_path.read_text()).get("metrics")
        except (OSError, json.JSONDecodeError):
            metrics = None
        if not isinstance(metrics, dict):
            skipped += 1
            continue
        rows.append((
            task,
            category,
            int(probe) if probe.isdigit() else probe,
            metrics.get("tokens_in", 0),
            metrics.get("tokens_out", 0),
            metrics.get("step_count", 0),
        ))
    if not rows:
        raise SystemExit(f"no probe trajectories with metrics under {run_dir}")
    return rows, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path, help="A BEAM or LongMemEval run directory.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--in-tokens", action="store_true",
                      help="Print only a bare 1-d list of tokens_in, one per probe.")
    mode.add_argument("--out-tokens", action="store_true",
                      help="Print only a bare 1-d list of tokens_out, one per probe.")
    mode.add_argument("--turns", action="store_true",
                      help="Print only a bare 1-d list of turns, one per probe.")
    mode.add_argument("--lists", action="store_true",
                      help="Emit parallel lists (labels/tokens_in/tokens_out/turns) "
                           "instead of one list of tuples.")
    args = ap.parse_args()

    rows, skipped = collect(args.run_dir)
    # Bare 1-d list modes: nothing but the list on stdout (skips go to stderr).
    for flag, col in (("in_tokens", 3), ("out_tokens", 4), ("turns", 5)):
        if getattr(args, flag):
            if skipped:
                print(f"# skipped {skipped} probe(s): missing trajectory.json "
                      f"or metrics", file=sys.stderr)
            print([r[col] for r in rows])
            return

    print(f"# run: {args.run_dir}  ({len(rows)} probes"
          + (f", {skipped} skipped: missing trajectory.json or metrics" if skipped else "")
          + ")")
    if args.lists:
        print(f"labels = {[f'{t}/{c}/{p}' for t, c, p, *_ in rows]}")
        print(f"tokens_in = {[r[3] for r in rows]}")
        print(f"tokens_out = {[r[4] for r in rows]}")
        print(f"turns = {[r[5] for r in rows]}")
        return
    print("cost = [")
    print("    # (task, category, probe, tokens_in, tokens_out, turns)")
    for r in rows:
        print(f"    {r!r},")
    print("]")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Piped into head/less and the reader went away: point stdout at
        # /dev/null so the interpreter's final flush can't re-raise on exit.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)
