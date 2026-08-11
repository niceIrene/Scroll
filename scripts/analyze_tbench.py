#!/usr/bin/env python3
"""Complete analysis of a scroll-eval terminal-bench run directory.

Usage:
    python3 scripts/analyze_run.py runs/<run-dir>
    # or via the venv:  .venv/bin/python scripts/analyze_run.py runs/<run-dir>

Reports, per run:
  - pass / fail / incomplete counts and pass rate
  - failure split: budget-exhausted vs submitted-but-wrong vs other
  - ms_ops usage (hist reads) — how often the agent uses the substrate
  - execute_python ModuleNotFoundError count (the "two Pythons" misuse)
  - the actionable task lists
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter


def _load(task_dir: str):
    rwd = glob.glob(f"{task_dir}/harbor-out/*/*/verifier/reward.txt")
    traj = glob.glob(f"{task_dir}/harbor-out/*/*/agent/trajectory.json")
    reward = None
    if rwd:
        try:
            reward = float(open(rwd[0]).read().strip())
        except (OSError, ValueError):
            pass
    d = None
    if traj:
        try:
            d = json.load(open(traj[0]))
        except (OSError, json.JSONDecodeError):
            pass
    return reward, d


def analyze(run_dir: str, max_steps: int | None = None) -> None:
    tasks_root = os.path.join(run_dir, "tasks")
    task_dirs = sorted(os.listdir(tasks_root))

    passed, failed, incomplete = [], [], []
    budget_fail, submit_wrong, other_fail = [], [], []
    ms_recorded = ms_used = 0
    modnotfound = []
    tok_in_total = steps_total = 0
    evict_recorded = evict_any = 0
    evict_turns_total = evict_msgs_total = 0
    filtered_long = []  # tasks excluded because their trajectory is too long
    task_steps = []     # (task, step_count) for every task with a trajectory

    for td in task_dirs:
        reward, d = _load(os.path.join(tasks_root, td))
        if d is not None:
            outcome = "INCOMPLETE" if reward is None else ("PASS" if reward >= 1.0 else "FAIL")
            task_steps.append((td, len(d.get("steps", [])), outcome))
        # Optionally exclude long-trajectory tasks from the whole analysis.
        if max_steps is not None and d is not None and len(d.get("steps", [])) > max_steps:
            filtered_long.append((td, len(d.get("steps", []))))
            continue
        if d:
            m = d.get("metrics", {})
            tok_in_total += int(m.get("tokens_in", 0) or 0)
            steps_total += len(d.get("steps", []))
            ms = m.get("ms_ops")
            if ms is not None:
                ms_recorded += 1
                if any(ms.get(k, 0) for k in (
                    "hist_read", "hist_fts", "hist_seq", "hist_scan",
                )):
                    ms_used += 1
            # The "two Pythons" misuse: ModuleNotFoundError raised *inside*
            # execute_python (runner REPL). Ignore the same error from bash —
            # there it's a normal "pip install it in the sandbox" situation.
            if any(
                (s.get("action") or {}).get("tool") == "execute_python"
                and "ModuleNotFoundError" in (s.get("observation") or "")
                for s in d.get("steps", [])
            ):
                modnotfound.append(td)
            ev = m.get("eviction")
            if ev is not None:
                evict_recorded += 1
                if ev.get("turns", 0) > 0:
                    evict_any += 1
                evict_turns_total += ev.get("turns", 0)
                evict_msgs_total += ev.get("msgs", 0)

        if reward is None:
            # No verifier reward. Record whether the agent ran (has a
            # trajectory) so we can distinguish "ran but ungraded" (verifier
            # timed out / errored) from "never ran" (infra/setup failure).
            steps = len(d.get("steps", [])) if d else None
            term = d.get("terminated") if d else None
            incomplete.append((td, steps, term))
        elif reward >= 1.0:
            passed.append(td)
        else:
            failed.append(td)
            term = (d or {}).get("terminated")
            if term == "success":
                submit_wrong.append(td)
            elif term == "budget":
                budget_fail.append(td)
            else:
                other_fail.append(td)

    n = len(task_dirs) - len(filtered_long)  # analyzed set (excludes filtered)
    graded = len(passed) + len(failed)
    print(f"RUN: {os.path.basename(run_dir)}")
    if max_steps is not None:
        print(f"[filter] excluding {len(filtered_long)} tasks with > {max_steps} steps; "
              f"analyzing {n} of {len(task_dirs)} task dirs")
    print(f"task dirs: {n}  |  graded: {graded}  |  incomplete: {len(incomplete)}")
    print(f"PASSED: {len(passed)}   FAILED: {len(failed)}")
    if graded:
        print(f"pass rate (graded): {len(passed)/graded*100:.1f}%   |   over all dirs: {len(passed)/n*100:.1f}%")
    print(f"tokens_in total: {tok_in_total:,}   steps total: {steps_total}")
    print()
    print(f"PASSED TASKS ({len(passed)}):")
    for t in passed:
        print(f"  {t}")
    print()
    print(f"INCOMPLETE TASKS ({len(incomplete)}):")
    for td, steps, term in incomplete:
        if steps is None:
            print(f"  {td}  (no trajectory — never ran / errored before the agent)")
        else:
            print(f"  {td}  (agent ran {steps} steps, terminated={term}; not graded "
                  "— verifier/infra failed)")
    print()
    print("FAILURE SPLIT:")
    print(f"  budget-exhausted:     {len(budget_fail)}")
    print(f"  submitted-but-wrong:  {len(submit_wrong)}")
    print(f"  other (error/gave_up):{len(other_fail)}")
    print()
    print("SUBSTRATE USE (from ms_ops metric):")
    print(f"  tasks with ms_ops recorded: {ms_recorded}")
    print(f"  tasks that used ms at all:  {ms_used}  ({ms_used}/{ms_recorded or 1} = "
          f"{(ms_used/ms_recorded*100 if ms_recorded else 0):.0f}%)")
    print(f"  execute_python ModuleNotFoundError tasks: {len(modnotfound)}")
    print()
    print("EVICTION (context trimmed because the window was too long):")
    print(f"  tasks with eviction recorded: {evict_recorded}")
    print(f"  tasks that evicted at least once: {evict_any}  "
          f"({(evict_any/evict_recorded*100 if evict_recorded else 0):.0f}%)")
    print(f"  total evict-turns: {evict_turns_total}   total Msgs evicted: {evict_msgs_total}")
    print()
    top_n = 20
    longest = sorted(task_steps, key=lambda x: -x[1])
    print(f"LONGEST TRAJECTORIES (top {min(top_n, len(longest))} of {len(longest)} by step count):")
    for td, st, oc in longest[:top_n]:
        print(f"  {st:>3} steps   [{oc:^10}]   {td}")
    print()
    print("LISTS")
    print("  budget-exhausted:", ", ".join(budget_fail))
    print()
    print("  submitted-but-wrong:", ", ".join(submit_wrong))
    print()
    print("  ModuleNotFoundError (two-Pythons misuse):", ", ".join(modnotfound))
    if filtered_long:
        print()
        print(f"  EXCLUDED (> {max_steps} steps):",
              ", ".join(f"{t}({s})" for t, s in sorted(filtered_long, key=lambda x: -x[1])))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Analyze a scroll-eval terminal-bench run dir.")
    p.add_argument("run_dir", help="runs/<run-dir>")
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="Exclude tasks whose trajectory has MORE than this many steps "
             "(filter out long-trajectory runs).",
    )
    args = p.parse_args()
    analyze(args.run_dir.rstrip("/"), max_steps=args.max_steps)
