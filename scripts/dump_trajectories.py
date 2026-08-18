#!/usr/bin/env python3
"""Print per-probe trajectories (tools used, thinking, observations) from a BEAM run.

Built for inspecting the ablation arms (e.g. how scroll_tools spends its
search_history/expand_turns budget vs scroll_react's execute_python), but works
on any run directory this harness produces.

Usage:
    uv run python scripts/dump_trajectories.py runs/<label>                 # everything
    uv run python scripts/dump_trajectories.py runs/<label> --task 100K-1
    uv run python scripts/dump_trajectories.py runs/<label> --category temporal_reasoning
    uv run python scripts/dump_trajectories.py runs/<label> --probe 3       # one index
    uv run python scripts/dump_trajectories.py runs/<label> --tool expand_turns
    uv run python scripts/dump_trajectories.py runs/<label> --max-obs 0     # full observations
    uv run python scripts/dump_trajectories.py runs/<label> --summary       # tool table only

Reads tasks/*/probes/<category>/<i>/trajectory.json, joining each probe with
its question (answers.json) and score (scores.json) when present. Thinking
(the step's `reasoning` field) is printed by default; suppress with
--no-thinking.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n… [{len(text) - limit} more chars — use --max-obs 0 for full text]"


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _task_lookup(task_dir: Path) -> tuple[dict, dict]:
    """(probe_id -> question, probe_id -> score) for one task, best-effort."""
    questions: dict[str, str] = {}
    scores: dict[str, float] = {}
    answers = _load_json(task_dir / "answers.json") or {}
    for entries in answers.values():
        for e in entries:
            if isinstance(e, dict) and e.get("id"):
                questions[e["id"]] = e.get("question", "")
    graded = _load_json(task_dir / "scores.json") or {}
    for entry in (graded.get("per_type") or {}).values():
        for p in entry.get("probes", []):
            if isinstance(p, dict) and p.get("id"):
                scores[p["id"]] = p.get("primary")
    return questions, scores


def _iter_probes(run_dir: Path, tasks: list[str], categories: list[str], probe: int | None):
    for task_dir in sorted((run_dir / "tasks").iterdir()):
        if not task_dir.is_dir():
            continue
        if tasks and task_dir.name not in tasks:
            continue
        questions, scores = _task_lookup(task_dir)
        probes_root = task_dir / "probes"
        if not probes_root.is_dir():
            continue
        for cat_dir in sorted(probes_root.iterdir()):
            if categories and cat_dir.name not in categories:
                continue
            for probe_dir in sorted(cat_dir.iterdir(), key=lambda p: (len(p.name), p.name)):
                if probe is not None and probe_dir.name != str(probe):
                    continue
                traj = _load_json(probe_dir / "trajectory.json")
                if traj is None:
                    continue
                probe_id = f"{cat_dir.name}-{probe_dir.name}"
                yield (
                    task_dir.name,
                    cat_dir.name,
                    probe_dir.name,
                    traj,
                    questions.get(probe_id, ""),
                    scores.get(probe_id),
                )


def _print_probe(task, category, idx, traj, question, score, *, max_obs, thinking):
    m = traj.get("metrics", {})
    header = f"{task}  {category}/{idx}"
    if score is not None:
        header += f"  score={score}"
    header += (
        f"  terminated={traj.get('terminated')}"
        f"  steps={len(traj.get('steps', []))}"
        f"  tokens={m.get('tokens_in', 0)}+{m.get('tokens_out', 0)}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    if question:
        print(f"Q: {question}\n")
    for step in traj.get("steps", []):
        action = step.get("action") or {}
        tool = action.get("tool", "(no tool call)") if action else "(no tool call)"
        print(f"--- step {step.get('index')}  [{tool}] ---")
        if thinking and step.get("reasoning"):
            print(f"  [thinking] {_clip(step['reasoning'], max_obs)}")
        if step.get("thought"):
            print(f"  [thought]  {_clip(step['thought'], max_obs)}")
        if action:
            args = action.get("args", {})
            rendered = json.dumps(args, ensure_ascii=False, indent=None)
            print(f"  [call]     {tool}({_clip(rendered, max_obs)})")
        if step.get("observation"):
            print(f"  [obs]      {_clip(step['observation'], max_obs)}")
        print()
    print(f"FINAL ANSWER: {traj.get('final_answer') or '(none)'}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="A run directory under runs/.")
    parser.add_argument("--task", action="append", default=[], help="Task name filter (repeatable), e.g. 100K-1.")
    parser.add_argument("--category", action="append", default=[], help="Probe category filter (repeatable).")
    parser.add_argument("--probe", type=int, default=None, help="Single probe index within each category.")
    parser.add_argument("--tool", default=None, help="Only probes whose trajectory used this tool.")
    parser.add_argument("--max-obs", type=int, default=600,
                        help="Clip long fields to N chars (default 600; 0 = no clipping).")
    parser.add_argument("--no-thinking", action="store_true", help="Omit the reasoning/thinking field.")
    parser.add_argument("--summary", action="store_true", help="Print only the aggregate tool-usage table.")
    args = parser.parse_args()

    if not (args.run_dir / "tasks").is_dir():
        print(f"error: {args.run_dir} has no tasks/ — not a run directory?", file=sys.stderr)
        return 2

    tool_calls: Counter[str] = Counter()
    tool_probes: Counter[str] = Counter()
    n_probes = 0
    n_steps = 0

    for task, category, idx, traj, question, score in _iter_probes(
        args.run_dir, args.task, args.category, args.probe
    ):
        tools_here = [
            (s.get("action") or {}).get("tool")
            for s in traj.get("steps", [])
            if s.get("action")
        ]
        if args.tool and args.tool not in tools_here:
            continue
        n_probes += 1
        n_steps += len(traj.get("steps", []))
        tool_calls.update(t for t in tools_here if t)
        tool_probes.update(set(t for t in tools_here if t))
        if not args.summary:
            _print_probe(task, category, idx, traj, question, score,
                         max_obs=args.max_obs, thinking=not args.no_thinking)

    if n_probes == 0:
        print("no matching trajectories found", file=sys.stderr)
        return 1

    print("#" * 60)
    print(f"TOOL USAGE — {n_probes} probes, {n_steps} steps")
    print(f"{'tool':<20}{'calls':>8}{'probes':>8}{'calls/probe':>14}")
    for tool, calls in tool_calls.most_common():
        print(f"{tool:<20}{calls:>8}{tool_probes[tool]:>8}{calls / n_probes:>14.2f}")
    no_call_steps = n_steps - sum(tool_calls.values())
    if no_call_steps:
        print(f"{'(no tool call)':<20}{no_call_steps:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
