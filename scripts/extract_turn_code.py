#!/usr/bin/env python3
"""Extract the code the model wrote each turn, for one BEAM run.

A BEAM run lays out one directory per probe::

    <run>/tasks/<chat>/probes/<category>/<i>/trajectory.json

Each ``trajectory.json`` has ``steps[]``; a step's ``action`` is
``{"tool": name, "args": {...}}``. The model "writes code" on
``execute_python`` turns (``args.source``) and, if a sandbox is attached,
``bash`` turns (``args.command``). This script walks every probe, pulls those
per-turn snippets in order — each followed by its result (the step's
``observation``) — and writes one file per probe so you can read what the agent
actually ran and what it got back, then prints the list of files written.

Usage::

    python3 scripts/extract_turn_code.py runs/full-headline-with-thinking
    python3 scripts/extract_turn_code.py runs/<run> -o /tmp/code      # custom out dir
    python3 scripts/extract_turn_code.py runs/<run> --per-turn        # one file per turn
    python3 scripts/extract_turn_code.py runs/<run> --include-bash    # bash turns too

By default one ``.py`` file is written per probe (all its turns, in order, each
under a header comment). ``--per-turn`` instead writes one file per turn. Output
defaults to ``<run>/extracted_code/``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def _iter_probes(run: str):
    """Yield (task, category, probe_id, trajectory_path) for every probe."""
    pattern = os.path.join(run, "tasks", "*", "probes", "*", "*", "trajectory.json")
    for tr in sorted(glob.glob(pattern)):
        parts = Path(tr).parts
        # .../tasks/<task>/probes/<category>/<probe>/trajectory.json
        task, category, probe = parts[-5], parts[-3], parts[-2]
        yield task, category, probe, tr


def _code_turns(traj_path: str, include_bash: bool) -> list[dict]:
    """Return the code-writing turns of one trajectory, in order.

    Each entry: ``{step, tool, code, output}``. ``execute_python`` contributes
    its ``source``; ``bash`` (only with ``include_bash``) its ``command``.
    ``output`` is the step's ``observation`` — the result of running that code.
    """
    traj = json.loads(Path(traj_path).read_text(encoding="utf-8"))
    turns: list[dict] = []
    for step in traj.get("steps", []):
        action = step.get("action") or {}
        tool = action.get("tool")
        args = action.get("args") or {}
        if tool == "execute_python":
            code = args.get("source", "")
        elif tool == "bash" and include_bash:
            code = args.get("command", "")
        else:
            continue
        if code.strip():
            turns.append({
                "step": step.get("index"),
                "tool": tool,
                "code": code,
                "output": step.get("observation") or "",
            })
    return turns


def _comment_block(text: str, max_output: int | None) -> list[str]:
    """Render an execution result as ``# `` -prefixed lines so the file stays
    readable as Python. Truncated to ``max_output`` chars when set (0/None = full)."""
    text = text.rstrip()
    if not text:
        return ["#   (no output)"]
    if max_output and len(text) > max_output:
        text = text[:max_output].rstrip() + f"\n... [truncated, {len(text)} chars total]"
    return ["#   " + line for line in text.splitlines()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory, e.g. runs/full-headline-with-thinking")
    ap.add_argument("-o", "--out", default=None,
                    help="output directory (default: <run>/extracted_code)")
    ap.add_argument("--per-turn", action="store_true",
                    help="one file per turn instead of one file per probe")
    ap.add_argument("--include-bash", action="store_true",
                    help="also extract bash command turns (default: execute_python only)")
    ap.add_argument("--max-output", type=int, default=0,
                    help="truncate each step's result to N chars (0 = full output)")
    args = ap.parse_args(argv)
    max_output = args.max_output or None

    run = args.run.rstrip("/")
    if not os.path.isdir(os.path.join(run, "tasks")):
        raise SystemExit(f"not a BEAM run dir (no tasks/ under {run!r})")
    out_dir = Path(args.out or os.path.join(run, "extracted_code"))
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    n_turns = 0
    for task, category, probe, tr in _iter_probes(run):
        turns = _code_turns(tr, args.include_bash)
        if not turns:
            continue
        stem = f"{task}__{category}__{probe}"
        if args.per_turn:
            for t in turns:
                path = out_dir / f"{stem}__step{t['step']:02d}__{t['tool']}.py"
                body = [t["code"].strip(), "",
                        f"# ----- output (step {t['step']}) -----",
                        *_comment_block(t["output"], max_output)]
                path.write_text("\n".join(body) + "\n", encoding="utf-8")
                written.append(path)
                n_turns += 1
        else:
            lines = [f"# {stem}", f"# source: {tr}",
                     f"# {len(turns)} code turn(s)", ""]
            for t in turns:
                lines.append(f"# ===== step {t['step']} · {t['tool']} =====")
                lines.append(t["code"].strip())
                lines.append("")
                lines.append(f"# ----- output (step {t['step']}) -----")
                lines.extend(_comment_block(t["output"], max_output))
                lines.append("")
            path = out_dir / f"{stem}.py"
            path.write_text("\n".join(lines), encoding="utf-8")
            written.append(path)
            n_turns += len(turns)

    # The requested output: the list of files written, plus a one-line summary.
    for p in written:
        print(p)
    print(f"\n# {len(written)} file(s), {n_turns} code turn(s) -> {out_dir}")


if __name__ == "__main__":
    main()
