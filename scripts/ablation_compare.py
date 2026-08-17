#!/usr/bin/env python3
"""Compare BEAM ablation runs side by side, per probe category.

Usage:
    uv run python scripts/ablation_compare.py runs/<run1> runs/<run2> [runs/<run3> ...] [--csv]

Reads each run's manifest.json (label = agent.id; falls back to the dir name)
and every tasks/*/scores.json, aggregating per_type means weighted by probe
count. Prints one table: rows = probe categories (+ ALL), one column per run.
When exactly THREE runs are given, they are assumed to be the ablation arms in
order (longctx baseline, scroll-tools, full scroll) and two delta columns are
appended: run2−run1 (persisted DB + agentic retrieval) and run3−run2 (the REPL:
code as retrieval interface + cross-step reorganization).

Fairness checklist (encode in the runs, not this script): same model +
thinking mode, same judge (SCROLL_JUDGE_MODEL), same budget and
memory.history_max_tokens, same task set and concurrency, same seed ingest.
Caveat: `beam_analysis.py index` under-reports the scroll-tools arm by
construction (its hl_queries signal greps execute_python SQL text, which that
arm never emits) — compare retrieval volume via metrics.ms_ops route counts
instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_run(run_dir: Path) -> tuple[str, dict[str, tuple[float, int]], tuple[float, int]]:
    """One run's (label, {category: (mean, n)}, (overall_mean, overall_n))."""
    label = run_dir.name
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        try:
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            label = meta.get("agent", {}).get("id") or label
        except (OSError, json.JSONDecodeError):
            pass

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for scores_path in sorted(run_dir.glob("tasks/*/scores.json")):
        try:
            scores = json.loads(scores_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"warning: unreadable {scores_path}, skipped", file=sys.stderr)
            continue
        for category, entry in (scores.get("per_type") or {}).items():
            mean, n = entry.get("mean"), int(entry.get("n") or 0)
            if mean is None or n <= 0:
                continue
            sums[category] = sums.get(category, 0.0) + float(mean) * n
            counts[category] = counts.get(category, 0) + n

    per_category = {c: (sums[c] / counts[c], counts[c]) for c in sums}
    total_n = sum(counts.values())
    overall = (sum(sums.values()) / total_n, total_n) if total_n else (float("nan"), 0)
    return label, per_category, overall


def _fmt(value: float | None) -> str:
    return "     -" if value is None else f"{value:6.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories.")
    parser.add_argument("--csv", action="store_true", help="CSV instead of a table.")
    args = parser.parse_args()

    loaded = []
    for run_dir in args.runs:
        if not run_dir.is_dir():
            print(f"error: not a run directory: {run_dir}", file=sys.stderr)
            return 2
        loaded.append(_load_run(run_dir))
    if not any(per_cat for _, per_cat, _ in loaded):
        print("error: no tasks/*/scores.json found in the given runs", file=sys.stderr)
        return 2

    labels = [label for label, _, _ in loaded]
    categories = sorted({c for _, per_cat, _ in loaded for c in per_cat})
    deltas = len(loaded) == 3

    def row_values(category: str | None) -> list[float | None]:
        vals: list[float | None] = []
        for _, per_cat, overall in loaded:
            if category is None:
                vals.append(overall[0] if overall[1] else None)
            else:
                entry = per_cat.get(category)
                vals.append(entry[0] if entry else None)
        return vals

    header = ["category"] + labels
    if deltas:
        header += [f"{labels[1]}-{labels[0]}", f"{labels[2]}-{labels[1]}"]

    rows: list[list[str]] = []
    for category in categories + [None]:
        vals = row_values(category)
        cells = [category or "ALL"] + [_fmt(v) for v in vals]
        if deltas:
            for a, b in ((1, 0), (2, 1)):
                d = None if vals[a] is None or vals[b] is None else vals[a] - vals[b]
                cells.append("     -" if d is None else f"{d:+6.3f}")
        rows.append(cells)

    if args.csv:
        print(",".join(header))
        for cells in rows:
            print(",".join(c.strip() for c in cells))
    else:
        width = max(len(c) for c in [r[0] for r in rows] + ["category"])
        print("  ".join([f"{'category':<{width}}"] + [f"{h:>12}" for h in header[1:]]))
        for cells in rows:
            print("  ".join([f"{cells[0]:<{width}}"] + [f"{c:>12}" for c in cells[1:]]))
        ns = [f"{label}: n={overall[1]}" for label, _, overall in loaded]
        print("\nprobes  " + "  ".join(ns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
