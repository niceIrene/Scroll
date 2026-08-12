#!/usr/bin/env python3
"""Analyses over a native BEAM run directory.

A BEAM run lays out one directory per probe (one chat x one probing question)::

    <run>/tasks/<chat>/probes/<category>/<i>/trajectory.json

Each ``trajectory.json`` carries ``metrics`` (ms_ops, tokens, steps, eviction)
plus the step-by-step actions. This script is a home for per-run analyses over
that tree; add new ones as subcommands.

Subcommands:
    ms-ops   Per-chat / per-category memory-substrate interaction counts
             (conversation-history reads, scratch reads/writes) plus how often
             the in-context history buffer was evicted to fit the token budget.
    cost     Per-probe input/output token cost and turn count — one row per
             probe (each probe is a separate agent chat, so same-type probes
             can take a different number of turns; they are shown on their own
             rows rather than summed). A per-chat ``ALL`` row totals the chat's
             probes. ``--avg`` collapses to a per-category mean per chat across
             the run.
    scores   Cumulative mean judge score per probing-question category over the
             chats finished so far (reads per-chat tasks/*/scores.json, so it
             works mid-run, before the final summary.json is written).
             ``--chat <chat>`` instead scores a single chat: its own per-category
             mean over that chat's probes (and the probe count), read straight
             from that chat's scores.json.
    index    How much each probe actually leaned on the headline/index — the
             in-context ``[memory]`` map of ``seq · headline`` lines is always
             present, but the agent only *uses* it when it expands a span by
             querying the ``headline`` column. Reports, per row, probes that did
             so (``idx_probes``) and the share they make up (``idx%``); the
             headline-expansion queries (``hl_queries``); the positional
             content reads (``seq_range_reads`` — ``WHERE seq BETWEEN…`` / ``seq
             IN (…)``, a broader signal that also covers widening around a search
             hit, so it is not folded into ``idx_probes``); and, for contrast,
             how often each ms primitive was issued — ``ms.search`` split into
             snippet-triage (``search_snip``) vs quiet ``snippet=False``
             aggregation (``search_nosnip``), and ``expand_calls``. Default aggregates the whole
             run, one row per chat; ``--chat <chat>`` drills into one chat, one
             row per probe type (category) in it.

Usage::

    python3 scripts/beam_analysis.py ms-ops runs/<run-dir>
    python3 scripts/beam_analysis.py ms-ops runs/<run-dir> --csv > ms_ops.csv
    python3 scripts/beam_analysis.py cost runs/<run-dir>
    python3 scripts/beam_analysis.py cost runs/<run-dir> --avg
    python3 scripts/beam_analysis.py cost runs/<run-dir> --avg --csv > cost.csv
    python3 scripts/beam_analysis.py scores runs/<run-dir>
    python3 scripts/beam_analysis.py scores runs/<run-dir> --chat 10M-6
    python3 scripts/beam_analysis.py index runs/<run-dir>
    python3 scripts/beam_analysis.py index runs/<run-dir> --chat 10M-6
    python3 scripts/beam_analysis.py index runs/<run-dir> --chat 10M-6 --csv > index.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Per-probe integer metrics summed per (chat, category). Each entry is
# (column_name, (metrics_block, key_within_block)):
#   hist_read     - conversation-history reads (ms.search + any ms.sql_query
#                   against hist.conversation_history)
#   evict_turns   - turns on which the in-context history buffer was evicted to
#                   fit the token budget (i.e. how many times we evicted)
#   evict_msgs    - total messages dropped across those evictions
_METRICS = (
    ("hist_read", ("ms_ops", "hist_read")),
    ("evict_turns", ("eviction", "turns")),
    ("evict_msgs", ("eviction", "msgs")),
)
_FIELDS = tuple(name for name, _ in _METRICS)


def _iter_probe_trajectories(run_dir: Path):
    """Yield ``(chat, category, probe, trajectory_dict)`` for every probe.

    ``probe`` is the probe's directory index ``<i>`` (its on-disk id within the
    run), so callers can keep probes of the same category distinct. Probes whose
    trajectory is missing/unreadable are skipped silently; the caller counts them
    via the difference if it cares.
    """
    tasks_root = run_dir / "tasks"
    if not tasks_root.is_dir():
        raise SystemExit(f"no tasks/ under {run_dir} (not a run dir?)")
    for traj_path in sorted(tasks_root.glob("*/probes/*/*/trajectory.json")):
        # .../tasks/<chat>/probes/<category>/<i>/trajectory.json
        probe = traj_path.parent.name
        category = traj_path.parent.parent.name
        chat = traj_path.parent.parent.parent.parent.name
        try:
            traj = json.loads(traj_path.read_text())
        except (OSError, json.JSONDecodeError):
            yield chat, category, probe, None
            continue
        yield chat, category, probe, traj


def _chat_sort_key(chat: str):
    """Order chats by scale then numeric id (``10M-2`` before ``10M-10``)."""
    scale, _, num = chat.partition("-")
    return (scale, int(num) if num.isdigit() else 0, chat)


# --- ms-ops ----------------------------------------------------------------

def _collect_ms_ops(run_dir: Path):
    """Sum per-probe metrics per (chat, category).

    Returns (rows, probe_counts, skipped). A probe is counted as long as it has
    a ``metrics`` block; a missing sub-block (e.g. ``eviction`` on an older
    trajectory) contributes 0 to its fields rather than dropping the probe.
    """
    rows: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {f: 0 for f in _FIELDS}
    )
    probe_counts: dict[tuple[str, str], int] = defaultdict(int)
    skipped = 0
    for chat, category, _probe, traj in _iter_probe_trajectories(run_dir):
        metrics = (traj or {}).get("metrics") if traj else None
        if not isinstance(metrics, dict):
            skipped += 1
            continue
        key = (chat, category)
        for col, (block, field) in _METRICS:
            sub = metrics.get(block)
            rows[key][col] += int(sub.get(field, 0)) if isinstance(sub, dict) else 0
        probe_counts[key] += 1
    return rows, probe_counts, skipped


def _print_grouped_table(rows: dict, probe_counts: dict, fields: tuple) -> None:
    """Per-(chat, category) table with a per-chat ``ALL`` subtotal and a TOTAL.

    ``fields`` are the integer columns to sum (e.g. memory ops, or token cost).
    """
    keys = sorted(rows, key=lambda k: (_chat_sort_key(k[0]), k[1]))
    header = ["chat", "category", "probes", *fields]
    widths = [len(h) for h in header]
    for chat, category in keys:
        widths[0] = max(widths[0], len(chat))
        widths[1] = max(widths[1], len(category))
    for i, f in enumerate(fields):
        col = 3 + i
        for chat, category in keys:
            widths[col] = max(widths[col], len(str(rows[(chat, category)][f])))

    def line(cells) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    print(line(header))
    print(line(["-" * w for w in widths]))

    grand = {f: 0 for f in fields}
    grand_probes = 0
    current_chat = None
    chat_sub = {f: 0 for f in fields}
    chat_probes = 0

    def flush_chat() -> None:
        if current_chat is not None:
            print(line([current_chat, "ALL", chat_probes,
                        *[chat_sub[f] for f in fields]]))
            print()

    for chat, category in keys:
        if chat != current_chat:
            flush_chat()
            current_chat = chat
            chat_sub = {f: 0 for f in fields}
            chat_probes = 0
        r = rows[(chat, category)]
        n = probe_counts[(chat, category)]
        print(line([chat, category, n, *[r[f] for f in fields]]))
        for f in fields:
            chat_sub[f] += r[f]
            grand[f] += r[f]
        chat_probes += n
        grand_probes += n
    flush_chat()
    print(line(["TOTAL", "", grand_probes, *[grand[f] for f in fields]]))


def _write_grouped_csv(rows: dict, probe_counts: dict, out, fields: tuple) -> None:
    w = csv.writer(out)
    w.writerow(["chat", "category", "probes", *fields])
    for chat, category in sorted(rows, key=lambda k: (_chat_sort_key(k[0]), k[1])):
        r = rows[(chat, category)]
        w.writerow([chat, category, probe_counts[(chat, category)],
                    *[r[f] for f in fields]])


def _average_by_category(rows: dict, fields: tuple):
    """Collapse the chat dimension: per-category mean over chats.

    Each (chat, category) cell already sums its probes. This averages those
    per-chat cell values over the chats that have the category. Returns
    ``(per_category, n_chats)`` where ``per_category`` is a sorted list of
    ``(category, chats_with_category, {field: mean})`` followed by an OVERALL
    row whose means are the average per-chat total across all categories.
    """
    cat_sum: dict[str, dict[str, int]] = defaultdict(lambda: {f: 0 for f in fields})
    cat_chats: dict[str, set] = defaultdict(set)
    all_chats: set = set()
    for (chat, category), r in rows.items():
        all_chats.add(chat)
        cat_chats[category].add(chat)
        for f in fields:
            cat_sum[category][f] += r[f]

    n_chats = len(all_chats)
    out = []
    for category in sorted(cat_sum):
        c = len(cat_chats[category])
        means = {f: (cat_sum[category][f] / c if c else 0.0) for f in fields}
        out.append((category, c, means))
    overall = {
        f: (sum(cat_sum[cat][f] for cat in cat_sum) / n_chats if n_chats else 0.0)
        for f in fields
    }
    out.append(("OVERALL", n_chats, overall))
    return out, n_chats


def _print_grouped_avg_table(avg_rows: list, fields: tuple) -> None:
    header = ["category", "chats", *fields]
    cat_w = max(len("category"), max(len(r[0]) for r in avg_rows))
    num_w = {f: max(len(f), 10) for f in fields}

    def line(category, chats, vals) -> str:
        cells = [category.ljust(cat_w), str(chats).rjust(5)]
        cells += [f"{vals[f]:.2f}".rjust(num_w[f]) for f in fields]
        return "  ".join(cells)

    print("  ".join(["category".ljust(cat_w), "chats".rjust(5),
                     *[f.rjust(num_w[f]) for f in fields]]))
    print("  ".join(["-" * cat_w, "-" * 5, *["-" * num_w[f] for f in fields]]))
    for category, chats, means in avg_rows:
        if category == "OVERALL":
            print("  ".join(["-" * cat_w, "-" * 5, *["-" * num_w[f] for f in fields]]))
        print(line(category, chats, means))
    print("\n(values are mean per chat, averaged across tasks)")


def _write_grouped_avg_csv(avg_rows: list, out, fields: tuple) -> None:
    w = csv.writer(out)
    w.writerow(["category", "chats", *fields])
    for category, chats, means in avg_rows:
        w.writerow([category, chats, *[f"{means[f]:.4f}" for f in fields]])


def cmd_ms_ops(args: argparse.Namespace) -> None:
    rows, probe_counts, skipped = _collect_ms_ops(args.run_dir)
    if not rows:
        raise SystemExit(f"no probe trajectories with ms_ops found under {args.run_dir}")
    if args.avg:
        avg_rows, _ = _average_by_category(rows, _FIELDS)
        if args.csv:
            _write_grouped_avg_csv(avg_rows, sys.stdout, _FIELDS)
        else:
            _print_grouped_avg_table(avg_rows, _FIELDS)
            if skipped:
                print(f"(skipped {skipped} probe(s): missing trajectory.json or ms_ops)")
        return
    if args.csv:
        _write_grouped_csv(rows, probe_counts, sys.stdout, _FIELDS)
    else:
        _print_grouped_table(rows, probe_counts, _FIELDS)
        if skipped:
            print(f"\n(skipped {skipped} probe(s): missing trajectory.json or ms_ops)")


# --- cost -------------------------------------------------------------------

# Per-probe token + turn cost, summed per (chat, category). All three live at
# the top level of the trajectory ``metrics`` block:
#   tokens_in  - cumulative prompt tokens the model was sent across all turns
#   tokens_out - cumulative completion tokens it generated
#   turns      - number of agent steps taken (metrics.step_count)
_COST_METRICS = (
    ("tokens_in", "tokens_in"),
    ("tokens_out", "tokens_out"),
    ("turns", "step_count"),
)
_COST_FIELDS = tuple(name for name, _ in _COST_METRICS)


def _collect_cost(run_dir: Path):
    """Sum per-probe token/turn cost per (chat, category) — the ``--avg`` path.

    Mirrors ``_collect_ms_ops`` but reads the top-level metrics keys. Returns
    ``(rows, probe_counts, skipped)``; a probe missing a key contributes 0 for
    it rather than being dropped. The default (non-avg) ``cost`` view keeps
    probes separate via ``_collect_cost_probes``; this per-category sum only
    feeds ``--avg``.
    """
    rows: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {f: 0 for f in _COST_FIELDS}
    )
    probe_counts: dict[tuple[str, str], int] = defaultdict(int)
    skipped = 0
    for chat, category, _probe, traj in _iter_probe_trajectories(run_dir):
        metrics = (traj or {}).get("metrics") if traj else None
        if not isinstance(metrics, dict):
            skipped += 1
            continue
        key = (chat, category)
        for col, mkey in _COST_METRICS:
            v = metrics.get(mkey, 0)
            rows[key][col] += int(v) if isinstance(v, (int, float)) else 0
        probe_counts[key] += 1
    return rows, probe_counts, skipped


def _collect_cost_probes(run_dir: Path):
    """Per-probe token/turn cost — one row per probe, nothing summed across them.

    Each probe is a separate agent chat, so same-type probes can take a
    different number of turns; keeping them on their own rows surfaces that
    instead of hiding it in a per-category sum. Returns ``(rows, skipped)`` where
    each row is ``{chat, category, probe, tokens_in, tokens_out, turns}``.
    """
    rows: list[dict] = []
    skipped = 0
    for chat, category, probe, traj in _iter_probe_trajectories(run_dir):
        metrics = (traj or {}).get("metrics") if traj else None
        if not isinstance(metrics, dict):
            skipped += 1
            continue
        row = {"chat": chat, "category": category, "probe": probe}
        for col, mkey in _COST_METRICS:
            v = metrics.get(mkey, 0)
            row[col] = int(v) if isinstance(v, (int, float)) else 0
        rows.append(row)
    return rows, skipped


def _probe_sort_key(row: dict):
    """Order rows by chat, then category, then numeric probe index."""
    probe = row["probe"]
    return (
        _chat_sort_key(row["chat"]),
        row["category"],
        int(probe) if str(probe).isdigit() else 0,
        str(probe),
    )


def _print_cost_probe_table(rows: list, fields: tuple = _COST_FIELDS) -> None:
    """One row per probe, grouped by chat with a per-chat ``ALL`` total + TOTAL.

    The ``probe`` column holds each probe's directory index on its own rows; on
    the ``ALL``/``TOTAL`` rows it instead holds the count of probes summed.
    """
    rows = sorted(rows, key=_probe_sort_key)
    header = ["chat", "category", "probe", *fields]
    widths = [len(h) for h in header]
    for r in rows:
        widths[0] = max(widths[0], len(r["chat"]))
        widths[1] = max(widths[1], len(r["category"]))
        widths[2] = max(widths[2], len(str(r["probe"])))
    for i, f in enumerate(fields):
        col = 3 + i
        for r in rows:
            widths[col] = max(widths[col], len(str(r[f])))

    def line(cells) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    print(line(header))
    print(line(["-" * w for w in widths]))

    grand = {f: 0 for f in fields}
    grand_probes = 0
    current_chat = None
    chat_sub = {f: 0 for f in fields}
    chat_probes = 0

    def flush_chat() -> None:
        if current_chat is not None:
            print(line([current_chat, "ALL", chat_probes,
                        *[chat_sub[f] for f in fields]]))
            print()

    for r in rows:
        if r["chat"] != current_chat:
            flush_chat()
            current_chat = r["chat"]
            chat_sub = {f: 0 for f in fields}
            chat_probes = 0
        print(line([r["chat"], r["category"], r["probe"], *[r[f] for f in fields]]))
        for f in fields:
            chat_sub[f] += r[f]
            grand[f] += r[f]
        chat_probes += 1
        grand_probes += 1
    flush_chat()
    print(line(["TOTAL", "", grand_probes, *[grand[f] for f in fields]]))


def _write_cost_probe_csv(rows: list, out, fields: tuple = _COST_FIELDS) -> None:
    w = csv.writer(out)
    w.writerow(["chat", "category", "probe", *fields])
    for r in sorted(rows, key=_probe_sort_key):
        w.writerow([r["chat"], r["category"], r["probe"], *[r[f] for f in fields]])


def cmd_cost(args: argparse.Namespace) -> None:
    # --avg collapses probes into a per-category mean per chat; the default view
    # keeps each probe on its own row so same-type probes' differing turn counts
    # are visible rather than summed away.
    if args.avg:
        rows, _probe_counts, skipped = _collect_cost(args.run_dir)
        if not rows:
            raise SystemExit(
                f"no probe trajectories with metrics found under {args.run_dir}"
            )
        avg_rows, _ = _average_by_category(rows, _COST_FIELDS)
        if args.csv:
            _write_grouped_avg_csv(avg_rows, sys.stdout, _COST_FIELDS)
        else:
            _print_grouped_avg_table(avg_rows, _COST_FIELDS)
            if skipped:
                print(f"(skipped {skipped} probe(s): missing trajectory.json or metrics)")
        return

    rows, skipped = _collect_cost_probes(args.run_dir)
    if not rows:
        raise SystemExit(f"no probe trajectories with metrics found under {args.run_dir}")
    if args.csv:
        _write_cost_probe_csv(rows, sys.stdout)
    else:
        _print_cost_probe_table(rows)
        if skipped:
            print(f"\n(skipped {skipped} probe(s): missing trajectory.json or metrics)")


# --- scores ----------------------------------------------------------------

def _collect_scores(run_dir: Path):
    """Cumulative mean judge score per category over all *finished* chats.

    Each chat writes its own ``tasks/<chat>/scores.json`` the moment it is graded
    (``{per_type: {category: {mean, n, probes}}, overall_reward, n_probes}``), so
    reading those per-chat files yields a cumulative score even while the run is
    still in progress — we don't wait for the final ``summary.json``, which is
    only written once every chat has finished. For each chat we take its per_type
    ``mean`` (that chat's mean over its probes of the category) and average those
    per-chat means across chats — the same way the runner forms a chat's per_type
    means in ``summary.json``. Returns (rows, scored_chats, skipped_chats) with
    ``rows[category] = {"sum": float, "chats": int}``.
    """
    tasks_root = run_dir / "tasks"
    if not tasks_root.is_dir():
        raise SystemExit(f"no tasks/ under {run_dir} (not a run dir?)")
    score_paths = sorted(tasks_root.glob("*/scores.json"))
    if not score_paths:
        raise SystemExit(
            f"no tasks/*/scores.json under {run_dir} (no chat finished/graded yet?)"
        )

    rows: dict[str, dict[str, float]] = defaultdict(lambda: {"sum": 0.0, "chats": 0})
    scored = skipped = 0
    for path in score_paths:
        try:
            scores = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue
        per_type = scores.get("per_type") if isinstance(scores, dict) else None
        if not isinstance(per_type, dict):
            skipped += 1
            continue
        scored += 1
        for category, d in per_type.items():
            mean = d.get("mean") if isinstance(d, dict) else None
            if mean is None:
                continue
            rows[category]["sum"] += float(mean)
            rows[category]["chats"] += 1
    return rows, scored, skipped


def _collect_chat_scores(run_dir: Path, chat: str):
    """Per-category judge score for a single chat — its own ``scores.json``.

    Unlike ``_collect_scores`` (which averages each chat's per_type ``mean``
    across chats), this reads one chat's ``tasks/<chat>/scores.json`` and reports
    that chat's per-category ``mean`` over its own probes directly, plus the probe
    count ``n``. Returns ``rows[category] = {"mean": float, "probes": int}``.
    """
    tasks_root = run_dir / "tasks"
    if not tasks_root.is_dir():
        raise SystemExit(f"no tasks/ under {run_dir} (not a run dir?)")
    path = tasks_root / chat / "scores.json"
    if not path.is_file():
        available = ", ".join(
            sorted((p.parent.name for p in tasks_root.glob("*/scores.json")),
                   key=_chat_sort_key)
        )
        raise SystemExit(
            f"no scores.json for chat {chat!r} under {run_dir} (available: {available})"
        )
    try:
        scores = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"could not read {path}: {e}")
    per_type = scores.get("per_type") if isinstance(scores, dict) else None
    if not isinstance(per_type, dict):
        raise SystemExit(f"no per_type block in {path}")

    rows: dict[str, dict] = {}
    for category, d in per_type.items():
        if not isinstance(d, dict):
            continue
        mean = d.get("mean")
        if mean is None:
            continue
        probes = d.get("n")
        if not isinstance(probes, int):
            p = d.get("probes")
            probes = len(p) if isinstance(p, list) else 0
        rows[category] = {"mean": float(mean), "probes": probes}
    return rows


def _score_table_rows(rows: dict) -> list[tuple]:
    """(category, chats, mean) per category + an unweighted OVERALL row, sorted."""
    out = []
    cat_means = []
    for category in sorted(rows):
        r = rows[category]
        c = int(r["chats"])
        mean = r["sum"] / c if c else 0.0
        out.append((category, c, mean))
        cat_means.append(mean)
    overall = sum(cat_means) / len(cat_means) if cat_means else 0.0
    out.append(("OVERALL", "", overall))
    return out


def _chat_score_table_rows(rows: dict) -> list[tuple]:
    """(category, probes, mean) per category + an unweighted OVERALL row, sorted."""
    out = []
    cat_means = []
    for category in sorted(rows):
        r = rows[category]
        out.append((category, r["probes"], r["mean"]))
        cat_means.append(r["mean"])
    overall = sum(cat_means) / len(cat_means) if cat_means else 0.0
    out.append(("OVERALL", "", overall))
    return out


def cmd_scores(args: argparse.Namespace) -> None:
    if args.chat:
        # Drill into one chat: that chat's own per-category mean over its probes.
        rows = _collect_chat_scores(args.run_dir, args.chat)
        if not rows:
            raise SystemExit(
                f"no graded per-type means for chat {args.chat!r} under {args.run_dir}"
            )
        table = _chat_score_table_rows(rows)
        count_col, note = "probes", None
    else:
        rows, scored, skipped = _collect_scores(args.run_dir)
        if not rows:
            raise SystemExit(f"no graded per-type means in tasks/*/scores.json under {args.run_dir}")
        table = _score_table_rows(rows)
        count_col = "chats"
        note = (f"({scored} chat(s) graded"
                + (f", {skipped} skipped" if skipped else "") + ")")

    if args.csv:
        w = csv.writer(sys.stdout)
        w.writerow(["category", count_col, "mean_score"])
        for category, count, mean in table:
            w.writerow([category, count, f"{mean:.4f}"])
        return

    cat_w = max(len(r[0]) for r in table)
    print(f"{'category'.ljust(cat_w)}  {count_col:>5}  {'mean_score':>10}")
    print(f"{'-' * cat_w}  {'-' * 5}  {'-' * 10}")
    for category, count, mean in table:
        if category == "OVERALL":
            print(f"{'-' * cat_w}  {'-' * 5}  {'-' * 10}")
        print(f"{category.ljust(cat_w)}  {str(count):>5}  {mean:>10.4f}")
    if note:
        print(f"\n{note}")


# --- index ------------------------------------------------------------------

# Per-(chat, category) headline/index-usage counts. Unlike ms-ops/cost these
# are not in the trajectory ``metrics`` block — index usage isn't a counter the
# runner records — so they're derived from the step actions instead. The
# documented coarse-to-fine map idiom (scroll_react/prompts/index.md) is two
# sql_query phases: (2) expand a span to per-turn ``headline`` digests, then
# (3) open the chosen turns' full ``content`` by seq range. We count those, plus
# how often each ``ms`` retrieval primitive is issued (call sites in the
# execute_python source) — for contrast between the map path and the rest:
#   idx_probes      - probes that used the index at all (>=1 headline query).
#                     Keyed on the headline phase only — the unambiguous signal.
#   hl_queries      - step-2 queries: a sql_query selecting/filtering ``headline``
#                     (expand a map span to its per-turn headlines).
#   seq_range_reads - step-3 queries: a sql_query opening ``content`` by seq
#                     range/list (WHERE seq BETWEEN…, seq IN (1,2,3)). This is
#                     "navigated by position" — a superset that ALSO fires when
#                     widening around an ms.search hit, so it's broader/noisier
#                     than hl_queries and is NOT folded into idx_probes.
#   search_snip     - ms.search() calls that triage with a snippet (snippet=True,
#                     the default) — i.e. ms.search() WITHOUT an explicit
#                     snippet=False. The wide-overview path.
#   search_nosnip   - ms.search(..., snippet=False) calls — quiet rows for
#                     aggregation, not printed.
#   expand_calls    - ms.expand() calls — full untruncated reads by seq.
# Each count is over call sites in the source (like the originals).
_INDEX_FIELDS = (
    "idx_probes", "hl_queries", "seq_range_reads",
    "search_snip", "search_nosnip", "expand_calls",
)

# A headline (step-2) query is the map-expansion idiom: a structured ms.sql_query
# that selects/filters on `headline`. Requiring the sql_query context avoids
# counting an FTS like ms.search("how to headline a gig") as index usage.
_HEADLINE_WORD = re.compile(r"\bheadline\b")

# A seq-range (step-3) read opens turns by position: `WHERE seq BETWEEN lo AND hi`
# or `seq IN (12,13,14)` / `seq IN ({rendered list})`. The (?!SELECT) lookahead
# excludes the FTS-via-sql form `seq IN (SELECT rowid FROM …_fts(…))`, which is a
# keyword search, not a positional read.
_SEQ_RANGE = re.compile(r"\bseq\s+BETWEEN\b|\bseq\s+IN\s*\(\s*(?!SELECT)", re.IGNORECASE)


# ms.search defaults to snippet=True, so a call is the quiet aggregation variant
# only when it explicitly passes snippet=False (snippet only exists on ms.search,
# so counting the kwarg attributes it to a search).
_SNIPPET_FALSE = re.compile(r"snippet\s*=\s*False")


def _probe_index_signals(traj: dict) -> dict[str, int]:
    """Per-probe index signals: ``hl_queries``, ``seq_range_reads``, and call
    counts for the ms primitives (``search_snip`` / ``search_nosnip`` /
    ``expand_calls``).

    Counts per execute_python step: ``hl_queries`` = a sql_query touching the
    ``headline`` column (step-2 map expansion); ``seq_range_reads`` = a sql_query
    opening ``content`` by a seq range/list (step-3 positional read) — both count
    at most once per step. ``ms.search(`` calls are split by whether they pass
    ``snippet=False`` (``search_nosnip``) or not (``search_snip``, the default
    snippet=True overview); ``expand_calls`` counts every ``ms.expand(``. A probe
    "used the index" iff ``hl_queries > 0``.
    """
    counts = {f: 0 for f in
              ("hl_queries", "seq_range_reads",
               "search_snip", "search_nosnip", "expand_calls")}
    for step in traj.get("steps") or []:
        action = step.get("action") or {}
        if action.get("tool") != "execute_python":
            continue
        src = (action.get("args") or {}).get("source", "") or ""
        n_search = src.count("ms.search(")
        # snippet=False only appears as an ms.search kwarg; clamp so a stray match
        # can't make nosnip exceed the calls in this step.
        n_nosnip = min(len(_SNIPPET_FALSE.findall(src)), n_search)
        counts["search_nosnip"] += n_nosnip
        counts["search_snip"] += n_search - n_nosnip
        counts["expand_calls"] += src.count("ms.expand(")
        if "sql_query" in src:
            if _HEADLINE_WORD.search(src):
                counts["hl_queries"] += 1
            if "content" in src and _SEQ_RANGE.search(src):
                counts["seq_range_reads"] += 1
    return counts


def _collect_index(run_dir: Path):
    """Sum per-probe index-usage signals per (chat, category).

    Returns ``(rows, probe_counts, skipped)`` where ``rows[(chat, category)]``
    holds the summed ``_INDEX_FIELDS`` and ``probe_counts[(chat, category)]`` the
    number of probes summed. A probe with no ``steps`` contributes 0 to every
    field rather than being dropped.
    """
    rows: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {f: 0 for f in _INDEX_FIELDS}
    )
    probe_counts: dict[tuple[str, str], int] = defaultdict(int)
    skipped = 0
    for chat, category, _probe, traj in _iter_probe_trajectories(run_dir):
        if not isinstance(traj, dict) or "steps" not in traj:
            skipped += 1
            continue
        counts = _probe_index_signals(traj)
        key = (chat, category)
        rows[key]["idx_probes"] += 1 if counts["hl_queries"] > 0 else 0
        for f in _INDEX_FIELDS:
            if f != "idx_probes":
                rows[key][f] += counts[f]
        probe_counts[key] += 1
    return rows, probe_counts, skipped


def _index_rows_by_chat(cells: dict, probe_counts: dict) -> list[dict]:
    """Collapse the category dimension: one row per chat, summed over its probes.

    This is the run-level aggregate — each row is ``{label: chat, probes, ...}``
    summing every probe of that chat. Sorted by scale then numeric chat id.
    """
    agg: dict[str, dict[str, int]] = defaultdict(
        lambda: {"probes": 0, **{f: 0 for f in _INDEX_FIELDS}}
    )
    for (chat, category), r in cells.items():
        agg[chat]["probes"] += probe_counts[(chat, category)]
        for f in _INDEX_FIELDS:
            agg[chat][f] += r[f]
    return [
        {"label": chat, **vals}
        for chat, vals in sorted(agg.items(), key=lambda kv: _chat_sort_key(kv[0]))
    ]


def _index_rows_by_category(cells: dict, probe_counts: dict, chat: str) -> list[dict]:
    """Drill into one chat: one row per probe-type (category) in that chat."""
    rows = [
        {"label": category, "probes": probe_counts[(c, category)],
         **{f: r[f] for f in _INDEX_FIELDS}}
        for (c, category), r in cells.items() if c == chat
    ]
    rows.sort(key=lambda x: x["label"])
    return rows


def _pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.1f}%" if den else "-"


def _index_columns(pct_label: str) -> list[str]:
    """The display columns after ``probes``: every ``_INDEX_FIELD`` in order, with
    the idx-percentage inserted right after ``idx_probes``. ``pct_label`` is
    ``idx%`` for the table, ``idx_pct`` for CSV."""
    cols: list[str] = []
    for f in _INDEX_FIELDS:
        cols.append(f)
        if f == "idx_probes":
            cols.append(pct_label)
    return cols


def _index_row_cells(label, probes, vals) -> list:
    """One row's cells: ``label``, ``probes``, then each ``_INDEX_FIELD`` value
    with the derived idx-percentage after ``idx_probes`` (matches
    ``_index_columns``). ``vals`` is a dict carrying every ``_INDEX_FIELD``."""
    cells = [label, probes]
    for f in _INDEX_FIELDS:
        cells.append(vals[f])
        if f == "idx_probes":
            cells.append(_pct(vals["idx_probes"], probes))
    return cells


def _print_index_table(label_col: str, rows: list, note: str | None = None) -> None:
    """Generic index table: ``label_col`` rows + a derived ``idx%`` and a TOTAL.

    Columns are driven by ``_INDEX_FIELDS`` (so new fields show up automatically).
    ``idx%`` is the share of probes that used the index (idx_probes / probes); it
    can't be summed, so it's recomputed on the TOTAL row from the totals.
    """
    header = [label_col, "probes", *_index_columns("idx%")]
    display = [_index_row_cells(r["label"], r["probes"], r) for r in rows]
    totals = {k: sum(r[k] for r in rows) for k in ("probes", *_INDEX_FIELDS)}
    total_row = _index_row_cells("TOTAL", totals["probes"], totals)

    widths = [len(h) for h in header]
    for cells in (*display, total_row):
        for i, c in enumerate(cells):
            widths[i] = max(widths[i], len(str(c)))

    def line(cells):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    print(line(header))
    print(line(["-" * w for w in widths]))
    for cells in display:
        print(line(cells))
    print(line(["-" * w for w in widths]))
    print(line(total_row))
    if note:
        print(f"\n{note}")


def _write_index_csv(label_col: str, rows: list, out) -> None:
    w = csv.writer(out)
    w.writerow([label_col, "probes", *_index_columns("idx_pct")])
    totals = {k: sum(r[k] for r in rows) for k in ("probes", *_INDEX_FIELDS)}
    for r in rows:
        w.writerow(_index_row_cells(r["label"], r["probes"], r))
    w.writerow(_index_row_cells("TOTAL", totals["probes"], totals))


def cmd_index(args: argparse.Namespace) -> None:
    cells, probe_counts, skipped = _collect_index(args.run_dir)
    if not cells:
        raise SystemExit(f"no probe trajectories with steps found under {args.run_dir}")

    if args.chat:
        # Drill into one chat: per-probe-type usage within that chat.
        rows = _index_rows_by_category(cells, probe_counts, args.chat)
        if not rows:
            available = ", ".join(sorted({c for c, _ in cells}, key=_chat_sort_key))
            raise SystemExit(
                f"no chat {args.chat!r} in {args.run_dir} (available: {available})"
            )
        label_col = "category"
    else:
        # Run-level aggregate: one row per chat, summed over its probe types.
        rows = _index_rows_by_chat(cells, probe_counts)
        label_col = "chat"

    if args.csv:
        _write_index_csv(label_col, rows, sys.stdout)
    else:
        note = (f"(skipped {skipped} probe(s): missing trajectory.json or steps)"
                if skipped else None)
        _print_index_table(label_col, rows, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p_ms = sub.add_parser(
        "ms-ops", help="Per-chat/per-category memory-substrate + eviction counts."
    )
    p_ms.add_argument("run_dir", type=Path, help="The BEAM run directory (contains tasks/).")
    p_ms.add_argument("--csv", action="store_true", help="Emit CSV to stdout instead of a table.")
    p_ms.add_argument("--avg", action="store_true",
                      help="Collapse chats: show per-category mean per chat across all tasks.")
    p_ms.set_defaults(func=cmd_ms_ops)

    p_cost = sub.add_parser(
        "cost", help="Per-chat/per-category input/output token cost and turn count."
    )
    p_cost.add_argument("run_dir", type=Path, help="The BEAM run directory (contains tasks/).")
    p_cost.add_argument("--csv", action="store_true", help="Emit CSV to stdout instead of a table.")
    p_cost.add_argument("--avg", action="store_true",
                        help="Collapse chats: show per-category mean per chat across all tasks.")
    p_cost.set_defaults(func=cmd_cost)

    p_sc = sub.add_parser(
        "scores", help="Cumulative mean judge score per category over finished chats."
    )
    p_sc.add_argument("run_dir", type=Path, help="The BEAM run directory (contains tasks/).")
    p_sc.add_argument("--csv", action="store_true", help="Emit CSV to stdout instead of a table.")
    p_sc.add_argument("--chat", metavar="CHAT",
                      help="Score one chat (e.g. 10M-6): its own per-category mean over its "
                           "probes. Default is the cumulative per-category mean across chats.")
    p_sc.set_defaults(func=cmd_scores)

    p_idx = sub.add_parser(
        "index", help="Headline/index usage: how often probes expand the [memory] map."
    )
    p_idx.add_argument("run_dir", type=Path, help="The BEAM run directory (contains tasks/).")
    p_idx.add_argument("--csv", action="store_true", help="Emit CSV to stdout instead of a table.")
    p_idx.add_argument("--chat", metavar="CHAT",
                       help="Drill into one chat (e.g. 10M-6): per-probe-type usage in it. "
                            "Default aggregates the whole run, one row per chat.")
    p_idx.set_defaults(func=cmd_index)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
