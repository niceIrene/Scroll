#!/usr/bin/env python3
"""Read back the headlines / index representations from a logged probe.

A scroll_agent_A probe writes three files we can inspect:

  - ``call_messages.jsonl`` — the exact prompt dumped each turn. Line 0 is the
    (constant) system prompt, which carries the **seed map** (prior-session
    index); later lines may carry the live **eviction index** placeholder
    (the ``name="memory"`` message that appears once eviction fires).
  - ``trajectory.json`` — the model's own turns, including any ``⟦…⟧`` headlines
    it authored.
  - ``history.db`` — the durable ``conversation_history`` rows, i.e. the raw
    ``headline`` column for both the ingested ``seed`` rows and this run's turns.
    This is a single DB shared by every probe of a chat, so it lives at the task
    dir (``tasks/<chat>/history.db``), not inside the probe dir; we walk up from
    the probe dir to find it.

This prints all four views for a probe directory::

    python3 scripts/read_index.py <probe_dir>
    python3 scripts/read_index.py <run_or_task_dir> --all
    .venv/bin/python scripts/read_index.py <probe_dir> --full

A probe dir looks like::

    runs/<run>/tasks/<chat>/probes/<category>/<i>/

Pass any ancestor (a task or whole run dir) and it discovers the probe dirs
beneath it: with one match it analyzes it, with several it lists them (use
``--all`` to analyze every match). Stdlib only — no scroll_eval import needed.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

_SEED_MARKER = "An index of your PRIOR conversation sessions"  # unique to the seed map header
_EVICT_MARKER = "[context compressed]"


def _text(content: Any) -> str:
    """Flatten a Msg ``content`` (str, or a list of block dicts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("text")
        )
    return ""


def _systeminfo_block(text: str, needle: str) -> str | None:
    """The ``<system-info>…</system-info>`` span that contains ``needle``."""
    i = text.find(needle)
    if i < 0:
        return None
    start = text.rfind("<system-info>", 0, i)
    start = start if start >= 0 else i
    end = text.find("</system-info>", i)
    end = end + len("</system-info>") if end >= 0 else len(text)
    return text[start:end]


def _clip(s: str, chars: int | None) -> str:
    if chars is None or len(s) <= chars:
        return s
    return s[:chars].rstrip() + f" …(+{len(s) - chars} chars)"


def _find_probe_dirs(root: Path) -> list[Path]:
    """Probe dirs (those holding a ``call_messages.jsonl``) at or under ``root``."""
    if (root / "call_messages.jsonl").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("call_messages.jsonl"))


# --- the four views --------------------------------------------------------


def show_seed_map(probe: Path, chars: int | None) -> None:
    print("\n=== 1. SEED MAP (prior-session index, in the system prompt) ===")
    f = probe / "call_messages.jsonl"
    if not f.exists():
        print("  (no call_messages.jsonl)")
        return
    with f.open(encoding="utf-8") as fh:
        first = fh.readline()
    sysmsg = json.loads(first).get("system", {}) if first else {}
    block = _systeminfo_block(_text(sysmsg.get("content", "")), _SEED_MARKER)
    print(
        _clip(block, chars)
        if block
        else "  (no seed map in system prompt — SCROLL_SEED_INDEX off or no seed rows)"
    )


def show_eviction_maps(probe: Path, chars: int | None) -> None:
    print('\n=== 2. LIVE EVICTION INDEX (the name="memory" placeholder) ===')
    f = probe / "call_messages.jsonl"
    if not f.exists():
        print("  (no call_messages.jsonl)")
        return
    last_text: str | None = None
    last_step = None
    n_turns = 0
    with f.open(encoding="utf-8") as fh:
        for ln in fh:
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "messages" not in rec:
                continue
            for m in rec["messages"]:
                t = _text(m.get("content", ""))
                if m.get("name") == "memory" or _EVICT_MARKER in t:
                    n_turns += 1
                    last_text, last_step = t, rec.get("step")
                    break
    if last_text is None:
        print("  (no eviction placeholder — eviction never fired this probe)")
        return
    print(f"  appeared in {n_turns} turn(s); showing the final one (step {last_step}):")
    print(_clip(last_text, chars))


def _find_history_db(probe: Path) -> Path | None:
    """Locate the run's ``history.db``.

    Since history is a single DB shared by every probe of a chat, it lives at the
    task dir (``tasks/<chat>/history.db``), not inside the probe dir. Check the
    probe dir first (legacy per-probe layout), then walk up to find it.
    """
    for d in (probe, *probe.parents):
        cand = d / "history.db"
        if cand.exists():
            return cand
    return None


def show_db_headlines(probe: Path, chars: int | None) -> None:
    print("\n=== 3. HEADLINE COLUMN (durable conversation_history) ===")
    db = _find_history_db(probe)
    if db is None:
        print("  (no history.db found in probe dir or any ancestor)")
        return
    conn = sqlite3.connect(str(db))
    try:
        groups = conn.execute(
            "SELECT run_id, COUNT(*) FROM conversation_history "
            "WHERE headline IS NOT NULL GROUP BY run_id ORDER BY run_id"
        ).fetchall()
        if not groups:
            print("  (no headlines in this DB)")
            return
        for run_id, count in groups:
            tag = "ingested prior sessions" if run_id == "seed" else "this run's own turns"
            print(f"  run_id={run_id!r}: {count} headlines  ({tag})")
            sample = conn.execute(
                "SELECT seq, headline FROM conversation_history "
                "WHERE headline IS NOT NULL AND run_id=? ORDER BY seq LIMIT 6",
                (run_id,),
            ).fetchall()
            for seq, hl in sample:
                print(f"      seq {seq:>5}  {_clip(hl, chars)}")
    finally:
        conn.close()


def show_model_headlines(probe: Path, chars: int | None) -> None:
    print("\n=== 4. MODEL-AUTHORED ⟦…⟧ HEADLINES (from its own turns) ===")
    f = probe / "trajectory.json"
    if not f.exists():
        print("  (no trajectory.json)")
        return
    tj = json.loads(f.read_text(encoding="utf-8"))
    steps = tj.get("steps", [])
    hits = [(s.get("index"), s.get("thought") or "") for s in steps if "⟦" in (s.get("thought") or "")]
    if not hits:
        print(f"  0 of {len(steps)} turns emitted a ⟦…⟧ headline")
        return
    print(f"  {len(hits)} turn(s) emitted a headline:")
    for i, th in hits:
        print(f"      step {i}: {_clip(th, chars)}")


def analyze(probe: Path, chars: int | None) -> None:
    print("#" * 78)
    print(f"# {probe}")
    print("#" * 78)
    show_seed_map(probe, chars)
    show_eviction_maps(probe, chars)
    show_db_headlines(probe, chars)
    show_model_headlines(probe, chars)
    print()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", help="a probe dir, or any ancestor (task / run dir) to search under")
    ap.add_argument("--all", action="store_true", help="analyze every probe dir found, not just one")
    ap.add_argument("--full", action="store_true", help="don't truncate long text")
    ap.add_argument("--chars", type=int, default=100, help="truncation width (default 100; --full overrides)")
    args = ap.parse_args(argv)

    root = Path(args.path)
    if not root.exists():
        raise SystemExit(f"path not found: {root}")
    probes = _find_probe_dirs(root)
    if not probes:
        raise SystemExit(f"no probe dirs (call_messages.jsonl) found under {root}")

    chars = None if args.full else args.chars
    if len(probes) == 1 or args.all:
        for p in probes:
            analyze(p, chars)
    else:
        print(f"{len(probes)} probe dirs found under {root} — pass one, or --all:")
        for p in probes:
            print(f"  {p}")
        sys.exit(0)


if __name__ == "__main__":
    main()
