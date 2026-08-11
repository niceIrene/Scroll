#!/usr/bin/env python3
"""Generate native scroll_eval local-tasks from a LongMemEval dataset file.

LongMemEval (https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) is
multi-session chat QA: each instance is a long conversation history (the
"haystack") plus a question and a gold answer. This turns each instance into a
self-contained task under ``local-tasks/<dataset>/<question_id>/`` for the
*native* scroll_eval eval (``scroll-eval longmemeval``), NOT the Harbor path.

Each generated task dir:
- ``task.toml``           — metadata (``runner="native"``, ``benchmark``, ``question_type``).
- ``sessions.json``       — the haystack (``has_answer`` flags stripped so the agent
                            can't cheat), sessions sorted chronologically by date, in
                            the shape ``ingest.build_seed_db`` reads.
- ``questions.json``      — the single agent-facing probe ``[{id, type, question,
                            question_date}]`` (no gold answer).
- ``tests/probing_questions.json`` — the held-out gold, grouped by question type
                            for the judge: ``{qtype: [{id, question, answer,
                            question_type, is_abstention}]}``.

Stdlib-only (no scroll_eval import needed).

Usage:
    uv run python scripts/gen_longmemeval_tasks.py \
        --src external/longmemeval/data/longmemeval_s.json \
        --dataset longmemeval --limit 10

    # or pick specific question ids / a question type:
    uv run python scripts/gen_longmemeval_tasks.py --src ... --qids q1,q2
    uv run python scripts/gen_longmemeval_tasks.py --src ... --question-type knowledge-update
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

_TASK_TOML = """\
schema_version = "1.1"

[task]
name = "longmemeval/{qid}"
description = "LongMemEval memory-QA over a multi-session conversation."

[metadata]
runner = "native"
benchmark = "longmemeval"
question_type = "{question_type}"
is_abstention = {is_abstention}

[verifier]
timeout_sec = 300.0
"""

# LongMemEval dates look like "2023/05/20 (Sat) 02:21"; pull Y/M/D [+ H:M] for a
# chronological sort key. Sessions with an unparseable date sort last (stable).
_DATE_RE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})(?:\D+(\d{1,2}):(\d{2}))?")


def _sort_key(raw: str | None) -> tuple:
    m = _DATE_RE.search(raw or "")
    if not m:
        return (9999, 12, 31, 23, 59)
    y, mo, d, h, mi = m.groups()
    return (int(y), int(mo), int(d), int(h or 0), int(mi or 0))


def _build_sessions(instance: dict[str, Any]) -> list[dict[str, Any]]:
    """Sessions (has_answer stripped), sorted chronologically by date."""
    sessions_in = instance.get("haystack_sessions", [])
    dates = instance.get("haystack_dates", [])
    session_ids = instance.get("haystack_session_ids", [])
    rows = []
    for idx, session in enumerate(sessions_in):
        turns = [
            {"role": turn.get("role", ""), "content": turn.get("content", "")}
            for turn in session
        ]
        rows.append(
            {
                "session_id": str(session_ids[idx]) if idx < len(session_ids) else None,
                "date": dates[idx] if idx < len(dates) else "",
                "turns": turns,
            }
        )
    rows.sort(key=lambda r: _sort_key(r.get("date")))
    return rows


def _write_task(out_root: Path, instance: dict[str, Any]) -> Path:
    qid = str(instance["question_id"])
    qtype = str(instance.get("question_type", ""))
    is_abstention = "_abs" in qid
    task_dir = out_root / qid
    if task_dir.exists():
        shutil.rmtree(task_dir)
    (task_dir / "tests").mkdir(parents=True)

    (task_dir / "task.toml").write_text(
        _TASK_TOML.format(
            qid=qid, question_type=qtype, is_abstention=str(is_abstention).lower()
        ),
        encoding="utf-8",
    )
    (task_dir / "sessions.json").write_text(
        json.dumps(
            {
                "question_date": instance.get("question_date", ""),
                "sessions": _build_sessions(instance),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Agent-facing probe: question only, no gold.
    (task_dir / "questions.json").write_text(
        json.dumps(
            [
                {
                    "id": qid,
                    "type": qtype,
                    "question": instance.get("question", ""),
                    "question_date": instance.get("question_date", ""),
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    # Held-out gold for the judge, grouped by type (aligns with answers.json).
    (task_dir / "tests" / "probing_questions.json").write_text(
        json.dumps(
            {
                qtype: [
                    {
                        "id": qid,
                        "question": instance.get("question", ""),
                        "answer": instance.get("answer", ""),
                        "question_type": qtype,
                        "is_abstention": is_abstention,
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return task_dir


def _select(instances: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.qids:
        wanted = {q.strip() for q in args.qids.split(",") if q.strip()}
        picked = [i for i in instances if str(i.get("question_id")) in wanted]
    elif args.question_type:
        picked = [i for i in instances if i.get("question_type") == args.question_type]
    else:
        picked = instances
    return picked[: args.limit] if args.limit and args.limit > 0 else picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="longmemeval_*.json")
    parser.add_argument("--dataset", default="longmemeval", help="local-tasks subdir")
    parser.add_argument("--out-root", default="local-tasks", type=Path)
    parser.add_argument("--limit", type=int, default=10, help="0 = no limit")
    parser.add_argument("--qids", default="", help="comma-separated question_ids")
    parser.add_argument("--question-type", default="", help="filter by question_type")
    args = parser.parse_args()

    instances = json.loads(args.src.read_text(encoding="utf-8"))
    if not isinstance(instances, list):
        raise SystemExit(f"{args.src}: expected a JSON list of instances")

    picked = _select(instances, args)
    out_root = args.out_root / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    written = [_write_task(out_root, inst) for inst in picked]
    print(f"Wrote {len(written)} task(s) to {out_root}/")
    for path in written:
        print(f"  {path.name}")
    if written:
        print("\nNext: uv run scroll-eval longmemeval configs/longmemeval.yaml")


if __name__ == "__main__":
    main()
