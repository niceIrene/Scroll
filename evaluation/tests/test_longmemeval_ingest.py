"""Ingest seeds a retrievable multi-session memory the later task session recalls."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from scroll_context._runtime.memoryspace import MemorySpace
from scroll_eval.evals.longmemeval.ingest import (
    SEED_RUN_ID,
    build_seed_db,
    to_iso_date,
)


def _sessions() -> list[dict]:
    return [
        {
            "session_id": "s0",
            "date": "2024/03/15 (Fri) 09:30",
            "turns": [
                {"role": "user", "content": "Let's plan a Flask budget tracker."},
                {"role": "assistant", "content": "Certainly! Use SQLite for storage."},
            ],
        },
        {
            "session_id": "s1",
            "date": "2024/04/01 (Mon) 12:00",
            "turns": [
                {"role": "user", "content": "A user-only follow-up session."},
            ],
        },
    ]


def test_to_iso_date_normalizes() -> None:
    assert to_iso_date("2024/03/15 (Fri) 09:30") == "2024-03-15"
    assert to_iso_date("2024/3/5") == "2024-03-05"
    assert to_iso_date("garbage") is None


def test_seed_is_retrievable_by_later_task_session(tmp_path: Path) -> None:
    task_id = "longmemeval/qa-1"
    seed = tmp_path / "seed.db"
    n = build_seed_db(_sessions(), task_id, seed)
    assert n == 3  # 2 + 1 turns

    ms = MemorySpace(history_db_path=seed, session_id=f"run:{task_id}", task_id=task_id)
    hits = ms.search("Flask budget tracker", scope="task", k=5, snippet=False)
    assert hits, "seeded conversation not retrievable"
    assert "[Session 1" in hits[0]["content"]  # session-tagged for chronology


def test_iso_date_queryable_and_tagged(tmp_path: Path) -> None:
    task_id = "longmemeval/date-1"
    seed = tmp_path / "seed.db"
    build_seed_db(_sessions(), task_id, seed)
    ms = MemorySpace(history_db_path=seed, session_id=f"r:{task_id}", task_id=task_id)

    rows = ms.sql_query(
        "SELECT content, json_extract(metadata, '$.date') AS date "
        "FROM hist.conversation_history "
        "WHERE json_extract(metadata, '$.date') BETWEEN '2024-03-01' AND '2024-03-31' "
        "AND content LIKE '%Flask%'"
    )
    assert rows, "ISO date not queryable"
    assert rows[0]["date"] == "2024-03-15"
    assert "2024-03-15" in rows[0]["content"]  # ISO date in the content tag


def test_seed_rows_headline_assistant_turns(tmp_path: Path) -> None:
    # Milestone headlines are sampled from a session's ASSISTANT turns; each is
    # dated with the session date; leading filler ("Certainly!") is stripped. A
    # session with no assistant turn falls back to its opener.
    task_id = "longmemeval/qa-1"
    seed = tmp_path / "seed.db"
    build_seed_db(_sessions(), task_id, seed)

    conn = sqlite3.connect(seed)
    conn.row_factory = sqlite3.Row
    by_session: dict[str, list] = {}
    for r in conn.execute(
        "SELECT session_id, seq, role, headline FROM conversation_history ORDER BY seq"
    ):
        by_session.setdefault(r["session_id"], []).append(r)
    conn.close()

    s1 = by_session[f"seed:{task_id}:s1"]
    headlined = [r for r in s1 if r["headline"]]
    assert len(headlined) == 1
    assert headlined[0]["role"] == "assistant"
    assert headlined[0]["headline"].startswith("Session 1 | 2024-03-15 —")
    assert "Certainly" not in headlined[0]["headline"]  # filler stripped
    assert "SQLite" in headlined[0]["headline"]

    # No assistant turn → fall back to the session opener so the map still has a leaf.
    s2 = by_session[f"seed:{task_id}:s2"]
    assert len([r for r in s2 if r["headline"]]) == 1


def test_seed_index_off_leaves_headline_column_null(tmp_path: Path) -> None:
    task_id = "longmemeval/qa-1"
    seed = tmp_path / "seed.db"
    n = build_seed_db(_sessions(), task_id, seed, seed_index=False)
    assert n == 3

    conn = sqlite3.connect(seed)
    populated = conn.execute(
        "SELECT COUNT(*) FROM conversation_history WHERE headline IS NOT NULL AND headline != ''"
    ).fetchone()[0]
    conn.close()
    assert populated == 0, "seed_index=False must leave every headline NULL"

    ms = MemorySpace(history_db_path=seed, session_id=f"r:{task_id}", task_id=task_id)
    assert ms.search("Flask budget tracker", scope="task", k=5, snippet=False)


def test_run_id_is_seed(tmp_path: Path) -> None:
    # Seeded rows carry run_id='seed' so ScrollContextManager.seed_index_map() and
    # the runner's shared_run_ids=(SEED_RUN_ID,) isolation both key off it.
    assert SEED_RUN_ID == "seed"
    task_id = "longmemeval/qa-1"
    seed = tmp_path / "seed.db"
    build_seed_db(_sessions(), task_id, seed)
    conn = sqlite3.connect(seed)
    run_ids = {r[0] for r in conn.execute("SELECT DISTINCT run_id FROM conversation_history")}
    conn.close()
    assert run_ids == {"seed"}
