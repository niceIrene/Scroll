"""Ingest seeds retrievable cross-session memory; probes share one DB, isolated by run id."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from scroll_context._runtime.history import HistoryStore
from scroll_context._runtime.memoryspace import MemorySpace
from scroll_context._runtime.types import LogEntry
from scroll_eval.evals.beam.ingest import (
    SEED_RUN_ID,
    build_seed_db,
    clean_content,
    iter_turns,
)


def _chat() -> list[dict]:
    return [
        {"batch_number": 1, "turns": [[
            {"role": "user", "content": "Let's build a Flask budget tracker. ->-> 1,1",
             "id": 0, "time_anchor": "March-15-2024"},
        ]]},
        {"batch_number": 2, "turns": [[
            {"role": "assistant", "content": "Add a category column to transactions.",
             "id": 0, "time_anchor": "April-01-2024"},
        ]]},
    ]


def test_clean_content_strips_marker() -> None:
    assert clean_content("Help me ->-> 1,1") == "Help me"
    assert clean_content("no marker here") == "no marker here"


def test_batch_date_propagates_to_every_turn() -> None:
    # 10M shape: batch-level time_anchor is null; the date is on the first
    # message only. Every turn must still inherit the session date.
    chat = [
        {"batch_number": 7, "time_anchor": None, "turns": [[
            {"role": "user", "content": "first", "id": 0, "time_anchor": "August-17-2024"},
            {"role": "assistant", "content": "second", "id": 1},   # no per-msg date
            {"role": "user", "content": "third", "id": 2, "time_anchor": None},
        ]]},
    ]
    turns = list(iter_turns(chat))
    assert [t["time_anchor"] for t in turns] == ["August-17-2024"] * 3


def test_propagated_date_is_queryable_and_tagged(tmp_path: Path) -> None:
    task_id = "beam/date-1"
    seed = tmp_path / "seed.db"
    chat = [
        {"batch_number": 3, "time_anchor": None, "turns": [[
            {"role": "user", "content": "opener", "id": 0, "time_anchor": "August-17-2024"},
            {"role": "assistant", "content": "follow-up reply with no own date", "id": 1},
        ]]},
    ]
    build_seed_db(chat, task_id, seed)
    ms = MemorySpace(history_db_path=seed, session_id=f"run:{task_id}", task_id=task_id)

    # The dateless follow-up row is date-filterable via the sortable ISO
    # metadata.date (BETWEEN works because it is lexically sortable).
    rows = ms.sql_query(
        "SELECT content, json_extract(metadata, '$.time_anchor') AS anchor "
        "FROM hist.conversation_history "
        "WHERE json_extract(metadata, '$.date') BETWEEN '2024-08-01' AND '2024-08-31' "
        "AND content LIKE '%follow-up%'"
    )
    assert rows, "propagated date not queryable on a non-opener turn"
    # The raw anchor is still kept for the human-readable form...
    assert rows[0]["anchor"] == "August-17-2024"
    # ...and the ISO date shows in that turn's content tag.
    assert "2024-08-17" in rows[0]["content"]


def test_seed_is_retrievable_by_later_task_session(tmp_path: Path) -> None:
    task_id = "beam/test-1"
    seed = tmp_path / "seed.db"
    n = build_seed_db(_chat(), task_id, seed)
    assert n == 2

    # A later session with the SAME task_id retrieves seeded turns via scope=task.
    ms = MemorySpace(history_db_path=seed, session_id=f"run1:{task_id}", task_id=task_id)
    hits = ms.search("Flask budget tracker", scope="task", k=5, snippet=False)
    assert hits, "seeded conversation not retrievable"
    # Session tagging is present so cross-session/temporal reasoning is possible.
    assert "[Session 1" in hits[0]["content"]


def test_search_truncates_to_preview_expandable_to_full(tmp_path: Path) -> None:
    # Each hit's content is a ~600-char preview with an expand pointer (cheap
    # triage on large-turn corpora); chars=None returns the full turn.
    task_id = "beam/test-1"
    seed = tmp_path / "seed.db"
    long_text = "encryption microservice " + "extra detail. " * 100  # > 600 chars
    chat = [{"batch_number": 1, "turns": [[
        {"role": "user", "content": long_text, "id": 0, "time_anchor": "March-15-2024"},
    ]]}]
    build_seed_db(chat, task_id, seed)

    ms = MemorySpace(history_db_path=seed, session_id=f"r:{task_id}", task_id=task_id)
    preview = ms.search("encryption", scope="task", k=1, snippet=False)[0]["content"]
    assert "ms.expand(" in preview                       # truncation pointer present
    assert len(preview) < len(long_text)                 # actually shortened
    full = ms.search("encryption", scope="task", k=1, chars=None, snippet=False)[0]["content"]
    assert "ms.expand(" not in full                      # full content, no pointer
    assert full.count("extra detail.") == 100            # nothing dropped
    ms.close()


def test_seed_rows_headline_assistant_turns(tmp_path: Path) -> None:
    # Milestone headlines are sampled from a session's ASSISTANT turns (mirroring
    # the live index, where only the model's turns carry a headline) — never from
    # user turns. Each is dated with the session anchor; the leading filler of an
    # assistant reply ("Certainly!") is stripped. A session with no assistant turn
    # falls back to its opener.
    task_id = "beam/test-1"
    seed = tmp_path / "seed.db"
    chat = [
        {"batch_number": 1, "turns": [[
            {"role": "user", "content": "Let's build a Flask budget tracker.",
             "id": 0, "time_anchor": "March-15-2024"},
            {"role": "assistant", "content": "Certainly! Use SQLite for storage.",
             "id": 1},
            {"role": "user", "content": "Now add CSV export.", "id": 2},
            {"role": "assistant", "content": "Add a /export route returning CSV.",
             "id": 3},
        ]]},
        {"batch_number": 2, "turns": [[
            {"role": "user", "content": "A user-only session.",
             "id": 0, "time_anchor": "April-01-2024"},
        ]]},
    ]
    build_seed_db(chat, task_id, seed)

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
    assert len(headlined) == 2                       # both assistant turns
    assert all(r["role"] == "assistant" for r in headlined)   # never user turns
    heads1 = [r["headline"] for r in headlined]
    assert all(h.startswith("Session 1 | March-15-2024 —") for h in heads1)  # dated
    assert "Certainly" not in heads1[0]              # filler stripped
    assert "SQLite" in heads1[0] and "CSV" in heads1[1]

    # No assistant turn → fall back to the session opener so the map still has a leaf.
    s2 = by_session[f"seed:{task_id}:s2"]
    assert len([r for r in s2 if r["headline"]]) == 1


def test_seed_index_off_leaves_headline_column_null(tmp_path: Path) -> None:
    # The --no-index ablation (seed_index=False) must not persist ANY headline, so
    # the index data is physically absent from the DB — otherwise the agent can
    # still reach it via `SELECT headline FROM …` even with the in-context map off.
    task_id = "beam/test-1"
    seed = tmp_path / "seed.db"
    chat = [
        {"batch_number": 1, "turns": [[
            {"role": "user", "content": "Let's build a Flask budget tracker.",
             "id": 0, "time_anchor": "March-15-2024"},
            {"role": "assistant", "content": "Certainly! Use SQLite for storage.",
             "id": 1},
        ]]},
    ]
    n = build_seed_db(chat, task_id, seed, seed_index=False)
    assert n == 2  # rows still ingested — only the headline column is suppressed

    conn = sqlite3.connect(seed)
    populated = conn.execute(
        "SELECT COUNT(*) FROM conversation_history WHERE headline IS NOT NULL AND headline != ''"
    ).fetchone()[0]
    conn.close()
    assert populated == 0, "seed_index=False must leave every headline NULL"

    # ...and the turns themselves stay fully retrievable (only the index is gone).
    ms = MemorySpace(history_db_path=seed, session_id=f"r:{task_id}", task_id=task_id)
    assert ms.search("Flask budget tracker", scope="task", k=5, snippet=False)


def test_shared_db_probes_isolated_by_run_id(tmp_path: Path) -> None:
    """Probes share ONE history DB; scope='task' isolation is by run id.

    Each probe writes its own turns into the same file (no per-probe copy). With
    ``shared_run_ids=(SEED_RUN_ID,)`` a probe's ``scope='task'`` search returns
    the seed tier plus its OWN turns, never a sibling probe's.
    """
    task_id = "beam/test-1"
    shared = tmp_path / "history.db"
    build_seed_db(_chat(), task_id, shared)  # both probes share this one file

    # Probe A and probe B each write a working row into the SAME db, under their
    # own unique run id / session id.
    store_a = HistoryStore(shared)
    store_a.append(session_id=f"a:{task_id}", run_id="a", task_id=task_id,
                   entry=LogEntry(kind="model_turn", role="assistant", content="LEAKMARKER_A"))
    store_a.close()
    store_b = HistoryStore(shared)
    store_b.append(session_id=f"b:{task_id}", run_id="b", task_id=task_id,
                   entry=LogEntry(kind="model_turn", role="assistant", content="LEAKMARKER_B"))
    store_b.close()

    ms_a = MemorySpace(history_db_path=shared, session_id=f"a:{task_id}",
                       task_id=task_id, shared_run_ids=(SEED_RUN_ID,))
    ms_b = MemorySpace(history_db_path=shared, session_id=f"b:{task_id}",
                       task_id=task_id, shared_run_ids=(SEED_RUN_ID,))

    # Each probe sees its OWN write (include_self=True: these are the live
    # session's own model_turn rows, self-excluded from discovery by default)
    # but NOT the sibling's.
    assert ms_a.search("LEAKMARKER_A", scope="task", k=5, include_self=True)
    assert ms_a.search("LEAKMARKER_B", scope="task", k=5) == []
    assert ms_b.search("LEAKMARKER_B", scope="task", k=5, include_self=True)
    assert ms_b.search("LEAKMARKER_A", scope="task", k=5) == []

    # Both still retrieve the SHARED seed tier (the prior conversation).
    assert ms_a.search("Flask budget tracker", scope="task", k=5)
    assert ms_b.search("Flask budget tracker", scope="task", k=5)


def test_shared_run_ids_unset_keeps_all_runs_visible(tmp_path: Path) -> None:
    """Without shared_run_ids, scope='task' is the plain "all runs" scan (back-compat)."""
    task_id = "beam/test-2"
    shared = tmp_path / "history.db"
    build_seed_db(_chat(), task_id, shared)

    store = HistoryStore(shared)
    store.append(session_id=f"a:{task_id}", run_id="a", task_id=task_id,
                 entry=LogEntry(kind="model_turn", role="assistant", content="SIBLINGMARKER"))
    store.close()

    # A different session, no shared_run_ids -> sees every run's turns of the task.
    ms = MemorySpace(history_db_path=shared, session_id=f"b:{task_id}", task_id=task_id)
    assert ms.search("SIBLINGMARKER", scope="task", k=5)
