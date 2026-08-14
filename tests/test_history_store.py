"""Tests for the durable, cross-session HistoryStore + read-only model attach."""
from __future__ import annotations

import sqlite3

import pytest

from scroll_context._runtime.history import HistoryStore
from scroll_context._runtime.memoryspace import MemorySpace
from scroll_context._runtime.types import LogEntry


def _entry(**kw) -> LogEntry:
    kw.setdefault("kind", "conversation")  # corpus rows, like the seed tier
    return LogEntry(**kw)


def test_append_assigns_increasing_seq(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    a = store.append(session_id="r:t", run_id="r", task_id="t", entry=_entry(content="a"))
    b = store.append(session_id="r:t", run_id="r", task_id="t", entry=_entry(content="b"))
    assert a == 1 and b == 2
    store.close()


def test_structured_columns_roundtrip(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    store.append(
        session_id="r:t", run_id="r", task_id="t",
        entry=_entry(
            kind="tool_result", role="tool", name="bash", content="out",
            tool_call_id="c1", tool_input={"command": "ls"},
            tool_state="success",
            blocks=[{"type": "tool_result", "id": "c1", "output": "out"}],
        ),
    )
    entry = store.query_log("r:t")[0]
    assert entry.tool_call_id == "c1"
    assert entry.tool_input == {"command": "ls"}
    assert entry.tool_state == "success"
    assert entry.blocks == [{"type": "tool_result", "id": "c1", "output": "out"}]
    store.close()


def test_headline_column_roundtrips(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    store.append(
        session_id="r:t", run_id="r", task_id="t",
        entry=_entry(content="body", headline="found the prod host"),
    )
    assert store.query_log("r:t")[0].headline == "found the prod host"
    store.close()


def test_query_log_tail_and_filter_are_session_scoped(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    store.append(session_id="s1", run_id="r", task_id="t", entry=_entry(content="one"))
    store.append(session_id="s1", run_id="r", task_id="t",
                 entry=_entry(kind="tool_result", name="bash", content="two"))
    store.append(session_id="s2", run_id="r", task_id="t", entry=_entry(content="other"))
    assert store.count("s1") == 2
    assert store.query_log("s1", tail=1)[0].content == "two"
    assert store.query_log("s1", kind="tool_result")[0].content == "two"
    assert store.count("s2") == 1
    store.close()


def test_cross_session_retrieval_via_readonly_attach(tmp_path):
    db = tmp_path / "history.db"
    store = HistoryStore(db)
    store.append(session_id="r1:tX", run_id="r1", task_id="tX", entry=_entry(content="from run1"))
    store.append(session_id="r2:tX", run_id="r2", task_id="tX", entry=_entry(content="from run2"))
    store.append(session_id="r3:tX", run_id="r3", task_id="tX", entry=_entry(content="from run3"))
    # A third session reads its own + prior runs via ms.task_id; ms.session_id
    # narrows to just this run.
    ms = MemorySpace(history_db_path=db, session_id="r3:tX", task_id="tX")
    assert ms.session_id == "r3:tX"
    assert ms.task_id == "tX"
    # bind ms.task_id -> all runs of the task (current + prior)
    by_task = ms.sql_query(
        "SELECT content FROM hist.conversation_history WHERE task_id=? ORDER BY seq",
        (ms.task_id,),
    )
    assert [r["content"] for r in by_task] == ["from run1", "from run2", "from run3"]
    # bind ms.session_id -> just this run
    by_session = ms.sql_query(
        "SELECT content FROM hist.conversation_history WHERE session_id=?",
        (ms.session_id,),
    )
    assert [r["content"] for r in by_session] == ["from run3"]
    ms.close()
    store.close()


def test_history_is_read_only_to_model(tmp_path):
    db = tmp_path / "history.db"
    store = HistoryStore(db)
    store.append(session_id="r:t", run_id="r", task_id="t", entry=_entry(content="x"))
    ms = MemorySpace(history_db_path=db, session_id="r:t")
    # hist is ATTACHed read-only, so any write through ms is rejected by SQLite.
    with pytest.raises(sqlite3.OperationalError):
        ms.sql_query(
            "INSERT INTO hist.conversation_history (session_id, kind) VALUES ('z','k')"
        )
    ms.close()
    store.close()



def test_wal_mode_enabled(tmp_path):
    db = tmp_path / "history.db"
    store = HistoryStore(db)
    store.append(session_id="r:t", run_id="r", task_id="t", entry=_entry(content="x"))
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    store.close()


def test_file_persists_and_reopens(tmp_path):
    db = tmp_path / "history.db"
    store = HistoryStore(db)
    store.append(session_id="r:t", run_id="r", task_id="t", entry=_entry(content="durable"))
    store.close()
    assert db.exists()
    # Re-open: schema is idempotent and prior rows are still there.
    store2 = HistoryStore(db)
    assert store2.count("r:t") == 1
    store2.close()


def test_fts_search_ranks_and_filters(tmp_path):
    db = tmp_path / "history.db"
    store = HistoryStore(db)
    assert store._fts is True  # FTS5 available in this build
    rows = [
        ("tool_result", "installed the nvidia display driver at /usr/lib"),
        ("tool_result", "configured the network bridge and firewall rules"),
        ("model_turn", "verify the display driver loaded"),
    ]
    for i, (k, txt) in enumerate(rows):
        store.append(session_id="r:t", run_id="r", task_id="t",
                     entry=_entry(kind=k, content=txt, step_index=i))
    ms = MemorySpace(history_db_path=db, session_id="r:t", task_id="t")
    assert ms._fts_available() is True

    # keyword search returns both 'display driver' rows (these seeded rows are
    # the reader's own scaffolding kinds, so self-search is opted into)
    hits = {r["step_index"] for r in ms.search("display driver", include_self=True)}
    assert hits == {0, 2}
    # prefix query
    assert {r["step_index"] for r in ms.search("fire*", include_self=True)} == {1}
    # boolean
    assert {r["step_index"] for r in ms.search("network OR firewall", include_self=True)} == {1}
    # kind filter narrows to tool_result only
    assert {r["step_index"] for r in ms.search("driver", kind="tool_result")} == {0}
    # search counts as an FTS-route hist read in the op stats
    assert ms.stats()["hist_fts"] >= 4
    ms.close()
    store.close()


def test_search_seq_range_bounds_hits_to_a_span(tmp_path):
    """seq_range: the map's coordinate as a first-class search filter — the
    ranked/sanitized replacement for raw FTS sub-selects over BETWEEN."""
    db = tmp_path / "history.db"
    store = HistoryStore(db)
    for i in range(6):
        store.append(session_id="seed:t", run_id="seed", task_id="t",
                     entry=_entry(content=f"the zeppelin budget item {i}", step_index=i))
    store.close()
    ms = MemorySpace(history_db_path=db, session_id="r:t", task_id="t")
    all_hits = ms.search("zeppelin", scope="task", k=10)
    assert len(all_hits) == 6
    bounded = ms.search("zeppelin", scope="task", k=10, seq_range=(2, 4))
    assert {h["seq"] for h in bounded} == {2, 3, 4}
    # Composes with the scaffolding exclusion (bounded rows are seed-tier).
    assert all(h["kind"] == "conversation" for h in bounded)
    ms.close()


def test_fts_search_scope_session_vs_task(tmp_path):
    db = tmp_path / "history.db"
    store = HistoryStore(db)
    store.append(session_id="r1:t", run_id="r1", task_id="t",
                 entry=_entry(content="firewall config from run one", step_index=0))
    store.append(session_id="r2:t", run_id="r2", task_id="t",
                 entry=_entry(content="firewall config from run two", step_index=0))
    ms = MemorySpace(history_db_path=db, session_id="r2:t", task_id="t")
    # default scope=session -> only this run
    assert len(ms.search("firewall")) == 1
    # scope=task -> both runs
    assert len(ms.search("firewall", scope="task")) == 2
    ms.close()
    store.close()


def test_memoryspace_op_stats_classify_scratch_vs_hist(tmp_path):
    db = tmp_path / "history.db"
    store = HistoryStore(db)
    store.append(session_id="r:t", run_id="r", task_id="t",
                 entry=_entry(content="hello world"))
    ms = MemorySpace(history_db_path=db, session_id="r:t")

    ms.sql_query("SELECT content FROM hist.conversation_history")  # hist_read
    ms.search("hello")                                            # hist_read

    assert ms.stats() == {
        "hist_fts": 1, "hist_seq": 1, "hist_scan": 0,
    }
    ms.close()
    store.close()


def test_history_store_quarantines_corrupt_db(tmp_path):
    db = tmp_path / "history.db"
    # A non-SQLite file + a stale WAL sidecar: opening must fail, then recover.
    db.write_bytes(b"this is definitely not a sqlite database " * 8)
    (tmp_path / "history.db-shm").write_bytes(b"x" * 64)

    hs = HistoryStore(db)  # must NOT raise
    # the corrupt file was moved aside, not deleted silently
    corrupt = [p.name for p in tmp_path.iterdir() if p.name.startswith("history.db.corrupt-")]
    assert corrupt, f"expected a quarantined file, dir has {[p.name for p in tmp_path.iterdir()]}"
    assert hs.quarantined_to is not None
    # a fresh, working store was created — appends and reads work
    seq = hs.append(session_id="r:t", run_id="r", task_id="t",
                    entry=_entry(content="recovered"))
    assert seq == 1
    assert hs.query_log("r:t")[0].content == "recovered"
    hs.close()


def test_history_store_healthy_db_not_quarantined(tmp_path):
    db = tmp_path / "history.db"
    HistoryStore(db).close()                 # create a valid DB
    hs = HistoryStore(db)                     # reopen — should be untouched
    assert hs.quarantined_to is None
    assert not any(p.name.startswith("history.db.corrupt-") for p in tmp_path.iterdir())
    hs.close()


def test_sql_query_truncation_is_out_of_band_and_type_safe(tmp_path):
    """A capped result contains ONLY data rows — truncation is signaled on the
    list (`rows.truncated`), never as an in-band marker row. The old marker's
    empty-string values poisoned typed column operations
    (`sorted(set(r['seq'] …))` → TypeError; `min(dates)` silently `""`)."""
    db = tmp_path / "history.db"
    store = HistoryStore(db)
    for i in range(5):
        store.append(session_id="r:t", run_id="r", task_id="t", entry=_entry(content=f"m{i}"))
    store.close()

    ms = MemorySpace(history_db_path=db, session_id="r:t", task_id="t", row_cap=3)
    rows = ms.sql_query(
        "SELECT seq, content FROM hist.conversation_history ORDER BY seq"
    )
    assert len(rows) == 3                      # exactly the cap — the in-data tell
    assert rows.truncated is True and rows.row_cap == 3
    # Every row is a real, type-homogeneous data row: the operations the old
    # marker crashed/corrupted now just work.
    assert sorted({r["seq"] for r in rows}) == [r["seq"] for r in rows]
    assert min(r["content"] for r in rows) == "m0"
    # An uncapped result says so too.
    two = ms.sql_query("SELECT seq FROM hist.conversation_history LIMIT 2")
    assert two.truncated is False and len(two) == 2
    ms.close()


def _seed_store(db, texts_with_headlines):
    store = HistoryStore(db)
    for content, headline in texts_with_headlines:
        store.append(
            session_id="r:t", run_id="r", task_id="t",
            entry=_entry(content=content, headline=headline),
        )
    store.close()


def test_fts_indexes_headline_column(tmp_path):
    db = tmp_path / "history.db"
    _seed_store(db, [
        ("I solved twenty problems from Section 14.2 today", "weak-area targeting and study planning"),
        ("we watched a movie and ate popcorn", None),
    ])
    ms = MemorySpace(history_db_path=db, session_id="r:t", task_id="t")
    # "targeting" appears only in the headline, never in any turn's prose
    hits = ms.search("targeting planning", scope="task")
    assert hits and hits[0]["via"] == "headline"
    assert "Section 14.2" in (hits[0].get("snippet") or "") or hits[0]["headline"]
    ms.close()


def test_fts_v2_to_v3_migration(tmp_path):
    db = tmp_path / "history.db"
    _seed_store(db, [("alpha content here", "bravo headline text")])
    # Downgrade to the v2 (prose/code) schema to simulate an old DB.
    con = sqlite3.connect(db)
    con.execute("DROP TABLE conversation_history_fts")
    con.execute(
        "CREATE VIRTUAL TABLE conversation_history_fts USING fts5(prose, code, "
        "content='conversation_history', content_rowid='seq', tokenize='porter')"
    )
    con.execute("INSERT INTO conversation_history_fts(conversation_history_fts) VALUES('rebuild')")
    con.commit(); con.close()
    # Re-open through HistoryStore: must detect v2 and rebuild as v3.
    store = HistoryStore(db); store.close()
    ms = MemorySpace(history_db_path=db, session_id="r:t", task_id="t")
    hits = ms.search("bravo", scope="task")   # headline-only term
    assert hits and hits[0]["via"] == "headline"
    ms.close()


def test_search_via_prose_for_text_matches(tmp_path):
    db = tmp_path / "history.db"
    _seed_store(db, [("the quick brown fox jumps", None)])
    ms = MemorySpace(history_db_path=db, session_id="r:t", task_id="t")
    hits = ms.search("quick fox", scope="task")
    assert hits and hits[0]["via"] == "prose"
    assert "broadened" not in hits[0]
    ms.close()


def test_search_auto_broadens_thin_and_query(tmp_path):
    db = tmp_path / "history.db"
    _seed_store(db, [
        ("Nancy recommended the hostel in Bangkok", None),
        ("we discussed the walking tour route", None),
        ("completely unrelated turn about cooking pasta", None),
    ])
    ms = MemorySpace(history_db_path=db, session_id="r:t", task_id="t")
    # AND query matches nothing (no single turn has both terms) -> auto-OR
    hits = ms.search("hostel tour", scope="task")
    assert hits, "auto-broaden should surface OR leads"
    assert all(h.get("broadened") for h in hits)
    seqs = {h["seq"] for h in hits}
    assert len(seqs) == 2  # both single-term turns, not the pasta one
    # ranked by matched-term count: both have 1 term, order stable — now check
    # a two-term candidate outranks one-term ones
    _seed_store_extra = HistoryStore(db)
    _seed_store_extra.append(session_id="r:t", run_id="r", task_id="t",
        entry=_entry(content="the hostel offered a free walking tour"))
    _seed_store_extra.close()
    ms2 = MemorySpace(history_db_path=db, session_id="r:t", task_id="t")
    hits2 = ms2.search("hostel tour", scope="task")
    # now an exact AND match exists -> no broadening needed, via prose
    assert hits2[0]["via"] == "prose" and "broadened" not in hits2[0]
    ms2.close(); ms.close()


def test_search_no_broaden_when_query_has_or(tmp_path):
    db = tmp_path / "history.db"
    _seed_store(db, [("nothing relevant at all", None)])
    ms = MemorySpace(history_db_path=db, session_id="r:t", task_id="t")
    hits = ms.search("zebra OR unicorn", scope="task")
    assert hits == []  # already-OR queries are not re-broadened
    ms.close()


def _seed_sessions(db, sessions):
    """sessions: {session_id: [(content, headline), ...]}"""
    store = HistoryStore(db)
    for sid, turns in sessions.items():
        for content, headline in turns:
            store.append(session_id=sid, run_id="r", task_id="t",
                         entry=_entry(content=content, headline=headline))
    store.close()


def test_headline_router_surfaces_in_span_turn_not_boundary(tmp_path):
    """A headline match routes to the best-matching turn INSIDE the session."""
    db = tmp_path / "history.db"
    _seed_sessions(db, {"r:s1": [
        ("welcome to the planning call", "catering budget discussion"),  # boundary/headlined
        ("we talked about flowers", None),
        ("the catering costs were finalized at $5,000", None),           # the real target
    ]})
    ms = MemorySpace(history_db_path=db, session_id="replay:t", task_id="t")
    hits = ms.search("catering budget", scope="task")
    routed = [h for h in hits if h["via"] == "headline"]
    assert routed, "headline router should fire"
    top = routed[0]
    assert "catering costs" in (top.get("snippet") or "")  # in-span turn, own snippet
    assert "⟦via summary of S" in (top.get("snippet") or "")  # provenance marker
    # the boundary (headlined) turn is not what got surfaced
    assert "welcome to the planning call" not in (top.get("snippet") or "")
    ms.close()


def test_headline_router_opener_fallback_on_vocabulary_gap(tmp_path):
    """No query term in any turn: session opener returned as marked lead."""
    db = tmp_path / "history.db"
    _seed_sessions(db, {"r:s1": [
        ("first turn of the region", "quarterly fitness milestones review"),
        ("second turn text", None),
    ]})
    ms = MemorySpace(history_db_path=db, session_id="replay:t", task_id="t")
    hits = ms.search("fitness milestones", scope="task")
    routed = [h for h in hits if h["via"] == "headline"]
    assert routed and "first turn of the region" in (routed[0].get("snippet") or "")
    assert "⟦via summary of S" in (routed[0].get("snippet") or "")
    ms.close()


def test_headline_router_overflow_marker_lists_unsurfaced_sessions(tmp_path):
    """More matched sessions than the routed budget -> last marker lists them."""
    db = tmp_path / "history.db"
    sessions = {}
    for i in range(6):
        sessions[f"r:s{i}"] = [(f"turn about travel destination {i}", f"travel itinerary phase {i}")]
    _seed_sessions(db, sessions)
    ms = MemorySpace(history_db_path=db, session_id="replay:t", task_id="t")
    hits = ms.search("travel itinerary", scope="task", k=4)  # budget = k//2 = 2 routed rows
    routed = [h for h in hits if h["via"] == "headline"]
    assert routed
    last_snip = routed[-1].get("snippet") or ""
    assert "more session(s) matched summaries" in last_snip
    ms.close()
