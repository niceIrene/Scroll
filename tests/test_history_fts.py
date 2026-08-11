"""Tests for the FTS5 keyword index on conversation_history."""
from __future__ import annotations

import sqlite3

from scroll_context._runtime.history import HistoryStore
from scroll_context._runtime.memoryspace import MemorySpace, or_terms
from scroll_context._runtime.types import LogEntry


def _entry(content, kind="model_turn"):
    return LogEntry(kind=kind, role="assistant", content=content)


def _seed(store):
    for text in [
        "the quick brown fox jumps",
        "lazy dog database index lookup",
        None,  # NULL content must not break the trigger
        "database normalization and indexing rules",
    ]:
        store.append(session_id="r:t", run_id="r", task_id="t", entry=_entry(text))


def test_fts_matches_via_writer_connection(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    _seed(store)
    # Query the FTS index directly on the writer connection.
    rows = store._conn.execute(
        "SELECT seq FROM conversation_history "
        "WHERE seq IN (SELECT rowid FROM conversation_history_fts('database')) "
        "ORDER BY seq"
    ).fetchall()
    assert [r["seq"] for r in rows] == [2, 4]
    store.close()


def test_fts_searchable_through_readonly_attach(tmp_path):
    """The model reaches FTS over MemorySpace's read-only `hist` attach."""
    store = HistoryStore(tmp_path / "history.db")
    _seed(store)
    store.close()

    ms = MemorySpace(history_db_path=tmp_path / "history.db", session_id="r:t")
    rows = ms.sql_query(
        "SELECT seq, content FROM hist.conversation_history "
        "WHERE seq IN (SELECT rowid FROM hist.conversation_history_fts('fox')) "
        "ORDER BY seq"
    )
    assert [r["seq"] for r in rows] == [1]
    # Porter stemming: "index" matches both "index" (row 2) and "indexing" (row 4).
    assert [r["seq"] for r in ms.sql_query(
        "SELECT seq FROM hist.conversation_history "
        "WHERE seq IN (SELECT rowid FROM hist.conversation_history_fts('index')) "
        "ORDER BY seq"
    )] == [2, 4]
    # Prefix query matches both "index" and "indexing".
    assert [r["seq"] for r in ms.sql_query(
        "SELECT seq FROM hist.conversation_history "
        "WHERE seq IN (SELECT rowid FROM hist.conversation_history_fts('index*')) "
        "ORDER BY seq"
    )] == [2, 4]
    ms.close()


def test_fts_index_stays_in_sync_on_new_appends(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    _seed(store)
    store.append(session_id="r:t", run_id="r", task_id="t",
                 entry=_entry("a brand new searchable sentinel token"))
    rows = store._conn.execute(
        "SELECT content FROM conversation_history "
        "WHERE seq IN (SELECT rowid FROM conversation_history_fts('sentinel'))"
    ).fetchall()
    assert len(rows) == 1 and "sentinel" in rows[0]["content"]
    store.close()


def test_fts_backfills_preexisting_rows(tmp_path):
    """An older DB with rows but no FTS table gets backfilled on next open."""
    db = tmp_path / "history.db"
    # Build a conversation_history with rows but WITHOUT the FTS table/triggers.
    raw = sqlite3.connect(db)
    raw.execute(
        "CREATE TABLE conversation_history ("
        "seq INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
        "run_id TEXT, task_id TEXT, step_index INTEGER, msg_index INTEGER, "
        "kind TEXT NOT NULL, role TEXT, name TEXT, content TEXT, tool_call_id TEXT, "
        "tool_input TEXT, tool_state TEXT, blocks TEXT, metadata TEXT, created_at TEXT)"
    )
    raw.execute(
        "INSERT INTO conversation_history (session_id, kind, content) "
        "VALUES ('r:t', 'model_turn', 'legacy row mentioning kubernetes')"
    )
    raw.commit()
    raw.close()

    # Opening via HistoryStore should create FTS and rebuild over the legacy row.
    store = HistoryStore(db)
    rows = store._conn.execute(
        "SELECT content FROM conversation_history "
        "WHERE seq IN (SELECT rowid FROM conversation_history_fts('kubernetes'))"
    ).fetchall()
    assert len(rows) == 1 and "kubernetes" in rows[0]["content"]
    store.close()


def _seed_ranged(store):
    """Four turns across distinct sessions/positions for range + snippet tests."""
    rows = [
        (1, 10, "designing a vector database cluster for embeddings"),
        (2, 20, "optimizing FAISS index on GPU for 200k docs"),
        (3, 30, "comparing batch vs streaming ingestion strategies"),
        (4, 40, "query rewriting to improve recall in vector retrieval"),
    ]
    for step, msg, text in rows:
        store.append(
            session_id="r:t", run_id="r", task_id="t",
            entry=LogEntry(kind="conversation", role="user", content=text,
                           step_index=step, msg_index=msg),
        )


def test_search_snippet_returns_compact_match_centred_digest(tmp_path):
    """snippet=True swaps full `content` for a match-centred `snippet` digest."""
    store = HistoryStore(tmp_path / "memory.db")
    _seed_ranged(store)
    store.close()

    ms = MemorySpace(history_db_path=tmp_path / "memory.db", session_id="r:t")
    hits = ms.search("vector", scope="session", snippet=True)
    assert {h["step_index"] for h in hits} == {1, 4}     # both vector turns
    assert all("snippet" in h and "content" not in h for h in hits)
    assert all("vector" in h["snippet"] for h in hits)
    # snippet=False returns full content (no snippet key), quietly, for aggregation.
    plain = ms.search("FAISS", scope="session", snippet=False)
    assert "content" in plain[0] and "snippet" not in plain[0]
    ms.close()


def test_search_step_and_msg_range_filter_the_window(tmp_path):
    """step_range / msg_range scope the keyword search to a session/chrono window."""
    store = HistoryStore(tmp_path / "memory.db")
    _seed_ranged(store)
    store.close()

    ms = MemorySpace(history_db_path=tmp_path / "memory.db", session_id="r:t")
    # "vector" matches sessions 1 and 4, but the range restricts to 1..2.
    assert {h["step_index"] for h in
            ms.search("vector", scope="session", step_range=(1, 2))} == {1}
    # msg_index window 25..45 keeps only the later half.
    assert {h["step_index"] for h in
            ms.search("vector OR ingestion", scope="session", msg_range=(25, 45))} == {3, 4}
    ms.close()


def test_search_steps_and_seqs_restrict_to_candidate_set(tmp_path):
    """steps=/seqs= scope the search to an explicit candidate set (the funnel)."""
    store = HistoryStore(tmp_path / "memory.db")
    _seed_ranged(store)
    store.close()

    ms = MemorySpace(history_db_path=tmp_path / "memory.db", session_id="r:t")
    # "vector" matches sessions 1 and 4; steps= narrows to session 4 only.
    assert {h["step_index"] for h in ms.search("vector", scope="session", steps=[4])} == {4}
    # seqs= narrows to a specific turn (seq 1 is session 1's vector turn).
    assert {h["seq"] for h in ms.search("vector", scope="session", seqs=[1])} == {1}
    # An explicit EMPTY candidate set matches nothing (not "no filter").
    assert ms.search("vector", scope="session", steps=[]) == []
    assert ms.search("vector", scope="session", seqs=[]) == []
    ms.close()


def test_or_terms_builds_safe_fts_or_expression():
    """or_terms ORs alternatives and quotes phrases/hyphenated terms."""
    assert (or_terms(["module", "message-passing", "event driven"])
            == 'module OR "message-passing" OR "event driven"')
    # Bare words & prefix terms pass through; blanks dropped; existing quotes kept.
    assert or_terms(["deploy*", "", "   ", "ci"]) == "deploy* OR ci"
    assert or_terms(['"already quoted"', "plain"]) == '"already quoted" OR plain'


def _seed_funnel(store):
    """Topic and the decision live in DIFFERENT turns of the same session."""
    rows = [
        (5, 50, "I'm designing a modular simulation framework with several components"),  # topic
        (5, 51, "I decided to go with message-passing for communication between them"),    # mechanism+decision
        (2, 20, "here is a modular design sketch for the data pipeline"),                  # topic only
        (9, 90, "I prefer detailed logging in all of my services"),                        # decision only, off-topic
    ]
    for step, msg, text in rows:
        store.append(session_id="r:t", run_id="r", task_id="t",
                     entry=LogEntry(kind="conversation", role="user", content=text,
                                    step_index=step, msg_index=msg))


def test_progressive_funnel_ranks_multi_axis_session_first(tmp_path):
    """Per-axis OR recall + coverage ranking + steps= read finds the cross-turn answer."""
    store = HistoryStore(tmp_path / "memory.db")
    _seed_funnel(store)
    store.close()

    ms = MemorySpace(history_db_path=tmp_path / "memory.db", session_id="r:t", task_id="t")
    axes = {
        "topic":     ["module", "modular", "component", "framework"],
        "mechanism": ["message-passing", "communication"],
        "decision":  ["decided", "prefer", "go with"],
    }
    # PHASE 1 — per-axis OR recall, keep KEYS only in a plain dict (the working
    # data now lives in a persisted namespace variable, not a scratch table).
    from collections import defaultdict

    axes_by_step: dict[int, set] = defaultdict(set)
    for axis, terms in axes.items():
        for h in ms.search(or_terms(terms), scope="task", kind="conversation",
                           k=60, snippet=False):
            if h["step_index"] > 0:
                axes_by_step[h["step_index"]].add(axis)

    # PHASE 2 — rank sessions by axis coverage (most distinct axes first).
    ranked = sorted(axes_by_step.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    # Session 5 covers all three axes (across two turns) -> ranks first.
    assert ranked[0][0] == 5 and len(ranked[0][1]) == 3
    assert len(ranked[0][1]) > len(ranked[1][1])  # strictly above topic-only / decision-only

    # PHASE 3 — read within the survivor; the decision turn is surfaced.
    best = [ranked[0][0]]
    got = ms.search(or_terms(sum(axes.values(), [])), scope="task", steps=best,
                    snippet=False)
    assert any("message-passing" in r["content"] for r in got)
    ms.close()


def test_expand_returns_full_prose_and_code_is_opt_in(tmp_path):
    """expand() gives untruncated prose by default; code only when asked."""
    store = HistoryStore(tmp_path / "memory.db")
    long_prose = "alpha " * 400  # well past any preview budget
    store.append(session_id="r:t", run_id="r", task_id="t",
                 entry=_entry(f"{long_prose}\n```python\nsecret = compute()\n```\ndone"))
    store.close()

    ms = MemorySpace(history_db_path=tmp_path / "memory.db", session_id="r:t")
    row = ms.expand([1])[0]
    assert row["content"].count("alpha") == 400        # untruncated prose
    assert "secret = compute()" not in row["content"]  # code elided
    assert "code" not in row                            # not pulled by default
    withcode = ms.expand([1], code=True)[0]
    assert "secret = compute()" in withcode["code"]     # opt-in code field
    assert ms.expand([]) == []
    ms.close()


def test_search_returns_quietly_without_printing(tmp_path, capsys):
    """search RETURNS rows and prints nothing — the model decides what to show."""
    store = HistoryStore(tmp_path / "memory.db")
    for text in ["vector database design", "solr cluster sizing", "vector cluster notes"]:
        store.append(session_id="r:t", run_id="r", task_id="t", entry=_entry(text))
    store.close()

    ms = MemorySpace(history_db_path=tmp_path / "memory.db", session_id="r:t")
    hits = ms.search("vector OR solr", scope="session")   # snippet=True by default
    assert isinstance(hits, list) and all(isinstance(h["seq"], int) for h in hits)
    assert {h["seq"] for h in hits} == {1, 2, 3}
    assert capsys.readouterr().out == ""                  # prints nothing
    quiet = ms.search("vector OR solr", scope="session", snippet=False)
    assert {h["seq"] for h in quiet} == {1, 2, 3}
    assert capsys.readouterr().out == ""                  # snippet=False also silent
    ms.close()


def test_expand_returns_full_text_without_printing(tmp_path, capsys):
    """expand() RETURNS the full untruncated rows and prints nothing."""
    store = HistoryStore(tmp_path / "memory.db")
    body = "beta " * 300
    store.append(session_id="r:t", run_id="r", task_id="t", entry=_entry(body))
    store.close()

    ms = MemorySpace(history_db_path=tmp_path / "memory.db", session_id="r:t")
    rows = ms.expand([1])
    assert {r["seq"] for r in rows} == {1}               # list of rows for reuse
    assert rows[0]["content"].count("beta") == 300       # full untruncated content
    assert capsys.readouterr().out == ""                 # prints nothing
    ms.close()


def test_expand_accepts_search_result_directly(tmp_path):
    """The search->expand carry needs no glue: expand() takes search()'s list of
    hit-dicts, a filtered subset, a {seq: row} dict, a bare seq, or a list of seqs."""
    store = HistoryStore(tmp_path / "memory.db")
    for text in ["vector database design", "solr cluster sizing", "vector cluster notes"]:
        store.append(session_id="r:t", run_id="r", task_id="t", entry=_entry(text))
    store.close()

    ms = MemorySpace(history_db_path=tmp_path / "memory.db", session_id="r:t")
    hits = ms.search("vector OR solr", scope="session")

    def expanded(arg):
        return {r["seq"] for r in ms.expand(arg)}

    assert expanded(hits) == {1, 2, 3}                                            # list of hit-dicts
    assert expanded([h["seq"] for h in hits
                     if "vector" in (h.get("snippet") or "")]) == {1, 3}          # subset
    assert expanded({h["seq"]: h for h in hits}) == {1, 2, 3}                     # {seq: row} dict
    assert expanded(2) == {2}                                                     # bare seq
    assert expanded([1, 2]) == {1, 2}                                             # list of seqs
    ms.close()
