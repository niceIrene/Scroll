"""Unit tests for the ScrollRuntime building blocks."""
from __future__ import annotations

import asyncio

from scroll_context._runtime import ScrollRuntime
from scroll_context._runtime.exec import Executor
from scroll_context._runtime.history import HistoryStore
from scroll_context._runtime.memoryspace import MemorySpace
from scroll_context._runtime.types import LogEntry


def _populate_hist(db, n):
    """Seed a HistoryStore with n model_turn rows; return the db path."""
    store = HistoryStore(db)
    for i in range(n):
        store.append(session_id="r:t", run_id="r", task_id="t",
                     entry=LogEntry(kind="model_turn", role="assistant",
                                    content=f"row{i}"))
    store.close()
    return db


# --- MemorySpace -------------------------------------------------------


def test_memoryspace_reads_hist(tmp_path):
    db = _populate_hist(tmp_path / "memory.db", 3)
    ms = MemorySpace(history_db_path=db, session_id="r:t", task_id="t")
    rows = ms.sql_query("SELECT content FROM hist.conversation_history ORDER BY seq")
    assert [r["content"] for r in rows] == ["row0", "row1", "row2"]
    assert ms.stats()["hist_seq"] == 1
    ms.close()


def test_memoryspace_row_cap_marks_truncation(tmp_path):
    db = _populate_hist(tmp_path / "memory.db", 5)
    ms = MemorySpace(history_db_path=db, row_cap=2)
    rows = ms.sql_query("SELECT content FROM hist.conversation_history ORDER BY seq")
    # Exactly the capped data rows; truncation is out-of-band on the list.
    assert len(rows) == 2
    assert rows.truncated is True and rows.row_cap == 2
    assert all(isinstance(r.get("content"), str) and r["content"] for r in rows)
    ms.close()


# --- Executor ----------------------------------------------------------


def test_executor_persists_namespace_between_calls():
    ns: dict = {}
    ex = Executor(ns)
    asyncio.run(ex.execute("x = 1\ny = 2"))
    result = asyncio.run(ex.execute("print(x + y)"))
    assert result.stdout.strip() == "3"
    assert result.error is None


def test_executor_captures_stderr_on_exception():
    ex = Executor({})
    result = asyncio.run(ex.execute("raise RuntimeError('boom')"))
    assert result.error == "execution error"
    assert "RuntimeError: boom" in result.stderr


def test_executor_reports_syntax_error_distinctly():
    ex = Executor({})
    result = asyncio.run(ex.execute("def : pass"))
    assert result.error == "SyntaxError"


def test_executor_supports_top_level_await():
    """Model can write `await bash(...)` at the top level of a cell."""
    async def fake_bash(cmd: str) -> str:
        return f"ran:{cmd}"

    ns: dict = {"bash": fake_bash}
    ex = Executor(ns)
    result = asyncio.run(
        ex.execute("out = await bash('ls')\nprint(out)")
    )
    assert result.stdout.strip() == "ran:ls"
    assert ns["out"] == "ran:ls"


def test_executor_timeout_fires():
    ex = Executor({}, timeout_s=0.1)
    result = asyncio.run(
        ex.execute("import asyncio\nawait asyncio.sleep(1.0)")
    )
    assert result.error is not None
    assert "timed out" in result.error


def test_executor_caps_oversized_stdout_with_retry_notice():
    """Output past the cap is trimmed to a head + an actionable overflow notice;
    under-cap output passes through untouched."""
    ex = Executor({}, max_stdout_chars=2000)
    big = asyncio.run(ex.execute("print('x' * 5000)"))
    assert len(big.stdout) < 5000                 # not the full dump
    assert "output too long" in big.stdout        # actionable notice present
    assert "2000" in big.stdout                   # the limit is reported
    small = asyncio.run(ex.execute("print('ok')"))
    assert small.stdout.strip() == "ok"           # under cap: untouched


def test_stdout_cap_scales_with_budget_and_clamps():
    """stdout_cap_for is ~history_max_tokens//4, clamped to [2k, 32k]; unknown
    budget falls back to the 32k ceiling."""
    from scroll_context._runtime.exec import stdout_cap_for
    assert stdout_cap_for(None) == 32_000         # unbounded -> ceiling
    assert stdout_cap_for(500_000) == 32_000      # huge window -> ceiling binds
    assert stdout_cap_for(64_000) == 16_000       # proportional region
    assert stdout_cap_for(8_000) == 2_000         # tiny window -> floor binds


# --- ScrollRuntime end-to-end -----------------------------------------


def test_runtime_namespace_contains_expected_handles(tmp_path):
    rt = ScrollRuntime(
        history_db_path=tmp_path / "memory.db",
    )
    try:
        ns = rt.namespace
        # The namespace is the recall + compute surface; actions (bash,
        # submit_answer) are the agent's top-level tools, not exposed here.
        assert {"ms", "days_between", "or_terms"} <= set(ns)
        assert "bash" not in ns
        assert "submit_answer" not in ns
    finally:
        rt.close()


def test_runtime_append_visible_via_hist_in_namespace(tmp_path):
    """A write-through append is immediately readable from hist.conversation_history."""
    rt = ScrollRuntime(
        history_db_path=tmp_path / "memory.db",
        session_id="r:t",
        run_id="r",
        task_id="t",
    )
    try:
        rt.append_log(LogEntry(kind="model_turn", role="assistant", content="thinking"))
        result = asyncio.run(rt.execute(
            "rows = ms.sql_query("
            "'SELECT content FROM hist.conversation_history WHERE session_id=?', "
            "(ms.session_id,))\n"
            "print(len(rows)); print(rows[0]['content'])"
        ))
        out = result.stdout.strip().splitlines()
        assert out == ["1", "thinking"]
        # The runtime's own helper sees it too.
        assert rt.log_entries()[0].content == "thinking"
    finally:
        rt.close()


def test_runtime_vars_persist_across_cells(tmp_path):
    """A variable assigned in one cell is readable in the next — the model's
    working memory is plain Python state in the persistent namespace."""
    rt = ScrollRuntime(
        history_db_path=tmp_path / "memory.db",
    )
    try:
        asyncio.run(rt.execute("notes = {'a': 'one'}\nhits = [1, 2, 3]"))
        result = asyncio.run(rt.execute("print(notes['a'], len(hits))"))
        assert result.stdout.strip() == "one 3"
    finally:
        rt.close()


def test_runtime_digest_lists_persisted_vars(tmp_path):
    """The working-memory digest names the model's stored variables, not the
    injected fixtures (ms/helpers), so it survives turns scrolling out."""
    rt = ScrollRuntime(history_db_path=tmp_path / "memory.db")
    try:
        assert rt.digest() == "vars: (empty)"
        asyncio.run(rt.execute("hits = [1, 2, 3]\ntotals = {'x': 1}"))
        digest = rt.digest()
        assert "hits (list, 3 items)" in digest
        assert "totals (dict, 1 keys)" in digest
        # injected fixtures are hidden from the digest
        assert "days_between" not in digest
        assert "or_terms" not in digest
    finally:
        rt.close()
