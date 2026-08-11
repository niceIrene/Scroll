"""Unit tests for scroll_context — the packaged context manager.

Covers the behaviors formerly tested at the agent level in
test_scroll_window.py (token estimation, eviction, observation aging) plus the
package's own surface: group eviction with the pinned placeholder map, the
index-off ablation placeholder, the working-memory digest, the prompt
protocol assembly (and its index-off headline strip), and the seeded
prior-sessions map.
"""
from __future__ import annotations

import uuid

import pytest

from scroll_context._runtime.types import LogEntry
from scroll_context import (
    HEADLINE_SCHEMA_FRAGMENTS,
    SCROLL_REPL_TOOL_SCHEMA,
    ScrollContextManager,
    core_prompt,
    index_prompt,
    protocol_prompt,
    strip_headline_schema,
)
from scroll_context.manager import (
    _est_tokens,
    _obs_keep_turns_from_env,
)


def _mgr(tmp_path, **kw):
    kw.setdefault("history_db_path", str(tmp_path / "history.db"))
    kw.setdefault("session_id", "r:t")
    kw.setdefault("run_id", "r")
    kw.setdefault("task_id", "t")
    kw.setdefault("history_max_tokens", 0)
    kw.setdefault("repl_name", "execute_python")
    return ScrollContextManager(**kw)


def _assistant(text: str, *, tool: str | None = None, output: str | None = None):
    """One recorded exchange: assistant dict (+ its tool-result dict)."""
    msgs = []
    a: dict = {"role": "assistant", "content": text}
    if tool is not None:
        call_id = uuid.uuid4().hex
        a["tool_calls"] = [
            {"id": call_id, "type": "function",
             "function": {"name": tool, "arguments": "{}"}}
        ]
        msgs.append(a)
        msgs.append({"role": "tool", "tool_call_id": call_id, "name": tool,
                     "content": output or ""})
    else:
        msgs.append(a)
    return msgs


def _record_exchange(mgr, history, text, *, tool=None, output=None):
    msgs = _assistant(text, tool=tool, output=output)
    history.extend(msgs)
    mgr.record_assistant_turn(msgs[0])
    if len(msgs) > 1:
        mgr.record_tool_result(msgs[1], tool_name=tool)
    return msgs


# --- token estimation ----------------------------------------------------- #

def test_est_tokens_char_heuristic():
    assert _est_tokens("x" * 400) == 101  # 400 // 4 + 1
    assert _est_tokens("") == 1


def test_est_tokens_counts_tool_call_arguments():
    msg = {
        "role": "assistant",
        "content": "hi",
        "tool_calls": [{"id": "1", "type": "function",
                        "function": {"name": "f", "arguments": "x" * 398}}],
    }
    assert _est_tokens(msg) > _est_tokens("hi")


# --- eviction ------------------------------------------------------------- #

def test_evict_keeps_pinned_head_and_drops_oldest_groups(tmp_path):
    mgr = _mgr(tmp_path, history_max_tokens=200, pinned=2)
    system = {"role": "system", "content": "S" * 40}
    task = {"role": "user", "content": "T" * 40}
    history = [system, task]
    mgr.record_initial_prompt(task)
    for i in range(6):
        _record_exchange(mgr, history, f"turn {i}", tool="execute_python",
                         output="o" * 300)
    events = mgr.manage(history)
    assert events["evicted_msgs"] > 0
    # Pinned head survives.
    assert history[0] is system and history[1] is task
    # The placeholder map sits right after the pinned head.
    assert history[2] is mgr._placeholder
    assert "[context compressed]" in history[2]["content"]
    # The newest exchange is never evicted.
    assert history[-2]["content"] == "turn 5"
    # Groups evict atomically: no orphan role:"tool" right after the placeholder.
    assert history[3].get("role") == "assistant"


def test_evict_never_drops_newest_group(tmp_path):
    mgr = _mgr(tmp_path, history_max_tokens=10, pinned=1)
    task = {"role": "user", "content": "T" * 400}
    history = [task]
    mgr.record_initial_prompt(task)
    _record_exchange(mgr, history, "only turn", tool="execute_python", output="o" * 400)
    mgr.manage(history)
    # Even hopelessly over budget, the newest group stays.
    assert any(m.get("role") == "assistant" for m in history)


def test_evict_noop_when_under_budget(tmp_path):
    mgr = _mgr(tmp_path, history_max_tokens=100_000, pinned=1)
    task = {"role": "user", "content": "task"}
    history = [task]
    mgr.record_initial_prompt(task)
    _record_exchange(mgr, history, "turn", tool="execute_python", output="out")
    events = mgr.manage(history)
    assert "evicted_msgs" not in events
    assert mgr._placeholder is None


def test_evicted_headlines_fold_into_index_map(tmp_path):
    mgr = _mgr(tmp_path, history_max_tokens=150, pinned=1)
    task = {"role": "user", "content": "task"}
    history = [task]
    mgr.record_initial_prompt(task)
    _record_exchange(mgr, history, "⟦ config.db_host = prod-3 ⟧",
                     tool="execute_python", output="o" * 400)
    for i in range(3):
        _record_exchange(mgr, history, f"later {i}", tool="execute_python",
                         output="o" * 400)
    mgr.manage(history)
    assert mgr._placeholder is not None
    assert "config.db_host = prod-3" in mgr._placeholder["content"]
    # The recovery instructions name the host's REPL tool.
    assert "execute_python" in mgr._placeholder["content"]


def test_index_off_placeholder_is_opaque_span(tmp_path):
    mgr = _mgr(tmp_path, history_max_tokens=150, pinned=1, enable_index=False)
    task = {"role": "user", "content": "task"}
    history = [task]
    mgr.record_initial_prompt(task)
    for i in range(4):
        _record_exchange(mgr, history, f"turn {i}", tool="execute_python",
                         output="o" * 400)
    mgr.manage(history)
    text = mgr._placeholder["content"]
    assert "[context compressed]" in text
    assert "seq" in text and "hist.conversation_history" in text
    assert "[L0]" not in text  # no leveled map in the ablation


def test_unrecorded_nudge_evicts_without_index_fold(tmp_path):
    mgr = _mgr(tmp_path, history_max_tokens=120, pinned=1)
    task = {"role": "user", "content": "task"}
    history = [task]
    mgr.record_initial_prompt(task)
    # An unpersisted scaffold nudge (host never called record_user_message).
    history.append({"role": "user", "content": "Call a tool." * 50})
    for i in range(3):
        _record_exchange(mgr, history, f"turn {i}", tool="execute_python",
                         output="o" * 400)
    events = mgr.manage(history)  # must not raise; nudge simply drops
    assert events["evicted_msgs"] > 0


# --- observation aging ---------------------------------------------------- #

def test_age_observations_stubs_only_old_long_outputs(tmp_path):
    mgr = _mgr(tmp_path, obs_keep_turns=3)
    task = {"role": "user", "content": "task"}
    history = [task]
    mgr.record_initial_prompt(task)
    old = _record_exchange(mgr, history, "old", tool="execute_python",
                           output="x" * 2000)
    short = _record_exchange(mgr, history, "short", tool="execute_python",
                             output="tiny")
    recent = [
        _record_exchange(mgr, history, f"recent {i}", tool="execute_python",
                         output="y" * 2000)
        for i in range(3)
    ]
    mgr.manage(history)
    assert "aged out of this prompt" in old[1]["content"]
    assert len(old[1]["content"]) < 2000
    assert short[1]["content"] == "tiny"                # too short to stub
    for msgs in recent:
        assert msgs[1]["content"] == "y" * 2000         # inside the keep window
    # Idempotent: a second sweep changes nothing.
    before = old[1]["content"]
    mgr.manage(history)
    assert old[1]["content"] == before


def test_age_observations_window_slides(tmp_path):
    mgr = _mgr(tmp_path, obs_keep_turns=2)
    task = {"role": "user", "content": "task"}
    history = [task]
    mgr.record_initial_prompt(task)
    first = _record_exchange(mgr, history, "t0", tool="execute_python",
                             output="z" * 2000)
    _record_exchange(mgr, history, "t1", tool="execute_python", output="z" * 2000)
    mgr.manage(history)
    assert first[1]["content"] == "z" * 2000            # still within keep=2
    _record_exchange(mgr, history, "t2", tool="execute_python", output="z" * 2000)
    mgr.manage(history)
    assert "aged out of this prompt" in first[1]["content"]  # slid out


def test_aging_disabled_when_none(tmp_path):
    mgr = _mgr(tmp_path, obs_keep_turns=None)
    task = {"role": "user", "content": "task"}
    history = [task]
    mgr.record_initial_prompt(task)
    old = _record_exchange(mgr, history, "old", tool="execute_python",
                           output="x" * 2000)
    for i in range(4):
        _record_exchange(mgr, history, f"r{i}", tool="execute_python", output="y")
    mgr.manage(history)
    assert old[1]["content"] == "x" * 2000


def test_obs_keep_turns_env(monkeypatch):
    monkeypatch.delenv("SCROLL_OBS_KEEP_TURNS", raising=False)
    assert _obs_keep_turns_from_env() == 3
    monkeypatch.setenv("SCROLL_OBS_KEEP_TURNS", "5")
    assert _obs_keep_turns_from_env() == 5
    for off in ("0", "off", "none"):
        monkeypatch.setenv("SCROLL_OBS_KEEP_TURNS", off)
        assert _obs_keep_turns_from_env() is None


# --- digest --------------------------------------------------------------- #

def test_digest_message_includes_reflection_prompt(tmp_path):
    mgr = _mgr(tmp_path)
    msg = mgr.digest_message()
    content = msg["content"]
    assert msg["role"] == "user"
    assert "[working memory]" in content
    assert "vars: (empty)" in content                  # the runtime digest
    assert "judge in one line" in content              # the reflection nudge
    assert "change approach" in content
    # No eviction yet -> no retrieval guidance (avoid generic boilerplate).
    assert "no longer in this prompt" not in content


def test_digest_message_surfaces_evicted_history_search(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.totals["evicted_msgs"] = 7
    content = mgr.digest_message()["content"]
    assert "7 earlier turn(s) are no longer in this prompt" in content
    assert "ms.search(" in content and "ms.expand(" in content  # recall cue
    assert "execute_python" in content                 # names the host's REPL
    assert "judge in one line" in content


def test_digest_message_carries_budget_note(tmp_path):
    mgr = _mgr(tmp_path)
    content = mgr.digest_message("BUDGET-SENTINEL")["content"]
    assert content.endswith("BUDGET-SENTINEL")


# --- REPL + persistence --------------------------------------------------- #

def test_execute_python_persists_and_recalls(tmp_path):
    mgr = _mgr(tmp_path)
    task = {"role": "user", "content": "the task"}
    mgr.record_initial_prompt(task)
    out = mgr.execute_python(
        "rows = ms.sql_query('SELECT kind FROM hist.conversation_history "
        "WHERE session_id=?', (ms.session_id,))\nprint(len(rows))"
    )
    assert "1" in out                                  # the task row is durable
    assert mgr.totals["repl_calls"] == 1
    mgr.close()


def test_records_write_through_all_kinds(tmp_path):
    import sqlite3

    db = tmp_path / "history.db"
    mgr = _mgr(tmp_path)
    task = {"role": "user", "content": "the task"}
    history = [task]
    mgr.record_initial_prompt(task)
    _record_exchange(mgr, history, "thinking\n⟦ found it ⟧",
                     tool="execute_python", output="42")
    mgr.record_tool_call("submit_answer", "42", tool_input={"answer": "42"})
    nudge = {"role": "user", "content": "note"}
    mgr.record_user_message(nudge, kind="user_message")
    mgr.close()

    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT kind, headline FROM conversation_history ORDER BY seq"
    ).fetchall()
    conn.close()
    kinds = [r[0] for r in rows]
    assert kinds == ["task", "model_turn", "tool_result", "tool_call", "user_message"]
    assert ("model_turn", "found it") in [(k, h) for k, h in rows if h]


# --- prompt protocol ------------------------------------------------------ #

def test_protocol_prompt_binds_repl_name():
    text = protocol_prompt("execute_python")
    assert "`execute_python(source)`" in text
    assert "{repl}" not in text
    assert "# Managing your own context" in text
    assert "# Your in-context memory map" in text      # index.md appended


def test_protocol_prompt_index_off_strips_headline_schema():
    on = protocol_prompt("execute_python", index=True)
    off = protocol_prompt("execute_python", index=False)
    assert "headline" in on.lower()
    assert "headline" not in off.lower()
    assert "# Your in-context memory map" not in off   # index.md omitted


def test_strip_headline_schema_fragments_match_core():
    """Byte-exact coupling: every strip fragment must appear in core.md."""
    base = core_prompt("execute_python")
    for with_col, _ in HEADLINE_SCHEMA_FRAGMENTS:
        assert with_col in base
    stripped = strip_headline_schema(base)
    assert stripped != base


def test_index_prompt_survives_templating_braces():
    # index.md contains a literal {prose} FTS filter; str.replace templating
    # must leave it intact.
    assert "{prose}" in index_prompt("execute_python")


def test_manager_protocol_prompt_follows_index_flag(tmp_path):
    on = _mgr(tmp_path, enable_index=True)
    off = _mgr(
        tmp_path, enable_index=False,
        history_db_path=str(tmp_path / "h2.db"), session_id="r:t2", task_id="t2",
    )
    assert "headline" in on.protocol_prompt().lower()
    assert "headline" not in off.protocol_prompt().lower()
    on.close()
    off.close()


def test_scroll_repl_tool_schema_shape():
    fn = SCROLL_REPL_TOOL_SCHEMA["function"]
    assert fn["name"] == "scroll_repl"
    assert fn["parameters"]["required"] == ["source"]


# --- seeded prior sessions ------------------------------------------------ #

def test_seed_index_map_renders_session_spans(tmp_path):
    from scroll_context._runtime.history import HistoryStore

    db = tmp_path / "history.db"
    store = HistoryStore(db)
    for sid, (first, last) in {
        "seed:t:s1": ("intro to the project", "wrapped up phase one"),
        "seed:t:s2": ("phase two kickoff", "phase two retro"),
    }.items():
        store.append(session_id=sid, run_id="seed", task_id="t", entry=LogEntry(
            kind="conversation", role="assistant", content="a", headline=first,
        ))
        store.append(session_id=sid, run_id="seed", task_id="t", entry=LogEntry(
            kind="conversation", role="assistant", content="b", headline=last,
        ))
    store.close()

    mgr = _mgr(tmp_path, shared_run_ids=("seed",))
    text = mgr.seed_index_map()
    assert text is not None
    assert "[memory]" in text
    assert "intro to the project - wrapped up phase one" in text
    assert "phase two kickoff - phase two retro" in text
    assert "execute_python" in text                    # recovery names the REPL
    mgr.close()


def test_seed_index_map_none_without_seed_rows(tmp_path):
    mgr = _mgr(tmp_path)
    assert mgr.seed_index_map() is None
    mgr.close()


# --- metrics -------------------------------------------------------------- #

def test_metrics_shape(tmp_path):
    mgr = _mgr(tmp_path)
    task = {"role": "user", "content": "task"}
    history = [task]
    mgr.record_initial_prompt(task)
    _record_exchange(mgr, history, "⟦ hl ⟧", tool="execute_python", output="o")
    m = mgr.metrics()
    assert m["assistant_turns"] == 1
    assert m["headlined_turns"] == 1
    assert m["headline_compliance_rate"] == 1.0
    assert m["index_enabled"] is True
    mgr.close()


def test_level_cap_below_three_rejected(tmp_path):
    with pytest.raises(ValueError):
        _mgr(tmp_path, index_level_cap=2)
