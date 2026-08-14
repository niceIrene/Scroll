"""Unit tests for scroll_context.ScrollContextManager.

Drives the manager over synthetic OpenAI-format message lists — no LLM, no
harness. Mirrors the invariants of scroll_agent_A's orchestration: grouped
eviction (assistant + paired tool results atomic), single mutated placeholder,
idempotent observation aging, ephemeral digest, headline→Leaf mapping.
"""

from __future__ import annotations

import pytest

from scroll_context import (
    SCROLL_PROMPT_PROTOCOL,
    SCROLL_REPL_TOOL_NAME,
    SCROLL_REPL_TOOL_SCHEMA,
    ScrollContextManager,
    shared_session_spans,
)
from scroll_context.manager import (
    _AGED_MARKER,
    _extract_headline,
    _obs_keep_turns_from_env,
)


def make_mgr(tmp_path, budget=200, **kw):
    return ScrollContextManager(
        history_db_path=tmp_path / "hist.db",
        session_id="r1:t1",
        run_id="r1",
        task_id="t1",
        history_max_tokens=budget,
        pinned=1,
        **kw,
    )


def turn(mgr, messages, text, tool_output=None, headline=None, call_id="c1"):
    """Append one assistant turn (+ optional tool result) and record it."""
    content = text if headline is None else f"{text}\n⟦ {headline} ⟧"
    msg = {"role": "assistant", "content": content}
    if tool_output is not None:
        msg["tool_calls"] = [
            {"id": call_id, "type": "function",
             "function": {"name": "python_execute", "arguments": "{}"}}
        ]
    messages.append(msg)
    mgr.record_assistant_turn(msg, usage=None)
    if tool_output is not None:
        tmsg = {"role": "tool", "tool_call_id": call_id, "content": tool_output}
        messages.append(tmsg)
        mgr.record_tool_result(tmsg, tool_name="python_execute")


def test_headline_extraction():
    assert _extract_headline("did stuff\n⟦ found the key ⟧") == "found the key"
    assert _extract_headline("no fence here") is None
    assert _extract_headline(None) is None


def test_eviction_pairs_tool_calls_and_folds_index(tmp_path):
    mgr = make_mgr(tmp_path, budget=120)
    messages = [{"role": "user", "content": "task prompt"}]
    mgr.record_initial_prompt(messages[0])

    for i in range(6):
        turn(mgr, messages, f"turn {i} " + "x" * 200,
             tool_output=f"result {i} " + "y" * 200,
             headline=f"milestone {i}", call_id=f"c{i}")

    events = mgr.manage(messages)
    assert events.get("evicted_msgs", 0) > 0
    # Pinned prompt intact, placeholder inserted at index 1.
    assert messages[0]["content"] == "task prompt"
    assert messages[1]["role"] == "user" and "[memory]" in messages[1]["content"]
    # No orphaned tool results: every role:"tool" msg follows an assistant
    # msg whose tool_calls contains its tool_call_id.
    for i, m in enumerate(messages):
        if m.get("role") == "tool":
            prev = messages[i - 1] if messages[i - 1].get("role") == "assistant" else messages[i - 2]
            ids = {tc["id"] for tc in prev.get("tool_calls", [])}
            assert m["tool_call_id"] in ids
    # Evicted headlines appear in the rendered index placeholder.
    assert "milestone 0" in messages[1]["content"]
    # The newest group is never evicted.
    assert any("turn 5" in str(m.get("content", "")) for m in messages)


def test_placeholder_is_single_and_mutated_in_place(tmp_path):
    mgr = make_mgr(tmp_path, budget=100)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    for i in range(4):
        turn(mgr, messages, "a" * 300, headline=f"h{i}", call_id=f"c{i}")
        mgr.manage(messages)
    ph = [m for m in messages if "[memory]" in str(m.get("content", ""))]
    assert len(ph) == 1
    assert messages.index(ph[0]) == 1


def test_observation_aging_idempotent(tmp_path):
    mgr = make_mgr(tmp_path, budget=10_000, obs_keep_turns=1)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    turn(mgr, messages, "old turn", tool_output="z" * 2000, call_id="c0")
    turn(mgr, messages, "new turn 1", call_id="c1")
    turn(mgr, messages, "new turn 2", call_id="c2")

    n1, saved1 = mgr._age_observations(messages)
    assert n1 == 1 and saved1 > 0
    aged = [m for m in messages if m.get("role") == "tool"][0]
    assert _AGED_MARKER in aged["content"] and len(aged["content"]) < 600
    n2, saved2 = mgr._age_observations(messages)
    assert n2 == 0 and saved2 == 0  # idempotent


def test_digest_is_ephemeral_and_lists_vars(tmp_path):
    mgr = make_mgr(tmp_path, budget=1000)
    out = mgr.execute_python("findings = [1, 2, 3]\nprint('stored')")
    assert "stored" in out
    d = mgr.digest_message()
    assert d["role"] == "user"
    assert "[working memory]" in d["content"] and "findings" in d["content"]
    # Recall guidance only appears once something was evicted.
    assert "recoverable" not in d["content"]
    mgr.totals["evicted_msgs"] = 3
    assert "recoverable" in mgr.digest_message()["content"]


def test_repl_recalls_persisted_history(tmp_path):
    mgr = make_mgr(tmp_path, budget=1000)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    turn(mgr, messages, "the secret config value is prod-3", headline="config is prod-3")
    out = mgr.execute_python(
        "rows = ms.sql_query(\"SELECT content FROM hist.conversation_history \"\n"
        "                    \"WHERE headline IS NOT NULL\")\n"
        "print(rows[0]['content'])"
    )
    assert "prod-3" in out


def test_headline_compliance_metric(tmp_path):
    mgr = make_mgr(tmp_path, budget=10_000)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    turn(mgr, messages, "plain turn", call_id="c0")
    turn(mgr, messages, "milestone turn", headline="found it", call_id="c1")
    m = mgr.metrics()
    assert m["assistant_turns"] == 2 and m["headlined_turns"] == 1
    assert m["headline_compliance_rate"] == pytest.approx(0.5)


def test_tool_schema_and_protocol_name_consistency():
    assert SCROLL_REPL_TOOL_SCHEMA["function"]["name"] == SCROLL_REPL_TOOL_NAME == "scroll_repl"
    assert "scroll_repl" in SCROLL_PROMPT_PROTOCOL
    assert "execute_python" not in SCROLL_PROMPT_PROTOCOL


def test_oversized_tool_result_capped_but_fully_persisted(tmp_path):
    mgr = make_mgr(tmp_path, budget=10_000, tool_result_cap_chars=500)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    huge = "row-data " + "ab-testing-needle " + "x" * 5000
    turn(mgr, messages, "reading spreadsheet", tool_output=huge, call_id="c0")

    tool_msg = [m for m in messages if m.get("role") == "tool"][0]
    # In-context copy stubbed: bounded, marked, and points at the seq
    assert len(tool_msg["content"]) < len(huge)
    assert "truncated in this prompt" in tool_msg["content"]
    assert "seq" in tool_msg["content"] and "scroll_repl" in tool_msg["content"]
    assert mgr.totals["capped_results"] == 1
    # Full text durable and recoverable via ms
    out = mgr.execute_python(
        "rows = ms.sql_query(\"SELECT content FROM hist.conversation_history \"\n"
        "                    \"WHERE kind='tool_result'\")\n"
        "print(len(rows[0]['content']), 'needle' if 'ab-testing-needle' in rows[0]['content'] else 'missing')"
    )
    assert "needle" in out and str(len(huge)) in out


def test_auto_cap_bounds_result_below_budget(tmp_path):
    mgr = make_mgr(tmp_path, budget=3000)  # auto cap -> stdout_cap_for(3000) = 2000 chars
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    turn(mgr, messages, "big read", tool_output="z" * 400_000, call_id="c0")
    tool_msg = [m for m in messages if m.get("role") == "tool"][0]
    # A ~100k-token result can no longer exceed the whole budget in-context
    assert len(tool_msg["content"]) < 3000


def test_manage_recomputes_est_after_external_mutation(tmp_path):
    mgr = make_mgr(tmp_path, budget=1000, tool_result_cap_chars=None)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    for i in range(3):
        turn(mgr, messages, "t" * 4000, call_id=f"c{i}")
    assert mgr.est_input > 1000
    # Simulate an API-layer hard trim mutating the list behind scroll's back
    del messages[1:-1]
    mgr.manage(messages)
    # Estimate re-anchored to the actual (now small) list, not the stale sum
    assert mgr.est_input < 3500


def test_eviction_noop_under_budget_and_never_evicts_newest_group(tmp_path):
    """Ported from the deleted test_scroll_window: the two boundary invariants."""
    mgr = make_mgr(tmp_path, budget=10_000)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    turn(mgr, messages, "recent", call_id="c0")
    assert mgr.manage(messages).get("evicted_msgs", 0) == 0  # under budget: noop
    assert len(messages) == 2

    # One huge newest group that alone exceeds the budget must not be evicted.
    mgr2 = make_mgr(tmp_path / "b", budget=10)
    messages2 = [{"role": "user", "content": "task"}]
    mgr2.record_initial_prompt(messages2[0])
    turn(mgr2, messages2, "x" * 40_000, call_id="c0")
    mgr2.manage(messages2)
    assert any("x" * 100 in str(m.get("content", "")) for m in messages2)


def test_aging_window_slides(tmp_path):
    """Ported from test_scroll_window: each new turn ages the next-oldest output."""
    mgr = make_mgr(tmp_path, budget=100_000, obs_keep_turns=3)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    for i in range(4):
        turn(mgr, messages, f"turn {i}", tool_output="B" * 2000, call_id=f"c{i}")
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    mgr._age_observations(messages)
    assert _AGED_MARKER in tool_msgs[0]["content"]
    assert _AGED_MARKER not in tool_msgs[1]["content"]
    # A new turn arrives; the second-oldest now falls out of the keep window.
    turn(mgr, messages, "turn 4", tool_output="B" * 2000, call_id="c4")
    mgr._age_observations(messages)
    assert _AGED_MARKER in tool_msgs[1]["content"]
    assert _AGED_MARKER not in tool_msgs[2]["content"]


def test_aging_keeps_all_when_fewer_steps_than_keep(tmp_path):
    """Regression: with FEWER assistant steps than keep, NOTHING ages — a large
    keep must not fall into an 'age everything' branch (the bug that inverted
    --obs-keep, aging more the higher it was set)."""
    mgr = make_mgr(tmp_path, budget=100_000, obs_keep_turns=20)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    for i in range(5):  # 5 assistant steps << keep=20
        turn(mgr, messages, f"turn {i}", tool_output="B" * 2000, call_id=f"c{i}")
    n_aged, _ = mgr._age_observations(messages)
    assert n_aged == 0, "no output should age when steps < keep"
    assert all(_AGED_MARKER not in m["content"]
               for m in messages if m.get("role") == "tool")
    # And once past the window, the oldest DOES age (sanity: the knob still works).
    mgr2 = make_mgr(tmp_path / "b", budget=100_000, obs_keep_turns=2)
    m2 = [{"role": "user", "content": "task"}]
    mgr2.record_initial_prompt(m2[0])
    for i in range(4):
        turn(mgr2, m2, f"t{i}", tool_output="B" * 2000, call_id=f"d{i}")
    mgr2._age_observations(m2)
    tool2 = [m for m in m2 if m.get("role") == "tool"]
    assert _AGED_MARKER in tool2[0]["content"] and _AGED_MARKER not in tool2[-1]["content"]


def test_aging_skips_short_outputs_and_non_tool_messages(tmp_path):
    mgr = make_mgr(tmp_path, budget=100_000, obs_keep_turns=1)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    turn(mgr, messages, "old " + "T" * 5000, tool_output="short", call_id="c0")
    turn(mgr, messages, "new 1", call_id="c1")
    n, saved = mgr._age_observations(messages)
    assert n == 0 and saved == 0
    # Assistant text (however long) is untouched; the short tool output too.
    assert "T" * 100 in messages[1]["content"]
    assert [m for m in messages if m.get("role") == "tool"][0]["content"] == "short"


def test_obs_keep_turns_env(monkeypatch):
    monkeypatch.delenv("SCROLL_OBS_KEEP_TURNS", raising=False)
    assert _obs_keep_turns_from_env() == 3
    monkeypatch.setenv("SCROLL_OBS_KEEP_TURNS", "5")
    assert _obs_keep_turns_from_env() == 5
    for off in ("0", "off", "none"):
        monkeypatch.setenv("SCROLL_OBS_KEEP_TURNS", off)
        assert _obs_keep_turns_from_env() is None


# --- host-parity surface (used by scroll_agent_A) ------------------------------


def test_repl_name_substituted_in_model_facing_texts(tmp_path):
    mgr = make_mgr(
        tmp_path, budget=10_000, repl_name="execute_python", tool_result_cap_chars=500
    )
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    turn(mgr, messages, "big read", tool_output="z" * 5000, call_id="c0")
    tool_msg = [m for m in messages if m.get("role") == "tool"][0]
    assert "execute_python" in tool_msg["content"]        # cap stub
    assert "scroll_repl" not in tool_msg["content"]
    mgr.totals["evicted_msgs"] = 2
    d = mgr.digest_message()["content"]
    assert "execute_python" in d and "scroll_repl" not in d  # digest recall cue


def test_record_parity_step_and_msg_index_flow_to_log_rows(tmp_path):
    mgr = make_mgr(tmp_path, budget=10_000)
    messages = [{"role": "user", "content": "probe question"}]
    mgr.record_initial_prompt(messages[0], step_index=-1, msg_index=1)
    asst = {"role": "assistant", "content": "thinking about it"}
    messages.append(asst)
    mgr.record_assistant_turn(asst, step_index=0, msg_index=2, reasoning="chain")
    mgr.record_tool_call("submit_answer", "final", tool_input={"answer": "final"},
                         tool_call_id="c9", msg_index=2)
    out = mgr.execute_python(
        "rows = ms.sql_query(\"SELECT kind, step_index, msg_index, tool_call_id \"\n"
        "                    \"FROM hist.conversation_history ORDER BY seq\")\n"
        "print([(r['kind'], r['step_index'], r['msg_index']) for r in rows])\n"
        "meta = ms.sql_query(\"SELECT metadata FROM hist.conversation_history \"\n"
        "                    \"WHERE kind='model_turn'\")\n"
        "print(meta[0]['metadata'])"
    )
    assert "('task', -1, 1)" in out
    assert "('model_turn', 0, 2)" in out
    assert "('tool_call', 0, 2)" in out
    assert "reasoning" in out and "chain" in out  # persisted, never re-prompted


def test_executor_bounded_output_not_double_capped(tmp_path):
    """REPL stdout the executor already trimmed must pass through the cap."""
    from scroll_context._runtime.exec import OVERFLOW_MARKER

    mgr = make_mgr(tmp_path, budget=10_000, tool_result_cap_chars=500)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    bounded = "head of output\n" + OVERFLOW_MARKER + " 99999 chars printed...]" + "x" * 600
    turn(mgr, messages, "ran code", tool_output=bounded, call_id="c0")
    tool_msg = [m for m in messages if m.get("role") == "tool"][0]
    assert tool_msg["content"] == bounded  # untouched: no second stub
    assert mgr.totals["capped_results"] == 0


def test_execute_python_async_inside_event_loop(tmp_path):
    import asyncio

    mgr = make_mgr(tmp_path, budget=1000)

    async def _go():
        return await mgr.execute_python_async("v = 41\nprint(v + 1)")

    assert "42" in asyncio.run(_go())
    assert mgr.totals["repl_calls"] == 1


# --- prior-session priming -----------------------------------------------------


def test_prime_prior_sessions_from_explicit_spans(tmp_path):
    mgr = make_mgr(tmp_path, budget=10_000)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    spans = [
        {"seq_lo": 1, "seq_hi": 40, "head": "kickoff planning", "tail": "sprint 1 done",
         "session": 1},
        {"seq_lo": 41, "seq_hi": 90, "head": "api design", "tail": "v2 shipped",
         "session": 2},
    ]
    assert mgr.prime_prior_sessions(messages, spans) is True
    # Placeholder pinned at `pinned` (=1 here) from turn one, carrying the map.
    ph = messages[1]
    assert ph["role"] == "user" and "[memory]" in ph["content"]
    assert "kickoff planning" in ph["content"] and "v2 shipped" in ph["content"]
    # Empty source: a fresh manager primes nothing and inserts no placeholder.
    mgr2 = make_mgr(tmp_path / "b", budget=10_000)
    messages2 = [{"role": "user", "content": "task"}]
    assert mgr2.prime_prior_sessions(messages2, []) is False
    assert len(messages2) == 1


def test_prime_prior_sessions_from_shared_history_db(tmp_path):
    """Default source: sessions under shared run_ids in the attached history DB."""
    db = tmp_path / "hist.db"
    # Seed two prior sessions under a shared run_id, with headlines.
    seeder = ScrollContextManager(
        history_db_path=db, session_id="prior:t1:s1", run_id="prior", task_id="t1",
        history_max_tokens=0,
    )
    m1 = {"role": "assistant", "content": "did the thing\n⟦ configured the pipeline ⟧"}
    seeder.record_assistant_turn(m1, step_index=1)
    seeder.close()
    seeder2 = ScrollContextManager(
        history_db_path=db, session_id="prior:t1:s2", run_id="prior", task_id="t1",
        history_max_tokens=0,
    )
    m2 = {"role": "assistant", "content": "wrapped up\n⟦ deployed v2 to staging ⟧"}
    seeder2.record_assistant_turn(m2, step_index=2)
    seeder2.close()

    mgr = ScrollContextManager(
        history_db_path=db, session_id="r9:t1", run_id="r9", task_id="t1",
        history_max_tokens=0, pinned=1, shared_run_ids=("prior",),
    )
    spans = shared_session_spans(
        mgr.runtime.memoryspace, run_ids=("prior",), task_id="t1"
    )
    assert [s["head"] for s in spans] == [
        "configured the pipeline", "deployed v2 to staging"
    ]
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    assert mgr.prime_prior_sessions(messages) is True  # default source = shared tier
    assert "configured the pipeline" in messages[1]["content"]
    assert "deployed v2 to staging" in messages[1]["content"]
    mgr.close()


def test_prime_prior_sessions_noop_when_index_disabled(tmp_path):
    mgr = make_mgr(tmp_path, budget=10_000, enable_index=False)
    messages = [{"role": "user", "content": "task"}]
    spans = [{"seq_lo": 1, "seq_hi": 5, "head": "h", "tail": "t", "session": 1}]
    assert mgr.prime_prior_sessions(messages, spans) is False
    assert len(messages) == 1


# --- var-context mode (SCROLL_VAR_CONTEXT ablation) ---------------------------


def var_turn(mgr, messages, text, source=None, tool_output=None, call_id="c1",
             tool_name="scroll_repl"):
    """One assistant turn; if ``source`` is given, run it through the manager's
    REPL and record the (possibly overridden) observation as its tool result."""
    import json as _json

    msg = {"role": "assistant", "content": text}
    if source is not None or tool_output is not None:
        args = _json.dumps({"source": source} if source is not None else {})
        msg["tool_calls"] = [
            {"id": call_id, "type": "function",
             "function": {"name": tool_name, "arguments": args}}
        ]
    messages.append(msg)
    mgr.record_assistant_turn(msg)
    if source is not None or tool_output is not None:
        obs = mgr.execute_python(source) if source is not None else tool_output
        tmsg = {"role": "tool", "tool_call_id": call_id, "content": tool_output or obs}
        messages.append(tmsg)
        mgr.record_tool_result(tmsg, tool_name=tool_name)


def test_var_context_changelog_virtualization_and_freeze(tmp_path):
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])

    var_turn(mgr, messages, "storing findings", call_id="c0",
             source="hits = [{'seq': i, 'date': '2024-07-0%d' % (i+1)} for i in range(9)]\nprint('ok')")
    var_turn(mgr, messages, "no retention here", call_id="c1",
             source="print('just looking at a big dump ' * 40)")
    var_turn(mgr, messages, "newest turn", call_id="c2", source="print('x')")
    mgr.manage(messages)

    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    # Old tool result with variable changes -> events-only changelog: names, no metadata.
    assert "⟦output virtualized to variables" in tool_msgs[0]["content"]
    assert "vars: + hits" in tool_msgs[0]["content"]
    assert "list[dict" not in tool_msgs[0]["content"]   # metadata lives in the digest…
    assert "seq" in tool_msgs[0]["content"]  # durable pointer
    d = mgr.digest_message()["content"]
    assert "hits (list[dict{seq,date}], 9 items)" in d  # …which stays authoritative
    # Old tool result with NO changes -> explicit retention nudge.
    assert "left NO variables" in tool_msgs[1]["content"]
    # Newest group stays verbatim.
    assert tool_msgs[2]["content"].startswith("stdout:")
    # Frozen: further manage() passes never touch the rewritten copies.
    frozen = [m["content"] for m in tool_msgs[:2]]
    var_turn(mgr, messages, "another turn", call_id="c3", source="print('y')")
    mgr.manage(messages)
    assert [m["content"] for m in tool_msgs[:2]] == frozen
    assert mgr.totals["virtualized_results"] >= 2
    assert mgr.totals["var_changes"] >= 1


def test_var_context_thought_distillation_with_fallback(tmp_path):
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    long = " This first line should survive distillation as the fallback." + " filler" * 50
    var_turn(mgr, messages, "found it\n⟦ launch date = 2025-02-15 ⟧", call_id="c0",
             source="print(1)")
    var_turn(mgr, messages, long, call_id="c1", source="print(2)")
    var_turn(mgr, messages, "recent A", call_id="c2", source="print(3)")
    var_turn(mgr, messages, "recent B", call_id="c3", source="print(4)")
    mgr.manage(messages)

    asst = [m for m in messages if m.get("role") == "assistant"]
    assert asst[0]["content"] == "⟦ launch date = 2025-02-15 ⟧"      # headline wins
    assert asst[1]["content"].startswith("⟦ (no headline) This first line")
    assert len(asst[1]["content"]) < len(long)
    assert asst[1]["tool_calls"], "tool_calls must survive distillation"
    assert asst[2]["content"] == "recent A" and asst[3]["content"] == "recent B"
    assert mgr.totals["distilled_thoughts"] == 2


def test_var_context_digest_auto_pin_and_note(tmp_path):
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True)
    mgr.execute_python("small = [1, 2, 3]\nbig = [{'seq': i} for i in range(50)]")
    d = mgr.digest_message()["content"]
    assert "small (list[int], 3 items)" in d
    assert "1" in d.split("small")[1][:120]          # small collection shows rows
    assert "big (list[dict{seq}], 50 items)" in d
    assert "{'seq': 0}" not in d                     # large collection: meta only
    out = mgr.execute_python("pin('big', 'head', note='candidate rows')")
    assert "pinned big" in out
    d2 = mgr.digest_message()["content"]
    assert "[pinned head]" in d2 and "note: candidate rows" in d2
    assert "{'seq': 0}" in d2                        # head rows now shown
    # derived op facts from ms flow into the meta line (no query gist)
    mgr.execute_python("rows = ms.sql_query('SELECT seq FROM hist.conversation_history')")
    d3 = mgr.digest_message()["content"]
    assert "← sql_query → " in d3 and "SELECT" not in d3.split("rows (")[1][:120]


def test_var_context_external_result_autobound(tmp_path):
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    var_turn(mgr, messages, "running bash", call_id="c0", tool_name="bash",
             tool_output="file1\nfile2\n" + "x" * 500)
    # External tool result auto-bound to a typed variable...
    ns = mgr.runtime.namespace
    obs_vars = [k for k in ns if k.startswith("obs_")]
    assert len(obs_vars) == 1 and ns[obs_vars[0]].startswith("file1")
    # ...and its changelog entry lands in the virtualized view.
    var_turn(mgr, messages, "next", call_id="c1", tool_output="ok")
    mgr.manage(messages)
    first_tool = [m for m in messages if m.get("role") == "tool"][0]
    assert f"vars: + {obs_vars[0]} (bash)" in first_tool["content"]


def test_var_context_coverage_and_overlap_flags(tmp_path):
    """Coverage lives in the digest only; ⚠ fires ONLY for a cross-step
    retrieval that re-fetched held rows — never for in-Python derivation."""
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    # Step 0: a real retrieval, stored.
    var_turn(mgr, messages, "broad sweep", call_id="c0", source=(
        "budget_rows = ms.sql_query('SELECT seq, step_index, content "
        "FROM hist.conversation_history ORDER BY seq LIMIT 2')\nprint(len(budget_rows))"
    ))
    # Step 1: derivation of a subset — NO retrieval ops → must not warn.
    var_turn(mgr, messages, "curate", call_id="c1", source=(
        "spa_rows = [dict(r) for r in budget_rows[:2]]\nprint('ok')"
    ))
    # Step 2: a RE-retrieval of the same rows → the genuine target, warns.
    var_turn(mgr, messages, "re-query", call_id="c2", source=(
        "again = ms.sql_query('SELECT seq, step_index, content "
        "FROM hist.conversation_history ORDER BY seq LIMIT 2')\nprint(len(again))"
    ))
    var_turn(mgr, messages, "newest", call_id="c3", source="print('x')")
    mgr.manage(messages)
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    # Changelog = events only: no coverage/schema text in stubs.
    assert "covers" not in tool_msgs[0]["content"]
    assert "vars: + spa_rows" in tool_msgs[1]["content"]
    assert "⚠" not in tool_msgs[1]["content"]          # derivation never warns
    assert "already in `budget_rows`" in tool_msgs[2]["content"]  # re-fetch warns
    assert mgr.totals["overlap_warnings"] == 1
    # Digest: metadata authority + the stashed verdict; superset unflagged.
    d = mgr.digest_message()["content"]
    assert "covers" in d
    again_line = next(ln for ln in d.splitlines() if ln.strip().startswith("- again"))
    assert "already in `budget_rows`" in again_line
    budget_line = next(ln for ln in d.splitlines() if ln.strip().startswith("- budget_rows"))
    assert "⚠" not in budget_line
    spa_line = next(ln for ln in d.splitlines() if ln.strip().startswith("- spa_rows"))
    assert "⚠" not in spa_line


def test_var_context_expand_after_search_is_upgrade_not_warning(tmp_path):
    """The snippet→full-text path: expanding your search hits is the INTENDED
    idiom — it renders as ↳ lineage, never as a ⚠ redundancy warning."""
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    var_turn(mgr, messages, "seeding distinctive content about zephyr budgets",
             call_id="c0", source="print('noted')")
    var_turn(mgr, messages, "searching", call_id="c1",
             source="hits = ms.search('zephyr', scope='session', k=5, include_self=True)\nprint(len(hits))")
    var_turn(mgr, messages, "reading in full", call_id="c2",
             source="full_rows = ms.expand([h['seq'] for h in hits])\n"
                    "anchors = [{'date': '2025-01-01', 'seq': full_rows[0]['seq']}]\n"
                    "print(len(full_rows))")
    var_turn(mgr, messages, "newest", call_id="c3", source="print('x')")
    mgr.manage(messages)
    third_tool = [m for m in messages if m.get("role") == "tool"][2]["content"]
    assert "↳ full_rows: full-text upgrade of `hits`" in third_tool
    # Neither the upgrade nor the hand-built fact record (anchors — no history
    # schema) may warn, even though both were created in a retrieval step.
    assert "⚠" not in third_tool
    assert mgr.totals["overlap_warnings"] == 0
    d = mgr.digest_message()["content"]
    full_line = next(ln for ln in d.splitlines() if ln.strip().startswith("- full_rows"))
    assert "full-text upgrade of `hits`" in full_line and "⚠" not in full_line


def test_var_context_digest_polish(tmp_path):
    """Regression pack for the first-run feedback: creation-order digest,
    data-keyed dict schemas, msg_index overlap fallback, and useful intents."""
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    # Turn whose first line is the mandated self-check: intent must skip it.
    var_turn(mgr, messages, "Last result: printed too much, truncated.\n"
             "Let me cast a wider net using SQL to find staff interactions.",
             call_id="c0", source=(
                 "zz_rows = [{'msg_index': 100+i, 'step_index': 61, 'date': '2024-10-13'} for i in range(20)]\n"
                 "seq_map = {13660 + i: 'x' for i in range(9)}\nprint('ok')"
             ))
    var_turn(mgr, messages, "narrowing", call_id="c1", source=(
        "aa_late = [dict(r) for r in zz_rows[:15]]\nprint('ok')"
    ))
    var_turn(mgr, messages, "newest", call_id="c2", source="print('x')")
    mgr.manage(messages)
    d = mgr.digest_message()["content"]
    # 1. The digest carries NO intent copies (the distilled stream line is the
    # single home); distillation skipped the self-check verdict and kept the
    # full forward-looking sentence.
    assert "while:" not in d and "Last result" not in d
    asst0 = [m for m in messages if m.get("role") == "assistant"][0]
    assert asst0["content"] == (
        "⟦ (no headline) Let me cast a wider net using SQL to find staff interactions. ⟧"
    )
    # 2. Creation order, not alphabetical: zz_rows (turn 0) before aa_late (turn 1).
    assert d.index("zz_rows") < d.index("aa_late")
    # 3. Data-keyed dict renders key→value types, not a spray of seqs.
    assert "seq_map (dict[int→str], 9 keys)" in d
    assert "13660" not in d
    # 4. aa_late is an in-Python derivation (its step ran no retrieval ops) —
    # under the corrected trigger it must NOT be flagged as overlap.
    assert "⚠" not in d


def test_var_context_scratch_collapsing(tmp_path):
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    var_turn(mgr, messages, "loop pass", call_id="c0", source=(
        "from collections import Counter\n"
        "findings = [{'seq': 5}]\n"
        "for h in findings: c = h['seq']\nprint('ok')"
    ))
    var_turn(mgr, messages, "next", call_id="c1", source="print('y')")
    var_turn(mgr, messages, "newest", call_id="c2", source="print('z')")
    mgr.manage(messages)
    first_tool = [m for m in messages if m.get("role") == "tool"][0]["content"]
    # Real variable first-class; temporaries and the imported class collapsed.
    assert "vars: + findings" in first_tool
    assert "· scratch: " in first_tool
    assert "Counter" in first_tool.split("· scratch:")[1]
    assert "+ h" not in first_tool and "+ c" not in first_tool.replace("+ findings", "")
    d = mgr.digest_message()["content"]
    assert "(scratch: Counter, c, h)" in d
    assert "\n  - h (" not in d


def test_var_context_auto_intent_and_note_override(tmp_path):
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    var_turn(mgr, messages, "found it\n⟦ narrowing to the finalized budget ⟧",
             call_id="c0", source="anchors = [{'seq': 9}]\nprint('ok')")
    # The digest never carries intent — the distilled stream line is its home.
    d0 = mgr.digest_message()["content"]
    assert "while:" not in d0
    var_turn(mgr, messages, "annotating", call_id="c1",
             source="note('anchors', 'both endpoints verified')")
    var_turn(mgr, messages, "newest", call_id="c2", source="print('x')")
    mgr.manage(messages)
    # The changelog does NOT repeat the intent — the distilled thought line
    # sits directly above the virtualized stub and already says it.
    first_tool = [m for m in messages if m.get("role") == "tool"][0]["content"]
    assert "while:" not in first_tool
    # The model-authored note (true state) renders in the digest; the turn's
    # headline survives as the distilled stream line.
    d = mgr.digest_message()["content"]
    assert "note: both endpoints verified" in d and "while:" not in d
    asst0 = [m for m in messages if m.get("role") == "assistant"][0]
    assert asst0["content"] == "⟦ narrowing to the finalized budget ⟧"


def test_var_context_origin_seq_pointer(tmp_path):
    """Every created variable's digest line points at its producing turn's seq,
    whose durable row holds the exact source (tool_input) verbatim."""
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    src = "anchors = [{'event': 'launch', 'seq': 71}]\nprint('ok')"
    msg = {"role": "assistant", "content": "storing",
           "tool_calls": [{"id": "c0", "type": "function",
                           "function": {"name": "scroll_repl", "arguments": "{}"}}]}
    messages.append(msg)
    mgr.record_assistant_turn(msg)
    obs = mgr.execute_python(src)
    tmsg = {"role": "tool", "tool_call_id": "c0", "content": obs}
    messages.append(tmsg)
    mgr.record_tool_result(tmsg, tool_name="scroll_repl", tool_input={"source": src})

    d = mgr.digest_message()["content"]
    line = next(ln for ln in d.splitlines() if "anchors" in ln)
    assert "← seq " in line  # plain-Python var still gets an origin pointer
    seq = int(line.split("← seq ")[1].split()[0].rstrip(","))
    # The pointed-at durable row recovers the exact producing source.
    out = mgr.execute_python(
        f"r = ms.sql_query('SELECT tool_input FROM hist.conversation_history "
        f"WHERE seq={seq}')\nprint(r[0]['tool_input'])"
    )
    assert "anchors = [{'event': 'launch'" in out
    # ms-produced vars carry derived op facts + the same pointer.
    mgr.execute_python("hits = ms.search('launch', k=5)")
    var_turn(mgr, messages, "next", call_id="c1", source="print('x')")
    d2 = mgr.digest_message()["content"]
    hits_line = next(ln for ln in d2.splitlines() if ln.strip().startswith("- hits"))
    assert "← search → " in hits_line and "snippets" in hits_line and ", seq " in hits_line


def test_var_context_ops_line_survives_accumulation_idiom(tmp_path):
    """The qwen pattern: bare tool call (no thought), `hits += ms.search(...)`
    loop (which strips value-carried provenance), intent in a code comment.
    The ops header + comment-intent must still land in the changelog."""
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    var_turn(mgr, messages, "", call_id="c0", source=(
        "# Search for the user's stated motivation across phrasings\n"
        "hits2 = []\n"
        "for q in ['alpha', 'beta', 'gamma']:\n"
        "    hits2 += ms.search(q, k=10)\n"
        "print(len(hits2))"
    ))
    var_turn(mgr, messages, "", call_id="c1", source="print('next')")
    var_turn(mgr, messages, "", call_id="c2", source="print('newest')")
    mgr.manage(messages)
    first_tool = [m for m in messages if m.get("role") == "tool"][0]["content"]
    # Aggregated ops header (3 identical-shape searches merged), no query text.
    assert "ops: 3× search → 0 hits total (snippets)" in first_tool
    assert "alpha" not in first_tool
    # Accumulated plain list carries no value tag, but the event is present.
    assert "vars: + hits2" in first_tool
    # The empty thought distilled to the code comment (adjacency restored)…
    asst = [m for m in messages if m.get("role") == "assistant"][0]
    assert asst["content"] == "⟦ (no headline) Search for the user's stated motivation across phrasings ⟧"
    # …and the digest does NOT duplicate it (single-home rule).
    d = mgr.digest_message()["content"]
    assert "while:" not in d


def test_var_context_ops_header_always_and_nudge_on_ops_only(tmp_path):
    """Events-only changelog: the ops header renders whenever the step
    retrieved (entries no longer carry ← tags, so it is never redundant);
    an ops-only step keeps the retention nudge alongside the ops record."""
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    var_turn(mgr, messages, "storing directly", call_id="c0",
             source="rows = ms.sql_query('SELECT seq FROM hist.conversation_history')")
    var_turn(mgr, messages, "just looking, storing nothing " * 20, call_id="c1",
             source="_ = ms.search('needle', k=5)\ndel _\nprint('looked ' * 60)")
    var_turn(mgr, messages, "newest", call_id="c2", source="print('x')")
    mgr.manage(messages)
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    # Single retrieval assignment: ops header present, metadata absent.
    assert "ops: sql_query → " in tool_msgs[0]["content"]
    assert "vars: + rows" in tool_msgs[0]["content"]
    assert "← sql_query" not in tool_msgs[0]["content"]
    # Ops-only turn: ops recorded AND the no-variables nudge retained.
    assert "ops: search → 0 hits (snippets)" in tool_msgs[1]["content"]
    assert "no variables stored" in tool_msgs[1]["content"]


def test_row_cap_note_appended_to_observation_same_turn(tmp_path):
    """The row-cap hedge: no marker row in the DATA, but the OBSERVATION the
    model reads gets a same-turn [note] — in both context modes."""
    for var_mode in (True, False):
        mgr = make_mgr(tmp_path / ("v" if var_mode else "b"), budget=100_000,
                       var_context=var_mode)
        # Seed rows so a query can exceed a tiny row cap.
        for i in range(5):
            m = {"role": "assistant", "content": f"turn {i}"}
            mgr.record_assistant_turn(m)
        mgr.runtime.memoryspace._row_cap = 3
        out = mgr.execute_python(
            "rows = ms.sql_query('SELECT seq, content FROM hist.conversation_history')\n"
            "print(len(rows), sorted(r['seq'] for r in rows)[-1])"  # type-safe now
        )
        assert "3-row cap" in out and "NOT returned" in out       # the hedge note
        assert "print less" not in out
        out2 = mgr.execute_python("print(rows.truncated, rows.row_cap)")
        assert "True 3" in out2
        assert "row cap" not in out2                               # note only on capped turns
        mgr.close()


def test_var_fallback_chars_configurable(tmp_path, monkeypatch):
    """Distillation fallback budget: constructor param > env > default (240)."""
    long_thought = ("Let me sweep every October session for staff interactions, "
                    "then compare the tip amounts against the app records "
                    "before ranking the ten distinct events chronologically.")

    def distilled(mgr):
        messages = [{"role": "user", "content": "task"}]
        mgr.record_initial_prompt(messages[0])
        var_turn(mgr, messages, long_thought, call_id="c0", source="print(1)")
        var_turn(mgr, messages, "b", call_id="c1", source="print(2)")
        var_turn(mgr, messages, "c", call_id="c2", source="print(3)")
        mgr.manage(messages)
        return [m for m in messages if m.get("role") == "assistant"][0]["content"]

    # Default (240): the whole ~190-char thought survives.
    assert long_thought in distilled(make_mgr(tmp_path / "a", budget=10**5, var_context=True))
    # Param: tight budget clips at a sentence/word boundary.
    tight = distilled(make_mgr(tmp_path / "b", budget=10**5, var_context=True,
                               var_fallback_chars=60))
    assert long_thought not in tight and len(tight) < 120
    # Env is honored when the param is omitted.
    monkeypatch.setenv("SCROLL_VAR_FALLBACK_CHARS", "80")
    env_d = distilled(make_mgr(tmp_path / "c", budget=10**5, var_context=True))
    assert long_thought not in env_d and len(env_d) < 140


def run_turn(mgr, messages, ask=None, n_steps=2, answer="the answer is 42", label="t"):
    """Simulate one user turn: optional new user msg, N tool steps, final answer."""
    if ask is not None:
        umsg = {"role": "user", "content": ask}
        mgr.record_user_message(umsg, messages=messages)
        messages.append(umsg)
    for k in range(n_steps):
        var_turn(mgr, messages, f"{label} step {k} thinking", call_id=f"{label}{k}",
                 source=f"print('{label}{k}')")
    amsg = {"role": "assistant", "content": answer}
    messages.append(amsg)
    mgr.record_assistant_turn(amsg)
    mgr.manage(messages)


def test_close_turn_folds_and_retains_qa(tmp_path):
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True,
                   keep_turns_verbatim=2, repl_name="scroll_repl")
    messages = [{"role": "user", "content": "What is the launch date?"}]
    mgr.record_initial_prompt(messages[0])
    run_turn(mgr, messages, n_steps=2, answer="Launch is 2025-02-15.", label="a")
    # Turn 2 begins: turn 1 folds.
    run_turn(mgr, messages, ask="And the deploy date?", n_steps=2,
             answer="Deploy prep starts 2025-03-01.", label="b")

    # Map: turn 1's record line, ask → answer, tagged T1.
    ph = next(m for m in messages if "[memory]" in str(m.get("content", "")))
    assert "T1" in ph["content"]
    assert "What is the launch date?" in ph["content"]
    assert "Launch is 2025-02-15." in ph["content"]
    # In-context: turn 1's step husks are GONE; its ask + answer stay verbatim.
    flat = " ".join(str(m.get("content", "")) for m in messages)
    assert "a step 0" not in flat and "a step 1" not in flat
    assert any(m.get("content") == "Launch is 2025-02-15." for m in messages)
    assert any(m.get("content") == "What is the launch date?" for m in messages)
    # DB: durable turn_record with span metadata + searchable headline.
    out = mgr.execute_python(
        "r = ms.sql_query(\"SELECT content, headline, metadata FROM "
        "hist.conversation_history WHERE kind='turn_record'\")\n"
        "print(len(r)); print(r[0]['content']); print(r[0]['metadata'])"
    )
    assert "Q: What is the launch date?" in out and "A: Launch is 2025-02-15." in out
    assert '"turn": 1' in out and '"seq_lo"' in out
    assert mgr.totals["turn_records"] == 1


def test_close_turn_retention_ages_out_after_M(tmp_path):
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True,
                   keep_turns_verbatim=1, repl_name="scroll_repl")
    messages = [{"role": "user", "content": "q1?"}]
    mgr.record_initial_prompt(messages[0])
    run_turn(mgr, messages, n_steps=1, answer="answer one", label="a")
    run_turn(mgr, messages, ask="q2?", n_steps=1, answer="answer two", label="b")
    # Closing turn 2 (M=1): turn 1's verbatim q+a must age out of context…
    run_turn(mgr, messages, ask="q3?", n_steps=1, answer="answer three", label="c")
    assert not any(m.get("content") == "answer one" for m in messages)
    # …turn 2's q+a still verbatim, and BOTH turns keep their map lines.
    assert any(m.get("content") == "answer two" for m in messages)
    ph = next(m for m in messages if "[memory]" in str(m.get("content", "")))
    assert "T1" in ph["content"] and "T2" in ph["content"]
    # Retained answer is exempt from distillation while retained.
    a2 = next(m for m in messages if m.get("content") == "answer two")
    assert "⟦" not in a2["content"]
    # Env knob wiring.
    import os
    os.environ["SCROLL_KEEP_TURNS_VERBATIM"] = "5"
    try:
        mgr2 = make_mgr(tmp_path / "e", budget=1000, var_context=True)
        assert mgr2._keep_turns_verbatim == 5
    finally:
        del os.environ["SCROLL_KEEP_TURNS_VERBATIM"]


def test_search_excludes_own_scaffolding_but_not_records_or_pointers(tmp_path):
    """The scoped self-exclusion: relevance search skips the live session's
    thoughts/tool outputs by default, while user turns + turn records stay
    searchable and every seq pointer still resolves."""
    mgr = make_mgr(tmp_path, budget=100_000, var_context=True)
    messages = [{"role": "user", "content": "Where did we discuss the zugzwang budget?"}]
    mgr.record_initial_prompt(messages[0])
    # A step whose thought and output both contain the distinctive term.
    var_turn(mgr, messages, "thinking hard about the zugzwang budget now",
             call_id="c0", source="print('zugzwang budget notes in output')")
    # A new user turn (searchable) + turn record (searchable) mentioning it.
    run_turn(mgr, messages, ask="More about the zugzwang budget please?",
             n_steps=1, answer="The zugzwang budget was 700.", label="b")

    out = mgr.execute_python(
        "hits = ms.search('zugzwang', scope='session', k=20)\n"
        "print(sorted(set(h['kind'] for h in hits)))\n"
        "own_seq = ms.sql_query(\"SELECT seq FROM hist.conversation_history \"\n"
        "                       \"WHERE kind='model_turn' LIMIT 1\")[0]['seq']\n"
        "full = ms.expand([own_seq])\n"
        "print('EXPAND_OK' if full and 'zugzwang' in full[0]['content'] else 'EXPAND_BROKEN')\n"
        "all_hits = ms.search('zugzwang', scope='session', k=20, include_self=True)\n"
        "print('SELF', sorted(set(h['kind'] for h in all_hits)))"
    )
    # Default search: no scaffolding kinds; user/turn_record present.
    line1 = out.splitlines()[1] if out.startswith("stdout") else out.splitlines()[0]
    assert "model_turn" not in line1 and "tool_result" not in line1 and "'task'" not in line1
    assert "turn_record" in line1 or "user_message" in line1
    # Addressed lookup untouched.
    assert "EXPAND_OK" in out
    # Opt-in restores self-search.
    assert "model_turn" in out.split("SELF", 1)[1]
    # Explicit scaffold-kind query is unambiguous self-intent: not blanked.
    out2 = mgr.execute_python(
        "mt = ms.search('zugzwang', scope='session', kind='model_turn', k=10)\n"
        "print('KIND_OK' if mt else 'KIND_BLANKED')"
    )
    assert "KIND_OK" in out2
    mgr.close()


def test_close_session_record_and_own_history_priming(tmp_path):
    """Session 1 records itself; session 2 of the same task primes a durable
    P1 map line from that record — the own-history analogue of the seed map."""
    from scroll_context import ScrollContextManager
    from scroll_context.manager import own_session_spans

    db = tmp_path / "hist.db"
    m1 = ScrollContextManager(
        history_db_path=db, session_id="r1:tX", run_id="r1", task_id="tX",
        history_max_tokens=0, pinned=1, var_context=True,
    )
    msgs1 = [{"role": "user", "content": "Find the launch date of the testing suite."}]
    m1.record_initial_prompt(msgs1[0])
    run_turn(m1, msgs1, n_steps=1, answer="The launch date is 2025-02-15 (S71).", label="a")
    assert m1.close_session(final_answer="The launch date is 2025-02-15 (S71).") is True
    assert m1.close_session(final_answer="dup") is False       # idempotent
    assert m1.totals["session_records"] == 1
    m1.close()

    m2 = ScrollContextManager(
        history_db_path=db, session_id="r2:tX", run_id="r2", task_id="tX",
        history_max_tokens=0, pinned=1, var_context=True,
    )
    spans = own_session_spans(m2.runtime.memoryspace, task_id="tX",
                              exclude_session_id="r2:tX")
    assert len(spans) == 1 and spans[0]["tag"] == "P1"
    msgs2 = [{"role": "user", "content": "Same task, second session."}]
    m2.record_initial_prompt(msgs2[0])
    assert m2.prime_prior_sessions(msgs2) is True
    ph = msgs2[1]["content"]
    assert "P1" in ph
    assert "Find the launch date of the testing suite." in ph
    assert "The launch date is 2025-02-15 (S71)." in ph
    # A session's own record never enters its own map.
    m2b_spans = own_session_spans(m2.runtime.memoryspace, task_id="tX",
                                  exclude_session_id="r1:tX")
    assert m2b_spans == []
    m2.close()


def test_close_session_is_mode_independent(tmp_path):
    """A BASELINE (non-var) session also writes its session_record at close —
    plain durable bookkeeping, so future sessions can prime P lines from any
    mode's history. Turn count reports 1 (turn tracking is var-only)."""
    import json as _json

    from scroll_context import ScrollContextManager
    from scroll_context.manager import own_session_spans

    db = tmp_path / "hist.db"
    m1 = ScrollContextManager(
        history_db_path=db, session_id="r1:tY", run_id="r1", task_id="tY",
        history_max_tokens=0, pinned=1, var_context=False,
    )
    # Before the initial prompt: nothing to record.
    assert m1.close_session(final_answer="early") is False
    msgs = [{"role": "user", "content": "What is the deploy cadence?"}]
    m1.record_initial_prompt(msgs[0])
    assert m1.close_session(final_answer="Weekly, on Thursdays.") is True
    assert m1.close_session(final_answer="dup") is False        # idempotent
    rows = m1.runtime.memoryspace.sql_query(
        "SELECT content, metadata FROM hist.conversation_history "
        "WHERE kind='session_record'"
    )
    assert len(rows) == 1
    assert "Weekly, on Thursdays." in rows[0]["content"]
    assert _json.loads(rows[0]["metadata"])["turns"] == 1
    m1.close()

    # A later session (any mode) primes a P line from the baseline record.
    m2 = ScrollContextManager(
        history_db_path=db, session_id="r2:tY", run_id="r2", task_id="tY",
        history_max_tokens=0, pinned=1, var_context=False,
    )
    spans = own_session_spans(m2.runtime.memoryspace, task_id="tY",
                              exclude_session_id="r2:tY")
    assert len(spans) == 1 and spans[0]["tag"] == "P1"
    m2.close()


def test_var_keep_thoughts_and_clip_knobs(tmp_path, monkeypatch):
    # var_keep_thoughts=3: with 4 assistant steps, only the oldest distills.
    mgr = make_mgr(tmp_path / "k", budget=100_000, var_context=True,
                   var_keep_thoughts=3)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    long = ("I will sweep the October sessions for staff interactions and then "
            "rank the ten distinct events chronologically before answering. "
            "This is thought number {k} with plenty of surplus prose to shrink. "
            + "After that I will double-check each candidate against the app "
              "records, discard the duplicates, and only then compose the final "
              "ordered list with one line per event and its date attached. " * 2)
    for k in range(4):
        var_turn(mgr, messages, long.format(k=k), call_id=f"c{k}",
                 source=f"print({k})")
    mgr.manage(messages)
    asst = [m for m in messages if m.get("role") == "assistant"]
    assert asst[0]["content"].startswith("⟦")               # oldest distilled
    assert all(a["content"] == long.format(k=k)             # window of 3 verbatim
               for k, a in enumerate(asst) if k >= 1)
    # Env wiring for the new knobs.
    monkeypatch.setenv("SCROLL_VAR_KEEP_THOUGHTS", "4")
    monkeypatch.setenv("SCROLL_TURN_ASK_CHARS", "50")
    monkeypatch.setenv("SCROLL_TURN_ANS_CHARS", "55")
    m2 = make_mgr(tmp_path / "e", budget=1000, var_context=True)
    assert m2._var_keep_thoughts == 4
    assert m2._turn_ask_chars == 50 and m2._turn_ans_chars == 55


def test_close_turn_inert_outside_var_context(tmp_path):
    mgr = make_mgr(tmp_path, budget=100_000)  # baseline
    messages = [{"role": "user", "content": "q?"}]
    mgr.record_initial_prompt(messages[0])
    turn(mgr, messages, "thinking", tool_output="obs", call_id="c0")
    before = list(messages)
    umsg = {"role": "user", "content": "next question"}
    mgr.record_user_message(umsg, messages=messages)
    assert messages == before                  # no collapse, no placeholder
    assert mgr.close_turn(messages) is False
    out = mgr.execute_python(
        "r = ms.sql_query(\"SELECT COUNT(*) AS n FROM hist.conversation_history "
        "WHERE kind='turn_record'\")\nprint(r[0]['n'])"
    )
    assert "0" in out


def test_var_context_off_leaves_everything_alone(tmp_path, monkeypatch):
    monkeypatch.delenv("SCROLL_VAR_CONTEXT", raising=False)
    mgr = make_mgr(tmp_path, budget=100_000)          # default off
    assert mgr.metrics()["var_context"] is False
    assert "pin" not in mgr.runtime.namespace          # ops not injected
    monkeypatch.setenv("SCROLL_VAR_CONTEXT", "1")
    mgr2 = make_mgr(tmp_path / "b", budget=100_000)   # env default picks it up
    assert mgr2.metrics()["var_context"] is True
    assert callable(mgr2.runtime.namespace["pin"])


def test_placeholder_at_before_task_protects_task_from_eviction(tmp_path):
    mgr = ScrollContextManager(
        history_db_path=tmp_path / "h.db", session_id="r:t", run_id="r", task_id="t",
        history_max_tokens=120, pinned=2, placeholder_at=1,
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "the task"},
    ]
    mgr.record_initial_prompt(messages[1])
    spans = [{"seq_lo": 1, "seq_hi": 9, "head": "prior stuff", "tail": "prior stuff",
              "session": 1}]
    assert mgr.prime_prior_sessions(messages, spans)
    assert "[memory]" in messages[1]["content"]        # map BEFORE the task
    assert messages[2]["content"] == "the task"
    for i in range(6):
        turn(mgr, messages, f"t{i} " + "x" * 300, call_id=f"c{i}")
        mgr.manage(messages)
    # Pinned head (system, map, task) all survive heavy eviction.
    assert messages[0]["content"] == "sys"
    assert "[memory]" in messages[1]["content"]
    assert messages[2]["content"] == "the task"


def test_aged_stub_preserves_seq_pointer_even_over_cap_stub(tmp_path):
    mgr = make_mgr(tmp_path, budget=10_000, obs_keep_turns=1, tool_result_cap_chars=800)
    messages = [{"role": "user", "content": "task"}]
    mgr.record_initial_prompt(messages[0])
    turn(mgr, messages, "big read", tool_output="needle-xyz " + "d" * 5000, call_id="c0")
    tool_msg = [m for m in messages if m.get("role") == "tool"][0]
    assert "truncated in this prompt" in tool_msg["content"]  # cap fired first
    import re
    seq = int(re.search(r"seq (\d+)", tool_msg["content"]).group(1))

    turn(mgr, messages, "later 1", call_id="c1")
    turn(mgr, messages, "later 2", call_id="c2")
    n, _ = mgr._age_observations(messages)
    assert n == 1
    # Aging rewrote the cap stub, but the seq pointer survived
    assert f"seq {seq}" in tool_msg["content"]
    assert "ms.expand" in tool_msg["content"]
    # And that seq really recovers the full original content
    out = mgr.execute_python(
        f"rows = ms.expand([{seq}])\nprint('needle-xyz' in rows[0]['content'], len(rows[0]['content']))"
    )
    assert "True" in out
