"""Guard against drift between the BEAM system prompt and the real DB schema.

The BEAM system prompt hard-codes the column names of
``hist.conversation_history`` so the agent queries memory directly instead of
probing ``sqlite_master`` at runtime. Those names are a hand-copied
transcription of the schema defined in ``src/scroll_context/_runtime/history.py``; if the
schema changes and the prompt doesn't (or vice-versa), the agent is back to
issuing queries against columns that don't exist. These tests fail when the two
fall out of sync, forcing both to change together.
"""
from __future__ import annotations

import re

from scroll_context._runtime.history import HistoryStore
from scroll_eval.evals.beam import prompts

# Columns the prompt promises do NOT exist (the trap the agent kept falling into).
_FORBIDDEN = {"timestamp", "date", "time"}


def _actual_columns(tmp_path) -> set[str]:
    """The real conversation_history columns, straight from a built schema."""
    store = HistoryStore(tmp_path / "schema_probe.db")
    try:
        cur = store._conn.execute("PRAGMA table_info(conversation_history)")
        return {row[1] for row in cur}
    finally:
        store.close()


def _prompt_columns() -> set[str]:
    """Backticked column names the system prompt documents.

    Matches the leading backticked column token at the start of a line in either
    layout the prompt has used: a markdown table row (``| `step_index` | ... |``)
    or a bullet (``- **`step_index`** — ...``, incl. ``- **`kind = 'conversation'`**``).
    The token must be a bare identifier followed by a closing backtick or a space,
    so a function reference like ``- **`json_extract(metadata, '$.date')`**`` (which
    points at the ``metadata`` column, not a column literally named ``json_extract``)
    is not mistaken for a column.
    """
    text = prompts.load("system")
    return set(re.findall(r"^[|\-]\s*(?:\*\*)?`([a-z_]+)(?:`| )", text, re.MULTILINE))


def test_prompt_columns_exist_in_schema(tmp_path) -> None:
    actual = _actual_columns(tmp_path)
    promised = _prompt_columns()

    # Sanity: the table parsed at all (a prompt reformat shouldn't silently
    # yield an empty set that vacuously passes the subset check below).
    assert {"content", "step_index", "msg_index", "kind"} <= promised, (
        f"prompt schema table not parsed as expected; got {promised}"
    )
    missing = promised - actual
    assert not missing, (
        f"system prompt names columns absent from conversation_history: {missing}. "
        "Update evaluation/scroll_eval/evals/beam/prompts/system.md or _runtime/history.py to match."
    )


def test_prompt_forbidden_columns_really_absent(tmp_path) -> None:
    # The prompt tells the agent there is no timestamp/date/time column; if the
    # schema ever grows one, that guidance becomes wrong and should be revisited.
    actual = _actual_columns(tmp_path)
    leaked = _FORBIDDEN & actual
    assert not leaked, (
        f"schema now has column(s) the prompt says don't exist: {leaked}. "
        "Update the BEAM system prompt."
    )
