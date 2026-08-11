"""The scroll context-management prompt protocol — single source of truth.

The model-facing text lives in two ``{repl}``-templated markdown files in this
package (canonical home; no other copy exists):

- ``core.md``  — the context-management core: the REPL, the ``ms`` memory API,
  the retrieve→reshape loop, search craft, and evidence-weighing discipline.
- ``index.md`` — the headline fence and the in-context memory map (line
  shapes, drill-down recipes). Only meaningful when the eviction index is
  enabled.

:func:`protocol_prompt` assembles them for a given configuration and REPL tool
name; ``ScrollContextManager.protocol_prompt()`` calls it with the manager's
own settings, so a host's system prompt is *harness preamble +
mgr.protocol_prompt() + harness finishing* and every prompt edit lands in one
file. ``{repl}`` binds the host's REPL tool name (``scroll_repl`` for
OpenAI-format harnesses, ``execute_python`` for scroll_agent_A). Templating is
a plain ``str.replace``, so other braces in the files (e.g. the literal
``{prose}`` FTS column filter in ``index.md``) pass through untouched.

With the index disabled, :func:`strip_headline_schema` removes the ``headline``
column from the schema docs so the ablation prompt matches the disabled
feature instead of describing a thing that isn't there.
"""

from pathlib import Path

_DIR = Path(__file__).parent


def _load(name: str, repl_name: str) -> str:
    return (_DIR / f"{name}.md").read_text(encoding="utf-8").replace(
        "{repl}", repl_name
    )


def core_prompt(repl_name: str = "scroll_repl") -> str:
    """The context-management core (``core.md``), REPL name bound."""
    return _load("core", repl_name)


def index_prompt(repl_name: str = "scroll_repl") -> str:
    """The headline + memory-map guidance (``index.md``), REPL name bound."""
    return _load("index", repl_name)


# The `core.md` fragments that advertise the `headline` column (the `ms.search`
# return-row shape, the routed-hit explanation, and the `ms.sql_query` column
# list), with their index-off replacements. Byte-exact pairs —
# `test_scroll_agent_A` fails loudly if a prompt edit breaks the match.
HEADLINE_SCHEMA_FRAGMENTS = (
    ("kind, role, name, headline, snippet|content, has_code, via",
     "kind, role, name, snippet|content, has_code, via"),
    # The headline-hit explanation only makes sense when headlines exist; with
    # the index OFF the headline FTS column is all-NULL and via='headline' can
    # never occur, so strip the sentence entirely.
    ("`via='headline'` marks a ROUTED row: a session's summary matched your query "
     "and this row is the best-matching turn found inside that session — its "
     "snippet carries `⟦via summary of S<n>: …⟧`; the summary text is index "
     "provenance, never quotable evidence. When the marker lists more matched "
     "sessions, sweep those `step_index` values too before answering "
     "aggregate/count questions. ",
     ""),
    ("role, name, content, headline, tool_call_id",
     "role, name, content, tool_call_id"),
)


def strip_headline_schema(text: str) -> str:
    """Remove the ``headline`` column from the core schema docs (index off)."""
    for with_col, without_col in HEADLINE_SCHEMA_FRAGMENTS:
        text = text.replace(with_col, without_col)
    return text


def protocol_prompt(repl_name: str = "scroll_repl", *, index: bool = True) -> str:
    """The full scroll protocol for one configuration, ready to embed.

    ``index`` appends the headline/map guidance; off, it also strips the
    ``headline`` column (and the routed-hit sentence) from the schema docs.
    """
    parts = [core_prompt(repl_name)]
    if index:
        parts.append(index_prompt(repl_name))
    out = "\n\n".join(p.rstrip() for p in parts)
    if not index:
        out = strip_headline_schema(out)
    return out


# Convenience constant for OpenAI-format hosts: the default protocol
# (core + index) under the `scroll_repl` tool name. scroll_agent_A instead
# calls `mgr.protocol_prompt()`, which binds `execute_python` and follows the
# manager's own index flag.
SCROLL_PROMPT_PROTOCOL = protocol_prompt("scroll_repl")
