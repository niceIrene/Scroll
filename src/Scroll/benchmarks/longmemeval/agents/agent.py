"""LongMemEval SCROLL agent.

The agent applies the SCROLL harness (:class:`Scroll.core.ScrollAgent`)
to LongMemEval:

- ``E`` is the conversation log; the harness mirrors each haystack
  session into it as ``chat_turn`` entries.
- ``W`` is a SQLite + vector memoryspace with five tables
  (``chat_turns``, ``user_preferences``, ``event_dates``, ``facts``,
  ``sessions``) plus a ``rounds`` view. The agent's view is read-only
  (the experimental contract: harness ingests, agent only queries).
- At probe time the agent runs Python through an ``execute_python``
  tool call against a persistent REPL with ``ms`` / ``log`` / ``rlm``
  / ``extract_time_range`` bound.

Cross-task procedural memory (``procedural_hints``) is the LME-specific
extra: after the judge scores each probe, ``_on_probe_complete``
distills a 1-3 hint case study via a one-shot sub-LM and persists it
to a per-run ``_shared_memoryspace.json`` (file-locked). The NEXT probe
of the same ``question_type`` sees the top-K most-relevant hints
injected at the top of its probe hint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from opentelemetry.trace import Status, StatusCode

from Scroll.core import ScrollAgent
from Scroll.core._agent import _extract_text_from_response
from Scroll.core._codeact_agent import (
    _ANSWER_LINE_RE,
    _detect_cell_ops,
    _is_abstention_text,
    _tracer,
)
from Scroll.benchmarks.longmemeval.agents._common import (
    handle_only_body,
    make_time_range_extractor,
    probe_session_body,
)
from Scroll.benchmarks.longmemeval.agents._namespace import (
    make_lme_namespace,
    write_chat_turn_entries,
)
from Scroll.benchmarks.longmemeval.ingestor import (
    LMEIngestor,
    ensure_schema as _lme_ensure_schema,
)


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Probe-time tool schema. Probe-mode Python runs through an OpenAI-style
# ``tools`` interface so the model can reason in natural-language message
# content while passing CODE through a structured ``code`` parameter.
# Session-loop runs are auto-advanced (no LM call), so this affects ONLY
# probe answering.
# ---------------------------------------------------------------------------

_EXECUTE_PYTHON_TOOL_NAME = "execute_python"

_EXECUTE_PYTHON_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": _EXECUTE_PYTHON_TOOL_NAME,
        "description": (
            "Execute a Python snippet in the persistent REPL to answer "
            "the probe. Use it to query ``ms`` / ``log`` and call "
            "``await rlm(...)``. REPL globals persist across calls, "
            "so a later call can use variables (``rows``, ``candidates``) "
            "defined earlier. The snippet's stdout / stderr / any "
            "traceback comes back as a tool_result. Pass ONLY Python "
            "source as ``code`` — no markdown fences, no prose."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python source. Use ``print(x)`` to surface "
                        "values (the REPL does NOT auto-display the "
                        "last expression)."
                    ),
                },
            },
            "required": ["code"],
        },
    },
}


# ---------------------------------------------------------------------------
# Prompt text — LME-specific framing + probe hint.
# ---------------------------------------------------------------------------

_LME_CONTEXT = """\
You are an autonomous chat assistant evaluated on long-term memory
across one QA item, run as N+1 days:

- Days 1..N: each session delivers ONE past chat session (with date).
- Day N+1: A single probe question is asked. Answer the user's question,
    or explicitly abstain only when the topic was never discussed.
"""


_LME_PROBE_HINT = """\
LOOP shape — each turn is EITHER a tool call OR a final answer:
  - To keep retrieving: emit an ``execute_python`` tool call with
    Python source as the ``code`` argument. ``print(x)`` to surface
    values; bare ``x`` is a no-op. The handler runs the cell and
    feeds stdout back as the next tool_result.
  - To finish: emit a plain-text answer in ``content`` and NO tool
    call. The handler treats "no tool call" as the termination
    signal. Don't wrap the answer in ``print()`` or
    ``execute_python``; don't carry a tool call alongside.

The notes below are awareness about failure modes and tool
semantics, not a script. If a cleaner approach (window functions,
hybrid pipelines, custom dedup) reaches the same answer, take it.

──────────────────────────────────────────────────────────────────
PROBE INTENT — what the question wants:

  FACTUAL ("what / when / where / how many did I X") — the answer
    is a literal value the user stated; retrieve + compute.
    Inferring across two stated facts is reading context, not
    fabricating ("uses Cartwheel app" + "Cartwheel = Target" →
    "redeemed at Target"). Abstain only when the chat genuinely
    contains nothing relevant AND no inferential path exists.

  RECOMMEND ("suggest / I'm thinking of X / any tips") — the
    literal subject is probe-only and won't appear in chat. You
    are recalling related preferences and constraints in the
    broader domain (NOT retrieving a literal answer). NEVER
    abstain on RECOMMEND. Drop off-topic prefs and anything
    explicitly disliked from your final answer.

──────────────────────────────────────────────────────────────────
PRIMITIVES — pick by what the answer demands:

  ``ms.sql_exec`` / ``log.findall`` — sub-second deterministic
    match (verbatim phrase, distinctive noun, regex pattern).
  ``ms.vector_query`` / ``log.semantic_search`` — paraphrase
    recall when the noun probably appears in different phrasings.
  ``rlm(query, context)`` — sub-agent with its own REPL + batched
    sub-LM calls. 5-20 sub-LM calls, 30-300s per invocation.
    For genuine LLM reasoning over evidence: filtering ambiguous
    candidates, span extraction across paragraph-length rows,
    multi-axis comparison/ranking. NOT a fallback for SQL misses.

Chain primitives in one cell when steps don't depend on seeing
prior output (SQL → rlm filter → Python merge/dedup). Split only
when the next QUERY needs the prior result.

──────────────────────────────────────────────────────────────────
WATCH-OUTS (awareness, not recipes):

  - Stdout cap ~4K chars (head 3K + tail 1K). A dozen full-content
    rows overflow it and the answer-row may sit in the dropped
    middle. Truncate per-row preview or LIMIT broad queries.
  - Values revised across sessions (salary, weight, address, PB
    time, mortgage amount): the most recent USER assertion is
    typically what the question asks for — even when the question
    phrases a single past event. **Default for any user-stated
    value question: end your query with ``ORDER BY session_idx
    DESC LIMIT 1`` and commit that row.** If you used a different
    ordering, do a sanity-check second query to verify no LATER
    assertion exists before committing. The wrong answer here is
    almost always the chronologically-FIRST hit.
  - Counting: same real-world item across sessions = ONE item;
    dedup before reporting. Apply any temporal scope BEFORE
    counting, not after.
  - RECOMMEND retrieval: anchor on the user's topical mentions in
    the broader domain, NOT on the recommendation verb
    ("publications", "tips", "suggestions") — those keywords
    match too broadly and pull the wrong session.

──────────────────────────────────────────────────────────────────
FINAL ANSWER — plain text in ``content``, NO ``execute_python``
tool call. The handler treats "no tool call" as the termination
signal, so the final turn must carry only ``content`` (don't pair
the answer with a tool call, and don't wrap it in ``print()``).

❌ WRONG:  execute_python({"code": "print('The play was ...')"})
✓ RIGHT:  The play was The Glass Menagerie (session 30, turn 4).

  Length: 1-3 sentences for FACTUAL, 3-6 short bullets for
    RECOMMEND. Mention session_idx inline only when disambiguating.
  ABSTENTION (FACTUAL only, after empty broaden):
    "I don't have that information from our conversations."
    Uncertainty about an exact value ≠ abstention — commit your
    best-grounded guess.
  COMPOSITION: if retrieval surfaced fragments (durations to sum,
    dates to subtract, facts to combine), do the arithmetic in
    your answer. "No single source stated the combined value" is
    not "no information".
  CONJUNCTION: "I'm Italian and Irish" → answer must keep BOTH;
    cherry-picking fails the judge.
  PAST-TENSE PROXY: "I attended" / "I just got back" → session
    date is the GT's event-date proxy. Don't abstain on "exact
    date wasn't stated".
"""


# Distilled playbook from prior 500-QA qwen3.6-plus run, optionally
# appended so weaker models inherit the strong model's query patterns
# without per-probe distillation cost.
from pathlib import Path as _PlaybookPath
_PLAYBOOK_PATH = _PlaybookPath(__file__).resolve().parents[4] / "playbook_longmemeval.md"
try:
    _PLAYBOOK = _PLAYBOOK_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    _PLAYBOOK = ""

_PLAYBOOK_BLOCK = (
    "\n\n"
    + "──────────────────────────────────────────────────────────────────\n"
    + "DISTILLED PLAYBOOK — synthesized from prior probe trajectories.\n"
    + "Apply by question SHAPE, not literally.\n\n"
    + _PLAYBOOK
) if _PLAYBOOK else ""


# ---------------------------------------------------------------------------
# JSON parser tolerant of markdown fences / leading prose. Used by L3
# distillation to consume sub-LM output.
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?(?P<body>.*?)\n?\s*```", re.DOTALL,
)


def _parse_json_tolerant(raw: str):
    if not raw or not raw.strip():
        return None
    s = raw.strip()
    try:
        return json.loads(s)
    except (ValueError, json.JSONDecodeError):
        pass
    m = _JSON_FENCE_RE.search(s)
    if m:
        try:
            return json.loads(m.group("body").strip())
        except (ValueError, json.JSONDecodeError):
            pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = s.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    chunk = s[start : i + 1]
                    try:
                        return json.loads(chunk)
                    except (ValueError, json.JSONDecodeError):
                        break
    return None


_STRATEGY_PROMPT = """\
The harness auto-ingests every chat session into ``memoryspace``
before each long-term-memory task fires. At probe time you have
ONE tool — ``execute_python(code: str)`` — that runs Python in a
persistent REPL with ``ms`` / ``log`` / ``rlm`` already bound
(full API in REPL GLOBALS section below).

Each turn is EITHER a tool call (still retrieving) OR a plain-text
``content`` answer (done). Don't carry both at once. The ``code``
argument takes ONLY Python source — no markdown fences, no
multi-line reasoning chains as comments. If your model has a
native thinking channel it will plan there; otherwise a short
``content`` string before the tool call is fine.

Memoryspace is READ-ONLY (writes raise PermissionError); REPL globals
persist across calls during the probe so call 2 can use variables
you bound in call 1.
"""


_LME_AUTO_LAYOUT = """\
AUTO-BUILT MEMORYSPACE (READ-ONLY). All tables regex-populated at
ingest. Types + enum values + format conventions matter — get them
wrong and your SQL silently returns 0 rows.

KEY NAMING — ``session_idx`` is the 1-based SESSION INDEX in
chronological order, NOT a calendar day. ``session_idx=7`` means
the 7th session in this QA's chat history; the actual calendar
date is in ``session_date_iso`` / ``session_ts_iso``. Use
``session_idx`` for ordering / joining / PKs; use the ``_iso``
columns for date arithmetic, BETWEEN ranges, and "how long ago".

  rounds(                                     -- ⭐ DEFAULT SURFACE
      session_idx      INTEGER,
      round_idx        INTEGER,               -- = user turn_idx
      session_date_iso TEXT,                  -- "YYYY-MM-DD"
      user_msg         TEXT,                  -- user side of round
      assistant_msg    TEXT)                  -- paired assistant reply
      ⭐ Each row = one full Q&A exchange (user + assistant together).
      Search this BY DEFAULT — single-row recall already carries both
      sides of context, no role filter needed. Use ``user_msg LIKE ...``
      for "what did I say about X"; ``assistant_msg LIKE ...`` for
      "what did you suggest"; bare ``content`` LIKE both via
      ``(user_msg LIKE ... OR assistant_msg LIKE ...)``.

  chat_turns(                                 -- raw turn-level access
      session_idx      INTEGER,
      turn_idx         INTEGER,
      role             TEXT,        -- 'user' | 'assistant'
      content          TEXT,
      session_id       TEXT,
      session_date_iso TEXT,
      session_ts_iso   TEXT)
      PRIMARY KEY (session_idx, turn_idx)
      -- Use only when you need ONE side in isolation. For most
      -- retrieval, prefer ``rounds``.

  user_preferences(
      session_idx INTEGER, turn_idx INTEGER,
      polarity    TEXT,             -- 'like' | 'dislike' | 'acquired'
      target      TEXT,             -- short noun phrase after the verb
      sentence    TEXT)             -- excerpt from the turn
      -- regex on "I like/want/dislike X" → like/dislike, plus
      --   "I (just|recently) (bought|got|signed up for|started using) X"
      --   → polarity='acquired' (implicit preference signal).
      -- Still SPARSE: on preference probes, ALSO LIKE chat_turns.content
      -- as fallback.

  event_dates(
      session_idx          INTEGER, turn_idx INTEGER,
      date_text            TEXT,    -- verbatim ("last week", "Mar 15")
      date_iso             TEXT,    -- resolved "YYYY-MM-DD" or NULL
      relative_offset_days INTEGER, -- signed offset from session_date_iso
      sentence             TEXT)    -- one-sentence excerpt (TRUNCATED)
      -- ``sentence`` is the source span; the actual EVENT description
      -- often lives in surrounding turns. JOIN chat_turns on
      -- (session_idx, turn_idx) when the question asks WHAT happened.

  facts(
      session_idx INTEGER, turn_idx INTEGER,
      fact_type   TEXT,             -- 'possession' | 'plan' | 'attribute' |
                                    --   'change' | 'current_state'
      fact_text   TEXT,
      sentence    TEXT)
      PRIMARY KEY (session_idx, turn_idx, fact_type, fact_text)
      -- Filter by fact_type to narrow. Sparse — fall back to chat_turns
      -- LIKE.

  sessions(
      session_idx      INTEGER PRIMARY KEY,   -- 1 row per session
      session_id       TEXT,
      session_date_iso TEXT,
      session_ts_iso   TEXT,
      session_text     TEXT)        -- full transcript concatenated
                                    --   capped at 20K chars.
      -- ⭐ One LIKE here = "which sessions mention X" in ONE round trip.
      --   For broad-noun probes, start here, get candidate session_idxs,
      --   then ``WHERE session_idx IN (...)`` on ``rounds``.

VECTORS — ``ms.vector_query(text, top_k=5)`` → list[(key, text, score)].
    Keys: "sess{N}_turn{K}_{role}" per-turn, "sess{N}_session"
    per-session. Embedded text is K = V + facts (raw content +
    regex-extracted prefs/dates/facts concatenated).
"""


_PROBE_EXAMPLE = """\
Don't call ``schema_inspect()`` — the schema is fixed (above). Your
first ``execute_python`` call should be a real retrieval query.
When SQL rows are paragraph-length and the answer span isn't visible
verbatim in stdout, chain ``rlm`` over the rows.

TURN SHAPE — each assistant turn is EITHER a tool call OR a final
answer, not both at once:

  • Retrieval turn: emit an ``execute_python`` tool call. The
    ``code`` argument is ONLY Python source — no markdown fences,
    no multi-line reasoning chains as ``#`` comments. If your model
    has a native thinking channel it will use that for planning;
    otherwise a short ``content`` string before the tool call is
    fine. Either way, don't fold reasoning into the code body.

  • Final-answer turn: emit plain text in ``content``, NO tool call.
    The presence of a tool call means "I'm still working"; absence
    means "this is my answer".
"""


_POST_HINTS_REMINDER = """\
COMMIT RULE — re-anchor (LAST thing before the question):
  Hints are PRIORS, not directives. Adopt them only if their
  ``when:`` trigger matches your probe's query shape; ignore the
  rest. The moment a tool_result shows the literal answer (keyword
  + value in one round, or ORDER BY DESC LIMIT 1 row, or all
  summands visible), STOP — your NEXT turn should be the final
  answer: plain text in ``content``, NO ``execute_python`` tool
  call. The handler treats "no tool call" as the termination
  signal, so a final-answer turn must not carry one. Each extra
  ``execute_python`` after the evidence is in burns one of your
  remaining budgeted turns; the harness will note ``[budget: N
  turns left]`` on each tool_result.
"""


BASE_PROMPT = (
    _LME_CONTEXT + "\n"
    + _STRATEGY_PROMPT
    + "\n" + _LME_AUTO_LAYOUT
)


_NAMESPACE_DOCS = """\
REPL GLOBALS available inside ``execute_python(code=...)`` —
``ms`` (memoryspace), ``log`` (event stream), ``rlm`` (sub-agent).
Full APIs below.

────────────────────────────────────────────────────────────────────
ms — memoryspace handle (read-only — see _STRATEGY_PROMPT):

  ms.sql_exec(statement: str, params: tuple | None = None) -> list[dict] | None
      SELECT / PRAGMA / EXPLAIN / WITH only (writes raise
      PermissionError). Returns list[dict] for SELECT (keys =
      column names), None for PRAGMA/EXPLAIN.
      ``params`` MUST be a tuple — single param needs trailing comma:
        ✓ ms.sql_exec("SELECT * FROM t WHERE session_idx=?", params=(5,))

  ms.vector_query(query: str, top_k: int = 5)
      -> list[tuple[str, str, float]]
      Bag-of-words cosine over auto-built vector index. Returns
      3-tuples — unpack, NOT dict.

  ms.json_read(key) -> Any              # KeyError if missing
  ms.json_list() -> list[str]
  ms.file_read(name) -> str             # KeyError if missing
  ms.file_list() -> dict[str, int]
  ms.schema_inspect() -> dict           # schema is fixed; rarely needed

────────────────────────────────────────────────────────────────────
log — LogHandle, unified event stream:

  Each ``LogEntry`` has attrs ``e.session_idx``, ``e.role``,
  ``e.content``, ``e.metadata`` (dict). ``e.session_idx`` is the
  cross-env session counter. ``kind`` lives in
  ``e.metadata.get('kind')``: {chat_turn, lm_turn, code, stdout,
  rlm_call, rlm_result}.

  log.slice(session_idx=None, role=None, kind=None) -> list[LogEntry]
  log.range_by_session(start: int, end: int)        -> list[LogEntry]  # inclusive
  log.search(query: str, k=20)                      -> list[LogEntry]
      # substring match (case-insensitive), newest-first
  log.semantic_search(
      query: str, k=10, *,
      session_idx=None, role=None, kind=None, min_score=0.0,
  ) -> list[tuple[LogEntry, float]]
      # bag-of-words cosine; NOT neural
  log.findall(pattern: str, entries=None, *, flags=0) -> list
  log.text(entries) -> str
      # serialize "[session N | role] content\\n…" — feed into rlm as ``context=``

────────────────────────────────────────────────────────────────────
rlm(query: str, *, context: str = "") -> str   (async — MUST ``await``)

  RECURSIVE LANGUAGE MODEL (backed by ``dspy.RLM``). The sub-agent
  receives only METADATA about ``context`` (type, length, preview),
  then writes Python in a sandboxed interpreter to search / filter /
  sample, calling its own ``llm_query`` / ``llm_query_batched``.
  Up to ~20 iterations / ~50 sub-LM calls per invocation.

  USE FOR:
    - Semantic operations SQL / vector can't express
    - Aggregation + commit over many candidate rows
    - Verification of SQL hits when synonyms / revisions may exist

  The sub-agent has NO access to your ``ms`` / ``log`` / outer
  ``rlm``. Whatever it needs MUST be in ``context``.

  Canonical usage — EXTRACT / RANK / SYNTHESIZE over evidence rows:

      rows = ms.sql_exec(
          "SELECT session_idx, user_msg, assistant_msg FROM rounds "
          "WHERE user_msg LIKE '%<topic>%' OR assistant_msg LIKE '%<topic>%'")
      body = "\\n".join(
          f"[sess {r['session_idx']}] U: {r['user_msg']}\\n"
          f"               A: {r['assistant_msg']}"
          for r in rows
      )
      print(await rlm(
          query=f"{probe_question}\\nReturn the most relevant 1-3 "
                f"[sess N] rounds with a verbatim quote each, or "
                f"'NOT FOUND'.",
          context=body,
      ))

  Each call mirrors into ``log`` as ``kind="rlm_call"`` /
  ``"rlm_result"`` so prior RLM answers are recoverable via
  ``log.semantic_search("<topic>", kind="rlm_result")``.

────────────────────────────────────────────────────────────────────
extract_time_range(question: str) -> dict | None   (async; MUST ``await``)

  One-shot LM helper that parses a date range from a question.
  Returns ``{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}`` when a
  range is detectable, or ``None`` when the question has no
  temporal scope (e.g. "what color does the user like?"). Knows
  what "today" means via the latest session date in your log, so
  relative phrases ("last month", "this week", "current") resolve
  to concrete ISO dates.

  USE FOR:
    - temporal-reasoning probes: filter sessions by date BEFORE
      reading content (avoids scanning the whole haystack)
    - Any time you want to cheaply narrow retrieval by date

  Canonical usage — DATE-FILTER, THEN READ:

      rng = await extract_time_range(probe_question)
      if rng:
          rows = ms.sql_exec(
              "SELECT session_idx, role, content "
              "FROM chat_turns "
              "WHERE session_ts_iso BETWEEN ? AND ? "
              "ORDER BY session_idx, turn_idx",
              (rng["start"], rng["end"]),
          )
      else:
          # No temporal scope — fall back to keyword / semantic
          # retrieval over the whole haystack.
          rows = ms.sql_exec(
              "SELECT ... FROM chat_turns WHERE content LIKE ?",
              (f"%{keyword}%",),
          )

  Implementation note: this is a system-side LM call — it does NOT
  appear in your ``log`` and does not count against your visible
  context. Cheap to call once at the start of a temporal probe.
"""


# ---------------------------------------------------------------------------
# Post-probe distillation prompt — consumed by L3's _on_probe_complete.
# ---------------------------------------------------------------------------

_DISTILL_PROMPT = """\
You are auditing a memory-system agent's probe answer. Distill 0-2
case-study hints that would help a future probe of similar SHAPE.
The agent's reasoning happened in a native thinking channel that you
do NOT see — only DO (the code the agent ran) and SEE (the resulting
stdout). Your job is to recover the SHAPE of the successful (or
failed) approach from those two surfaces.

Return JSON list ONLY (no prose), each entry:

  {{
    "qtype": "{qtype}",
    "applies_to_qtypes": "<comma-separated qtypes this case ALSO
                          applies to. Use 'ALL' for universal
                          lessons. Empty '' means this qtype only.>",
    "polarity": "success" | "failure",
    "question": "<the original probe question, paraphrased to remove
                  user-specific nouns. Aim for the TRANSFERABLE shape,
                  e.g. 'How many distinct X events happened between
                  date A and date B' rather than the literal probe.>",
    "pattern": "<ONE-line concrete code/SQL shape that captures the
                 actionable, copyable piece. This is the most
                 important field — it must be specific enough that a
                 future probe can adapt it by swapping nouns.
                 GOOD: 'SELECT ... FROM rounds WHERE user_msg LIKE
                   ''%<topic>%'' ORDER BY session_idx DESC LIMIT 1'
                 GOOD: 'rlm(query, context=body_of_rows) for paraphrase
                   classification before SQL'
                 BAD: 'use SQL'  (too abstract)
                 BAD: full multi-line code block  (too verbose)>",
    "summary": "<ONE imperative sentence saying what to do or NOT do,
                 with a brief reason. Must be specific and actionable.
                 GOOD success: 'Anchor knowledge-update probes on
                   session_idx DESC LIMIT 1 — the most recent
                   user-stated value supersedes earlier ones.'
                 GOOD failure: 'Do not abstain on 0 SQL hits when the
                   answer can be inferred from two stated facts
                   (e.g. Cartwheel app + coupon → Target).'
                 BAD: 'be careful with knowledge-update' (vague)>",
    "trajectory": "<the full DO/SEE trajectory, ONE step per turn,
                    ordered. Cap each SEE block at ~300 chars.
                    Skip steps that contributed nothing. This is for
                    offline audit only — it is NOT shown to future
                    probes, so don't golf it.>"
  }}

Required fields: ``question``, ``pattern``, ``summary``. Hints
missing any of these are dropped.

Skip cases where the agent succeeded in one trivial query OR
thrashed across many — those don't carry generalizable shape.
Prefer cases where DO + SEE shows a non-obvious primitive choice or
recovery move (broaden → narrow, SQL → rlm, etc.).

PROBE TYPE: {qtype}
PROBE QUESTION: {question}
AGENT ANSWER: {answer}
GROUND TRUTH: {gt}
SCORE: {score}

AGENT'S TRAJECTORY (DO / SEE per step):
{trajectory}
"""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class LongMemEvalAgent(ScrollAgent):
    """SCROLL agent for LongMemEval.

    Read-only memoryspace + dspy.RLM sub-agent + cross-task procedural
    memory (``procedural_hints`` distilled post-probe and injected into
    future probes of the same ``question_type``).
    """

    # SCROLL feature flags
    readonly_memoryspace = True
    expose_rlm = True

    sys_prompt = BASE_PROMPT
    probe_max_iters = 10

    # Cross-task procedural memory: orchestrator wires ``cfg.shared_memoryspace_path``
    # to a per-run shared JSON file. The base mechanism loads
    # ``procedural_hints`` at agent init and merges back after every session-end.
    shared_memoryspace_keys = ["procedural_hints"]

    procedural_hints_cap_total = 500
    procedural_hints_per_qtype_cap = 50
    procedural_hints_in_prompt = 5

    def __init__(self, cfg, storage, data) -> None:
        # Probe-time bookkeeping (set in probe_user_hint, read by L3
        # distillation in _on_probe_complete).
        self._active_probe = None
        self._probe_cells: list[str] = []
        # One-shot LM helper used SYSTEM-SIDE only (L3 distillation +
        # extract_time_range). Distinct from agent-facing ``rlm``.
        # Populated in extra_namespace.
        self._oneshot = None
        super().__init__(cfg, storage, data)

    # ----- SCROLL hooks -----

    ingestor_cls = LMEIngestor

    def _ensure_schema(self, memoryspace) -> None:
        _lme_ensure_schema(memoryspace)

    def extra_namespace(self) -> dict:
        # System-side one-shot LM (NOT exposed to the agent). Feeds
        # ``extract_time_range`` and L3 distillation. Direct ``self._model``
        # call — no log mirror (the harness doesn't query these back).
        async def _oneshot(prompt: str) -> str:
            response = await self._model(
                [{"role": "user", "content": str(prompt)}]
            )
            return _extract_text_from_response(response)

        self._oneshot = _oneshot
        return {
            "extract_time_range": make_time_range_extractor(
                self._oneshot, today_iso_provider=self._latest_session_iso,
            ),
        }

    def _base_namespace(self) -> dict:
        return make_lme_namespace(self._tool_state, agent=self)

    def namespace_docs(self) -> str:
        return _NAMESPACE_DOCS

    def session_prompt(self, session_idx: int) -> str:
        # Auto-advance: agent only acts at probe time.
        env = self._tool_state.env
        if not getattr(env, "current_session", None):
            return probe_session_body(session_idx)
        return handle_only_body()

    def _latest_session_iso(self) -> str | None:
        try:
            rows = self.memoryspace.sql_exec(
                "SELECT session_ts_iso FROM chat_turns "
                "WHERE session_ts_iso IS NOT NULL "
                "ORDER BY session_idx DESC, turn_idx DESC LIMIT 1"
            )
        except Exception:  # noqa: BLE001
            return None
        return rows[0]["session_ts_iso"] if rows else None

    # ----- Probe-mode system prompt -----
    # Drop the base CodeActAgent ``PROBE_SUBSTRATE_PROMPT`` prelude — in
    # this tool-call substrate the loop shape is structural (text-only
    # response = final answer), so the prelude is redundant.

    def _probe_sys_prompt(self) -> str:
        parts: list[str] = []
        env_probe = self._env_probe_prompt()
        if env_probe:
            parts.append(env_probe.strip())
        parts.append(self.sys_prompt.strip())
        ns_docs = self.namespace_docs().strip()
        if ns_docs:
            parts.append(ns_docs)
        extra = self.extra_system_prompt().strip()
        if extra:
            parts.append(extra)
        return "\n\n".join(p for p in parts if p)

    # ----- Probe loop — OpenAI tool-call format -----

    async def _call_probe_model(self, msgs: list):
        """Wrapper around ``self._model`` that passes ``tools=`` and
        quota-aware retry."""
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                return await self._model(
                    msgs,
                    tools=[_EXECUTE_PYTHON_TOOL_SCHEMA],
                    tool_choice="auto",
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                err_str = str(e).lower()
                is_quota = any(
                    s in err_str
                    for s in ("429", "quota", "rate", "insufficient")
                )
                if attempt < 3 and is_quota:
                    delay = 15 * attempt
                    _log.warning(
                        "LongMemEvalAgent probe model quota error "
                        "(attempt %d/3): %.150s — backing off %ds",
                        attempt, e, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
        raise last_err  # type: ignore[misc]

    async def _answer_probe_inner(self, max_iters: int) -> str:
        """Probe loop using ``execute_python`` tool calls instead of
        fenced ``python`` blocks.
        """
        last_text = ""
        last_stdout = ""
        cells_executed = 0
        zero_cell_abstain_forced = False

        for it in range(max_iters):
            response = await self._call_probe_model(self._history)

            text_parts: list[str] = []
            tool_use_blocks: list[dict] = []
            content = getattr(response, "content", None)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type")
                        if btype == "text":
                            text_parts.append(block.get("text", "") or "")
                        elif btype == "tool_use":
                            tool_use_blocks.append(block)
                    elif isinstance(block, str):
                        text_parts.append(block)
            elif isinstance(content, str):
                text_parts.append(content)
            text = "\n".join(p for p in text_parts if p).strip()

            if not text and not tool_use_blocks:
                # Empty response — terminate with what we have.
                break

            # Diagnostic: the prompt asks for EITHER content OR tool_calls
            # each turn. Track turns where both showed up — reasoning
            # models that natively think shouldn't also dump prose into
            # content alongside a tool call. This is debug-only; the
            # handler still keeps both for trace fidelity.
            if text and tool_use_blocks:
                _log.debug(
                    "LongMemEvalAgent probe iter %d emitted content "
                    "(%d chars) AND %d tool call(s) — prompt expects "
                    "either-or",
                    it, len(text), len(tool_use_blocks),
                )

            assistant_msg: dict = {
                "role": "assistant",
                "content": text if text else "",
            }
            tool_calls_payload: list[dict] = []
            for idx, block in enumerate(tool_use_blocks):
                tc_id = block.get("id") or f"probe-tc-{it}-{idx}"
                raw_input = block.get("input")
                if isinstance(raw_input, dict):
                    code = raw_input.get("code", "") or ""
                else:
                    raw_str = block.get("raw_input") or ""
                    try:
                        parsed = json.loads(raw_str) if raw_str else {}
                        code = parsed.get("code", "") if isinstance(parsed, dict) else ""
                    except Exception:  # noqa: BLE001
                        code = ""
                tool_calls_payload.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": block.get("name") or _EXECUTE_PYTHON_TOOL_NAME,
                        "arguments": json.dumps({"code": code}),
                    },
                })
            if tool_calls_payload:
                assistant_msg["tool_calls"] = tool_calls_payload
            self._history.append(assistant_msg)

            if not tool_use_blocks:
                last_text = text
                # Reject zero-cell abstentions once.
                if (
                    cells_executed == 0
                    and not zero_cell_abstain_forced
                    and _is_abstention_text(text)
                ):
                    zero_cell_abstain_forced = True
                    self._history.append({
                        "role": "user",
                        "content": (
                            "[ABSTENTION REJECTED — zero tool calls.] "
                            "You answered \"I don't have that "
                            "information\" without calling "
                            "``execute_python`` ANY memoryspace query. "
                            "Per the SEARCH BEFORE YOU REFUSE rule, a "
                            "zero-call abstention is wrong by default "
                            "— the data may be in your store even when "
                            "nothing in your recent chat history "
                            "mentions it.\n\n"
                            "Make at least ONE ``execute_python`` "
                            "call now: ``ms.sql_exec(...)`` or "
                            "``log.semantic_search(...)`` or "
                            "``log.findall(...)`` using keywords from "
                            "the probe question. If the query genuinely "
                            "returns nothing relevant, you may abstain "
                            "in your NEXT response based on what you "
                            "observed."
                        ),
                    })
                    last_text = ""
                    continue
                break

            day_ended = False
            for idx, block in enumerate(tool_use_blocks):
                tc_id = tool_calls_payload[idx]["id"]
                tc_args = json.loads(
                    tool_calls_payload[idx]["function"]["arguments"]
                )
                code = tc_args.get("code", "") or ""
                if not code.strip():
                    self._history.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": (
                            "[error: execute_python called with empty "
                            "code. Pass actual Python source as the "
                            "``code`` argument.]"
                        ),
                    })
                    continue

                cells_executed += 1
                ops = _detect_cell_ops(code)
                ops_label = ", ".join(ops) if ops else "no-ops"
                line_count = code.count("\n") + 1
                span_name = (
                    f"probe_cell d{self._current_session}.i{it}.{idx} "
                    f"({line_count}L) [{ops_label}]"
                )
                code_preview = code if len(code) <= 3000 else (
                    code[:2900]
                    + "\n\n# ... [truncated; see cell.code_full]"
                )
                with _tracer.start_as_current_span(span_name) as span:
                    span.set_attributes({
                        "tool.name": _EXECUTE_PYTHON_TOOL_NAME,
                        "cell.session": self._current_session,
                        "cell.iter": it,
                        "cell.tool_idx": idx,
                        "cell.kind": "probe",
                        "cell.ops": ",".join(ops),
                        "cell.line_count": line_count,
                        "cell.code_chars": len(code),
                        "cell.code_full": code,
                        "input.value": code_preview,
                        "input.mime_type": "text/plain",
                    })
                    result = await self._runtime.execute_cell(code)
                    output_parts = []
                    if result.stdout:
                        output_parts.append(result.stdout)
                    if result.stderr:
                        output_parts.append("[stderr]\n" + result.stderr)
                    if result.exception:
                        output_parts.append(
                            "[exception]\n" + result.exception
                        )
                    summary_bits = [
                        f"stdout={len(result.stdout or '')}c",
                        f"ops=[{ops_label}]",
                    ]
                    if result.exception:
                        summary_bits.append("EXCEPTION")
                    cell_summary = " | ".join(summary_bits)
                    output_preview = (
                        ("\n\n".join(output_parts))[:2000]
                        if output_parts
                        else (
                            f"[probe cell ran, no stdout]\n\n"
                            f"summary: {cell_summary}\n\n"
                            f"code (first 200c):\n{code[:200]}"
                        )
                    )
                    stdout_preview = (
                        (result.stdout or "")[:200].replace("\n", " ⏎ ")
                    )
                    stdout_lines = (
                        result.stdout.count("\n") if result.stdout else 0
                    ) + (1 if result.stdout else 0)
                    span.set_attributes({
                        "openinference.span.kind": "TOOL",
                        "output.value": output_preview,
                        "output.mime_type": "text/plain",
                        "cell.summary": cell_summary,
                        "cell.stdout_preview": stdout_preview or "[empty]",
                        "cell.stdout_lines": stdout_lines,
                        "cell.stdout_chars": len(result.stdout or ""),
                        "code_exec.has_exception": (
                            result.exception is not None
                        ),
                    })
                    if result.exc is not None:
                        span.record_exception(result.exc)
                        span.set_status(Status(
                            StatusCode.ERROR,
                            f"{type(result.exc).__name__}: {result.exc}"[:200],
                        ))

                last_stdout = result.stdout or ""
                self._record_probe_cell(code, result, it, thought=text)

                # Re-anchor question + budget reminder next to latest
                # tool_result.
                turns_left = max_iters - (it + 1)
                probe_q = (
                    getattr(self._active_probe, "question", "") or ""
                ).strip()
                m_q = re.search(r"Question:\s*(.+?)(?:\n|If the chat)",
                                probe_q, re.DOTALL)
                question_only = (m_q.group(1).strip() if m_q else probe_q)[:300]
                question_anchor = (
                    f"\n[QUESTION: {question_only}]"
                    if question_only else ""
                )
                if turns_left == 0:
                    budget_note = (
                        "\n[turns remaining: 0 — your NEXT response "
                        "should be the final answer (plain text, no "
                        "tool call).]"
                    )
                elif turns_left <= 2:
                    budget_note = f"\n[turns remaining: {turns_left}]"
                else:
                    budget_note = ""
                budget_tag = f"{question_anchor}{budget_note}"
                self._history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result.to_user_message() + budget_tag,
                })
                if result.day_ended:
                    day_ended = True
                    break
            if day_ended:
                break

        # stdout fallback safety net.
        if last_stdout and not _ANSWER_LINE_RE.search(last_text):
            return last_text + "\n\n[stdout]\n" + last_stdout
        return last_text

    # ----- Probe-time hint composition -----

    def probe_user_hint(self, probe=None) -> str:
        """Compose probe-time hint = base + procedural_hints + example +
        probe context. The ``probe`` arg is read for ``question_type`` so
        we can filter hints by qtype.
        """
        self._active_probe = probe
        self._probe_cells = []

        qtype = self._resolve_qtype(probe)
        probe_question = getattr(probe, "question", "") if probe else ""

        base = _LME_PROBE_HINT
        if _PLAYBOOK_BLOCK and getattr(self.cfg, "enable_playbook", True):
            base = base + _PLAYBOOK_BLOCK
        parts: list[str] = [base]
        hints_block = self._format_procedural_hints(qtype, probe_question)
        if hints_block:
            parts.append(hints_block)
            # Re-anchor commit rule AFTER hints (model otherwise gets
            # absorbed in "try the suggested pattern" and forgets to
            # commit when stdout already has the answer).
            parts.append(_POST_HINTS_REMINDER)
        parts.append(_PROBE_EXAMPLE)
        ctx = self._format_probe_context()
        if ctx:
            parts.append(ctx)
        return "\n\n".join(parts)

    def _resolve_qtype(self, probe) -> str | None:
        attr = getattr(probe, "question_type", None) if probe else None
        if attr:
            return attr
        try:
            from Scroll.benchmarks.longmemeval.tasks.probes import active_question_type
            return active_question_type()
        except Exception:  # noqa: BLE001
            return None

    def _format_probe_context(self) -> str:
        """Most-recent session pointer (anchors knowledge-update probes)."""
        try:
            rows = self.memoryspace.sql_exec(
                "SELECT session_idx, session_date_iso FROM chat_turns "
                "ORDER BY session_idx DESC, turn_idx DESC LIMIT 1"
            )
        except Exception:  # noqa: BLE001
            return ""
        if not rows:
            return ""
        last = rows[0]
        last_idx = last["session_idx"]
        last_date = last["session_date_iso"]
        first_user_snippet = ""
        try:
            first = self.memoryspace.sql_exec(
                "SELECT content FROM chat_turns "
                "WHERE session_idx = ? AND role = 'user' "
                "ORDER BY turn_idx LIMIT 1",
                params=(last_idx,),
            )
            if first:
                first_user_snippet = (first[0]["content"] or "")[:200]
        except Exception:  # noqa: BLE001
            pass
        lines = [
            "PROBE CONTEXT — most recent ingested session:",
            f"  session_idx={last_idx}, session_date={last_date!r}",
        ]
        if first_user_snippet:
            lines.append(f"  first user turn: {first_user_snippet!r}")
        return "\n".join(lines)

    def _format_procedural_hints(
        self, qtype: str | None, probe_question: str = "",
    ) -> str:
        """Inject top-K relevant hints for ``qtype``.

        Filter: qtype match + polarity in (success, failure) + source_qid
        not equal to the active probe's qid (leakage prevention). Then
        rank by BoW cosine between probe question and hint
        question/trigger.
        """
        span_name = f"probe_hints_inject qtype={qtype}"
        with _tracer.start_as_current_span(span_name) as span:
            span.set_attributes({
                "openinference.span.kind": "CHAIN",
                "inject.qtype": qtype or "",
            })
            return self._format_procedural_hints_inner(qtype, probe_question, span)

    def _format_procedural_hints_inner(
        self, qtype: str | None, probe_question: str, span,
    ) -> str:
        if not qtype:
            span.set_attribute("inject.skip_reason", "no_qtype")
            return ""
        try:
            hints = self.memoryspace.json_read("procedural_hints")
        except (KeyError, Exception):  # noqa: BLE001
            span.set_attribute("inject.skip_reason", "no_store")
            return ""
        if not isinstance(hints, list) or not hints:
            span.set_attribute("inject.store_size", 0)
            return ""
        span.set_attribute("inject.store_size", len(hints))

        active_qid = getattr(self._active_probe, "question_id", None)

        def _hint_matches_qtype(h: dict) -> bool:
            if h.get("qtype") == qtype:
                return True
            applies = (h.get("applies_to_qtypes") or "").strip()
            if not applies:
                return False
            applies_lc = applies.lower()
            if applies_lc == "all":
                return True
            extra = {q.strip() for q in applies_lc.split(",") if q.strip()}
            return qtype.lower() in extra

        def _has_takeaway(h: dict) -> bool:
            # v2 schema needs both pattern AND summary to render
            # something useful; legacy schema (lesson / anti_pattern)
            # still counts as "has takeaway" so old shared_memoryspace
            # entries survive.
            new_has = (h.get("pattern") or "").strip() and (h.get("summary") or "").strip()
            legacy_has = (
                (h.get("lesson") or "").strip()
                or (h.get("anti_pattern") or "").strip()
            )
            return bool(new_has or legacy_has)

        candidates = [
            h for h in hints
            if isinstance(h, dict)
            and _hint_matches_qtype(h)
            and h.get("polarity") in ("success", "failure")
            and _has_takeaway(h)
            and (active_qid is None or h.get("source_qid") != active_qid)
        ]
        span.set_attribute("inject.candidates_same_qtype", len(candidates))
        if active_qid:
            span.set_attribute("inject.self_qid_filtered", True)
            span.set_attribute("inject.active_qid", active_qid)
        if not candidates:
            span.set_attribute("inject.skip_reason", "no_same_qtype")
            return ""

        def _toks(s: str) -> set:
            return {t for t in re.findall(r"\w+", (s or "").lower()) if len(t) > 2}
        q_toks = _toks(probe_question)
        if q_toks:
            def _score(h: dict) -> float:
                tt = _toks(h.get("question", "") + " " + h.get("trigger", ""))
                if not tt:
                    return 0.0
                inter = len(q_toks & tt)
                return inter / ((len(q_toks) * len(tt)) ** 0.5) if inter else 0.0
            ranked = sorted(candidates, key=_score, reverse=True)
        else:
            ranked = candidates
        k = getattr(self.cfg, "procedural_hints_in_prompt", None) or self.procedural_hints_in_prompt
        matching = ranked[:k]
        span.set_attribute("inject.ranking_mode", "qtype_relevance_bow")

        if not matching:
            span.set_attribute("inject.skip_reason", "no_match")
            return ""
        span.set_attributes({
            "inject.injected_count": len(matching),
            "inject.source_qids": ",".join(
                str(h.get("source_qid", "")) for h in matching
            ),
            "inject.questions": " | ".join(
                (h.get("question") or h.get("trigger") or "")[:80]
                for h in matching
            ),
        })
        lines = [
            "──────────────────────────────────────────────────────────────────",
            f"LESSONS FROM PRIOR PROBES ({len(matching)} similar) — what "
            "worked, what didn't.",
            "Each entry: paraphrased question + a concrete PATTERN that",
            "captures the actionable code / query shape + a one-line",
            "TAKEAWAY. Adapt the shape; do NOT copy literal nouns. These",
            "are PRIORS to consider, not directives.",
            "──────────────────────────────────────────────────────────────────",
        ]
        for h in matching:
            polarity = h.get("polarity", "?")
            question = (h.get("question") or h.get("trigger") or "").strip()
            pattern = (h.get("pattern") or "").strip()
            summary = (h.get("summary") or "").strip()
            # Legacy schema (pre-v2) hints — pattern/summary may be
            # absent. Fall back to older field names so the shared
            # memoryspace.json from prior runs still surfaces useful hints
            # instead of getting filtered out by the renderer.
            legacy_lesson = (h.get("lesson") or "").strip()
            legacy_anti = (h.get("anti_pattern") or "").strip()
            legacy_code = (h.get("code") or "").strip()

            marker = "✓" if polarity == "success" else "⚠"
            lines.append(f"  ── [{polarity}] Q: {question}")
            if pattern:
                lines.append(f"      PATTERN: {pattern}")
            if summary:
                lines.append(f"      {marker} TAKEAWAY: {summary}")
            # Legacy fallback (only when the new fields are absent —
            # avoids double-rendering for v2 hints).
            if not pattern and not summary:
                if legacy_lesson:
                    lines.append(f"      {legacy_lesson}")
                if legacy_anti:
                    lines.append(f"      ⚠ {legacy_anti}")
                if legacy_code:
                    code_lines = [
                        ln for ln in legacy_code.splitlines() if ln.strip()
                    ][:4]
                    for ln in code_lines:
                        lines.append(f"      {ln}")
        return "\n".join(lines)

    # ----- L3 — post-probe distillation -----

    async def _on_probe_complete(
        self, *, probe, agent_answer: str, score: float, ground_truth,
    ) -> None:
        """Distill 0-3 procedural hints from this probe's trajectory.

        Skips on:
          - distillation disabled
          - middle ambiguous scores
          - empty trajectory (zero-cell abstain)
          - trivial trajectory (1 cell → success)
          - pathological trajectory (≥7 cells)
        """
        if not getattr(self.cfg, "enable_distillation", True):
            return
        qid = getattr(probe, "question_id", "") if probe else ""
        qtype = self._resolve_qtype(probe) or "unknown"
        cells_count = self._count_probe_cells()
        span_name = f"probe_distill d{self._current_session} qid={qid}"
        with _tracer.start_as_current_span(span_name) as span:
            span.set_attributes({
                "openinference.span.kind": "CHAIN",
                "probe.qid": qid,
                "probe.qtype": qtype,
                "probe.score": score,
                "probe.cells_count": cells_count,
            })
            if self._oneshot is None:
                span.set_attribute("distill.skip_reason", "no_oneshot_lm")
                return
            if 0.1 < score < 0.9:
                span.set_attribute("distill.skip_reason", "ambiguous_score")
                return
            if cells_count == 0:
                span.set_attribute("distill.skip_reason", "no_cells")
                return
            if cells_count == 1 and score >= 0.9:
                span.set_attribute("distill.skip_reason", "trivial_success")
                return
            if cells_count >= 7:
                span.set_attribute("distill.skip_reason", "pathological")
                return
            trajectory = self._summarize_trajectory()
            if not trajectory:
                span.set_attribute("distill.skip_reason", "empty_trajectory")
                return
            span.set_attribute("distill.trajectory_chars", len(trajectory))
            question = getattr(probe, "question", "")
            prompt = _DISTILL_PROMPT.format(
                qtype=qtype,
                question=question,
                answer=(agent_answer or "")[:800],
                gt=str(ground_truth)[:300],
                score=score,
                trajectory=trajectory[:16000],
            )
            try:
                raw = await self._oneshot(prompt)
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "LongMemEvalAgent distillation oneshot LM failed: %s", exc
                )
                span.set_attribute("distill.skip_reason", "oneshot_lm_error")
                span.record_exception(exc)
                return
            parsed = _parse_json_tolerant(raw or "")
            if not isinstance(parsed, list):
                span.set_attribute("distill.skip_reason", "parse_failed")
                span.set_attribute("distill.raw_preview", (raw or "")[:300])
                return
            new_hints: list[dict] = []
            for entry in parsed:
                if not isinstance(entry, dict):
                    continue
                summary = (entry.get("summary") or "").strip()
                pattern = (entry.get("pattern") or "").strip()
                question = (entry.get("question") or "").strip()
                # Required fields for the case-study schema. Hints that
                # miss any of these have no actionable signal — drop.
                if not summary or not pattern or not question:
                    continue
                new_hints.append({
                    # Schema marker — bumped from the legacy ``THINK/DO/SEE``
                    # trajectory-heavy format to a question + pattern +
                    # summary triple. Old hints (no ``schema_version``)
                    # are still readable in injection; the renderer falls
                    # back to the legacy display when ``pattern`` is
                    # missing.
                    "schema_version": 2,
                    "qtype": entry.get("qtype") or qtype,
                    "applies_to_qtypes": (
                        entry.get("applies_to_qtypes") or ""
                    ).strip(),
                    "polarity": entry.get("polarity") or (
                        "success" if score >= 0.9 else "failure"
                    ),
                    "question": question[:300],
                    "pattern": pattern[:400],
                    "summary": summary[:400],
                    # ``trajectory`` is preserved for offline audit but
                    # NOT shown to future probes (the renderer skips it
                    # by default). Capped at 3K chars.
                    "trajectory": (entry.get("trajectory") or "")[:3000],
                    "source_qid": getattr(probe, "question_id", ""),
                })
            span.set_attribute("distill.hints_emitted", len(new_hints))
            if not new_hints:
                return
            try:
                existing = self.memoryspace.json_read("procedural_hints")
                if not isinstance(existing, list):
                    existing = []
            except KeyError:
                existing = []
            span.set_attribute("distill.hints_existing", len(existing))
            merged = (existing + new_hints)[-self.procedural_hints_cap_total:]
            self.memoryspace.json_write("procedural_hints", merged)
            # Persist immediately so parallel sibling QAs see this before
            # our session-end hook fires (post-probe is the last meaningful
            # event in LME).
            self._persist_shared_memoryspace_state()
            span.set_attribute("distill.hints_total_after", len(merged))

    def _count_probe_cells(self) -> int:
        try:
            tail = self.log.entries[-30:]
        except Exception:  # noqa: BLE001
            return 0
        count = 0
        for entry in tail:
            meta = entry.metadata or {}
            kind = meta.get("kind") if isinstance(meta, dict) else None
            if kind == "probe_code":
                count += 1
        return count

    def _summarize_trajectory(self) -> str:
        """Render this probe's trajectory as DO / SEE pairs for
        distillation.

        Under the either-or prompt design, reasoning lives in the
        model's native thinking channel (not in ``entry.content``), so
        THINK is no longer captured here. DO (the code) and SEE (the
        stdout) are the two surfaces the distillation sub-LM uses to
        recover the SHAPE of what worked.
        """
        try:
            tail = self.log.entries[-30:]
        except Exception:  # noqa: BLE001
            return ""
        last_code_idx = -1
        for i, entry in enumerate(tail):
            meta = entry.metadata or {}
            if isinstance(meta, dict) and meta.get("kind") == "probe_code":
                last_code_idx = i
        parts: list[str] = []
        step_num = 0
        for i, entry in enumerate(tail):
            meta = entry.metadata or {}
            kind = meta.get("kind") if isinstance(meta, dict) else None
            if kind == "probe_code":
                step_num += 1
                tc = entry.tool_call or {}
                args = tc.get("arguments", {}) if isinstance(tc, dict) else {}
                code = args.get("code", "") if isinstance(args, dict) else ""
                label = (
                    f"STEP {step_num} [FINAL]"
                    if i == last_code_idx else f"STEP {step_num}"
                )
                if code:
                    parts.append(f"=== {label} ===\nDO:\n{code}")
            elif kind == "probe_stdout":
                content = (entry.content or "")[:2000]
                if content:
                    parts.append(f"SEE:\n{content}")
        return "\n\n".join(parts)

    # ----- Cross-task store hook -----

    def get_memoryspace(self):  # type: ignore[override]
        return self.memoryspace

    def _filter_for_shared_write(self, key: str, value):  # type: ignore[override]
        """Dedup + per-qtype cap before writing to cross-task store.

        Identity is (qtype, polarity, pattern OR lesson) — the new
        ``pattern`` field is the discriminator under the v2 schema;
        legacy ``lesson`` text falls through for pre-v2 entries so
        de-duplication still works across the schema bump.
        """
        if key != "procedural_hints" or not isinstance(value, list):
            return value
        seen: set[str] = set()
        deduped: list = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            identity = (
                entry.get("pattern")
                or entry.get("lesson")
                or entry.get("summary")
                or ""
            )
            sig = (
                f"{entry.get('qtype')}|"
                f"{entry.get('polarity')}|"
                f"{identity}"
            )
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(entry)
        by_qtype: dict[str, list] = {}
        for entry in deduped:
            by_qtype.setdefault(entry.get("qtype", ""), []).append(entry)
        capped: list = []
        for qtype, rows in by_qtype.items():
            capped.extend(rows[-self.procedural_hints_per_qtype_cap:])
        return capped[-self.procedural_hints_cap_total:]

    # ----- Auto-ingest -----

    def run_session(self, env) -> list[str]:
        """Mirror the env's chat session into ``E``. The attached
        :class:`LMEIngestor` will catch up on the next ``ms`` read; we
        trigger it eagerly here so any same-session probe sees fresh W.
        """
        write_chat_turn_entries(self.log, env, env.session_idx + 1)
        try:
            self.memoryspace._maybe_catch_up()
        except Exception:  # noqa: BLE001
            _log.warning(
                "LongMemEvalAgent ingest catch-up failed (session_idx=%s)",
                env.session_idx + 1, exc_info=True,
            )
        return []
