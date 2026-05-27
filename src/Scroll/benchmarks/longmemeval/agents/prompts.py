"""LongMemEval prompt constants — single source of truth.

================================================================
CONTEXT COMPOSITION — three categories
================================================================

  Part 1 — SYSTEM_PROMPT          (static; goes in the system message)
  Part 2 — SYSTEM_REMINDERS       (dynamic, probe-time context placed
                                   BEFORE the question — mirrors
                                   Anthropic claude-code's
                                   <system-reminder> pattern: reminders
                                   are priors the model reads first,
                                   then processes the question through
                                   that lens)
  Part 3 — auto-appended messages (the question itself, with a date-anchor
                                   wrapper around it and postscripts after)

────────────────────────────────────────────────────────────────
Part 1 — SYSTEM_PROMPT
────────────────────────────────────────────────────────────────

  LongMemEvalAgent.sys_prompt = SYSTEM_PROMPT   (class attribute)

      SYSTEM_PROMPT = _LME_CONTEXT        task framing
                    + _STRATEGY_PROMPT    tool + turn shape
                    + _LME_AUTO_LAYOUT    SQL schema docs
                    + _NAMESPACE_DOCS     REPL globals
                                          (ms / log / rlm / helpers)

  Static for the whole run — never depends on which probe is being asked.

────────────────────────────────────────────────────────────────
Part 2 — SYSTEM_REMINDERS  (probe-time, placed BEFORE the question)
────────────────────────────────────────────────────────────────

  Assembled by ``LongMemEvalAgent.probe_user_hint``. The framework
  inserts this block at the START of the user-turn message, before
  ``[PROBE — qid]`` and the question. Mirrors Anthropic claude-code's
  ``<system-reminder>`` content-block pattern.

      [qtype template]        ← dispatcher picks ONE of:
                                    _MULTI_SESSION_COUNT_TEMPLATE
                                    _KNOWLEDGE_UPDATE_TEMPLATE
                                    _TEMPORAL_REASONING_TEMPLATE
      _LME_PROBE_HINT         ← retrieval shapes + intent docs
        + _PLAYBOOK_BLOCK      (optional: playbook file present
                                AND enable_playbook=True)
      procedural_hints        ← per-qtype distilled lessons (top-K)
      ANSWER_GENERATION_PROMPT    ← mem0-style commit-time rules of evidence
                                  (incl. LME-specific equivalences in
                                  its Misc Rules section)

────────────────────────────────────────────────────────────────
Part 3 — auto-appended messages  (the question itself + wrappers)
────────────────────────────────────────────────────────────────

  Order in the user-turn message:

      [probe time anchor]              ← from ``probe_user_question_prefix``:
                                         ``Today's Date: YYYY-MM-DD``
                                         ``(latest session_idx=N)``
                                         ``Question:``
      <PROBE QUESTION TEXT>            ← from the dataset
      LME_PROBE_USER_POSTSCRIPT        ← universal reminder
      + _QTYPE_POSTSCRIPT[qtype]       ← per-qtype nudge (TR / KU / pref)
      OR _ABSTENTION_POSTSCRIPT         ← overrides qtype postscript when
                                         qid ends in ``_abs``

  Postscripts composed by ``tasks/probes.py::compose_user_postscript``;
  prefix wired by ``inject_probe`` via ``agent.probe_user_question_prefix``.

────────────────────────────────────────────────────────────────
Out-of-band — post-probe distillation (agent never sees this)
────────────────────────────────────────────────────────────────

  _DISTILL_PROMPT  ← one-shot LM call after a probe scores; extracts
                     0-2 case-study hints from the probe's trajectory
                     and persists them to the cross-task
                     procedural_hints store. Picked up in Part 2 on
                     the NEXT probe of the same question_type.

NOTE: The judge prompt (LLM-as-judge that scores the final answer)
lives in ``tasks/probes.py`` alongside the judge call logic
(``_judge_score`` / ``_parse_judge_verdict``). It's not a prompt
the AGENT sees, so it's intentionally kept next to its caller
rather than centralized here.

================================================================
WHERE TO MAKE CHANGES
================================================================

  Agent role / tool surface     →  Part 1 (SYSTEM_PROMPT pieces)
  Per-qtype forcing pattern     →  Part 2 qtype template + dispatcher
                                   in agent.py:_format_qtype_specific_hint
  Commit-time rules of evidence →  ANSWER_GENERATION_PROMPT (semi-general;
                                   Misc Rules subsection holds the
                                   LME-specific judge equivalences)
  Postscript wording            →  Part 3 constants
  Distillation rubric           →  _DISTILL_PROMPT
  Judge lenience / strictness   →  tasks/probes.py judge templates
"""

from __future__ import annotations

from pathlib import Path


# =============================================================================
# Part 1 — SYSTEM_PROMPT (static; system message)
# =============================================================================

_LME_CONTEXT = """\
You are a memory-recall assistant evaluated on one QA item. The
user's past chat sessions (each with a date) have already been
ingested into your memoryspace by the harness; you are NOT shown
them session-by-session. You are invoked ONCE: a probe question is
asked, you retrieve from memory using ``execute_python``, then commit
a plain-text answer — or explicitly abstain only when the topic was
never discussed.
"""


_STRATEGY_PROMPT = """\
The harness auto-ingests every chat session into ``memoryspace``
before each long-term-memory task fires. At probe time you have
ONE tool — ``execute_python(code: str)`` — that runs Python in a
persistent REPL with ``ms`` / ``log`` / ``rlm`` already bound
(full API in REPL GLOBALS section below). Memoryspace is READ-ONLY
(writes raise PermissionError); REPL globals persist across calls
during the probe so call 2 can use variables you bound in call 1.

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

  sessions(
      session_idx      INTEGER PRIMARY KEY,   -- 1 row per session
      session_id       TEXT,
      session_date_iso TEXT,
      session_ts_iso   TEXT)
      -- Date/index metadata only. The chat content lives in
      -- ``chat_turns`` / ``rounds`` — JOIN if you need both.

  chat_turns_fts (FTS5 virtual table over chat_turns.content)
      -- Use MATCH for keyword search — indexed (O(matches), not
      -- O(rows)). On large haystacks (M-split etc) this is 10-100x
      -- faster than ``WHERE content LIKE '%X%'`` full-table scans.
      -- Porter stemmer: "running" matches "run"; case-insensitive.
      -- Syntax: AND is implicit, ``OR`` explicit, ``"phrase"`` for
      -- exact, ``term*`` prefix, ``-term`` exclude.
      -- Canonical pattern (JOIN back to chat_turns for full row data):
      --   SELECT ct.session_idx, ct.role, ct.content
      --   FROM chat_turns_fts f
      --   JOIN chat_turns ct ON ct.rowid = f.rowid
      --   WHERE chat_turns_fts MATCH 'tank OR aquarium OR gallon'
      --   ORDER BY rank LIMIT 30
      -- Prefer this over ``rounds LIKE`` for broad keyword searches.
      -- ``LIKE`` still works (no index, but exact substring); keep it
      -- for matching distinctive multi-word verbatim phrases that
      -- FTS tokenization would split (e.g. exact dollar amounts).

VECTORS — ``ms.vector_query(text, top_k=5)`` → list[(key, text, score)].
    Keys: "sess{N}_turn{K}_{role}" per-turn, "sess{N}_session"
    per-session. Bag-of-words cosine (no neural embedder) — use
    ``chat_turns_fts MATCH`` for synonym-light keyword recall, not
    this. Mainly useful when you don't remember which session a
    topic was in and want a quick "top-5 sessions by token overlap".
"""


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

────────────────────────────────────────────────────────────────────
days_between(d1_iso: str, d2_iso: str, inclusive: bool = False) -> int

  Pure-Python date arithmetic. Returns the absolute number of days
  between two dates (always non-negative). Set ``inclusive=True``
  when the question says "including the last day" / GT counts
  endpoints.

  USE FOR every "how many days/weeks/months between A and B" or
  "how many days ago did X happen" question. LLM calendar math is
  unreliable past 2-week deltas — always compute with this helper.

  Inputs: ISO ``YYYY-MM-DD`` strings (only the first 10 chars are
  parsed, so full ISO timestamps work too). For human-phrased dates
  ("last Thursday", "yesterday"), first resolve them via
  ``extract_time_range`` to ISO, then call this.

  Example:

      anchor = "2022-04-15"            # probe-time "today"
      event  = "2022-03-20"            # date pulled from chat
      delta  = days_between(anchor, event)         # → 26
      delta_inc = days_between(anchor, event, True)  # → 27 (counts
                                                     # both endpoints)
      print(f"event was {delta} days before anchor")

  GT for temporal probes often accepts BOTH ``N`` and ``N+1`` days
  (inclusive); state your convention in the final answer.
"""


SYSTEM_PROMPT = (
    _LME_CONTEXT + "\n"
    + _STRATEGY_PROMPT + "\n"
    + _LME_AUTO_LAYOUT + "\n"
    + _NAMESPACE_DOCS
)


# =============================================================================
# Part 2 — SYSTEM_REMINDERS (probe-time; prepended to the user question)
# =============================================================================

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
    For LLM reasoning over evidence: filtering ambiguous
    candidates, span extraction across paragraph-length rows,
    multi-axis comparison/ranking.

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
  - BUDGET: each ``execute_python`` after the evidence is in burns
    one of your remaining turns (``[budget: N turns left]`` is
    shown on each tool_result). The moment stdout shows the literal
    answer — keyword + value in one round, ORDER BY DESC LIMIT 1
    row, or all summands visible — STOP and commit on the next turn.
    Any procedural hints you see are PRIORS, not directives; adopt
    only when their ``when:`` trigger matches your query shape.

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
# without per-probe distillation cost. Loaded once at module init.
_PLAYBOOK_PATH = Path(__file__).resolve().parents[4] / "playbook_longmemeval.md"
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


# ----- Per-qtype forcing templates -----

_MULTI_SESSION_COUNT_TEMPLATE = """\
MULTI-SESSION COUNT — forcing pattern for "how many X" probes:

Single-pass counting from a `SELECT COUNT(*)` is wrong by default
on this benchmark. Off-by-one is the modal multi-session failure:
the agent retrieves SOME of the items, never sees the last one
hiding past the LIMIT, and commits N when GT is N+1. Use this
shape instead, in two cells:

  cell 1 — BROAD RETRIEVAL via FTS5 MATCH (indexed, fast on big
  haystacks). Cast a wide net by SUBJECT, not by the counting noun.
  Use AT LEAST 4-6 OR terms covering synonyms, related nouns, slang,
  and indirect mentions. Examples of MATCH queries:
      "how many tanks":
        'tank OR aquarium OR gallon OR "fish bowl" OR reef'
      "how many weddings":
        'wedding OR married OR ceremony OR vows OR
         reception OR bridesmaid OR groom OR nuptials'
      "how many art events":
        'exhibit OR gallery OR "museum show" OR "art fair"
         OR opening OR vernissage OR "painting workshop"'

    Pattern (JOIN back to chat_turns for the row data the enumerate
    step in cell 2 needs):
      rows = ms.sql_exec(\"\"\"
        SELECT ct.session_idx, ct.session_date_iso, ct.role, ct.content
        FROM chat_turns_fts f
        JOIN chat_turns ct ON ct.rowid = f.rowid
        WHERE chat_turns_fts MATCH ?
        ORDER BY rank LIMIT 100
      \"\"\", ('tank OR aquarium OR gallon OR "fish bowl" OR reef',))

    Print all rows. Off-by-one almost always traces to a keyword
    miss in cell 1, not an enumeration error in cell 2.

  cell 1.5 — DEDUP VIA rlm (REQUIRED when cell 1 returns ≥5 rows):
    Many failures come from counting two mentions of the SAME real-world
    item as two distinct things ("my new bike" + "the road bike I just
    got" = same bike). SQL/Python can't judge same-instance semantically.
    Hand the rows to rlm for a focused dedup pass:

      body = "\\n\\n".join(
          f"[sess {r['session_idx']} | {r['session_date_iso']}] "
          f"U: {r['user_msg'][:300]}"
          for r in rows
      )
      kept = await rlm(
          query=(
              "From the rows below, identify distinct real-world "
              "<item_type> the user actually has/did. Two mentions are "
              "the SAME instance if they describe the same physical/"
              "conceptual thing across sessions (e.g. 'my new bike' "
              "and 'the road bike I just got' = same bike). "
              "Return a numbered list: each distinct item with "
              "(a) one-line description, (b) first-seen session_idx, "
              "(c) which sess rows refer to it. End with 'COUNT = N'."
          ),
          context=body,
      )
      print(kept)

    Why rlm: dedup of "same real-world item across sessions" is a
    semantic-judgment task over paragraph-length rows. rlm runs
    30-60s for ~10 rows — budget for it; this is the primary use
    case. Skip this cell ONLY if you have <5 rows (no dedup needed).

  cell 2 — ENUMERATE + SCOPE + DEDUP + COUNT (in Python, not SQL):
    Print each candidate as ONE line:
      ``[sess N] <one-line identifier>  <kept|dropped: reason>``
    Then apply the scope/dedup rules EXPLICITLY:
      * Temporal scope (e.g. "last month"): use the probe's
        anchor date (most-recent session) and drop items outside
        the window; show the date check per item. Default to
        CALENDAR scope ("the month of March") when the question
        names a month; default to ROLLING 30-day only when the
        question says "the past month".
      * Same item across sessions = ONE item. Dedup by
        normalized identifier (lowercased noun stem), not by row.
      * Question wording matters: "tanks INCLUDING the one I set
        up for my friend's kid" → don't drop the friend-tank;
        "how many MY plants" → drop gifts.
    Print final ``Count = N`` and commit `N` in your answer.

If cell 1 returns < (your gut estimate for N), broaden the keyword
set further — multi-session questions rarely have all evidence in
one session, so missing rows is usually a keyword miss, not absence.
"""


_TEMPORAL_REASONING_TEMPLATE = """\
TEMPORAL-REASONING — never compute dates in your head:

LLM calendar arithmetic is unreliable past 2-week deltas — modal
failure on this qtype is "March 20 → April 15" answered as 26
days when GT is 21. Use the ``days_between`` helper EVERY time;
do not commit a delta computed in prose.

Mandatory shape:
  cell 1 — RESOLVE EACH ANCHOR to an ISO date:
    For each date phrase in the question (and each candidate event
    in the chat), get an ISO date. Either:
      - Use ``session_date_iso`` columns from the retrieved rows, or
      - Call ``rng = await extract_time_range("<phrase>")`` for
        relative phrases ("last Thursday", "two months ago").
    Print each anchor as ``label = "YYYY-MM-DD"`` before computing.

  cell 2 — COMPUTE WITH ``days_between``, NOT IN HEAD:
    For "how many days between A and B":
        delta = days_between(a_iso, b_iso)
        delta_inc = days_between(a_iso, b_iso, inclusive=True)
        print(f"delta={delta} (inclusive={delta_inc})")
    Commit the value from the helper. State the inclusive
    convention in your answer ("21 days, or 22 if counting both
    endpoints").

For "what is the order of N events":
  Build ``[(date_iso, label), ...]``, sort by date_iso in Python,
  print the sorted list before answering. Never claim an ordering
  without printing the sorted list — manual ordering of 4+ events
  fails reliably.

  When N ≥ 3 events, ALSO verify the order via rlm before answering.
  Sub-LLM sorting catches "I scanned wrong" mistakes the agent's own
  sort doesn't (the agent often returns rows in stdout-order, not
  date-order, even after the sort cell). Pattern:

      body = "\\n".join(f"{d}: {l}" for d, l in sorted_events)
      verified = await rlm(
          query=(
              "Sort these events strictly by date ascending (earliest "
              "first). Return as a numbered list, restating each "
              "date_iso. If two events share a date, keep the input "
              "order between them."
          ),
          context=body,
      )
      print(verified)

  Commit the rlm-verified ordering. Why rlm: ordering 3+ items with
  distractor text is a focused task — rlm runs ~30s for ~10 items
  and reliably outputs the sorted list. Skip this only for N < 3.

For "how many months / weeks since":
  Convert days_between output to the right unit explicitly
  (e.g. ``months = days // 30``); don't mix units in head.

Anti-patterns:
  - Stating a delta without showing the days_between call.
  - "X happened about N weeks ago" without a printed computation.
  - Sorting 4+ events without an explicit sorted-list print.
"""


_KNOWLEDGE_UPDATE_TEMPLATE = """\
KNOWLEDGE-UPDATE — DO NOT COMMIT ON ONE CELL.

The modal failure here is the agent committing on a 1-cell hit:
sees the first stated value, ignores that the user revised it
later. ``ORDER BY session_idx DESC`` in SQL is NECESSARY but NOT
SUFFICIENT — your Python loop naturally prints/commits in
seen-order, which biases toward whichever row you read first.
You MUST run TWO cells and EXPLICITLY compare values across rows
before committing.

Mandatory shape (do not skip cell 2):
  cell 1 — BROAD RETRIEVAL across all mentions:
    rows = ms.sql_exec(\"\"\"
        SELECT session_idx, session_date_iso, user_msg
        FROM rounds
        WHERE user_msg LIKE '%<subject>%'
           OR user_msg LIKE '%<synonym>%'
        ORDER BY session_idx DESC LIMIT 10
    \"\"\")
    Print all 10 rows. If <2 hits, broaden the keyword set
    (synonyms, related noun, slang) before giving up.

  cell 2 — EXTRACT + COMPARE + COMMIT BY QUESTION QUALIFIER:
    Print one line per (session_idx, value-bearing snippet):
        [sess 32] "three times a week"
        [sess 12] "twice a week"
        [sess  4] "twice a week"

    Then pick by the question's TEMPORAL QUALIFIER:

      * Default ("what is my X" / "how often"): commit the value
        from the HIGHEST session_idx (latest assertion supersedes
        earlier ones). State "latest from session N, supersedes M".

      * "initially" / "first" / "originally" / "when I started":
        commit the value from the LOWEST session_idx (original
        assertion). NEVER commit a later/revised value when the
        question asks for the original.

      * "how long have I been Xing": this implies duration from
        FIRST mention to probe-anchor date. Pull the earliest
        mention's date and compute with days_between against the
        probe-anchor (latest session_date_iso).

      * If only one row but the question implies an updatable
        value → re-broaden the keyword and re-query before committing.

Anti-patterns:
  - Committing in 1 cell. 1-cell commits are the leading failure
    here; the template requires 2 cells, period.
  - SELECT ... LIMIT 1 without ORDER BY DESC.
  - Ignoring "initially" / "first" qualifiers and committing the
    latest value anyway.
  - "I don't have that information" in the same answer that lists
    concrete facts about the subject — commit the facts.
"""


# ----- Synthesis blocks (mem0 parity, fire on every probe) -----
# The two blocks below are LME-tuned answer-synthesis guidance, ported and
# adapted from mem0's published LongMemEval prompts:
#   https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/longmemeval/prompts.py
#
# Intentionally separated from the codegen-side prompt (``_LME_PROBE_HINT``
# + qtype templates) so the SCROLL retrieval framework stays general:
#   - Codegen prompt teaches the agent how to write SQL / call helpers /
#     pace itself — generalizes to other memory benchmarks.
#   - SYNTHESIS_RULES = memory-benchmark heuristics that apply broadly
#     (recency wins, scan-all-rows-twice, etc.) — semi-general.
#   - DOMAIN_EQUIVALENCES = pure LME-specific quirks (chandelier=jewelry,
#     scratch grains = new layer feed, etc.) — clearly labeled overfit,
#     for paper-comparable parity with mem0 / Letta / Memori baselines.
#
# Both fire at probe time, after qtype templates and procedural_hints,
# right before the commit rule re-anchor — so they're the last
# substantive guidance the model reads before drafting its answer.

ANSWER_GENERATION_PROMPT = """\
FINAL ANSWER SYNTHESIS — rules of evidence at commit time:

1. ALWAYS TRY TO ANSWER. Topic appears anywhere in retrieved rows
   → answer with what you have; don't refuse for one missing detail.

2. MOST RECENT WINS. Same fact, multiple values → use the row at
   HIGHEST session_idx. For current counts/scores/status the latest
   value REPLACES earlier ones (don't sum/average). Two numbers on
   the same date ("1,250" and "close to 1,300") → prefer HIGHER.
   Memories about different people/contexts are NOT conflicting.

3. TIME-BOUNDED QUESTIONS. Compute the INCLUSIVE date window first
   (use ``days_between``), then check every candidate's date. "Last
   weekend" may mean up to 10 days ago. "Last month" includes
   current month-to-date + previous month. If strict window yields
   nothing, check the preceding period before abstaining.

4. TEMPORAL REFERENCE POINTS. "How many days ago did X when Y
   happened" = interval X→Y, NOT X→today.

5. COUNTING. Scan ALL rows first to last; build a numbered list
   with date + position. **Do a SECOND scan after enumerate** —
   items at stdout positions 30-200 are commonly missed. Multiple
   items in one row count separately. Count + "added X items" on
   the SAME date → count already includes them.

6. USE ONLY THE ROWS. Don't invent numbers, prices, or addresses.

7. WHEN TO ABSTAIN. Abstain only when (a) topic is genuinely
   unmentioned, or (b) the question uses a WRONG role/title/entity
   that doesn't match retrieved rows (e.g., "Sales Manager" but
   rows say "Senior Sales Engineer" — different things, abstain
   rather than substitute). Keyword-scan ALL rows before abstaining.

8. YES/NO + COMPARISON. "Did I ever do X" with no matching row →
   "No." For comparisons, if one side is missing → abstain on the
   COMPARISON (not the side you have).

9. ACTIONS vs INTENTIONS. Use execution date, not plan date.
   "Decided to" / "took X for servicing" = action initiated.
   Future-tense only ("plans to", "will") = plan. A plan with a
   specific date + no later contradiction = assume completed.

10. USER FACTS > ASSISTANT ADVICE. "User:" lines are actual
    experience; "Assistant:" lines are generic advice. Don't
    convert currencies unless the user stated the conversion.

11. CROSS-TOPIC FACTS. Facts for one computation are often in
    unrelated conversations (age in travel chat + relative's age in
    birthday chat). Search each fact independently.

12. PERSONALIZATION. For recommendations: scan ALL rows for tools
    the user OWNS (apps, cards, gear) — mention them, not just the
    most obvious one. Check every suggestion against stated dislikes.

13. REASONABLE DEDUCTION. Similar items in the same sentence default
    to same type ("ate lunch, which was my third meal with chicken
    fajitas" → other meals were also lunches). No name but a
    description → answer with the description.

14. CONTRADICTION. Direct contradiction (not an update) → trust the
    row at higher session_idx. Same day → trust the later turn.

15. CONTEXT CHECK. Before using a row's value, verify it applies to the
    SAME context as the question. "Wake-up while traveling" ≠ "weekday
    wake-up"; "general daily schedule" ≠ "specific weekday schedule".
    Prefer the more specific context-matching row over a generic one.

ROW GROUPING. Same session_date_iso = same conversation. Past-tense
event verbs ("attended", "went to", "just got back from",
"completed") = happened on/near that date.

# Misc Rules
- Count class projects too when asked about users' projects. Class projects = projects.
- Most old (Eg. ancestral, vintage, heritage) items count as antiques too!
- If you don't have chords for a song (but have notes), output the notes. Song notes count as chord progressions.
- Starting a *diorama project* (eg. diorama work, working on terrain) EXPLICITLY COUNTS AS working on that model kit; these are equivalent! Always count such items.
- Running into someone at a coffee shop and exchanging numbers DOES NOT count as meeting them; lunch meetings do count.
- Potlucks/feasts/birthday parties count as dinner parties (BBQ doesn't).
- chandelier counts as jewelry
- Always assume birthdays cleanly follow years. Ie. User was 22 in 2022; they will be 23 in 2023.
- "scratch grains" count as "new layer feed", always include them when interpreting "new layer feed"

"""



# =============================================================================
# Out-of-band — post-probe distillation (system-side; agent never sees this)
# =============================================================================

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


# =============================================================================
# Part 3 — auto-appended messages (postscripts; appended AFTER the question)
# =============================================================================
#
# Appended AFTER the question text inside ``inject_probe``. Sits in the
# same user-turn message as the question itself. Composed by
# ``tasks/probes.py::compose_user_postscript`` which layers the
# universal postscript + qtype nudge (or abstention override).

LME_PROBE_USER_POSTSCRIPT = (
    "Write Python cells to query, compute, or call ``rlm`` — "
    "whatever combination you need to gather and combine "
    "evidence. Commit by replying with a plain-text answer when you "
    "have enough. If the search genuinely returns nothing, abstain "
    "explicitly: \"I don't have that information from our "
    "conversations\" — the judge scores that correct."
)


# Per-question-type nudges, layered on top of LME_PROBE_USER_POSTSCRIPT.
# These mirror the dimensions the paper identifies as separately
# tested (TR, KU, IE-preference) and lift the abstention judge's
# explicit-refusal requirement up to the user-turn message so the
# agent can see it without re-reading the system prompt.
_QTYPE_POSTSCRIPT: dict[str, str] = {
    "temporal-reasoning": (
        "Time-sensitive question. Pattern: extract a date range from "
        "the question first, filter sessions by that range "
        "(``session_ts_iso``) BEFORE reading content — don't keyword-"
        "scan the whole haystack. Off-by-one on day/week/month is not "
        "penalized."
    ),
    "knowledge-update": (
        "Recency-sensitive question. The user may have stated multiple "
        "values over time; the MOST RECENT statement is the current "
        "truth. Pattern: order matching sessions by ``session_ts_iso`` "
        "descending and take the latest value. Mentioning prior values "
        "is fine, but do not state an out-of-date value as current."
    ),
    "single-session-preference": (
        "Preference question. The literal subject of the question may "
        "never have been discussed verbatim — that is the point. Exact "
        "keyword / SQL match will miss; use semantic retrieval to "
        "surface related preferences (likes, dislikes, constraints, "
        "recurring interests) and ground the recommendation in those. "
        "\"I have no information about <topic>\" is the wrong frame "
        "here."
    ),
}


# Abstention applies to ANY qtype when the question_id has the `_abs`
# suffix (per dataset convention). It overrides the qtype postscript.
_ABSTENTION_POSTSCRIPT = (
    "This may be an unanswerable question — the user may never have "
    "stated the relevant information. After ≥2 distinct retrieval "
    "queries (different keywords AND different surfaces) return "
    "nothing useful, abstain EXPLICITLY using the phrasing: "
    "\"I don't have that information from our conversations.\" The "
    "judge requires explicit refusal to score abstention correct."
)
