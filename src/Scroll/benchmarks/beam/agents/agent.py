"""BEAM SCROLL agent.

Applies :class:`Scroll.core.ScrollAgent` to BEAM: each batch is ingested
into a SQLite + vector memoryspace (same schema and regex extractors as
LongMemEval — both are chat-memory benchmarks); at probe time the
agent runs Python via an ``execute_python`` tool against a persistent
REPL with ``ms`` / ``log`` / ``rlm`` bound.

Differences from :class:`Scroll.benchmarks.longmemeval.agents.agent.LongMemEvalAgent`:
- No procedural_hints / L3 distillation (BEAM is single-pass; no
  cross-task store to write to).
- No qtype-conditional probe-hint composition (BEAM uses one base hint;
  per-category nudges are added by the env's ``probe_user_postscript``).
- 20 probes fire on the probe day (one per probing question), not 1.
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
from Scroll.benchmarks.beam.ingestor import (
    BeamIngestor,
    ensure_schema as _beam_ensure_schema,
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


_log = logging.getLogger(__name__)


# Memoryspace schema lives in :mod:`Scroll.benchmarks.beam.ingestor` —
# identical shape to LongMemEval (both are chat-memory benchmarks).


# ---------------------------------------------------------------------------
# Probe-time tool schema (identical to LongMemEval — same retrieval shape).
# ---------------------------------------------------------------------------

_EXECUTE_PYTHON_TOOL_NAME = "execute_python"

_EXECUTE_PYTHON_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": _EXECUTE_PYTHON_TOOL_NAME,
        "description": (
            "Execute a Python snippet in the persistent REPL to answer "
            "the probe. Use it to query ``ms`` / ``log`` and call "
            "``await rlm(...)``. REPL globals persist across calls. "
            "Pass ONLY Python source as ``code`` — no markdown fences, "
            "no prose."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python source. Use ``print(x)`` to surface values."
                    ),
                },
            },
            "required": ["code"],
        },
    },
}


# ---------------------------------------------------------------------------
# Prompt text — BEAM-specific framing.
# ---------------------------------------------------------------------------

_BEAM_CONTEXT = """\
You are an autonomous chat assistant evaluated on long-term memory
over a single long conversation (BEAM: 100K-10M tokens), structured
as N time-anchored batches:

- Days 1..N: each day delivers ONE past chat batch (with time anchor).
- Day N+1: A sequence of probing questions fires across 10 memory
    ability categories. Answer each on its own terms; abstain
    explicitly when the topic was never discussed.
"""


_BEAM_PROBE_HINT = """\
LOOP shape per probe — each turn is EITHER a tool call OR a final
answer:
  - To keep retrieving: emit an ``execute_python`` tool call with
    Python source as the ``code`` argument. ``print(x)`` to surface
    values; bare ``x`` is a no-op. The handler runs the cell and
    feeds stdout back as the next tool_result.
  - To finish: emit a plain-text answer in ``content`` and NO tool
    call. The handler treats "no tool call" as the termination
    signal. Don't wrap the answer in ``print()`` or
    ``execute_python``; don't carry a tool call alongside.

The notes below are awareness about failure modes and tool semantics,
not a script.

──────────────────────────────────────────────────────────────────
PROBE INTENT — what the question wants:

  INFORMATION_EXTRACTION / MULTI_SESSION_REASONING — retrieve a
    literal value the user stated (single-batch or across batches).
    Use ms.sql_exec / log.semantic_search; compose across batches
    in Python or via rlm.

  KNOWLEDGE_UPDATE — the user revised a value over time. The MOST
    RECENT statement wins. Sort by ``session_idx DESC LIMIT 1`` or
    by ``session_ts_iso DESC``.

  TEMPORAL_REASONING / EVENT_ORDERING — extract the date scope from
    the question (each batch has a time_anchor stored as
    session_date_iso); filter by date BEFORE reading content.

  ABSTENTION — the question is unanswerable from the chat. After
    ≥2 retrieval attempts (different keywords AND different surfaces)
    return nothing, abstain with the exact pattern:
    "Based on the provided chat, there is no information related to
    <topic>."

  PREFERENCE_FOLLOWING / INSTRUCTION_FOLLOWING — ground answers in
    the user's explicit preferences / instructions (queryable via
    user_preferences or chat_turns LIKE on 'I prefer / I'd rather
    / my favorite / I avoid').

  CONTRADICTION_RESOLUTION — pull BOTH conflicting statements
    (cite their batches), then commit to whichever the user
    explicitly resolved or most recently stated.

  SUMMARIZATION — synthesize across the whole chat. Cite the
    batches you draw from.

──────────────────────────────────────────────────────────────────
PRIMITIVES — pick by what the answer demands:

  ``ms.sql_exec`` / ``log.findall`` — sub-second deterministic
    match (verbatim phrase, distinctive noun, regex pattern).
  ``ms.vector_query`` / ``log.semantic_search`` — paraphrase
    recall when the noun appears in different phrasings.
  ``rlm(query, context)`` — sub-agent with its own REPL + batched
    sub-LM calls. For genuine LLM reasoning over evidence:
    span extraction, multi-axis ranking, dedup.

Chain primitives in one cell when steps don't depend on prior
output. Split only when the next QUERY needs the prior result.

──────────────────────────────────────────────────────────────────
WATCH-OUTS:

  - Stdout cap ~4K chars (head 3K + tail 1K). Long rows overflow
    it; truncate per-row preview or LIMIT broad queries.
  - Counting events across batches: same real-world event mentioned
    in two batches = ONE event; dedup before reporting.
  - Time anchors are stored on every turn (``session_date_iso``,
    ``session_ts_iso``). Use them; do NOT rely on the model's
    internal date knowledge.

──────────────────────────────────────────────────────────────────
FINAL ANSWER — plain text in ``content``, NO ``execute_python``
tool call. Never wrap the answer in ``print()`` or
``execute_python``. The handler treats "no tool call" as the
termination signal, so the final turn must carry only ``content``.

  Length: 1-4 sentences for most categories; 4-8 short bullets for
    summarization. Cite batch_idx inline only when disambiguating.
  ABSTENTION exact pattern: "Based on the provided chat, there is
    no information related to <topic>."
"""


_STRATEGY_PROMPT = """\
The harness auto-ingests every chat batch into ``memoryspace`` before
the probe day fires. At probe time you have ONE tool —
``execute_python(code: str)`` — that runs Python in a persistent
REPL with ``ms`` / ``log`` / ``rlm`` already bound (full API in REPL
GLOBALS section below).

Each turn is EITHER a tool call (still retrieving) OR a plain-text
``content`` answer (done). Don't carry both at once. The ``code``
argument takes ONLY Python source — no markdown fences, no
multi-line reasoning chains as comments. If your model has a
native thinking channel it will plan there; otherwise a short
``content`` string before the tool call is fine.

Memoryspace is READ-ONLY (writes raise PermissionError); REPL globals
persist across calls within one probe (but are reset between probes
— don't expect variables from probe 1 to be visible in probe 2).
"""


_BEAM_AUTO_LAYOUT = """\
AUTO-BUILT MEMORYSPACE (READ-ONLY). All tables regex-populated at
ingest. Types + format conventions matter — get them wrong and
your SQL silently returns 0 rows.

KEY NAMING — ``session_idx`` is the 1-based BATCH INDEX (BEAM
batches are time-anchored chunks of the conversation). The calendar
date / time_anchor lives in ``session_date_iso`` / ``session_ts_iso``.

  rounds(                                     -- ⭐ DEFAULT SURFACE
      session_idx      INTEGER,
      round_idx        INTEGER,
      session_date_iso TEXT,                  -- batch's time_anchor
      user_msg         TEXT,
      assistant_msg    TEXT)
      ⭐ Each row = one full user+assistant exchange. Search by
      default; single-row recall carries both sides of context.

  chat_turns(session_idx, turn_idx, role, content,
             session_id, session_date_iso, session_ts_iso)
      PRIMARY KEY (session_idx, turn_idx)

  user_preferences(session_idx, turn_idx, polarity, target, sentence)
      -- 'like' | 'dislike' | 'acquired' polarity.

  event_dates(session_idx, turn_idx, date_text, date_iso,
              relative_offset_days, sentence)

  facts(session_idx, turn_idx, fact_type, fact_text, sentence)
      -- fact_type ∈ {possession, plan, attribute, change, current_state}.

  sessions(session_idx, session_id, session_date_iso,
           session_ts_iso, session_text)        -- whole-batch transcript

VECTORS — ``ms.vector_query(text, top_k=5)`` returns 3-tuples
``(key, text, score)``. Keys: "sess{N}_turn{K}_{role}" or
"sess{N}_session" — embedded text is K = V + regex-extracted facts.
"""


_NAMESPACE_DOCS = """\
REPL GLOBALS available inside ``execute_python(code=...)``:

ms — memoryspace handle (read-only):
  ms.sql_exec(statement, params=None) -> list[dict] | None
      SELECT / PRAGMA / EXPLAIN / WITH only.
      params MUST be a tuple (trailing comma for single param).
  ms.vector_query(query, top_k=5) -> list[tuple[key, text, score]]
  ms.json_read(key) -> Any         (KeyError if missing)
  ms.json_list() -> list[str]
  ms.file_read(name) -> str
  ms.file_list() -> dict[str, int]
  ms.schema_inspect() -> dict

log — LogHandle (read-only view of E):
  log.slice(day=None, role=None, kind=None) -> list[LogEntry]
  log.range_by_day(start, end) -> list[LogEntry]   # inclusive
  log.search(query, k=20) -> list[LogEntry]        # substring
  log.semantic_search(query, k=10, *, day=None, role=None,
                      kind=None, min_score=0.0)
                       -> list[(LogEntry, float)]   # BoW cosine
  log.findall(pattern, entries=None, *, flags=0) -> list
  log.text(entries) -> str

rlm(query: str, *, context: str = "") -> str   (async — MUST `await`)
  Recursive sub-agent (dspy.RLM). Use for span extraction, ranking,
  synthesis over a body of evidence rows. Pass evidence as
  ``context=``, the ask as ``query=``. The sub-agent has NO access
  to ms/log/outer rlm — everything it needs must be in ``context``.

extract_time_range(question: str) -> dict | None   (async; `await`)
  Returns {"start", "end"} ISO dates or None.

wait_for_next_day()
  Ends day; auto-advanced for BEAM, rarely needed in practice.
"""


BASE_PROMPT = (
    _BEAM_CONTEXT + "\n"
    + _STRATEGY_PROMPT
    + "\n" + _BEAM_AUTO_LAYOUT
)


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class BeamAgent(ScrollAgent):
    """SCROLL agent for BEAM.

    Read-only memoryspace (the harness ingests every batch; the agent
    only queries). dspy.RLM exposed for semantic operations over
    candidate rows. No procedural_hints / cross-task store — BEAM
    runs are single-pass.
    """

    # SCROLL feature flags
    readonly_memoryspace = True
    expose_rlm = True

    sys_prompt = BASE_PROMPT
    probe_max_iters = 10

    def __init__(self, cfg, storage, data) -> None:
        # Probe-time bookkeeping.
        self._active_probe = None
        self._oneshot = None
        super().__init__(cfg, storage, data)

    # ----- SCROLL hooks -----

    ingestor_cls = BeamIngestor

    def _ensure_schema(self, memoryspace) -> None:
        _beam_ensure_schema(memoryspace)

    def extra_namespace(self) -> dict:
        # System-side one-shot LM (NOT exposed to the agent). Feeds
        # ``extract_time_range`` only. Direct ``self._model`` call —
        # no log mirror (the harness doesn't query these back).
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

    def _probe_sys_prompt(self) -> str:
        # Same as LongMemEval — drop the base ``PROBE_SUBSTRATE_PROMPT``
        # prelude (built for fenced-block substrate; we're using
        # tool-calls).
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

    # ----- Probe loop (tool-call format, identical to LongMemEval) -----

    async def _call_probe_model(self, msgs: list):
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
                        "BeamAgent probe model quota error (attempt %d/3): "
                        "%.150s — backing off %ds",
                        attempt, e, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
        raise last_err  # type: ignore[misc]

    async def _answer_probe_inner(self, max_iters: int) -> str:
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
                break

            # Diagnostic: the prompt asks for EITHER content OR tool_calls
            # each turn. Track turns where both showed up — reasoning
            # models that natively think shouldn't also dump prose into
            # content alongside a tool call. Debug-only; the handler
            # still keeps both for trace fidelity.
            if text and tool_use_blocks:
                _log.debug(
                    "BeamAgent probe iter %d emitted content (%d chars) "
                    "AND %d tool call(s) — prompt expects either-or",
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
                            "Make at least ONE ``execute_python`` "
                            "call now: ``ms.sql_exec(...)`` or "
                            "``log.semantic_search(...)`` using keywords "
                            "from the probe question. If the query "
                            "genuinely returns nothing relevant, you may "
                            "abstain in your NEXT response based on what "
                            "you observed."
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
                        "content": "[error: execute_python called with empty code.]",
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
                    code[:2900] + "\n\n# ... [truncated; see cell.code_full]"
                )
                with _tracer.start_as_current_span(span_name) as span:
                    span.set_attributes({
                        "tool.name": _EXECUTE_PYTHON_TOOL_NAME,
                        "cell.day": self._current_session,
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
                        output_parts.append("[exception]\n" + result.exception)
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

                turns_left = max_iters - (it + 1)
                probe_q = (
                    getattr(self._active_probe, "question", "") or ""
                ).strip()
                m_q = re.search(r"\[([^\]]+)\]\s*(.+)", probe_q, re.DOTALL)
                if m_q:
                    cat = m_q.group(1)
                    q_only = m_q.group(2).strip()[:300]
                    question_anchor = f"\n[{cat} | QUESTION: {q_only}]"
                else:
                    question_anchor = f"\n[QUESTION: {probe_q[:300]}]" if probe_q else ""
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

        if last_stdout and not _ANSWER_LINE_RE.search(last_text):
            return last_text + "\n\n[stdout]\n" + last_stdout
        return last_text

    # ----- Probe-time hint composition -----

    def probe_user_hint(self, probe=None) -> str:
        """Base probe hint. Per-category nudge is injected by the env's
        ``probe_user_postscript`` (which reads the last-injected probe's
        category from :mod:`Scroll.benchmarks.beam.tasks.probes`).
        """
        self._active_probe = probe
        return _BEAM_PROBE_HINT

    # ----- Auto-ingest -----

    def run_session(self, env) -> list[str]:
        """Mirror the env's chat session into ``E``. The attached
        :class:`BeamIngestor` will catch up on the next ``ms`` read; we
        trigger it eagerly here so any same-day probe sees fresh W.
        """
        write_chat_turn_entries(self.log, env, env.session_idx + 1)
        try:
            self.memoryspace._maybe_catch_up()
        except Exception:  # noqa: BLE001
            _log.warning(
                "BeamAgent ingest catch-up failed (day=%s)",
                env.session_idx + 1, exc_info=True,
            )
        return []
