"""LongMemEval SCROLL agent.

The agent applies the SCROLL harness (:class:`Scroll.core.ScrollAgent`)
to LongMemEval:

- ``E`` is the conversation log; the harness mirrors each haystack
  session into it as ``chat_turn`` entries.
- ``W`` is a SQLite + vector memoryspace with two tables
  (``chat_turns``, ``sessions``) plus a ``rounds`` view. The agent's
  view is read-only (the experimental contract: harness ingests,
  agent only queries). BEAM ships a richer 5-table variant; LME
  shipped the same originally but the v4 audit showed >98% of
  agent queries hit ``rounds``/``chat_turns`` only, so the typed
  extraction layer is disabled here.
- At probe time the agent runs Python through an ``execute_python``
  tool call against a persistent REPL with ``ms`` / ``log`` / ``rlm``
  / ``extract_time_range`` bound.

Procedural memory (``procedural_hints``) is the LME-specific extra:
after the judge scores each probe, ``_on_probe_complete`` distills a
1-3 hint case study via a one-shot sub-LM and writes it to the
per-task memoryspace. Subsequent probes of the same ``question_type``
in the same task see the top-K most-relevant hints injected at the
top of the probe hint.
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
    split_response_into_text_and_tools,
)
from Scroll.core._codeact_agent import EXECUTE_PYTHON_TOOL_NAME
from Scroll.tools.chat_memory import (
    handle_only_body,
    make_time_range_extractor,
    probe_session_body,
)
from Scroll.tools.chat_memory import (
    make_chat_memory_namespace,
    write_chat_turn_entries,
)
from Scroll.benchmarks.longmemeval.ingestor import (
    LMEIngestor,
    ensure_schema as _lme_ensure_schema,
)


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt text lives in ``prompts.py`` (see that file's module docstring for
# the assembly pipeline). Re-imported here so the agent class can reference
# them as bare names without a prefix.
# ---------------------------------------------------------------------------

from Scroll.benchmarks.longmemeval.agents.prompts import (  # noqa: E402
    ANSWER_GENERATION_PROMPT,
    SYSTEM_PROMPT,
    _DISTILL_PROMPT,
    _KNOWLEDGE_UPDATE_TEMPLATE,
    _LME_PROBE_HINT,
    _MULTI_SESSION_COUNT_TEMPLATE,
    _PLAYBOOK_BLOCK,
    _TEMPORAL_REASONING_TEMPLATE,
)


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




# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class LongMemEvalAgent(ScrollAgent):
    """SCROLL agent for LongMemEval.

    Read-only memoryspace + dspy.RLM sub-agent + procedural memory
    (``procedural_hints`` distilled post-probe and injected into
    future probes of the same ``question_type`` within the same task).
    """

    # SCROLL feature flag
    expose_rlm = True

    sys_prompt = SYSTEM_PROMPT
    probe_max_iters = 10

    procedural_hints_cap_total = 500
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
        # Per-task counter — incremented each time L3 distillation
        # appends new hints. Surfaced to the orchestrator via
        # ``augment_efficiency`` so the run-level summary's
        # ``total_lessons_written`` is non-zero.
        self._lessons_written = 0
        super().__init__(cfg, storage, data)

    def augment_efficiency(self, efficiency: dict) -> None:
        efficiency["lessons_count"] = int(self._lessons_written)

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

        # Pure-Python date arithmetic so the agent doesn't compute
        # "how many days between X and Y" in its head. Eval-set
        # comparison showed temporal-reasoning failures cluster on
        # LLM date math errors (e.g. "March 20 → April 15" answered
        # as 26 days instead of 21). Accepts ISO ("YYYY-MM-DD") or
        # the common 4-letter / numeric variants ``date.fromisoformat``
        # parses.
        def _days_between(d1: str, d2: str, inclusive: bool = False) -> int:
            from datetime import date as _date
            try:
                a = _date.fromisoformat(str(d1).strip()[:10])
                b = _date.fromisoformat(str(d2).strip()[:10])
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"days_between: expected ISO YYYY-MM-DD inputs, "
                    f"got d1={d1!r} d2={d2!r} ({exc})"
                ) from exc
            delta = abs((b - a).days)
            return delta + 1 if inclusive else delta

        return {
            "extract_time_range": make_time_range_extractor(
                self._oneshot, today_iso_provider=self._latest_session_iso,
            ),
            "days_between": _days_between,
        }

    def _base_namespace(self) -> dict:
        return make_chat_memory_namespace(self._tool_state, agent=self)

    # NOTE: REPL globals docs are inlined into SYSTEM_PROMPT (Part 1), so we
    # don't override ``namespace_docs()`` — the base class returns "" and
    # ``_probe_sys_prompt`` simply skips it.

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
    # In this tool-call substrate the loop shape is structural
    # (no-tool-call response = final answer), so no separate "probe
    # mode" prelude is needed. We just splice in any env-supplied
    # probe-format hints (rare these days; ride on the user-msg
    # postscript instead) plus the regular sys_prompt + extras.

    def _probe_sys_prompt(self) -> str:
        parts: list[str] = []
        env_probe = self._env_probe_prompt()
        if env_probe:
            parts.append(env_probe.strip())
        parts.append(self.sys_prompt.strip())
        extra = self.extra_system_prompt().strip()
        if extra:
            parts.append(extra)
        return "\n\n".join(p for p in parts if p)

    # ----- Probe loop — OpenAI tool-call format -----

    async def _answer_probe_inner(self, max_iters: int) -> str:
        """LME probe loop. Inherited base ``_call_model`` already passes
        ``tools=[EXECUTE_PYTHON_TOOL_SCHEMA]`` (from ``CodeActAgent``);
        this override adds the LME-specific failure-mode handling:
        abstention rejection, evidence-anchored grace turn, stdout
        fallback.
        """
        last_text = ""
        last_stdout = ""
        cells_executed = 0
        zero_cell_abstain_forced = False

        for it in range(max_iters):
            response = await self._call_model(self._history)

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
                        "name": block.get("name") or EXECUTE_PYTHON_TOOL_NAME,
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
                        "tool.name": EXECUTE_PYTHON_TOOL_NAME,
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

        # Budget-exhaustion grace turn. ~45% of pre-fix failures were
        # "agent kept emitting tool calls for all 10 iters and never
        # committed a natural-language answer" — the safety net then
        # returned bare stdout which the judge can't score. Give one
        # more LM call with tools disabled + a stern commit prompt;
        # this rescues most of those.
        if not last_text.strip():
            # Pull the last few tool-result stdouts into the grace prompt
            # so the model anchors on retrieved evidence instead of
            # defaulting to "I don't have that information". The
            # ``wrongly_abstained`` mode in the failure report shows the
            # model often abstains in the grace turn even when earlier
            # cells surfaced relevant rows.
            evidence_snippets: list[str] = []
            for hm in reversed(self._history):
                if hm.get("role") != "tool":
                    continue
                content = hm.get("content") or ""
                if not isinstance(content, str) or not content.strip():
                    continue
                evidence_snippets.append(content[:600].strip())
                if len(evidence_snippets) >= 3:
                    break
            if evidence_snippets:
                evidence_block = (
                    "Your prior cells surfaced this evidence (most "
                    "recent first):\n\n"
                    + "\n\n---\n\n".join(evidence_snippets)
                    + "\n\n"
                )
            else:
                evidence_block = ""
            self._history.append({
                "role": "user",
                "content": (
                    "[BUDGET EXHAUSTED — no more tool calls.] Reply NOW "
                    "with your best-grounded plain-text answer. Any tool "
                    "call in this turn will be dropped.\n\n"
                    f"{evidence_block}"
                    "If the evidence above mentions the subject of the "
                    "question, COMMIT a value derived from it — even "
                    "low-confidence answers grounded in retrieved rows "
                    "score better than abstention here. Only abstain "
                    "with \"I don't have that information from our "
                    "conversations.\" when the evidence is truly silent "
                    "on the subject."
                ),
            })
            try:
                grace = await self._model(
                    self._history, tools=None, tool_choice="none",
                )
                grace_text = _extract_text_from_response(grace).strip()
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "LongMemEvalAgent probe grace-commit turn failed: %s",
                    exc,
                )
                grace_text = ""
            if grace_text:
                self._history.append(
                    {"role": "assistant", "content": grace_text}
                )
                last_text = grace_text

        # stdout fallback safety net. If the agent still gave us nothing
        # (grace turn also failed), prefer an explicit abstention over
        # a raw stdout dump — the judge can score the former; the
        # latter scores 0.
        if not last_text.strip():
            return (
                "I don't have that information from our conversations."
            )
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
        # qtype-specific template comes BEFORE the base hint so it's
        # the first non-system content the model sees — KU and
        # temporal-reasoning failures cluster on 1-2 cell early commits
        # where the model ignored later-appearing template guidance.
        parts: list[str] = []
        qtype_block = self._format_qtype_specific_hint(qtype, probe_question)
        if qtype_block:
            parts.append(qtype_block)
        parts.append(base)
        hints_block = self._format_procedural_hints(qtype, probe_question)
        if hints_block:
            parts.append(hints_block)
            # NOTE: commit-rule re-anchor was previously a separate
            # _POST_HINTS_REMINDER block here. Folded into _LME_PROBE_HINT's
            # WATCH-OUTS / BUDGET item — fires unconditionally now.
        # Benchmark-tuned synthesis guidance (mem0 parity). Both fire
        # on every probe — see header above the constants for the
        # codegen-vs-synthesis split rationale.
        parts.append(ANSWER_GENERATION_PROMPT)
        # NOTE: the probe time anchor moved to ``probe_user_question_prefix``
        # so it wraps the question directly ("Today's Date: ... / Question: ...")
        # instead of trailing at the end of this hint block.
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

    # Triggers the multi-session enumerate-then-count template.
    # Matches "how many", "how often", "number of", "total ... I X-ed",
    # and bare "count(ed/ing)?" — covers all multi-session counting
    # questions in LME-s without false-positive matching on the verb
    # ``count`` inside answers.
    _COUNT_QUESTION_RE = re.compile(
        r"(?i)\b(how many|how often|number of|count(?:ed|ing)?|"
        r"total (?:number|count)|how much)\b"
    )

    def _format_qtype_specific_hint(
        self, qtype: str | None, probe_question: str
    ) -> str:
        """Per-qtype forcing templates for known failure modes.

        single-session-user / single-session-assistant fall through
        (already near-ceiling at 0.96-1.00 acc, no template helped).
        single-session-preference also falls through: the v2 500-QA
        run measured a preference template at -10pp (over-search,
        recommendation abandonment when no perfect quote existed).
        """
        if not qtype:
            return ""
        if qtype == "multi-session" and self._COUNT_QUESTION_RE.search(
            probe_question or ""
        ):
            return _MULTI_SESSION_COUNT_TEMPLATE
        if qtype == "knowledge-update":
            return _KNOWLEDGE_UPDATE_TEMPLATE
        if qtype == "temporal-reasoning":
            return _TEMPORAL_REASONING_TEMPLATE
        return ""

    # NOTE: no ``probe_user_question_prefix`` override — the dataset's
    # ``probe.question`` already has its own ``"As of YYYY/MM/DD (Day) HH:MM,
    # please answer..."`` preamble (assembled in ``tasks/probes.py``).
    # Adding a separate ``Today's Date: ...`` wrapper duplicated the date
    # and lost the (Day) + HH:MM precision, which TR probes need for date
    # arithmetic. The framework hook is intact; we just don't override it.

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
            # still counts as "has takeaway" so old playbook entries
            # survive.
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
            # absent. Fall back to older field names so legacy playbook
            # entries still surface useful hints instead of getting
            # filtered out by the renderer.
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
            self._lessons_written += len(new_hints)
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
