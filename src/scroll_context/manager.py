"""ScrollContextManager — scroll's context management over OpenAI dict messages.

The canonical implementation of scroll's orchestration (eviction to a token
budget folded into the `EvictionIndex`, observation aging, var-context
virtualization/distillation, the pinned index placeholder, the per-step
working-memory digest, write-through persistence), over plain OpenAI
chat-completions messages — `scroll_agent_A` delegates here:

    {"role": "user"|"assistant"|"tool", "content": str|list,
     "tool_calls": [...]?, "tool_call_id": str?}

Terminology (canonical for all scroll code and prompts):

- **session** — one host ``run()``; one manager lifetime; ``session_id``.
- **turn** — a unit of the conversation WITH THE USER, bounded by user
  inputs. The seeded prior history is made of turns; an interactive host
  interleaves new user turns via :meth:`record_user_message`.
- **step** — one iteration of the agent loop within a turn: one assistant
  message plus its contiguous tool results (an *exchange* — also the atomic
  eviction group). ``step_index`` counts these on the agent's own rows; on
  seed-tier rows ingest reuses the column as the prior-conversation SESSION
  number (that is what ``S<n>`` refers to).
- **row / seq** — one persisted ``LogEntry``; ``seq`` is the table-wide
  primary key, the only globally unique coordinate, and therefore the currency
  of every cross-reference (index spans, stub pointers, origin provenance).

Multi-turn contract: every context-reduction mechanism operates on STEP
artifacts only — aging and virtualization rewrite ``role:"tool"`` results,
distillation rewrites assistant text, eviction pops whole exchange groups.
User-turn messages are never rewritten: an interleaved user input stays
verbatim until (at most) evicted as its own group, and the keep-K windows
(``obs_keep_turns``, ``_VAR_KEEP_THOUGHTS``) count assistant STEPS, not user
turns — a multi-turn host gets step-based windows that roll across turn
boundaries by design.

Two structural adaptations vs the AgentScope original:

- OpenAI tool results are separate ``role:"tool"`` messages rather than blocks
  inside the assistant message, and an assistant message with ``tool_calls``
  must never be left in context without its paired results (the API rejects
  orphans). Eviction therefore works on *groups*: an assistant message plus
  its contiguous following tool-result messages evict atomically.
- The pinned head is configurable (``pinned``): LOCA-bench pins only the
  initial user prompt (index 0), scroll_agent_A pins system+task (2).

Python >= 3.10. No imports from any host harness — only ``scroll_context._runtime``
and the stdlib — so any OpenAI-format loop can reuse this class.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from scroll_context._runtime import ExecutionResult, LogEntry, ScrollRuntime
from scroll_context._runtime.exec import OVERFLOW_MARKER, stdout_cap_for
from scroll_context._runtime.index import EvictionIndex, Leaf
from scroll_context._runtime.memoryspace import MemorySpace
from scroll_context._runtime.namespace import (
    VarViews,
    diff_names,
    fingerprint,
    install_var_ops,
    is_scratch,
    model_vars,
    overlap_verdict,
    preview_value,
    var_meta_line,
)

_CHARS_PER_TOKEN = 4

# Milestone headline fence — mirrors scroll_agent_A/agent.py.
_HEADLINE_RE = re.compile(r"^[ \t]*⟦[ \t]*(.+?)[ \t]*⟧[ \t]*$", re.MULTILINE)
_HEADLINE_MAX = 200

# Observation aging — mirrors scroll_agent_A/agent.py.
_OBS_KEEP_TURNS = 3
_OBS_AGE_MIN_CHARS = 600
_OBS_AGE_HEAD_CHARS = 200
_AGED_MARKER = "[…tool output aged out of this prompt"

# Oversized-tool-result cap. External (harness) tools don't flow through
# scroll's Executor, so they bypass its stdout cap — a single unbounded result
# (e.g. a full-spreadsheet read) can exceed the whole context budget while
# sitting in the never-evicted newest group. The full text is persisted to the
# history DB first; only the in-context copy is stubbed.
_CAP_MARKER = "[…tool output truncated in this prompt"

# Sessions/spans per index level before folding into a chunk line (see
# EvictionIndex._fold). Raised well above the old default of 10 so seeded
# prior-session spans (prime_prior_sessions) stay individually resident with
# their per-session date/summary instead of collapsing into sparse endpoint-
# bracket chunks at benchmark-scale session counts (~100 seeded sessions) —
# folding measurably hurt discovery/temporal recall on BEAM. Override with
# the SCROLL_INDEX_LEVEL_CAP env var.
_INDEX_LEVEL_CAP = 120

# --- var-context mode knobs ---------------------------------------------------
# Thoughts older than the last _VAR_KEEP_THOUGHTS assistant STEPS distill to
# their ⟦headline⟧ (fallback: truncated first line); tool results older than
# the newest group virtualize to a changelog + metadata view. Both are ONE-SHOT
# rewrites at fixed shallow depth, then frozen — KV-cache friendly by design.
_VAR_KEEP_THOUGHTS = 2
# Distillation fallback budget. Generous on purpose: the distilled line is the
# ONLY surviving copy of a fallback thought (intent is not duplicated into the
# digest), and at ~60 tokens frozen-once it is cheap against the re-derivation
# a lost plan costs. Clipped at a sentence boundary, not mid-thought.
_VAR_FALLBACK_CHARS = 240
_VAR_AUTO_FULL_ITEMS = 5        # collections at/below this render fully in the digest
# Multi-turn knobs: closed turns keep their ask + final answer verbatim for the
# last _KEEP_TURNS_VERBATIM turns (then only their map line + turn_record row).
_KEEP_TURNS_VERBATIM = 2
_TURN_ASK_CHARS = 120
_TURN_ANS_CHARS = 160
_VAR_HEAD_ROWS = 3              # rows for a 'head' view
_VIRT_MARKER = "⟦output virtualized to variables"

# The digest's standing nudge asks the model to OPEN each turn with a one-line
# verdict on its last result — which makes the first line of a thought a
# backward-looking judgment, not intent. Skip such lines when picking the
# distillation fallback / auto-intent so "while:" says what the turn was FOR.
_SELF_CHECK_RE = re.compile(
    r"^\s*(?:my |the )?last (?:result|search|query|output)\b", re.IGNORECASE
)


def _clip_words(text: str, limit: int) -> str:
    """Clip at a word boundary — a mid-word cut reads as garbage in a digest."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:—-") + "…"


def _clip_sentences(text: str, limit: int) -> str:
    """Clip at the last sentence boundary under ``limit``; word boundary as
    fallback. A distilled line that ends mid-sentence reads as lost thought."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if dot > limit // 3:
        return cut[: dot + 1]
    return _clip_words(text, limit)


def _distill_text(text: str) -> str:
    """The forward-looking content of a thought, flattened.

    Skips leading self-check verdict lines (the digest nudge mandates them),
    then keeps EVERYTHING from the first forward-looking line on — not just
    that line — so multi-sentence plans survive distillation up to the
    sentence-clipped budget.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if not _SELF_CHECK_RE.match(ln):
            return " ".join(lines[i:])
    return lines[0] if lines else ""


def _tool_call_source(msg: dict) -> str:
    """The ``source`` argument of an assistant message's first REPL tool call."""
    for tc in msg.get("tool_calls") or []:
        args = (tc.get("function") or {}).get("arguments") if isinstance(tc, dict) else None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if isinstance(args, dict) and isinstance(args.get("source"), str):
            return args["source"]
    return ""


_COMMENT_RE = re.compile(r"^\s*#\s*(.+?)\s*$")


def _comment_intent(source: str) -> str | None:
    """Intent fallback for models that emit bare tool calls with no thought:
    the first ``#`` comment of the executed source, which such models
    reliably write as an intent header ("# Search for the user's motivation…")."""
    for ln in source.splitlines():
        m = _COMMENT_RE.match(ln)
        if m and m.group(1):
            return m.group(1)
    return None


# One derived-facts string per ms call: "search → 15 hits (snippets, k-saturated)".
_OPS_RE = re.compile(r"^(\w+) → (\d+) (\w+)(?: \((.+)\))?$")


def _aggregate_ops(ops: list[str]) -> str:
    """Collapse a turn's op strings into one line, repeats merged by shape.

    ``4× search → 40 hits total (snippets) · expand → 3 rows (full text)`` —
    op name, counts, and completeness flags only; never queries or SQL (the
    turn's ``seq`` pointer recovers those verbatim from ``tool_input``).
    """
    groups: dict[tuple, list[int]] = {}
    order: list[tuple] = []
    passthrough: list[str] = []
    for o in ops:
        m = _OPS_RE.match(o)
        if not m:
            passthrough.append(o)
            continue
        key = (m.group(1), m.group(3), m.group(4) or "")
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(int(m.group(2)))
    parts: list[str] = []
    for op, unit, flags in order:
        ns = groups[(op, unit, flags)]
        f = f" ({flags})" if flags else ""
        if len(ns) == 1:
            parts.append(f"{op} → {ns[0]} {unit}{f}")
        else:
            parts.append(f"{len(ns)}× {op} → {sum(ns)} {unit} total{f}")
    return " · ".join(parts + passthrough)

# `{repl}` is the host's REPL tool name (``repl_name``): "scroll_repl" for
# OpenAI-format harnesses, "execute_python" for scroll_agent_A.
_INDEX_HEADER_TMPL = (
    "<system-info>[memory] Your eviction index — earlier history (prior "
    "conversation turns with this user, and your own steps compacted out of "
    "this prompt) was folded into this map over rounds of compaction. "
    "Newest/finest entries are at the bottom; older spans are chunked upward "
    "into single lines whose endpoints bracket their era. The full rows are durable in "
    "hist.conversation_history: to look up earlier history, find the relevant "
    "span here and expand it inside {repl}."
)


def own_session_spans(
    ms: MemorySpace,
    *,
    task_id: str | None = None,
    exclude_session_id: str | None = None,
) -> list[dict]:
    """This agent's own prior sessions, from their ``session_record`` rows.

    One span per prior session of the same task (excluding the current
    session), tagged ``P1, P2, …`` in chronological order — the flat
    one-row-per-session read that :meth:`ScrollContextManager.close_session`
    materializes so priming never needs a group-by over raw rows. Endpoints
    come from the record's metadata (``ask``/``answer`` extractive clips).
    Empty when no prior sessions recorded — priming from nothing is a no-op.
    """
    sql = (
        "SELECT session_id, metadata FROM hist.conversation_history "
        "WHERE kind='session_record' "
        + ("AND task_id = ? " if task_id is not None else "")
        + ("AND session_id != ? " if exclude_session_id is not None else "")
        + "ORDER BY seq"
    )
    params = tuple(p for p in (task_id, exclude_session_id) if p is not None)
    spans: list[dict] = []
    for i, r in enumerate(ms.sql_query(sql, params), start=1):
        try:
            meta = json.loads(r.get("metadata") or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(meta.get("seq_lo"), int) or not isinstance(meta.get("seq_hi"), int):
            continue
        head = meta.get("ask") or f"prior session {i}"
        spans.append(
            {
                "seq_lo": meta["seq_lo"],
                "seq_hi": meta["seq_hi"],
                "head": head,
                "tail": meta.get("answer") or head,
                "tag": f"P{i}",
            }
        )
    return spans


def shared_session_spans(
    ms: MemorySpace,
    *,
    run_ids: tuple[str, ...] | list[str] = (),
    task_id: str | None = None,
) -> list[dict]:
    """Prior-session spans from the shared tiers of an attached history DB.

    One span per session whose rows carry a ``run_id`` in ``run_ids`` (an
    eval's seeded prior conversation, an earlier agent's shared tier, …):
    the session's full ``seq`` range with its FIRST and LAST milestone
    headline as endpoints, tagged with its session number (``step_index``).
    Returns ``[]`` when ``run_ids`` is empty or nothing matches — priming
    from an empty source is a no-op, not an error.

    This is the default *source* for :meth:`ScrollContextManager.prime_prior_sessions`;
    a host with a different notion of prior history can pass its own spans
    (``{"seq_lo", "seq_hi", "head", "tail", "session"}`` dicts) instead.
    """
    if not run_ids:
        return []
    marks = ",".join("?" for _ in run_ids)
    sql = (
        "SELECT MIN(ch.step_index) AS session, MIN(ch.seq) AS seq_lo, "
        "MAX(ch.seq) AS seq_hi, "
        "(SELECT headline FROM hist.conversation_history h2 "
        " WHERE h2.session_id = ch.session_id AND h2.headline IS NOT NULL "
        " ORDER BY h2.seq LIMIT 1) AS head, "
        "(SELECT headline FROM hist.conversation_history h3 "
        " WHERE h3.session_id = ch.session_id AND h3.headline IS NOT NULL "
        " ORDER BY h3.seq DESC LIMIT 1) AS tail "
        f"FROM hist.conversation_history ch WHERE ch.run_id IN ({marks}) "
        + ("AND ch.task_id = ? " if task_id is not None else "")
        + "GROUP BY ch.session_id ORDER BY seq_lo"
    )
    params = tuple(run_ids) + ((task_id,) if task_id is not None else ())
    spans: list[dict] = []
    for r in ms.sql_query(sql, params):
        if r.get("seq_lo") is None:  # defensive: the _truncated marker row
            continue
        head = r.get("head") or f"session {r.get('session')}"
        spans.append(
            {
                "seq_lo": int(r["seq_lo"]),
                "seq_hi": int(r["seq_hi"]),
                "head": head,
                # A session with one (or zero) headline collapses to a single
                # endpoint rather than a dangling pair.
                "tail": r.get("tail") or head,
                "session": r.get("session"),
            }
        )
    return spans


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    return default


def _obs_keep_turns_from_env() -> int | None:
    """Aging window from the env: int STEPS (assistant exchanges) to keep, or
    None = aging disabled. The env name says "turns" for historical reasons."""
    raw = os.environ.get("SCROLL_OBS_KEEP_TURNS", "").strip().lower()
    if raw in ("off", "none", "0"):
        return None
    if raw.isdigit():
        return int(raw)
    return _OBS_KEEP_TURNS


def _content_text(msg: dict) -> str:
    """Flatten an OpenAI message's content field to text."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multi-part content
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "") or ""))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _msg_chars(msg: dict) -> int:
    n = len(_content_text(msg))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        args = fn.get("arguments", "")
        n += len(fn.get("name", "") or "") + len(args if isinstance(args, str) else json.dumps(args))
    return n


def _est_tokens(msg_or_text: Any) -> int:
    if isinstance(msg_or_text, str):
        return len(msg_or_text) // _CHARS_PER_TOKEN + 1
    return _msg_chars(msg_or_text) // _CHARS_PER_TOKEN + 1


def _extract_headline(text: str | None) -> str | None:
    """The turn's durable index line: the model's own ``⟦ … ⟧`` fence, or None.

    Intentionally no extractive fallback (mirrors upstream): a turn with no
    fence does not become a leaf of the eviction map — it stays durably stored
    and recallable by seq range or ``ms.search``.
    """
    if text:
        m = _HEADLINE_RE.search(text)
        if m and m.group(1).strip():
            return m.group(1).strip()[:_HEADLINE_MAX]
    return None


def _format_execute_observation(result: ExecutionResult) -> str:
    """Compose stdout/stderr/error into one observation string (mirrors upstream)."""
    parts: list[str] = []
    if result.stdout:
        parts.append(f"stdout:\n{result.stdout.rstrip()}")
    if result.stderr:
        parts.append(f"stderr:\n{result.stderr.rstrip()}")
    if result.error:
        parts.append(f"error: {result.error}")
    if not parts:
        parts.append("(no output)")
    return "\n".join(parts)


class ScrollContextManager:
    """Per-task scroll context management for an OpenAI-format message list.

    Intended call pattern per loop step (see LOCA-bench's run_react wiring):

        mgr.record_assistant_turn(assistant_msg, usage)   # after each response
        mgr.record_tool_result(tool_msg)                   # after each tool result
        events = mgr.manage(messages)                      # before next API call
        call_messages = messages + [mgr.digest_message()]  # ephemeral digest

    ``scroll_repl`` tool calls are answered via :meth:`execute_python`.
    """

    def __init__(
        self,
        *,
        history_db_path: str | Path,
        session_id: str,
        run_id: str | None = None,
        task_id: str | None = None,
        history_max_tokens: int,
        pinned: int = 1,
        enable_index: bool | None = None,
        index_level_cap: int | None = None,
        obs_keep_turns: int | None = "env",  # type: ignore[assignment]
        execute_timeout_s: float = 60.0,
        tool_result_cap_chars: int | None = -1,  # -1 = auto (budget-scaled)
        shared_run_ids: tuple[str, ...] = (),
        repl_name: str = "scroll_repl",
        index_header: str | None = None,
        placeholder_name: str | None = None,
        var_context: bool | None = None,
        placeholder_at: int | None = None,
        var_fallback_chars: int | None = None,
        keep_turns_verbatim: int | None = None,
        var_keep_thoughts: int | None = None,
        turn_ask_chars: int | None = None,
        turn_ans_chars: int | None = None,
    ) -> None:
        # ``repl_name`` is the host's name for the REPL tool — it appears in
        # every model-facing recovery text (cap stubs, aged stubs, digest,
        # index header) so the instructions name a tool that actually exists.
        self._repl_name = repl_name
        self._index_header = (
            index_header if index_header is not None
            else _INDEX_HEADER_TMPL.format(repl=repl_name)
        )
        self._shared_run_ids = tuple(shared_run_ids)
        # Optional OpenAI `name` field stamped on the placeholder message so
        # hosts that surface message names (prompt dumps, AgentScope Msgs) can
        # label it (scroll_agent_A uses "memory").
        self._placeholder_name = placeholder_name
        self._runtime = ScrollRuntime(
            history_db_path=history_db_path,
            session_id=session_id,
            run_id=run_id,
            task_id=task_id,
            # Shared-tier run ids (e.g. an eval's seeded prior sessions) so a
            # shared history DB keeps scope='task' isolated to "shared tier +
            # own session".
            shared_run_ids=self._shared_run_ids,
            execute_timeout_s=execute_timeout_s,
            history_max_tokens=history_max_tokens,
        )
        self.history_max_tokens = int(history_max_tokens)
        self._pinned = int(pinned)
        if enable_index is None:
            enable_index = _env_flag("SCROLL_EVICTION_INDEX", True)
        self._enable_index = bool(enable_index)
        if index_level_cap is None:
            raw = os.environ.get("SCROLL_INDEX_LEVEL_CAP", "").strip()
            index_level_cap = int(raw) if raw.isdigit() else _INDEX_LEVEL_CAP
        self._index = EvictionIndex(session_id=session_id, level_cap=index_level_cap)
        self._obs_keep_turns = (
            _obs_keep_turns_from_env() if obs_keep_turns == "env" else obs_keep_turns
        )
        # Var-context ablation: the variable store is the curated context.
        # Namespace changes render as an append-only changelog, old tool
        # results virtualize to metadata views (replacing observation aging),
        # old thoughts distill to their headline, and the tail digest carries
        # typed per-variable views (auto policy + model pin/note/show ops).
        if var_context is None:
            var_context = _env_flag("SCROLL_VAR_CONTEXT", False)
        self._var_context = bool(var_context)
        # Distillation fallback budget (chars): param > SCROLL_VAR_FALLBACK_CHARS
        # env > default. Sentence-clipped; the model's own ⟦headline⟧ (capped at
        # _HEADLINE_MAX) always wins over the fallback regardless of this knob.
        if var_fallback_chars is None:
            raw = os.environ.get("SCROLL_VAR_FALLBACK_CHARS", "").strip()
            var_fallback_chars = int(raw) if raw.isdigit() else _VAR_FALLBACK_CHARS
        self._var_fallback_chars = max(40, int(var_fallback_chars))
        # Multi-turn state (var-context mode; see close_turn). M = how many
        # closed turns keep their ask + final answer verbatim in-context:
        # param > SCROLL_KEEP_TURNS_VERBATIM env > default 2.
        if keep_turns_verbatim is None:
            raw = os.environ.get("SCROLL_KEEP_TURNS_VERBATIM", "").strip()
            keep_turns_verbatim = int(raw) if raw.isdigit() else _KEEP_TURNS_VERBATIM
        self._keep_turns_verbatim = max(0, int(keep_turns_verbatim))
        # Live-window size for thought distillation (steps kept verbatim).
        if var_keep_thoughts is None:
            raw = os.environ.get("SCROLL_VAR_KEEP_THOUGHTS", "").strip()
            var_keep_thoughts = int(raw) if raw.isdigit() else _VAR_KEEP_THOUGHTS
        self._var_keep_thoughts = max(1, int(var_keep_thoughts))
        # Extractive clip budgets for turn/session record endpoints.
        if turn_ask_chars is None:
            raw = os.environ.get("SCROLL_TURN_ASK_CHARS", "").strip()
            turn_ask_chars = int(raw) if raw.isdigit() else _TURN_ASK_CHARS
        self._turn_ask_chars = max(40, int(turn_ask_chars))
        if turn_ans_chars is None:
            raw = os.environ.get("SCROLL_TURN_ANS_CHARS", "").strip()
            turn_ans_chars = int(raw) if raw.isdigit() else _TURN_ANS_CHARS
        self._turn_ans_chars = max(40, int(turn_ans_chars))
        self._session_ask = ""
        self._session_start_seq = 0
        self._session_closed = False
        self._turn_no = 0
        self._turn_ask = ""
        self._turn_user_id: int | None = None
        self._turn_start_seq = 0
        self._turn_folded_hi = 0
        self._retained_turns: list[dict] = []
        self._retained_ids: set[int] = set()
        self._views = VarViews()
        if self._var_context:
            install_var_ops(self._runtime.namespace, self._views)
        self._pending_changes: list[str] = []
        self._changes_by_id: dict[int, list[str]] = {}
        # Overlap/upgrade verdicts stashed at creation time (see _note_changes)
        # — the digest renders these verbatim instead of recomputing, so a
        # variable can never acquire a warning retroactively.
        self._overlap_by_name: dict[str, str] = {}
        # Origin pointer per variable: the seq of the tool_result row of the
        # turn that created/rebound it — that durable row holds the exact
        # query (tool_input) and raw output verbatim, so the prompt carries a
        # pointer instead of a query gist. Names accumulate here between the
        # REPL diff and the record_tool_result that learns the seq.
        self._var_seq: dict[str, int] = {}
        self._pending_var_names: list[str] = []
        self._headline_or_fallback: dict[int, str] = {}
        self._distilled_ids: set[int] = set()
        self._virtualized_ids: set[int] = set()
        # Where the pinned map/placeholder is inserted. Defaults to `pinned`
        # (right after the pinned head). An earlier slot (e.g. 1 = before the
        # task message) makes the map a constant prompt PREFIX shared across
        # sibling sessions of one task — a large KV-cache win when the map is
        # big and static. Must be <= pinned so the eviction base stays correct.
        self._placeholder_at = (
            self._pinned if placeholder_at is None else min(int(placeholder_at), self._pinned)
        )
        # Same budget-scaled cap scroll applies to its own REPL stdout
        # (clamped 2k-32k chars); None disables capping.
        self._tool_result_cap = (
            stdout_cap_for(history_max_tokens) if tool_result_cap_chars == -1
            else tool_result_cap_chars
        )

        # In-context bookkeeping. Keyed by id(msg dict) — the host loop must
        # append the same dict objects it passed to record_*() to its list.
        self._seq_by_id: dict[int, tuple[int, int]] = {}
        self._leaf_by_id: dict[int, Leaf] = {}
        self._aged_ids: set[int] = set()
        self._last_assistant_id: int | None = None
        self._placeholder: dict | None = None
        # Index-OFF ablation span (opaque [context compressed] placeholder).
        self._evicted_lo: int | None = None
        self._evicted_hi: int | None = None

        self.est_input = 0
        self._step_index = 0
        # Totals for metrics / trajectory events.
        self.totals = {
            "evict_sweeps": 0,
            "evicted_msgs": 0,
            "evicted_tokens_est": 0,
            "aged_blocks": 0,
            "aged_tokens_est": 0,
            "assistant_turns": 0,
            "headlined_turns": 0,
            "repl_calls": 0,
            "max_in_context": 0,
            "capped_results": 0,
            "capped_tokens_est": 0,
            # var-context mode
            "var_changes": 0,
            "virtualized_results": 0,
            "distilled_thoughts": 0,
            "overlap_warnings": 0,
            "turn_records": 0,
            "session_records": 0,
        }
        # Real-prompt-vs-estimate overhead (tool schemas, digest, wire format),
        # learned from reported usage; added on top of the recomputed base.
        self._est_overhead = 0
        self._last_manage_base = 0

    # ------------------------------------------------------------------ #
    # write-through recording

    def record_initial_prompt(
        self, msg: dict, *, step_index: int = 0, msg_index: int = 0
    ) -> None:
        """Persist the pinned task prompt (kind='task') and prime the estimate.

        ``step_index``/``msg_index`` let a host keep its own log conventions
        (scroll_agent_A stamps the task at ``step_index=-1, msg_index=1``).
        """
        text = _content_text(msg)
        self._runtime.append_log(
            LogEntry(
                kind="task",
                role="user",
                content=text,
                step_index=step_index,
                msg_index=msg_index,
                blocks=[{"type": "text", "text": text}],
            )
        )
        # Session-level tracking is mode-independent: every session writes a
        # `session_record` at close (so future sessions can prime `P` lines
        # from it) regardless of the var-context flag.
        self._session_ask = text
        self._session_start_seq = self._runtime.persisted_seq
        if self._var_context:
            # Turn 1 opens with the task prompt; turn-granularity tracking
            # (and close_turn) belongs to the var-context retention gradient.
            self._turn_no = 1
            self._turn_ask = text
            self._turn_user_id = id(msg)
            self._turn_start_seq = self._runtime.persisted_seq
        self.est_input += _est_tokens(msg)

    def record_assistant_turn(
        self,
        msg: dict,
        usage: dict | None = None,
        *,
        step_index: int | None = None,
        msg_index: int | None = None,
        reasoning: str | None = None,
    ) -> None:
        """Write-through one assistant turn; track seq/leaf; re-anchor the estimate.

        ``step_index`` (when given) overrides — and re-anchors — the internal
        step counter, so a host loop's own numbering flows through to the log
        rows of this turn and its tool results. ``reasoning`` (thinking-mode
        chain of thought) is persisted for inspection but never re-rendered
        into any prompt by this class.
        """
        if step_index is not None:
            self._step_index = step_index
        else:
            self._step_index += 1
        text = _content_text(msg)
        headline = _extract_headline(text)
        self._runtime.append_log(
            LogEntry(
                kind="model_turn",
                role="assistant",
                content=text,
                metadata={
                    "step_index": self._step_index,
                    **({"reasoning": reasoning} if reasoning else {}),
                },
                step_index=self._step_index,
                msg_index=self._step_index if msg_index is None else msg_index,
                headline=headline,
                blocks=[{k: v for k, v in msg.items() if k != "reasoning_details"}],
            )
        )
        seq = self._runtime.persisted_seq
        self._seq_by_id[id(msg)] = (seq, seq)
        if headline:
            self._leaf_by_id[id(msg)] = Leaf(seq=seq, headline=headline)
        if self._var_context:
            # What this turn distills to once it ages past the keep window: the
            # model's own ⟦headline⟧, else its first useful line (leading
            # self-check verdicts skipped, clipped at a word boundary) — so low
            # headline compliance degrades to a crude summary, never to erased
            # history. Models that emit bare tool calls with NO thought text
            # (qwen with thinking off) fall back further to the first ``#``
            # comment of the call's source — reliably an intent header.
            fallback = _clip_sentences(_distill_text(text), self._var_fallback_chars)
            if not fallback:
                c = _comment_intent(_tool_call_source(msg))
                if c:
                    fallback = _clip_sentences(c, self._var_fallback_chars)
            self._headline_or_fallback[id(msg)] = (
                f"⟦ {headline} ⟧" if headline
                else (f"⟦ (no headline) {fallback} ⟧" if fallback else "⟦ (no headline) ⟧")
            )
        self._last_assistant_id = id(msg)
        self.totals["assistant_turns"] += 1
        if headline:
            self.totals["headlined_turns"] += 1

        # Learn the estimate-vs-reality overhead (tool schemas, digest, wire
        # format) from reported usage: the next manage() recomputes the base
        # from the actual list and adds this on top (adaptation of the
        # re-anchoring in agent.py:988 that survives external list mutation).
        prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
        completion_tokens = int((usage or {}).get("completion_tokens") or 0)
        if prompt_tokens > 0 and self._last_manage_base > 0:
            self._est_overhead = max(0, prompt_tokens - self._last_manage_base)
        self.est_input += completion_tokens if completion_tokens > 0 else _est_tokens(msg)

    def record_tool_result(
        self,
        msg: dict,
        *,
        tool_name: str | None = None,
        tool_input: Any = None,
        tool_state: str | None = None,
        msg_index: int | None = None,
    ) -> None:
        """Write-through one ``role:"tool"`` result; widen the turn's seq span.

        The FULL result text is persisted to the durable history first. If it
        exceeds the in-context cap, the message's content is then replaced in
        place with a head + recovery note pointing at the persisted row — so an
        unbounded external tool result can never flood (or exceed) the window,
        mirroring the stdout cap scroll applies to its own REPL. Output the
        REPL's executor already bounded (it carries the "print less" notice) is
        left alone — stubbing it again would only chop off that notice.
        """
        text = _content_text(msg)
        self._runtime.append_log(
            LogEntry(
                kind="tool_result",
                role="tool",
                name=tool_name,
                content=text,
                step_index=self._step_index,
                msg_index=self._step_index if msg_index is None else msg_index,
                tool_call_id=msg.get("tool_call_id"),
                tool_input=tool_input,
                tool_state=tool_state,
                blocks=[dict(msg)],
            )
        )
        seq = self._runtime.persisted_seq
        self._seq_by_id[id(msg)] = (seq, seq)
        # Also widen the owning assistant turn's range so evicting that group
        # folds a span covering both durable rows (mirrors agent.py:1209).
        if self._last_assistant_id is not None and self._last_assistant_id in self._seq_by_id:
            lo, _hi = self._seq_by_id[self._last_assistant_id]
            self._seq_by_id[self._last_assistant_id] = (lo, seq)

        if self._var_context:
            # External (non-REPL) results are auto-bound to a namespace
            # variable, so "tool output" and "model data" share one substrate:
            # a typed variable with metadata, curatable like any other.
            if tool_name and tool_name != self._repl_name and isinstance(text, str) and text:
                vname = f"obs_{self._step_index}"
                self._runtime.namespace[vname] = text
                self._pending_changes.append(f"vars: + {vname} ({tool_name})")
                self._pending_var_names.append(vname)
                self.totals["var_changes"] += 1
            # This result claims the changelog accumulated since the last one;
            # the entries surface in-prompt when this message is virtualized.
            # Its durable seq becomes the origin pointer of the variables the
            # turn created — the row's tool_input holds the exact query.
            for vname in self._pending_var_names:
                self._var_seq[vname] = seq
            self._pending_var_names = []
            self._changes_by_id[id(msg)] = self._pending_changes
            self._pending_changes = []

        cap = self._tool_result_cap
        if (
            cap is not None
            and isinstance(msg.get("content"), str)
            and len(msg["content"]) > cap
            and _CAP_MARKER not in msg["content"]
            and OVERFLOW_MARKER not in msg["content"]
        ):
            full_len = len(msg["content"])
            stub = (
                msg["content"][:cap].rstrip()
                + f"\n{_CAP_MARKER} to protect your context window: "
                f"{tool_name or 'tool'} returned {full_len} chars "
                f"(~{full_len // _CHARS_PER_TOKEN} tokens); only the first {cap} chars are shown. "
                f"The FULL output is durably persisted at seq {seq} in "
                "hist.conversation_history. Do NOT re-issue the same call hoping to "
                f"see more — instead read it in slices inside {self._repl_name} "
                f"(e.g. rows = ms.expand([{seq}]) then slice/filter in code and "
                "print only what you need), or re-issue the tool call with "
                "narrower arguments (a range, a filter, a limit).]"
            )
            self.totals["capped_results"] += 1
            self.totals["capped_tokens_est"] += (full_len - len(stub)) // _CHARS_PER_TOKEN
            msg["content"] = stub
        self.est_input += _est_tokens(msg)

    def record_user_message(
        self, msg: dict, kind: str = "user_message", *, messages: list[dict] | None = None
    ) -> None:
        """Write-through an interleaved user message (nudges, notices — or, in a
        multi-turn host, a NEW USER TURN). User messages are never aged,
        distilled, or virtualized; they stay verbatim until folded at turn
        close (or evicted as their own group).

        When ``kind == "user_message"`` and ``messages`` is passed, this IS the
        turn boundary: the previous turn is folded via :meth:`close_turn` first,
        then this message opens the next turn. Scaffolding (notices, nudges)
        should use a different ``kind`` or omit ``messages``.
        """
        if messages is not None and kind == "user_message":
            self.close_turn(messages)
        self._runtime.append_log(
            LogEntry(kind=kind, role="user", content=_content_text(msg), step_index=self._step_index)
        )
        seq = self._runtime.persisted_seq
        self._seq_by_id[id(msg)] = (seq, seq)
        if self._var_context and kind == "user_message":
            self._turn_no = max(1, self._turn_no) + 1 if self._turn_no else 1
            self._turn_ask = _content_text(msg)
            self._turn_user_id = id(msg)
            self._turn_start_seq = seq
            self._turn_folded_hi = 0
        self.est_input += _est_tokens(msg)

    # ------------------------------------------------------------------ #
    # per-turn management (aging + eviction + placeholder)

    def manage(self, messages: list[dict]) -> dict:
        """Run the pre-call pipeline in place; return this sweep's event dict.

        The estimate is recomputed from the ACTUAL message list every call
        (plus the usage-learned overhead) rather than trusted incrementally —
        incremental accounting silently diverges if the harness ever mutates
        the list behind our back (e.g. an API-layer hard trim).
        """
        base = sum(_est_tokens(m) for m in messages)
        self._last_manage_base = base
        self.est_input = base + self._est_overhead
        events: dict[str, Any] = {}
        if self._var_context:
            # Var-context mode replaces observation aging: one-shot, shallow,
            # deterministic rewrites (cache-friendly), then frozen forever.
            n_virt, n_dist, tok_saved = self._virtualize_and_distill(messages)
            if n_virt:
                events["virtualized_results"] = n_virt
            if n_dist:
                events["distilled_thoughts"] = n_dist
            if tok_saved:
                events["var_tokens_est"] = tok_saved
        else:
            n_aged, tok_saved = self._age_observations(messages)
            if n_aged:
                events["aged_blocks"] = n_aged
                events["aged_tokens_est"] = tok_saved
        self.totals["max_in_context"] = max(self.totals["max_in_context"], len(messages))
        if self.history_max_tokens > 0:
            n_ev, tok_ev, span = self._evict_to_budget(messages)
            if n_ev:
                self.totals["evict_sweeps"] += 1
                events["evicted_msgs"] = n_ev
                events["evicted_tokens_est"] = tok_ev
                events["evicted_seq_span"] = span
                events["in_context_after"] = len(messages)
        return events

    def _age_observations(self, messages: list[dict]) -> tuple[int, int]:
        """Stub tool outputs older than the last ``keep_turns`` assistant steps.

        Mutates ``role:"tool"`` message content in place (in-context only — the
        durable copies are already persisted). Idempotent via ``_aged_ids``.
        """
        keep = self._obs_keep_turns
        if keep is None:
            return 0, 0
        assistant_idxs = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        if not assistant_idxs:
            return 0, 0
        # Everything before the keep-th most recent assistant turn is "old". With
        # FEWER than `keep` assistant turns so far, none is old enough to age —
        # cutoff 0 (NOT len(messages), which would age every output; that inverted
        # the knob so a larger keep aged MORE, since more probes stay under it).
        if keep > 0:
            cutoff = assistant_idxs[-keep] if len(assistant_idxs) >= keep else 0
        else:
            cutoff = assistant_idxs[-1] + 1
        n_aged = 0
        saved = 0
        for msg in messages[:cutoff]:
            if msg.get("role") != "tool" or id(msg) in self._aged_ids:
                continue
            out = msg.get("content")
            if not isinstance(out, str):
                continue
            if len(out) <= _OBS_AGE_MIN_CHARS or _AGED_MARKER in out:
                self._aged_ids.add(id(msg))
                continue
            # Preserve the durable-row pointer in the stub: aging may overwrite
            # a cap stub whose seq note sat at the (now dropped) tail, and even
            # for ordinary outputs a direct seq makes recovery one step cheaper
            # than re-discovering it via ms.search.
            rng = self._seq_by_id.get(id(msg))
            seq_note = (
                f"persisted at seq {rng[0]} in hist.conversation_history "
                f"(ms.expand([{rng[0]}]) in {self._repl_name} returns it in full)"
                if rng is not None
                else "persisted in hist.conversation_history (find it via ms.search)"
            )
            stub = (
                out[:_OBS_AGE_HEAD_CHARS].rstrip()
                + f"\n{_AGED_MARKER} to save context: "
                f"{len(out)} chars total. The full output is {seq_note}, and any "
                "variables you assigned in scroll_repl still hold the data — "
                "print from those variables (or recall by seq) if you need it again.]"
            )
            saved += max(0, (len(out) - len(stub)) // _CHARS_PER_TOKEN)
            msg["content"] = stub
            self._aged_ids.add(id(msg))
            n_aged += 1
        if saved:
            self.est_input = max(0, self.est_input - saved)
        if n_aged:
            self.totals["aged_blocks"] += n_aged
            self.totals["aged_tokens_est"] += saved
        return n_aged, saved

    def _virtualize_and_distill(self, messages: list[dict]) -> tuple[int, int, int]:
        """The var-context in-place pipeline: both rewrites are one-shot.

        - **Virtualize** every ``role:"tool"`` message older than the newest
          assistant group: its content becomes the changelog of variable
          changes its turn produced, plus a durable-seq pointer — the raw
          output was verbatim for exactly one prompt (the turn it landed).
        - **Distill** every assistant message older than the last
          ``_VAR_KEEP_THOUGHTS``: its text becomes its ⟦headline⟧ (or
          truncated first line), ``tool_calls`` untouched.

        Each message is rewritten at most once at a fixed shallow depth and
        never touched again, so the frozen prefix stays KV-cache stable.
        """
        assistant_idxs = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        if not assistant_idxs:
            return 0, 0, 0
        saved = 0
        n_virt = 0
        n_dist = 0
        # Tool results: everything before the newest assistant message is old.
        for msg in messages[: assistant_idxs[-1]]:
            if msg.get("role") != "tool" or id(msg) in self._virtualized_ids:
                continue
            if id(msg) in self._retained_ids:  # verbatim q+a of recent turns
                continue
            out = msg.get("content")
            if not isinstance(out, str):
                continue
            self._virtualized_ids.add(id(msg))
            changes = self._changes_by_id.pop(id(msg), [])
            rng = self._seq_by_id.get(id(msg))
            seq_note = (
                f"full text at seq {rng[0]} in hist.conversation_history "
                f"(ms.expand([{rng[0]}]) in {self._repl_name})"
                if rng is not None
                else "full text persisted in hist.conversation_history (ms.search)"
            )
            if changes:
                # The changelog is the durable record of where variables came
                # from — always surface it, even when the raw output was tiny.
                body = "\n".join(changes)
                if all(e.startswith("ops:") for e in changes):
                    # Ops ran but nothing was stored — record what was tried
                    # AND keep the retention nudge.
                    body += (
                        "\n(no variables stored — anything you need from this "
                        "later must be re-derived)"
                    )
                stub = f"{_VIRT_MARKER} — {seq_note}⟧\n" + body
            else:
                stub = (
                    f"{_VIRT_MARKER} — {seq_note}⟧\n"
                    "(this output left NO variables — anything you need from it "
                    "later must be re-derived; next time store findings in a "
                    "variable before moving on)"
                )
                if len(stub) >= len(out):
                    # No changes and the output is already smaller than the
                    # nudge: leave the tiny output verbatim forever.
                    continue
            saved += max(0, (len(out) - len(stub))) // _CHARS_PER_TOKEN
            msg["content"] = stub
            n_virt += 1
        # Thoughts: keep the last _VAR_KEEP_THOUGHTS verbatim, distill older.
        for i in assistant_idxs[: -self._var_keep_thoughts]:
            msg = messages[i]
            if id(msg) in self._distilled_ids or id(msg) in self._retained_ids:
                continue
            self._distilled_ids.add(id(msg))
            line = self._headline_or_fallback.get(id(msg))
            if line is None or not isinstance(msg.get("content"), str):
                continue
            # Rewrite when it shrinks the message — and also when the thought
            # was EMPTY (bare tool call): growing it into the fence line keeps
            # the stream's narrative adjacency (the line above each virtualized
            # stub says what that turn was doing).
            if len(line) < len(msg["content"]) or not msg["content"].strip():
                saved += max(0, (len(msg["content"]) - len(line))) // _CHARS_PER_TOKEN
                msg["content"] = line
                n_dist += 1
        if saved:
            self.est_input = max(0, self.est_input - saved)
        if n_virt:
            self.totals["virtualized_results"] += n_virt
        if n_dist:
            self.totals["distilled_thoughts"] += n_dist
        return n_virt, n_dist, saved

    def _base_index(self) -> int:
        return self._pinned + (1 if self._placeholder is not None else 0)

    def _group_end(self, messages: list[dict], start: int) -> int:
        """End (exclusive) of the eviction group starting at ``start``.

        An assistant message with tool_calls groups with its contiguous
        following ``role:"tool"`` results (they must evict atomically); any
        other message is a group of one.
        """
        msg = messages[start]
        end = start + 1
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            ids = {tc.get("id") for tc in msg["tool_calls"] if isinstance(tc, dict)}
            while end < len(messages) and messages[end].get("role") == "tool" and (
                messages[end].get("tool_call_id") in ids or not ids
            ):
                end += 1
        return end

    def _evict_to_budget(self, messages: list[dict]) -> tuple[int, int, tuple | None]:
        """Evict oldest non-pinned groups until the estimate fits the budget.

        Every message was write-through-persisted the turn it was produced, so
        dropped messages remain durable and recoverable via ``ms``. Each
        evicted message's Leaf (if headlined) folds into the eviction index.
        """
        base = self._base_index()
        leaves: list[Leaf] = []
        span_lo: int | None = None
        span_hi: int | None = None
        n_evicted = 0
        tokens_evicted = 0
        # Keep at least one group beyond the pinned head (mirrors upstream's
        # "at least one recent Msg").
        while self.est_input > self.history_max_tokens:
            end = self._group_end(messages, base) if base < len(messages) else base
            if end >= len(messages):  # never evict the newest group
                break
            for removed in [messages.pop(base) for _ in range(end - base)]:
                dropped = _est_tokens(removed)
                self.est_input = max(0, self.est_input - dropped)
                n_evicted += 1
                tokens_evicted += dropped
                rng = self._seq_by_id.pop(id(removed), None)
                leaf = self._leaf_by_id.pop(id(removed), None)
                self._aged_ids.discard(id(removed))
                if rng is None:  # unpersisted message — nothing durable to fold
                    continue
                span_lo = rng[0] if span_lo is None else min(span_lo, rng[0])
                span_hi = rng[1] if span_hi is None else max(span_hi, rng[1])
                if leaf is not None:
                    leaves.append(leaf)
        if n_evicted:
            self.totals["evicted_msgs"] += n_evicted
            self.totals["evicted_tokens_est"] += tokens_evicted
            if span_lo is not None:
                self._turn_folded_hi = max(self._turn_folded_hi, span_hi or 0)
                if self._enable_index:
                    self._index.add_eviction(leaves, seq_lo=span_lo, seq_hi=span_hi)
                else:
                    self._evicted_lo = span_lo if self._evicted_lo is None else min(self._evicted_lo, span_lo)
                    self._evicted_hi = span_hi if self._evicted_hi is None else max(self._evicted_hi, span_hi)
                self._refresh_placeholder(messages)
        span = (span_lo, span_hi) if span_lo is not None else None
        return n_evicted, tokens_evicted, span

    def _render_placeholder(self) -> str:
        if self._enable_index:
            return self._index.render(header=self._index_header)
        return (
            f"<system-info>[context compressed] Turns seq {self._evicted_lo}–{self._evicted_hi} "
            "were evicted from this window but are durable in hist.conversation_history. "
            f"Recall inside {self._repl_name}, e.g. ms.search(\"keywords\") or "
            "ms.sql_query(\"SELECT seq, kind, role, content FROM hist.conversation_history "
            f"WHERE session_id='{self._runtime.session_id}' AND seq BETWEEN <lo> AND <hi> ORDER BY seq\")."
            "</system-info>"
        )

    def _refresh_placeholder(self, messages: list[dict]) -> None:
        """Create (lazily) or update the single placeholder message in place.

        On first fold it is inserted at ``messages[pinned]``; afterwards the
        SAME dict is mutated (stable id, fixed slot) and only the net token
        delta is applied, so its bounded growth isn't double-counted.
        """
        text = self._render_placeholder()
        if self._placeholder is None:
            self._placeholder = {"role": "user", "content": text}
            if self._placeholder_name:
                self._placeholder["name"] = self._placeholder_name
            messages.insert(self._placeholder_at, self._placeholder)
            self.est_input += _est_tokens(self._placeholder)
            return
        old = _est_tokens(self._placeholder)
        self._placeholder["content"] = text
        self.est_input += _est_tokens(self._placeholder) - old

    # ------------------------------------------------------------------ #
    # digest + REPL

    def _var_digest(self) -> str:
        """Var-context digest: one typed metadata line per variable, plus its
        auto- or pin-selected data view. Auto policy: scalars inline, small
        collections in full, large ones metadata-only; ``pin`` overrides."""
        ns = self._runtime.namespace
        ns_vars = model_vars(ns)
        if not ns_vars:
            return (
                "vars: (empty) — store what you find in variables; they are "
                "your durable working context (old tool output leaves this prompt)."
            )
        # Scratch tier: loop temporaries and imported classes collapse to one
        # line (an explicit pin promotes them back to first-class).
        scratch = sorted(
            n for n, v in ns_vars.items()
            if is_scratch(n, v) and n not in self._views.pins
        )
        lines = []
        # Creation order (origin seq), oldest first — the digest reads as the
        # same narrative as the changelog, with the newest work nearest the
        # model's next action. Variables from the still-unclaimed live turn
        # sort last; alphabetical only as tie-break.
        ordered = sorted(
            ns_vars,
            key=lambda n: (self._var_seq.get(n, float("inf")), n),
        )
        prev_seq: int | None = None
        for name in ordered:
            if name in scratch:
                continue
            val = ns_vars[name]
            level = self._views.pins.get(name)
            if level is None:  # auto policy
                if isinstance(val, (list, tuple, set, frozenset, dict)):
                    level = "full" if len(val) <= _VAR_AUTO_FULL_ITEMS else "meta"
                else:
                    level = "meta"  # scalars carry their value in the meta line
            tag = f"  [pinned {level}]" if name in self._views.pins else ""
            # Same-turn group compaction: creation ordering makes variables
            # from one turn contiguous, so the seq pointer and the shared
            # "while:" intent print once, on the group's first line — later
            # lines inherit them by adjacency (model-authored notes always
            # render per variable).
            seq = self._var_seq.get(name)
            first_of_turn = seq is None or seq != prev_seq
            prev_seq = seq
            meta = var_meta_line(
                name, val, self._views,
                origin_seq=seq if first_of_turn else None,
                overlap=self._overlap_by_name.get(name),
            )
            lines.append(f"  - {meta}{tag}")
            if level in ("head", "full") and not isinstance(val, (bool, int, float)):
                lines.append(
                    preview_value(val, _VAR_HEAD_ROWS if level == "head" else None)
                )
        if scratch:
            lines.append("  - (scratch: " + ", ".join(scratch) + ")")
        return "vars:\n" + "\n".join(lines)

    def protocol_prompt(self) -> str:
        """The model-facing context-management protocol for THIS configuration.

        Assembles the canonical prompt files (``core.md`` [+ ``index.md``]
        [+ ``vars.md``] — see :mod:`scroll_context.prompts`) with this
        manager's own REPL tool name and feature flags, applying the index-off
        headline-schema strip. Hosts embed it between their own preamble and
        finishing policy, so prompt text has exactly one source of truth.
        """
        from scroll_context.prompts import protocol_prompt

        return protocol_prompt(
            self._repl_name,
            index=self._enable_index,
            var_context=self._var_context,
        )

    def digest_message(self, budget_note: str | None = None) -> dict:
        """The ephemeral per-turn ``[working memory]`` message (never persisted)."""
        if self._var_context:
            parts = ["[working memory] " + self._var_digest()]
        else:
            parts = ["[working memory] " + self._runtime.digest()]
        evicted = self.totals["evicted_msgs"]
        if evicted > 0:
            parts.append(
                f"{evicted} earlier step(s) are no longer in this prompt but are "
                "recoverable. If you need facts from earlier, recover them in "
                f"{self._repl_name}: ms.search('a phrasing', scope='task') for a wide "
                "overview of the relevant turns, then ms.expand([the seqs that matter]) "
                "to read them in full — keep the returned lists in a variable and don't "
                "re-fetch a seq you already pulled."
            )
        parts.append(
            "Before your next action, judge in one line: does your last result look "
            "correct and actually move toward the task goal? If it looks wrong, "
            "off-track, or stuck, change approach instead of repeating."
        )
        if budget_note:
            parts.append(budget_note)
        return {"role": "user", "content": "\n".join(parts)}

    def execute_python(self, source: str) -> str:
        """Run one REPL call in the persistent namespace (sync bridge).

        For hosts already inside an event loop, use :meth:`execute_python_async`
        — ``asyncio.run`` raises when called from a running loop.
        """
        self.totals["repl_calls"] += 1
        before = fingerprint(self._runtime.namespace) if self._var_context else None
        result = asyncio.run(self._runtime.execute(source))
        ops = self._runtime.memoryspace.drain_op_log()
        if before is not None:
            self._note_changes(before, ops=ops)
        return _format_execute_observation(result) + self._row_cap_note(ops)

    async def execute_python_async(self, source: str) -> str:
        """Async variant of :meth:`execute_python` for async host loops."""
        self.totals["repl_calls"] += 1
        before = fingerprint(self._runtime.namespace) if self._var_context else None
        result = await self._runtime.execute(source)
        ops = self._runtime.memoryspace.drain_op_log()
        if before is not None:
            self._note_changes(before, ops=ops)
        return _format_execute_observation(result) + self._row_cap_note(ops)

    def _row_cap_note(self, ops: list[str]) -> str:
        """Same-turn truncation visibility, in the OBSERVATION channel only.

        The row cap no longer plants a marker row inside the data (it poisoned
        typed operations over the columns); the model still deserves an
        interrupt the turn it happens, so a capped query appends one line to
        the observation instead — in-band for the channel the model reads,
        never for the data it computes on.
        """
        if any("(row-capped)" in o for o in ops):
            cap = self._runtime.memoryspace.row_cap
            return (
                f"\n[note] a sql_query hit the {cap}-row cap — matching rows beyond "
                f"the first {cap} were NOT returned; narrow with WHERE or page with "
                "LIMIT/OFFSET. (rows.truncated is True on the capped result.)"
            )
        return ""

    def _note_changes(self, before: dict, ops: list[str] = ()) -> None:  # type: ignore[assignment]
        """Diff the namespace against ``before`` into the pending changelog.

        Entries accumulate until the next :meth:`record_tool_result` claims
        them for its message — they surface in the prompt when that tool
        result is virtualized (and stay visible in the durable copy).

        Ownership split: the changelog carries EVENTS only — the ``ops:``
        header (what the step retrieved), compact ``vars: + a, b  ~ c  - d``
        name lists, scratch churn, and overlap/upgrade verdicts. All variable
        METADATA (schema, coverage, origin, notes, intent) lives solely in the
        digest, which is always current — the changelog's frozen copies used
        to go stale the moment a variable mutated.

        The overlap verdict is computed HERE, once, at creation time — only
        for steps that actually ran retrieval ops, and only against variables
        that existed before the step (same-step siblings are derivation, the
        behavior the prompt teaches, never a redundancy). ``expand`` over a
        snippets-holding counterpart renders as ``↳`` lineage, not ``⚠``.
        Verdicts are stashed per variable for the digest to reuse verbatim.
        """
        ns = self._runtime.namespace
        names = diff_names(before, ns)
        ops = list(ops)
        if not any(names.values()) and not ops:
            return
        # Pre-step variables (current values) — the only legal overlap
        # comparison set.
        pre_step = {n: ns[n] for n in before if n in ns}
        changed: dict[str, list[str]] = {"+": [], "~": [], "±": [], "-": []}
        scratch: list[str] = []
        verdicts: list[str] = []
        n_changes = 0
        for kind, mark in (("created", "+"), ("reassigned", "~"), ("mutated", "±")):
            for name in names[kind]:
                n_changes += 1
                if kind != "mutated":
                    self._pending_var_names.append(name)
                    self._overlap_by_name.pop(name, None)
                    if ops and not is_scratch(name, ns[name]):
                        verdict = overlap_verdict(name, ns[name], pre_step)
                        if verdict is not None:
                            v_kind, v_text = verdict
                            self._overlap_by_name[name] = v_text
                            verdicts.append(v_text)
                            if v_kind == "warn":
                                self.totals["overlap_warnings"] += 1
                if is_scratch(name, ns[name]):
                    scratch.append(name)
                else:
                    changed[mark].append(name)
        for name in names["deleted"]:
            n_changes += 1
            self._views.drop(name)
            self._var_seq.pop(name, None)
            self._overlap_by_name.pop(name, None)
            if len(name) == 1:  # value is gone; the name-based rule still holds
                scratch.append(f"del {name}")
            else:
                changed["-"].append(name)
        entries: list[str] = []
        if ops:
            entries.append("ops: " + _aggregate_ops(ops))
        var_parts = [
            f"{mark} {', '.join(ns_list)}"
            for mark, ns_list in changed.items() if ns_list
        ]
        if var_parts:
            entries.append("vars: " + "  ".join(var_parts))
        if scratch:
            entries.append("· scratch: " + ", ".join(scratch))
        entries.extend(verdicts)
        if entries:
            self._pending_changes.extend(entries)
            self.totals["var_changes"] += n_changes

    # ------------------------------------------------------------------ #
    # turn folding

    def close_turn(self, messages: list[dict]) -> bool:
        """Fold the current turn: one map line + one durable record, husks popped.

        Called at a turn boundary (usually via :meth:`record_user_message`).
        The closed turn becomes: (1) a ``kind='turn_record'`` DB row —
        ``Q: <ask> / A: <answer>`` with the turn's seq span in metadata, the
        durable, searchable anchor of the turn; (2) an index span tagged
        ``T<n>`` (``ask → answer`` endpoints — extractive clips, never
        summaries), entering the same fold/carry as prior sessions but WITHOUT
        their line-survival privilege, so merged old turns decay to endpoint
        pairs; (3) in-context, everything of the turn is popped EXCEPT its ask
        and final-answer messages, which stay verbatim (and exempt from
        distillation/virtualization) for the last ``keep_turns_verbatim``
        closed turns, then pop too. Var-context mode only; returns True when a
        turn was actually closed.
        """
        if not self._var_context or self._turn_no < 1:
            return False
        asst_idxs = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        if not asst_idxs:
            return False
        n = self._turn_no
        ask_clip = _clip_sentences(self._turn_ask, self._turn_ask_chars)
        # Hold the final-answer MESSAGE OBJECT, not its index — the placeholder
        # refresh below inserts into the list and shifts positions.
        ans_msg = messages[asst_idxs[-1]]
        ans_text = _content_text(ans_msg)
        ans_clip = _clip_sentences(_distill_text(ans_text), self._turn_ans_chars) or "(no answer text)"
        end_seq = self._runtime.persisted_seq
        self._runtime.append_log(
            LogEntry(
                kind="turn_record",
                content=f"Q: {ask_clip}\nA: {ans_clip}",
                headline=_clip_words(f"{ask_clip} → {ans_clip}", _HEADLINE_MAX),
                metadata={"turn": n, "seq_lo": self._turn_start_seq, "seq_hi": end_seq},
            )
        )
        self.totals["turn_records"] += 1
        if self._enable_index:
            # Skip whatever a mid-turn eviction sweep already folded, so index
            # spans stay disjoint (the partition-losslessness invariant).
            lo = max(self._turn_start_seq, self._turn_folded_hi + 1)
            if lo <= end_seq:
                self._index.add_span(
                    seq_lo=lo, seq_hi=end_seq,
                    head=ask_clip, tail=ans_clip, tag=f"T{n}",
                )
            self._refresh_placeholder(messages)
        # Verbatim survivors: the ask + the final-answer exchange (re-located
        # by identity, post-insertion).
        keep_ids: set[int] = set()
        if self._turn_user_id is not None:
            keep_ids.add(self._turn_user_id)
        la = next(i for i, m in enumerate(messages) if m is ans_msg)
        for m in messages[la: self._group_end(messages, la)]:
            keep_ids.add(id(m))
        base = self._base_index()
        i = base
        while i < len(messages):
            end = self._group_end(messages, i)
            group = messages[i:end]
            if any(id(g) in keep_ids or id(g) in self._retained_ids for g in group):
                i = end
                continue
            del messages[i:end]
            for g in group:
                self._forget_msg(id(g))
        self._retained_turns.append({"turn": n, "ids": list(keep_ids)})
        self._retained_ids |= keep_ids
        while len(self._retained_turns) > self._keep_turns_verbatim:
            old_ids = set(self._retained_turns.pop(0)["ids"])
            self._retained_ids -= old_ids
            j = self._base_index()
            while j < len(messages):
                if id(messages[j]) in old_ids:
                    end = self._group_end(messages, j)
                    dropped = messages[j:end]
                    del messages[j:end]
                    for g in dropped:
                        self._forget_msg(id(g))
                else:
                    j += 1
        return True

    def _forget_msg(self, mid: int) -> None:
        """Drop all in-context bookkeeping for a popped message (durable rows
        and the turn record keep everything recoverable)."""
        self._seq_by_id.pop(mid, None)
        self._leaf_by_id.pop(mid, None)
        self._aged_ids.discard(mid)
        self._virtualized_ids.discard(mid)
        self._distilled_ids.discard(mid)
        self._headline_or_fallback.pop(mid, None)
        self._changes_by_id.pop(mid, None)

    def close_session(self, final_answer: str | None = None) -> bool:
        """Write the session's durable ``session_record`` row (once, at end).

        The session-granularity fold producer: ``Q: <task ask> / A: <final
        answer>`` — extractive clips (the headline column is set for map/drill
        listings; headlines are not FTS-matchable), the session's own seq span
        and turn count in metadata. It is NOT folded into this
        session's live map (it describes the whole session, which is ending);
        its consumers are FUTURE sessions, whose
        :meth:`prime_prior_sessions` reads these rows via
        :func:`own_session_spans` and folds one durable ``P<n>`` line each —
        exactly how seeded prior conversations enter via ``S<n>``. Written in
        EVERY mode (unlike :meth:`close_turn`, which belongs to the
        var-context retention gradient, this is plain durable bookkeeping);
        the turn count is exact under var-mode turn tracking and reported as 1
        otherwise. Idempotent; no-op before the initial prompt is recorded.
        """
        if self._session_closed or not self._session_ask:
            return False
        ask_clip = _clip_sentences(self._session_ask, self._turn_ask_chars)
        ans_clip = (
            _clip_sentences(_distill_text(final_answer or ""), self._turn_ans_chars)
            or "(no final answer)"
        )
        self._runtime.append_log(
            LogEntry(
                kind="session_record",
                content=f"Q: {ask_clip}\nA: {ans_clip}",
                headline=_clip_words(f"{ask_clip} → {ans_clip}", _HEADLINE_MAX),
                metadata={
                    "seq_lo": self._session_start_seq,
                    "seq_hi": self._runtime.persisted_seq,
                    "turns": max(1, self._turn_no),
                    "ask": ask_clip,
                    "answer": ans_clip,
                },
            )
        )
        self.totals["session_records"] += 1
        self._session_closed = True
        return True

    # ------------------------------------------------------------------ #
    # prior-session priming

    def prime_prior_sessions(
        self, messages: list[dict], spans: list[dict] | None = None
    ) -> bool:
        """Fold prior-conversation spans into the index; pin the placeholder.

        ``spans`` is an external source of prior history — an iterable of
        ``{"seq_lo", "seq_hi", "head", "tail", "session"|"tag"}`` dicts,
        possibly empty. When ``None``, spans merge TWO sources, seq-ordered:
        the shared tiers of the attached DB (``shared_run_ids`` →
        :func:`shared_session_spans`, tagged ``S<n>``) and this agent's own
        prior sessions of the same task (their ``session_record`` rows →
        :func:`own_session_spans`, tagged ``P<n>``). Each span becomes one
        ``EvictionIndex.add_span`` block, entering the SAME index that live
        evictions later extend, and the placeholder message is inserted at
        ``messages[pinned]`` immediately so the model sees one continuous map
        of its history from turn one. No-op (returns False) when the index is
        disabled or the source yields nothing.
        """
        if not self._enable_index:
            return False
        if spans is None:
            ms = self._runtime.memoryspace
            spans = shared_session_spans(
                ms, run_ids=self._shared_run_ids, task_id=ms.task_id
            ) + own_session_spans(
                ms, task_id=ms.task_id, exclude_session_id=ms.session_id
            )
            spans.sort(key=lambda s: int(s["seq_lo"]))
        folded = False
        for s in spans:
            snum = s.get("session")
            self._index.add_span(
                seq_lo=int(s["seq_lo"]),
                seq_hi=int(s["seq_hi"]),
                head=s["head"],
                tail=s.get("tail") or s["head"],
                session=int(snum) if snum is not None else None,
                tag=s.get("tag"),
            )
            folded = True
        if folded:
            self._refresh_placeholder(messages)
        return folded

    # ------------------------------------------------------------------ #

    def record_tool_call(
        self,
        name: str,
        content: str,
        *,
        tool_input: Any = None,
        tool_call_id: str | None = None,
        msg_index: int | None = None,
    ) -> None:
        """Write-through a terminal tool call that produces no tool result.

        A ``submit_answer``-style call ends the loop before any result message
        exists; hosts persist it as its own ``kind='tool_call'`` row so the
        durable log still records what was submitted.
        """
        self._runtime.append_log(
            LogEntry(
                kind="tool_call",
                role="assistant",
                name=name,
                content=content,
                step_index=self._step_index,
                msg_index=self._step_index if msg_index is None else msg_index,
                tool_call_id=tool_call_id,
                tool_input=tool_input,
            )
        )

    @property
    def runtime(self) -> ScrollRuntime:
        """The owned runtime (namespace, memoryspace, durable history)."""
        return self._runtime

    @property
    def obs_keep_turns(self) -> int | None:
        return self._obs_keep_turns

    @property
    def index_enabled(self) -> bool:
        return self._enable_index

    def metrics(self) -> dict:
        out = dict(self.totals)
        turns = out["assistant_turns"]
        out["headline_compliance_rate"] = (out["headlined_turns"] / turns) if turns else None
        out["est_input_tokens"] = self.est_input
        out["history_max_tokens"] = self.history_max_tokens
        out["index_enabled"] = self._enable_index
        out["var_context"] = self._var_context
        out["pinned_views"] = len(self._views.pins)
        return out

    def close(self) -> None:
        self._runtime.close()
