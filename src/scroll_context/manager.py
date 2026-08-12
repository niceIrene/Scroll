"""ScrollContextManager — scroll's context management over OpenAI dict messages.

The canonical implementation of scroll's orchestration (write-through
persistence, observation aging, token-budget eviction folded into the
`EvictionIndex`, the pinned index placeholder, the per-step working-memory
digest, the seeded prior-sessions map), over plain OpenAI chat-completions
messages — `scroll_react` delegates here:

    {"role": "system"|"user"|"assistant"|"tool", "content": str,
     "tool_calls": [...]?, "tool_call_id": str?, "name": str?}

Terminology (canonical for all scroll code and prompts):

- **session** — one host ``run()``; one manager lifetime; ``session_id``.
- **step** — one iteration of the agent loop: one assistant message plus its
  contiguous tool results (an *exchange* — also the atomic eviction group).
  ``step_index`` counts these on the agent's own rows; on seed-tier rows
  ingest reuses the column as the prior-conversation SESSION number (that is
  what ``S<n>`` refers to).
- **row / seq** — one persisted ``LogEntry``; ``seq`` is the table-wide
  primary key, the only globally unique coordinate, and therefore the currency
  of every cross-reference (index spans, stub pointers).

One structural note for the OpenAI format: tool results are separate
``role:"tool"`` messages rather than blocks inside the assistant message, and
an assistant message with ``tool_calls`` must never be left in context without
its paired results (the API rejects orphans). Eviction therefore works on
*groups*: an assistant message plus its contiguous following tool-result
messages evict atomically.

No imports from any host harness — only ``scroll_context._runtime`` and the stdlib —
so any OpenAI-format agent loop can reuse this class.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from scroll_context._runtime import ScrollRuntime
from scroll_context._runtime.index import EvictionIndex, Leaf
from scroll_context._runtime.types import ExecutionResult, LogEntry

_CHARS_PER_TOKEN = 4

# The model marks a milestone turn with a fenced single line ``⟦ text ⟧``
# (U+27E6 / U+27E7 — rare brackets chosen to almost never collide with
# code/markdown the model might also emit). Those become the ``seq · headline``
# leaves of the in-context eviction map; most turns carry none.
_HEADLINE_RE = re.compile(r"^[ \t]*⟦[ \t]*(.+?)[ \t]*⟧[ \t]*$", re.MULTILINE)
_HEADLINE_MAX = 200  # chars — a headline is an index entry, not a paragraph

# Observation aging: every past tool output is replayed verbatim in each
# subsequent prompt, so per-run input cost grows ~quadratically with steps
# while contributing little — by design the model keeps extracted data in REPL
# variables and every output is write-through-persisted the turn it is
# produced. Once a step is more than ``obs_keep_turns`` assistant steps old,
# its tool outputs are replaced in-context with a short head plus a recovery
# note; durable copies are untouched. Tune/disable via SCROLL_OBS_KEEP_TURNS
# (integer; "0" or "off" disables aging).
_OBS_KEEP_TURNS = 3
_OBS_AGE_MIN_CHARS = 600   # outputs at or below this aren't worth stubbing
_OBS_AGE_HEAD_CHARS = 200  # how much of an aged output stays visible
_AGED_MARKER = "[…tool output aged out of this prompt"

# Eviction-index level cap (blocks per level before it carries up).
_INDEX_LEVEL_CAP = 10

_SEED_MAP_HEADER = (
    "<system-info>[memory] An index of your PRIOR conversation sessions (your "
    "long-term memory) — one line per session, newest at the bottom, older ones "
    "carried up as endpoint pairs. The full turns are durable in "
    "hist.conversation_history; browse it coarse-to-fine and expand any span "
    "inside {repl}, exactly like the [context compressed] map."
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    return default


def _obs_keep_turns_from_env() -> int | None:
    """Aging window from the env: int steps to keep, or None = aging disabled."""
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
        n += len(fn.get("name", "") or "") + len(args if isinstance(args, str) else str(args))
    return n


def _est_tokens(msg_or_text: Any) -> int:
    if isinstance(msg_or_text, str):
        return len(msg_or_text) // _CHARS_PER_TOKEN + 1
    return _msg_chars(msg_or_text) // _CHARS_PER_TOKEN + 1


def _extract_headline(text: str | None) -> str | None:
    """The turn's durable index line: the model's own ``⟦ … ⟧`` fence, or None.

    There is intentionally no extractive fallback — a turn with no fence gets
    ``headline=None`` and so does not become a leaf of the eviction map (it
    stays durably stored and recallable by ``seq`` range or ``ms.search``).
    """
    if text:
        m = _HEADLINE_RE.search(text)
        if m and m.group(1).strip():
            return m.group(1).strip()[:_HEADLINE_MAX]
    return None


def _format_execute_observation(result: ExecutionResult) -> str:
    """Compose stdout/stderr/error into one observation string."""
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
    """Per-session scroll context management for an OpenAI-format message list.

    Intended call pattern per loop step:

        mgr.record_assistant_turn(assistant_msg, usage)   # after each response
        mgr.record_tool_result(tool_msg, ...)             # after each tool result
        events = mgr.manage(messages)                     # before next API call
        call_messages = messages + [mgr.digest_message()] # ephemeral digest

    The host appends the SAME dict objects it passed to ``record_*()`` to its
    message list — bookkeeping is keyed by ``id(msg)``. REPL tool calls are
    answered via :meth:`execute_python` / :meth:`execute_python_async`.
    """

    def __init__(
        self,
        *,
        history_db_path: str | Path | None,
        session_id: str,
        run_id: str | None = None,
        task_id: str | None = None,
        history_max_tokens: int,
        pinned: int = 1,
        enable_index: bool | None = None,
        index_level_cap: int | None = None,
        obs_keep_turns: int | None = "env",  # type: ignore[assignment]
        execute_timeout_s: float = 60.0,
        shared_run_ids: tuple[str, ...] = (),
        repl_name: str = "scroll_repl",
        index_header: str | None = None,
        placeholder_name: str | None = None,
    ) -> None:
        # ``repl_name`` is the host's name for the REPL tool — it appears in
        # every model-facing recovery text (map header, aged stubs, digest) so
        # the instructions name a tool that actually exists.
        self._repl_name = repl_name
        self._index_header = index_header
        self._shared_run_ids = tuple(shared_run_ids)
        # Optional OpenAI `name` field stamped on the placeholder message so
        # hosts that surface message names (prompt dumps, AgentScope Msgs) can
        # label it (scroll_react uses "memory").
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
            # Scale the per-call stdout cap to the in-context budget so one
            # print can't flood the window.
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
        self._index_level_cap = index_level_cap
        self._index = EvictionIndex(session_id=session_id, level_cap=index_level_cap)
        self._obs_keep_turns = (
            _obs_keep_turns_from_env() if obs_keep_turns == "env" else obs_keep_turns
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
        (scroll_react stamps the task at ``step_index=-1, msg_index=1``).
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
        """Write-through one assistant turn; track seq/leaf; learn the overhead.

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
                blocks=[dict(msg)],
            )
        )
        seq = self._runtime.persisted_seq
        self._seq_by_id[id(msg)] = (seq, seq)
        if headline:
            self._leaf_by_id[id(msg)] = Leaf(seq=seq, headline=headline)
        self._last_assistant_id = id(msg)
        self.totals["assistant_turns"] += 1
        if headline:
            self.totals["headlined_turns"] += 1

        # Learn the estimate-vs-reality overhead (tool schemas, digest, wire
        # format) from reported usage: the next manage() recomputes the base
        # from the actual list and adds this on top, so estimate error can't
        # accumulate and external list mutation can't desync the accounting.
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
        """Write-through one ``role:"tool"`` result; widen the step's seq span.

        The FULL result text is persisted to the durable history; the
        in-context copy is left verbatim until observation aging stubs it.
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
        # folds a span covering both durable rows — a range query over the
        # evicted span then recovers both.
        if self._last_assistant_id is not None and self._last_assistant_id in self._seq_by_id:
            lo, _hi = self._seq_by_id[self._last_assistant_id]
            self._seq_by_id[self._last_assistant_id] = (lo, seq)
        self.est_input += _est_tokens(msg)

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

    def record_user_message(self, msg: dict, kind: str = "user_message") -> None:
        """Write-through an interleaved user message (a new user turn, a notice).

        Scaffolding the host chooses NOT to persist (e.g. a no-tool-call nudge)
        should simply skip this call — unrecorded messages have no seq mapping
        and are skipped when an eviction folds into the index.
        """
        self._runtime.append_log(
            LogEntry(kind=kind, role="user", content=_content_text(msg), step_index=self._step_index)
        )
        seq = self._runtime.persisted_seq
        self._seq_by_id[id(msg)] = (seq, seq)
        self.est_input += _est_tokens(msg)

    # ------------------------------------------------------------------ #
    # per-step management (aging + eviction + placeholder)

    def manage(self, messages: list[dict]) -> dict:
        """Run the pre-call pipeline in place; return this sweep's event dict.

        The estimate is recomputed from the ACTUAL message list every call
        (plus the usage-learned overhead) rather than trusted incrementally —
        incremental accounting silently diverges if the harness ever mutates
        the list behind our back.
        """
        base = sum(_est_tokens(m) for m in messages)
        self._last_manage_base = base
        self.est_input = base + self._est_overhead
        events: dict[str, Any] = {}
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
        """Stub tool outputs older than the last ``obs_keep_turns`` assistant steps.

        Mutates ``role:"tool"`` message content in place (in-context only — the
        durable copies are already persisted). Idempotent via ``_aged_ids``.
        """
        keep = self._obs_keep_turns
        if keep is None:
            return 0, 0
        assistant_idxs = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        if not assistant_idxs:
            return 0, 0
        # Everything before the keep-th most recent assistant turn is "old".
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
            stub = (
                out[:_OBS_AGE_HEAD_CHARS].rstrip()
                + f"\n{_AGED_MARKER} to save context: "
                f"{len(out)} chars total. The full output is persisted in "
                "hist.conversation_history and any variables you assigned still "
                "hold the data — print from those variables (or re-query) if you "
                "need it again.]"
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
        evicted message's Leaf (if headlined) folds into the eviction index;
        the whole sweep folds as ONE ``add_eviction`` span. Always keeps at
        least the newest group beyond the pinned head.
        """
        base = self._base_index()
        leaves: list[Leaf] = []
        span_lo: int | None = None
        span_hi: int | None = None
        n_evicted = 0
        tokens_evicted = 0
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
                if rng is None:  # unpersisted nudge — nothing durable to fold
                    continue
                span_lo = rng[0] if span_lo is None else min(span_lo, rng[0])
                span_hi = rng[1] if span_hi is None else max(span_hi, rng[1])
                if leaf is not None:
                    leaves.append(leaf)
        if n_evicted:
            self.totals["evicted_msgs"] += n_evicted
            self.totals["evicted_tokens_est"] += tokens_evicted
            if span_lo is not None:
                if self._enable_index:
                    self._index.add_eviction(leaves, seq_lo=span_lo, seq_hi=span_hi)
                else:
                    self._evicted_lo = (
                        span_lo if self._evicted_lo is None else min(self._evicted_lo, span_lo)
                    )
                    self._evicted_hi = (
                        span_hi if self._evicted_hi is None else max(self._evicted_hi, span_hi)
                    )
                self._refresh_placeholder(messages)
        span = (span_lo, span_hi) if span_lo is not None else None
        return n_evicted, tokens_evicted, span

    def _render_placeholder(self) -> str:
        """The single in-context memory placeholder for the evicted middle.

        With the index on it's the structured ``seq · headline`` map; with it
        off (the ablation baseline) it's one opaque [context compressed] line
        carrying the evicted ``seq`` span and the same recall idiom — durable,
        nothing lost.
        """
        if self._enable_index:
            return self._index.render(header=self._index_header, repl_name=self._repl_name)
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
            messages.insert(self._pinned, self._placeholder)
            self.est_input += _est_tokens(self._placeholder)
            return
        old = _est_tokens(self._placeholder)
        self._placeholder["content"] = text
        self.est_input += _est_tokens(self._placeholder) - old

    # ------------------------------------------------------------------ #
    # seeded prior sessions

    def seed_index_map(self) -> str | None:
        """Render durable ``run_id='seed'`` sessions as a [L0]/[L1] memory map.

        One span per seed session showing its FIRST and LAST milestone headline
        (``head - tail``) over a ``seq`` range covering the whole session, so a
        range query recovers every turn. Returns the placeholder text — meant
        to be appended to the (pinned, never-evicted) system prompt — or None
        when this task has no seed rows.
        """
        ms = self._runtime.memoryspace
        rows = ms.sql_query(
            "SELECT ch.session_id AS sid, MIN(ch.seq) AS lo, MAX(ch.seq) AS hi, "
            "(SELECT headline FROM hist.conversation_history h2 "
            " WHERE h2.session_id = ch.session_id AND h2.headline IS NOT NULL "
            " ORDER BY h2.seq LIMIT 1) AS head, "
            "(SELECT headline FROM hist.conversation_history h3 "
            " WHERE h3.session_id = ch.session_id AND h3.headline IS NOT NULL "
            " ORDER BY h3.seq DESC LIMIT 1) AS tail "
            "FROM hist.conversation_history ch "
            "WHERE ch.task_id = ? AND ch.run_id = 'seed' "
            "GROUP BY ch.session_id ORDER BY lo",
            (ms.task_id,),
        )
        rows = [r for r in rows if r.get("lo") is not None and "sid" in r]
        if not rows:
            return None
        index = EvictionIndex(
            session_id=f"seed:{ms.task_id}", level_cap=self._index_level_cap
        )
        for r in rows:
            head = r.get("head") or f"session {r['sid']}"
            # Fall back to head when a session has only one (or zero) headline,
            # so the line collapses to a single headline rather than a dangling
            # pair.
            tail = r.get("tail") or head
            index.add_span(seq_lo=int(r["lo"]), seq_hi=int(r["hi"]), head=head, tail=tail)
        return index.render(
            header=_SEED_MAP_HEADER.replace("{repl}", self._repl_name),
            repl_name=self._repl_name,
        )

    # ------------------------------------------------------------------ #
    # digest + REPL

    def digest_message(self, budget_note: str | None = None) -> dict:
        """The ephemeral per-step ``[working memory]`` message (never persisted)."""
        parts = ["[working memory] " + self._runtime.digest()]
        # Only surface retrieval guidance once history has actually been
        # evicted — and frame it as a CONTENT search (the model knows what it's
        # looking for, not which step number), shown for the rest of the run
        # since evicted turns stay out of the window.
        evicted = self.totals["evicted_msgs"]
        if evicted > 0:
            parts.append(
                f"{evicted} earlier turn(s) are no longer in this prompt but are "
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
        # Budget pressure is surfaced only near the limit (budget_note is None
        # the rest of the run), nudging the model to consolidate and submit.
        if budget_note:
            parts.append(budget_note)
        return {"role": "user", "content": "\n".join(parts)}

    def execute_python(self, source: str) -> str:
        """Run one REPL call in the persistent namespace (sync bridge).

        For hosts already inside an event loop, use :meth:`execute_python_async`
        — ``asyncio.run`` raises when called from a running loop.
        """
        self.totals["repl_calls"] += 1
        result = asyncio.run(self._runtime.execute(source))
        return _format_execute_observation(result)

    async def execute_python_async(self, source: str) -> str:
        """Async variant of :meth:`execute_python` for async host loops."""
        self.totals["repl_calls"] += 1
        result = await self._runtime.execute(source)
        return _format_execute_observation(result)

    # ------------------------------------------------------------------ #

    def protocol_prompt(self) -> str:
        """The model-facing context-management protocol for THIS configuration.

        Assembles the canonical prompt files (``core.md`` [+ ``index.md``] —
        see :mod:`scroll_context.prompts`) with this manager's own
        REPL tool name and index flag, applying the index-off headline-schema
        strip. Hosts embed it between their own preamble and finishing policy,
        so prompt text has exactly one source of truth.
        """
        from scroll_context.prompts import protocol_prompt

        return protocol_prompt(self._repl_name, index=self._enable_index)

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
        # Retrieval-op counters (hist_fts / hist_seq / hist_scan …) so hosts
        # report memory usage without reaching into the runtime.
        out["ms_ops"] = self._runtime.memoryspace.stats()
        return out

    def close(self) -> None:
        self._runtime.close()
