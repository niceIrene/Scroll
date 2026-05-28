"""Symbolic handle on the unified ConversationLog (E).

This is the Python object the LM operates on under the RLM-style
substrate. The log itself is jsonl-backed in
``conversation_log.jsonl``; ``LogHandle`` exposes a small read-only
API that the agent calls from inside REPL cells.

The agent's strategy under X = f(E) is whatever code it writes
against this handle plus its other namespace objects (``memoryspace``,
``rlm``, ``rlm``).
"""

from __future__ import annotations

import json
import re
from typing import Iterable, Iterator

from Scroll.core import LogEntry
from Scroll.log import ConversationLog
from Scroll.tools._embed import _cos, _embed


def _entry_haystack(entry: LogEntry) -> str:
    """Per-entry text payload for substring / regex extraction.

    Mirrors what :meth:`ConversationLog.search` matches against, but
    returns a single concatenated string so :func:`re.findall` can be
    run uniformly across mixed-shape entries:

      - briefings / LM turns / outcomes → ``entry.content``
      - cell stdout mirrors             → ``entry.tool_result["output"]``
      - cell code mirrors               → JSON of ``tool_call["arguments"]``
                                          (the code text)

    All three are concatenated when present so a regex doesn't need to
    know which entry kind a particular line came from.
    """
    parts: list[str] = []
    if entry.content:
        parts.append(entry.content)
    if entry.tool_result:
        out = entry.tool_result.get("output", "")
        if out:
            parts.append(str(out))
    if entry.tool_call:
        args = entry.tool_call.get("arguments", {})
        if args:
            try:
                parts.append(json.dumps(args, default=str))
            except (TypeError, ValueError):
                parts.append(str(args))
    return "\n".join(parts)


class LogHandle:
    """Read-only Python view onto a ``ConversationLog``.

    Methods are sync — the underlying log is in-process and cheap to
    scan. The agent can iterate, slice by session/role, substring-search,
    or render to a single string for passing to ``rlm()``.

    All filter kwargs are inclusive and support ``None`` as "any."
    """

    def __init__(self, log: ConversationLog) -> None:
        self._log = log
        # Lazy parallel cache aligned to ``self._log.entries`` by index.
        # Populated on first ``semantic_search`` call and incrementally
        # extended as the log grows (the log is append-only). Kept on
        # the handle, not the log, so multiple handles don't fight over
        # ownership and the cache evaporates with the REPL globals at
        # session boundaries (cheap to rebuild — bag-of-words is fast).
        self._embeds: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Filter / iteration
    # ------------------------------------------------------------------

    def slice(
        self,
        session_idx: int | None = None,
        role: str | None = None,
        kind: str | None = None,
        *,
        day: int | None = None,  # back-compat alias for session_idx
    ) -> list[LogEntry]:
        """Return entries matching ``session_idx`` and/or ``role`` and/or ``kind``.

        Args:
            session_idx: Restrict to entries whose ``.turn_idx == session_idx``.
                (Param name kept for agent-prompt back-compat; reads
                from the renamed ``LogEntry.turn_idx`` field.)
            role: Restrict to entries whose ``.role == role``
                (one of ``"system"``, ``"user"``, ``"assistant"``, ``"tool"``).
            kind: Restrict to entries whose ``metadata.kind == kind``
                (e.g. ``"code"``, ``"stdout"``, ``"compression_summary"``).
            day: deprecated alias for ``session_idx`` (kept so old agent
                prompts that say ``log.slice(day=N)`` still work).
        """
        if session_idx is None and day is not None:
            session_idx = day
        out: list[LogEntry] = []
        for e in self._log.entries:
            if session_idx is not None and e.turn_idx != session_idx:
                continue
            if role is not None and e.role != role:
                continue
            if kind is not None and (e.metadata or {}).get("kind") != kind:
                continue
            out.append(e)
        return out

    def range_by_session(self, start: int, end: int) -> list[LogEntry]:
        """Return entries with ``start <= entry.turn_idx <= end``."""
        return [e for e in self._log.entries if start <= e.turn_idx <= end]

    # Back-compat alias — agent system prompts mention ``range_by_day``.
    range_by_day = range_by_session

    def search(self, query: str, k: int = 20) -> list[LogEntry]:
        """Substring + tool-payload search, newest-first.

        Case-insensitive match against ``.content``, ``.tool_call`` (JSON),
        and ``.tool_result`` (JSON). Returns at most ``k`` matches.

        For recall-driven probes where the question's wording does NOT
        match the log's wording (paraphrase, multi-word query with
        partial overlap), use :meth:`semantic_search` instead — substring
        needs an exact lexical hit.
        """
        q = query.lower().strip()
        if not q:
            return list(reversed(self._log.entries))[:k]
        hits: list[LogEntry] = []
        for entry in reversed(self._log.entries):
            haystack = entry.content.lower()
            if entry.tool_call:
                haystack += " " + json.dumps(entry.tool_call, default=str).lower()
            if entry.tool_result:
                haystack += " " + json.dumps(entry.tool_result, default=str).lower()
            if q in haystack:
                hits.append(entry)
                if len(hits) >= k:
                    break
        return hits

    def semantic_search(
        self,
        query: str,
        k: int = 10,
        *,
        session_idx: int | None = None,
        role: str | None = None,
        kind: str | None = None,
        entries: Iterable[LogEntry] | None = None,
        min_score: float = 0.0,
        day: int | None = None,  # back-compat alias for session_idx
    ) -> list[tuple[LogEntry, float]]:
        """Vector-similarity search over E. NEAREST-NEIGHBOR, not substring.

        The semantic counterpart to :meth:`search`. Returns the entries
        whose token distribution is closest to ``query`` under the
        bag-of-words cosine metric used elsewhere in the codebase
        (:func:`Scroll.tools._embed._embed`). This is LEXICAL (token
        overlap) rather than deep neural — it tolerates word order /
        partial overlap (``"how many sessions did sales exceed quota"``
        will rank a line saying ``"sold 20 units above target on session
        5"`` higher than substring would) but does not understand
        true synonymy (``"dog"`` ↔ ``"puppy"``). Treat it as a
        recall-rerank tool, not an oracle.

        Args:
            query: free-text question or topic phrase.
            k: max results returned.
            session_idx, role, kind: same filter semantics as :meth:`slice`.
            entries: pre-filtered slice to rerank instead of the
                whole log (e.g. results from :meth:`range_by_session`).
                When supplied, embeddings for those entries are
                computed ad-hoc — the per-handle cache only covers
                the full log, by index.
            min_score: drop candidates with cosine ≤ this threshold.
                Default 0.0 keeps anything with at least one shared
                token; raise to ~0.1+ to filter near-noise hits.

        Returns:
            ``list[(LogEntry, score)]`` ranked by score descending.
            Length ≤ ``k``. Empty if the log is empty, the query
            embeds to nothing (no alphanumeric tokens), or all
            candidates fall at/under ``min_score``.

        Example::

            # Recall-rerank: pull a wide candidate set, then rank.
            for entry, score in log.semantic_search(
                "customer complaints about delivery delays", k=5,
                role="user",
            ):
                print(f"session {entry.turn_idx} score={score:.2f} :: "
                      f"{(entry.content or '')[:80]}")
        """
        if session_idx is None and day is not None:
            session_idx = day
        q_vec = _embed(query)
        if not q_vec:
            return []

        if entries is not None:
            cands: list[tuple[LogEntry, float]] = []
            for e in entries:
                if not self._matches_filters(e, session_idx=session_idx, role=role, kind=kind):
                    continue
                v = _embed(_entry_haystack(e))
                cands.append((e, _cos(q_vec, v)))
        else:
            self._ensure_embeds()
            cands = []
            for i, e in enumerate(self._log.entries):
                if not self._matches_filters(e, session_idx=session_idx, role=role, kind=kind):
                    continue
                cands.append((e, _cos(q_vec, self._embeds[i])))

        threshold = float(min_score)
        cands = [(e, s) for e, s in cands if s > threshold]
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands[: int(k)]

    @staticmethod
    def _matches_filters(
        entry: LogEntry,
        *,
        session_idx: int | None,
        role: str | None,
        kind: str | None,
    ) -> bool:
        if session_idx is not None and entry.turn_idx != session_idx:
            return False
        if role is not None and entry.role != role:
            return False
        if kind is not None and (entry.metadata or {}).get("kind") != kind:
            return False
        return True

    def _ensure_embeds(self) -> None:
        """Embed any entries appended since the last :meth:`semantic_search`.

        Cache is index-aligned to ``self._log.entries``. If the
        underlying log was truncated (e.g. checkpoint resume rewinds
        ``entries`` to ``entry_count``), drop the stale cache and
        rebuild — otherwise indices into ``_embeds`` would point at
        old, since-deleted entries.
        """
        n = len(self._log.entries)
        if len(self._embeds) > n:
            self._embeds = []
        while len(self._embeds) < n:
            i = len(self._embeds)
            self._embeds.append(_embed(_entry_haystack(self._log.entries[i])))

    def findall(
        self,
        pattern: str,
        entries: Iterable[LogEntry] | None = None,
        *,
        flags: int = 0,
    ) -> list:
        """Run a regex over ``entries`` and concatenate matches. NO LLM call.

        DETERMINISTIC field extraction — use this in place of
        ``rlm`` whenever the answer is structurally extractable
        (numbers, ids, units, dates). Cheaper (zero LLM tokens) and
        more reliable (no rounding / hallucination) than asking a
        sub-LM to "sum the values after rev=" or similar.

        Per-entry haystack mirrors :meth:`ConversationLog.search`:
        ``entry.content`` + ``tool_result["output"]`` +
        JSON of ``tool_call["arguments"]``, joined with newlines.
        So you can run a regex over a heterogeneous slice (briefings,
        cell stdouts, cell code mirrors) without thinking about the
        entry shape.

        Return shape — follows :func:`re.findall` plus a named-group
        ergonomic extension:

          - 0 capture groups        → list[str]   (each whole match)
          - 1 unnamed capture group → list[str]   (the captured group)
          - 2+ unnamed groups       → list[tuple[str, ...]]
          - ANY named groups        → list[dict[str, str]] keyed by name

        Args:
            pattern: regex pattern (``re`` module syntax).
            entries: iterable of LogEntry to scan; defaults to the whole log.
            flags: ``re`` flags (e.g. ``re.IGNORECASE``).

        Examples::

            # Total revenue over a session range, no LLM call.
            sale_entries = log.search("sale sku=", k=200)
            revs = log.findall(r"rev=([\\d.]+)", sale_entries)
            total = sum(float(r) for r in revs)

            # Structured rows via named groups.
            rows = log.findall(
                r"sku=(?P<sku>\\w+) units=(?P<units>\\d+) "
                r"rev=(?P<rev>[\\d.]+)",
                sale_entries,
            )
            # rows = [{"sku": "cola", "units": "5", "rev": "12.50"}, ...]
        """
        rx = re.compile(pattern, flags)
        seq = list(entries) if entries is not None else list(self._log.entries)
        has_named = bool(rx.groupindex)
        out: list = []
        for e in seq:
            text = _entry_haystack(e)
            if not text:
                continue
            if has_named:
                for m in rx.finditer(text):
                    out.append({k: m.group(k) for k in rx.groupindex})
            else:
                out.extend(rx.findall(text))
        return out

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def text(self, entries: Iterable[LogEntry] | None = None) -> str:
        """Render ``entries`` (or the whole log) to a single string.

        Use this when you want to pass a chunk of E to ``rlm(...)`` for
        sub-LM analysis. Each entry becomes one line:

            [session N | role] content
            [session N | tool] tool_name -> output

        The whole-log form is intentionally lossy — drop it through
        slicing first if E is large, or you'll blow the sub-LM's
        context.
        """
        seq = list(entries) if entries is not None else list(self._log.entries)
        lines: list[str] = []
        for e in seq:
            prefix = f"[session {e.turn_idx} | {e.role}]"
            if e.tool_call:
                name = (e.tool_call or {}).get("name", "?")
                args = (e.tool_call or {}).get("arguments", {})
                lines.append(f"{prefix} call {name}({args})")
            elif e.tool_result:
                name = (e.tool_result or {}).get("name", "?")
                output = (e.tool_result or {}).get("output", "")
                lines.append(f"{prefix} {name} -> {output}")
            elif e.content:
                lines.append(f"{prefix} {e.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Python container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._log.entries)

    def __getitem__(self, key: int | slice):
        return self._log.entries[key]

    def __iter__(self) -> Iterator[LogEntry]:
        return iter(self._log.entries)

    def __repr__(self) -> str:
        return f"<LogHandle entries={len(self._log.entries)}>"


__all__ = ["LogHandle"]
