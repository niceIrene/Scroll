from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable

from opentelemetry import trace as _otel_trace


_DEFAULT_ROW_CAP = 1000

# Tokens FTS5 treats as bare operators — leave them untouched when sanitizing.
_FTS_OPERATORS = {"AND", "OR", "NOT", "NEAR"}


def _fts_sanitize(query: str) -> str:
    """Make a raw query safe to pass as an FTS5 MATCH expression.

    FTS5 parses punctuation (``.`` ``-`` ``(`` ``:`` …) as syntax, so a literal
    term like ``franc v6.1.0`` or ``DeepL API v2`` raises a syntax error. Quote
    each non-operator whitespace token as a phrase (doubling any embedded
    quote) so the version/punctuation is matched literally instead of parsed.
    Bare operators, already-quoted phrases, and ``prefix*`` terms pass through.
    """
    out: list[str] = []
    for tok in query.split():
        if tok.upper() in _FTS_OPERATORS or tok.startswith('"'):
            out.append(tok)
        elif re.fullmatch(r"[A-Za-z0-9_]+\*?", tok):
            out.append(tok)  # bare word or prefix term — already FTS-safe
        else:
            out.append('"' + tok.replace('"', '""') + '"')
    return " ".join(out)


# Default per-hit content preview length in ms.search (snippet=False mode).
# Previews exist purely for triage — the model picks which seqs to ms.expand —
# and 300 chars is enough to recognize a turn; halving from 600 measurably cuts
# per-observation replay cost on big-turn corpora without changing the flow.
_SEARCH_PREVIEW_CHARS = 300


def _preview_hits(rows: list[dict], chars: int | None) -> list[dict]:
    """Truncate each hit's ``content`` to a ``chars``-long preview in place.

    Keeps the headline plus a pointer to expand the full turn by ``seq``, so the
    model can triage on cheap previews and pull full content only for the hits
    that matter. ``chars=None`` keeps full content; a turn already shorter than
    ``chars`` passes through untouched — a no-op on small-turn corpora (1M tier)
    and a large saving on big-turn ones (10M tier, median ~2.3k chars/turn).
    """
    if chars is None:
        return rows
    for r in rows:
        c = r.get("content") or ""
        if len(c) > chars:
            r["content"] = (
                c[:chars].rstrip()
                + f"… ⟨+{len(c) - chars} more chars — pull the full untruncated "
                f"turn (and its code) with ms.expand([{r.get('seq')}])⟩"
            )
    return rows


def sanitize_suffix(session_id: str | None) -> str:
    """Turn a session id into a SQL-identifier-safe table suffix.

    ``session_id`` is ``f"{run_id}:{task_id}"`` — the ``:`` and any other
    punctuation are not valid in a bare table name, so collapse anything that
    isn't alphanumeric/underscore to ``_``.
    """
    if not session_id:
        return "scratch"
    return re.sub(r"[^0-9A-Za-z_]", "_", session_id)

# A term is "multi-word" (needs quoting as an FTS5 phrase) if it contains
# anything other than word chars — whitespace, hyphens, slashes, etc. Bare
# hyphenated tokens like `message-passing` otherwise parse as `message NOT
# passing` (or error), and spaces become an implicit AND.
_FTS_NEEDS_QUOTE = re.compile(r"[^\w*]")


def or_terms(terms: Iterable[str]) -> str:
    """Build a safe FTS5 ``OR`` match expression from a list of terms.

    Joining alternatives with ``OR`` is how you get *recall* out of FTS5: a bare
    space-separated query is an implicit AND of every token, so a multi-word
    string usually matches nothing. Each term that isn't a single bare word
    (phrases, hyphenated tokens like ``message-passing``, ``a/b``) is wrapped in
    double quotes so it matches as a phrase instead of AND-ing or erroring.
    Prefix terms (``deploy*``) are passed through unquoted. Example::

        or_terms(["module", "message-passing", "event driven"])
        # -> 'module OR "message-passing" OR "event driven"'
    """
    parts: list[str] = []
    for raw in terms:
        t = (raw or "").strip()
        if not t:
            continue
        already_quoted = t.startswith('"') and t.endswith('"')
        if already_quoted or not _FTS_NEEDS_QUOTE.search(t):
            parts.append(t)
        else:
            parts.append('"' + t.replace('"', "") + '"')
    return " OR ".join(parts)
# Compact-digest sizing for snippet=True. FTS5's snippet() caps at 64 tokens, so
# we ask for the full window. The LIKE fallback can't centre on the match, so it
# returns a head-of-prose slice of this many characters instead.
_SNIPPET_TOKENS = 64
_SNIPPET_LIKE_CHARS = 160

# BM25 column weights for (prose, code, headline). Prose dominates rank; code
# still contributes a little when search_code=True so a code-only hit can
# surface. The headline weight only matters for the headline-scoped sub-query
# (column filters keep the paths separate).
_BM25_WEIGHTS = (10.0, 0.5, 5.0)

# Fusion / auto-broaden knobs for `search`.
_RRF_K = 60                 # reciprocal-rank-fusion constant
_HEADLINE_CAP_FRACTION = 2  # at most k/2 headline-only hits in a fused result
_BROADEN_MIN_HITS = 3       # exact hits below this trigger the OR fallback
_ROUTE_TURNS_PER_SESSION = 2  # headline-routed turn rows surfaced per session
_BROADEN_CANDIDATE_MULT = 3 # OR candidates fetched per k before reranking
_FTS_STOP_TOKENS = {"OR", "AND", "NOT", "NEAR", "the", "and", "for", "with", "from"}


def _broaden_tokens(query: str) -> list[str]:
    """Content tokens of a query worth OR-ing (words ≥3 chars, numbers ≥2)."""
    toks = re.findall(r"[A-Za-z][A-Za-z']{2,}|\d{2,}", query)
    out: list[str] = []
    for t in toks:
        if t in _FTS_STOP_TOKENS or t.lower() in _FTS_STOP_TOKENS:
            continue
        if t not in out:
            out.append(t)
    return out


def _rrf_fuse(primary: list[dict], secondary: list[dict], *, cap_secondary: int) -> list[dict]:
    """Reciprocal-rank fusion of two ranked row lists keyed by ``seq``.

    Rows in both lists get ``via='both'``; secondary-only rows keep their own
    ``via`` and are capped at ``cap_secondary`` so a short-field match (headlines
    score high under BM25 length normalization) can't swamp the exact hits.
    """
    scores: dict[int, float] = {}
    rows: dict[int, dict] = {}
    for rank, r in enumerate(primary):
        scores[r["seq"]] = scores.get(r["seq"], 0.0) + 1.0 / (_RRF_K + rank)
        rows.setdefault(r["seq"], r)
    n_secondary = 0
    for rank, r in enumerate(secondary):
        if r["seq"] in rows:
            rows[r["seq"]]["via"] = "both"
            scores[r["seq"]] += 1.0 / (_RRF_K + rank)
            continue
        if n_secondary >= cap_secondary:
            continue
        n_secondary += 1
        scores[r["seq"]] = 1.0 / (_RRF_K + rank)
        rows[r["seq"]] = r
    return [rows[s] for s in sorted(scores, key=lambda s: -scores[s])]


def _scoped_match(query: str, *, search_code: bool, code_only: bool) -> str:
    """Wrap an FTS5 match expression in a column filter (prose / code / both).

    By default a search is scoped to the ``prose`` column so code tokens can't
    match at all — the cleanest way to keep a code dump from out-ranking the
    turns that actually answer the query. ``search_code`` widens to both
    columns; ``code_only`` targets ``code`` (e.g. "find the turn with this
    snippet"). The user expression is parenthesised so OR/phrase grouping is
    preserved under the column filter.
    """
    cols = "code" if code_only else ("prose code" if search_code else "prose")
    return f"{{{cols}}} : ({query})"


def _to_seq_list(x) -> list[int]:
    """Coerce ``expand()``'s argument into a list of seq ints.

    So the triage→read carry needs no glue: ``expand`` accepts a single seq, an
    iterable of seqs, an iterable of hit/row dicts (each carrying ``'seq'``), or a
    ``{seq: row}`` dict (its keys are the seqs). This is why
    ``ms.expand(ms.search(...))`` and ``ms.expand([h["seq"] for h in hits if …])``
    both just work.
    """
    if isinstance(x, int):
        return [x]
    if isinstance(x, dict):
        return [int(s) for s in x]  # find()'s {seq: hit} — iterating yields seqs
    return [int(i["seq"]) if isinstance(i, dict) else int(i) for i in x]


class MemorySpace:
    """The model's read-only window onto its durable conversation history.

    This is a query surface, **not** a place to store working data — the model
    keeps intermediate findings in plain Python variables in the persistent
    ``execute_python`` namespace (which survive across turns), and uses ``ms``
    only to *retrieve* past turns into those variables.

    When ``history_db_path`` is given, the durable ``conversation_history``
    file is ATTACHed **read-only** as schema ``hist``. The model can then
    ``ms.search(...)`` (FTS5) or ``ms.sql_query("SELECT ... FROM
    hist.conversation_history ...")`` to retrieve across sessions; any write to
    ``hist.*`` is rejected by SQLite itself. The runtime writes history through
    a *separate* connection (see ``HistoryStore``); under WAL the read-only
    attach sees committed turns. The in-memory connection here exists only to
    host that attach and run the read queries.

    Returned rows are capped (``row_cap``) so a runaway SELECT can't bomb the
    model's context; truncation is flagged with a trailing ``_truncated`` row.
    """

    def __init__(
        self,
        *,
        history_db_path: str | Path | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        shared_run_ids: tuple[str, ...] = (),
        row_cap: int = _DEFAULT_ROW_CAP,
    ) -> None:
        # uri=True so the read-only ATTACH below can use a file: URI.
        self._conn = sqlite3.connect(":memory:", uri=True)
        self._conn.row_factory = sqlite3.Row
        self._row_cap = row_cap
        self._session_id = session_id
        self._task_id = task_id
        # Run ids whose rows form a *shared background tier*: under
        # ``scope='task'`` they are visible to every session of the task, while
        # all OTHER task rows are limited to this session. Empty (default) keeps
        # the plain "all runs of this task" scan. The caller names the tier (an
        # eval may seed prior turns under a sentinel run id and pass it here so
        # sibling sessions sharing one history DB don't leak into each other);
        # this class stays agnostic about what the value means.
        self._shared_run_ids = tuple(shared_run_ids)
        self._fts_ok: bool | None = None  # cached FTS5-availability check
        # Per-op interaction counters over the file-backed persisted store
        # (`hist`), split by read route — FTS keyword search, seq/structured
        # query, and the LIKE fallback scan. Exposed via stats() and emitted as
        # span events so each interaction is visible in Phoenix.
        self._stats = {
            "hist_fts": 0, "hist_seq": 0, "hist_scan": 0,
        }
        if history_db_path is not None:
            abs_path = Path(history_db_path).expanduser().resolve()
            self._conn.execute(
                "ATTACH DATABASE ? AS hist", (f"file:{abs_path}?mode=ro",)
            )

    @property
    def session_id(self) -> str | None:
        """The current session id (this run only) for ``hist.conversation_history``."""
        return self._session_id

    @property
    def task_id(self) -> str | None:
        """The current task id — scopes ``hist`` across ALL runs of this task."""
        return self._task_id

    def stats(self) -> dict:
        """Interaction counts: hist_fts/seq/scan (the hist read routes)."""
        return dict(self._stats)

    @staticmethod
    def _classify(sql: str) -> str:
        low = sql.lower()
        if "hist." in low or "conversation_history" in low:
            if "_fts" in low or " match " in low or "bm25(" in low:
                return "hist_fts"
            if " like " in low:
                return "hist_scan"
            return "hist_seq"
        return "scratch"

    def _record(self, sql: str, *, rows: int | None = None) -> None:
        """Count the op and emit a span event (no-op if tracing is off)."""
        kind = self._classify(sql)
        self._stats[kind] = self._stats.get(kind, 0) + 1
        try:
            attrs: dict = {"ms.kind": "hist_read", "ms.sql": sql[:300]}
            if rows is not None:
                attrs["ms.rows"] = rows
            _otel_trace.get_current_span().add_event("ms.hist_read", attrs)
        except Exception:  # noqa: BLE001 - monitoring must never break the loop
            pass

    def sql_query(self, sql: str, params: tuple | dict | None = None) -> list[dict]:
        """Run a read query over ``hist.conversation_history``. Returns ≤ row_cap rows.

        Rows come back as plain dicts. If more than ``row_cap`` rows match, only
        the first ``row_cap`` are returned plus a trailing ``_truncated`` marker.
        Writes are not supported — ``hist`` is read-only; keep working data in
        Python variables in your ``execute_python`` namespace instead.
        """
        cur = self._conn.execute(sql, params or ())
        rows: list[dict] = []
        for i, row in enumerate(cur):
            if i >= self._row_cap:
                # Shape-compatible truncation marker: carry the SAME keys as the
                # data rows (empty strings) so `for r in rows: r["seq"] …` loops
                # don't KeyError on it — a recurring source of wasted agent turns
                # — with the notice in the first column so it still surfaces in
                # any printed view. `_truncated`/`_row_cap` stay for explicit
                # checks (documented behavior).
                marker: dict = {k: "" for k in rows[-1]} if rows else {}
                first_col = next(iter(marker), None)
                if first_col is not None:
                    marker[first_col] = (
                        f"[truncated: only the first {self._row_cap} matching rows "
                        "are returned — narrow the query with WHERE, or page with "
                        "LIMIT/OFFSET]"
                    )
                marker["_truncated"] = True
                marker["_row_cap"] = self._row_cap
                rows.append(marker)
                break
            rows.append({k: row[k] for k in row.keys()})
        self._record(sql, rows=len(rows))
        return rows

    def search(
        self,
        query: str,
        *,
        scope: str = "session",
        kind: str | None = None,
        k: int = 10,
        snippet: bool = True,
        search_code: bool = False,
        code_only: bool = False,
        step_range: tuple[int, int] | None = None,
        msg_range: tuple[int, int] | None = None,
        steps: Iterable[int] | None = None,
        seqs: Iterable[int] | None = None,
        chars: int | None = _SEARCH_PREVIEW_CHARS,
    ) -> list[dict]:
        """Full-text search over ``hist.conversation_history`` (FTS5).

        Returns up to ``k`` rows ``{seq, step_index, kind, role, name, headline,
        content, has_code, via[, broadened]}`` ranked by relevance (bm25) —
        ``role`` and ``headline`` let the model judge a hit (and locate its
        milestone) without a second query. ``query`` is an FTS5 match expression
        (plain words, phrases ``"..."``, ``OR``, prefix ``term*``).

        Two recall layers beyond the plain prose match:
        - ``via``: ``'prose'`` = the turn's text matched; ``'headline'`` = the
          turn was ROUTED here — a session's model-written summary matched the
          query, and this row is the best-matching turn found *inside* that
          session (marker ``⟦via summary of S<n>: …⟧`` appended to its
          snippet/content). A routed row is an honest turn (its own seq and
          text); the summary is provenance, not content. When more sessions
          matched than could be surfaced, the last routed row's marker lists
          them for sweeping. ``'both'`` = prose-matched and routed.
        - ``broadened: True``: the exact query returned fewer than
          ``_BROADEN_MIN_HITS`` rows, so it was automatically retried with
          OR-joined terms; these extra rows are ranked by how many of the
          original terms they contain. Treat them as leads to verify, not
          confirmed matches.

        Code blocks are indexed in a **separate column** from prose and the
        search is scoped to prose by default, so code tokens never inflate BM25
        rank, and the returned ``content`` is the prose with fenced blocks elided
        to ``‹code›`` markers — the preview budget holds reasoning text instead of
        a code dump. ``has_code`` flags hits that carry code; pull the full
        untruncated turn (prose + code) with ``expand([seq])``. Set
        ``search_code=True`` to match prose+code, or ``code_only=True`` to search
        only the code column (e.g. locating a turn by a code snippet). ``scope``
        limits to
        ``'session'`` (this run, default), ``'task'`` (all runs of this task —
        or, when ``shared_run_ids`` is set, just that shared tier plus this
        session), or ``'all'``. ``kind`` optionally filters (e.g. ``'tool_result'``).
        Each hit's ``content`` is truncated to a ``chars``-long preview (default
        600) with a pointer to expand the full turn by ``seq`` — triage cheaply,
        then pull full content only for the hits that matter; ``chars=None``
        returns full content. Falls back to a ``LIKE`` scan if SQLite lacks FTS5.

        Nothing is printed — ``search`` only **returns** rows; you decide what to
        show. ``snippet`` (default **True**) picks the row shape. **``snippet=True``
        — wide retrieval:** each hit's ``content`` is replaced with a compact,
        match-centred ``snippet`` (FTS5 ``snippet()``); print a line per hit
        (`for h in hits: print(h["seq"], h["snippet"])`) to triage which turns are
        relevant, then ``expand`` the seqs that matter. **``snippet=False`` —
        for aggregation:** rows carry the (preview-truncated) ``content`` for
        filtering / counting / intersecting in code — **do not print these rows
        directly** (that dumps bulk text into your context); aggregate, then print
        only the result. ``step_range`` / ``msg_range`` are inclusive ``(lo, hi)``
        filters on ``step_index`` (session) / ``msg_index`` (chronology) — use them
        to keyword-triage the turns inside a time/session window you have scoped.

        ``steps`` / ``seqs`` restrict the search to an explicit *set* of
        sessions (``step_index IN ...``) / turns (``seq IN ...``) — the scope a
        prior pass narrowed to. This is how you run a second, more precise
        search only over the candidates a generous first pass surfaced
        (progressive narrowing) instead of re-scanning the whole task. An empty
        iterable matches nothing (returns ``[]``).
        """
        steps = None if steps is None else [int(s) for s in steps]
        seqs = None if seqs is None else [int(s) for s in seqs]
        if steps is not None and not steps or seqs is not None and not seqs:
            return []  # an explicit empty candidate set matches nothing
        if not self._fts_available():
            # No FTS5 in this SQLite build — degrade to a LIKE scan.
            sql, params, rows = self._search_like(
                query, scope, kind, int(k), snippet, step_range, msg_range, steps, seqs
            )
            self._record(sql, rows=len(rows))
            return _preview_hits(rows, chars)
        # FTS5 auxiliary funcs (bm25/snippet) and the `tbl MATCH` syntax need the
        # table NAME, not an alias — so reference `conversation_history_fts`
        # directly (resolves to hist.* since scratch has no such table).
        fts = "conversation_history_fts"
        # Shared non-MATCH filters, reused by every sub-query below.
        filters: list[str] = []
        fparams: list = []
        if scope == "session" and self._session_id:
            filters.append("ch.session_id = ?")
            fparams.append(self._session_id)
        elif scope == "task" and self._task_id:
            filters.append("ch.task_id = ?")
            fparams.append(self._task_id)
            if self._shared_run_ids:
                # Shared background tier + this session only — don't surface a
                # sibling session's turns from a shared history DB.
                ph = ",".join("?" * len(self._shared_run_ids))
                filters.append(f"(ch.run_id IN ({ph}) OR ch.session_id = ?)")
                fparams.extend(self._shared_run_ids)
                fparams.append(self._session_id)
        if kind:
            filters.append("ch.kind = ?")
            fparams.append(kind)
        if step_range is not None:
            filters.append("ch.step_index BETWEEN ? AND ?")
            fparams.extend([int(step_range[0]), int(step_range[1])])
        if msg_range is not None:
            filters.append("ch.msg_index BETWEEN ? AND ?")
            fparams.extend([int(msg_range[0]), int(msg_range[1])])
        if steps is not None:
            filters.append(f"ch.step_index IN ({','.join('?' * len(steps))})")
            fparams.extend(steps)
        if seqs is not None:
            filters.append(f"ch.seq IN ({','.join('?' * len(seqs))})")
            fparams.extend(seqs)
        # snippet() centres a short excerpt on the match: col 0 is `prose`, col 1
        # `code`. Non-snippet hits return prose (code elided to ‹code› markers)
        # as `content`, with has_code flagging turns whose code is retrievable.
        snip_col = 1 if code_only else 0
        text_col = (
            f"snippet({fts}, {snip_col}, '', '', ' … ', {_SNIPPET_TOKENS}) AS snippet"
            if snippet
            else "ch.prose AS content"
        )
        w_prose, w_code, w_head = _BM25_WEIGHTS
        select_cols = (
            f"SELECT ch.seq, ch.step_index, ch.kind, ch.role, ch.name, "
            f"ch.headline, {text_col}, ch.prose AS _prose, ch.session_id AS _sid, "
            f"(ch.code IS NOT NULL AND ch.code != '') AS has_code "
            f"FROM hist.{fts} JOIN hist.conversation_history ch "
            f"ON ch.seq = {fts}.rowid "
        )

        def _run(
            match_expr: str,
            limit: int,
            via: str,
            extra_where: str = "",
            extra_params: tuple = (),
        ) -> list[dict] | None:
            """One FTS sub-query; None on an unparseable match expression."""
            clauses = [f"{fts} MATCH ?"] + filters + ([extra_where] if extra_where else [])
            sql = (
                select_cols + "WHERE " + " AND ".join(clauses)
                + f" ORDER BY bm25({fts}, {w_prose}, {w_code}, {w_head}) LIMIT ?"
            )
            try:
                cur = self._conn.execute(
                    sql, [match_expr, *fparams, *extra_params, limit]
                )
            except sqlite3.OperationalError:
                return None
            out = []
            for r in cur:
                row = {kk: r[kk] for kk in r.keys()}
                row["via"] = via
                out.append(row)
            return out

        # 1) Exact match, scoped to prose (or code) as before; on an FTS syntax
        # error retry once with each term phrase-quoted (literal punctuation).
        effective_query = query
        exact = _run(_scoped_match(query, search_code=search_code, code_only=code_only), int(k), "prose")
        if exact is None:
            effective_query = _fts_sanitize(query)
            exact = _run(
                _scoped_match(effective_query, search_code=search_code, code_only=code_only),
                int(k), "prose",
            )
            if exact is None:
                # Still unparseable — degrade to a substring scan.
                sql, params, rows = self._search_like(
                    query, scope, kind, int(k),
                    search_code=search_code, code_only=code_only,
                )
                self._record(sql, rows=len(rows))
                return _preview_hits(rows, chars)

        # 2) Headline ROUTER. Session summaries are a second vocabulary register
        # (question-altitude phrasing), but a headline describes a whole SESSION,
        # not the boundary turn it happens to be pinned to — so headline matches
        # are used to route, never returned as evidence themselves:
        #   stage 1 (uncapped): OR-match the query terms over ALL headlines and
        #     rank the matched sessions by distinct-term coverage (BM25 on a
        #     12-word field over-rewards a single rare term), then by how many
        #     of the session's headlines matched, then first-seen BM25 order;
        #   stage 2 (capped, <=_ROUTE_TURNS_PER_SESSION rows per session,
        #     <=k/2 routed rows total): search the query WITHIN each top
        #     session's turns — strict AND first, then OR reranked by matched-
        #     term count — and surface honest turn rows (their own seq and
        #     snippet), marked `⟦via summary of S<n>: …⟧`. If nothing in the
        #     span matches any term, fall back to the session's opening turn as
        #     an explicitly-marked low-confidence lead. The last routed row's
        #     marker lists matched-but-unsurfaced sessions so aggregation
        #     questions can sweep them by step_index.
        routed: list[dict] = []
        overflow_note = ""
        route_toks = [] if code_only else _broaden_tokens(effective_query)
        if route_toks:
            or_expr_h = or_terms(route_toks)
            h_hits = _run(f"{{headline}} : ({or_expr_h})", 200, "headline") or []
            low_toks = [t.lower() for t in route_toks]
            sess: dict = {}
            for rank, r in enumerate(h_hits):
                s = sess.setdefault(r["_sid"], {"terms": set(), "n_heads": 0,
                                                "rank": rank, "head": r["headline"] or "",
                                                "step": r["step_index"]})
                s["n_heads"] += 1
                hl = (r["headline"] or "").lower()
                s["terms"].update(t for t in low_toks if t in hl)
            ranked = sorted(
                sess.items(),
                key=lambda kv: (-len(kv[1]["terms"]), -min(kv[1]["n_heads"], 2), kv[1]["rank"]),
            )
            budget = max(1, int(k) // _HEADLINE_CAP_FRACTION)
            surfaced_sids: list = []
            for sid, info in ranked:
                if len(routed) >= budget:
                    break
                extra, ep = "ch.session_id = ?", (sid,)
                rows = _run(
                    _scoped_match(effective_query, search_code=search_code, code_only=False),
                    _ROUTE_TURNS_PER_SESSION, "headline", extra, ep,
                ) or []
                if not rows:
                    cand = _run(
                        _scoped_match(or_expr_h, search_code=search_code, code_only=False),
                        10, "headline", extra, ep,
                    ) or []
                    def _tc(row: dict) -> int:
                        text = (row.get("_prose") or "").lower()
                        return sum(1 for t in low_toks if t in text)
                    cand = [r for r in cand if _tc(r) > 0]
                    cand.sort(key=_tc, reverse=True)  # stable: bm25 within ties
                    rows = cand[:_ROUTE_TURNS_PER_SESSION]
                if not rows:
                    # Vocabulary-gap fallback: the session is ABOUT the query per
                    # its summary but no turn contains any query term — surface
                    # its opening turn as a marked low-confidence lead.
                    op = self._conn.execute(
                        "SELECT seq, step_index, kind, role, name, headline, "
                        "prose AS content, prose AS _prose, session_id AS _sid, "
                        "(code IS NOT NULL AND code != '') AS has_code "
                        "FROM hist.conversation_history WHERE session_id = ? "
                        "ORDER BY seq LIMIT 1", (sid,),
                    ).fetchone()
                    if op:
                        row = {kk: op[kk] for kk in op.keys()}
                        if snippet:
                            row["snippet"] = (row.pop("content") or "")[:120]
                        row["via"] = "headline"
                        rows = [row]
                for r in rows[: max(0, budget - len(routed))][:_ROUTE_TURNS_PER_SESSION]:
                    r["via"] = "headline"
                    r["_marker"] = (
                        f" ⟦via summary of S{info['step']}: {info['head'][:90]}⟧"
                    )
                    routed.append(r)
                surfaced_sids.append(sid)
            overflow = [info for sid, info in ranked if sid not in surfaced_sids]
            if overflow:
                labels = " ".join(f"S{o['step']}" for o in overflow[:12])
                more = "…" if len(overflow) > 12 else ""
                overflow_note = (
                    f" ⟦{len(overflow)} more session(s) matched summaries: {labels}{more}"
                    " — sweep by step_index for aggregate questions⟧"
                )
        # Routed rows are ADDITIVE leads, not competitors: they must never
        # displace an exact prose hit from the top-k (the replay gate showed
        # displacement costs golden-turn coverage), and they must not suppress
        # the broaden fallback (whose trigger counts EXACT hits only, below).
        # A routed row duplicating an exact hit upgrades it to via='both' and
        # donates its provenance marker instead of appearing twice.
        fused = list(exact)
        exact_by_seq = {r["seq"]: r for r in fused}
        routed_extra: list[dict] = []
        for r in routed:
            dup = exact_by_seq.get(r["seq"])
            if dup is not None:
                # The row matched on its own text — provenance marker would be
                # redundant noise on real content; just record the dual match.
                dup["via"] = "both"
            else:
                routed_extra.append(r)

        # 3) Auto-broaden on thin results: a bare multi-word query is an implicit
        # AND, and trajectories show thin AND results trigger multi-turn manual
        # rewording chains (or give-ups). Retry in-call with OR-joined terms,
        # rerank by how many distinct original terms each candidate actually
        # contains (pure OR ranks common-token turns too high), and mark the
        # extras `broadened=True` so the model treats them as leads.
        exact_seqs = {r["seq"] for r in exact}
        if len(exact_seqs) < _BROADEN_MIN_HITS and " OR " not in query.upper():
            toks = _broaden_tokens(effective_query)
            if len(toks) >= 2:
                or_expr = or_terms(toks)
                # Prose-only: headline recall is the router's job (stage 1/2
                # above); a headline OR-match here would just re-surface the
                # boundary rows the router deliberately does not return.
                b_prose = _run(
                    _scoped_match(or_expr, search_code=search_code, code_only=code_only),
                    int(k) * _BROADEN_CANDIDATE_MULT, "prose",
                ) or []
                low = [t.lower() for t in toks]

                def _term_hits(row: dict) -> int:
                    text = ((row.get("_prose") or "") + " " + (row.get("headline") or "")).lower()
                    return sum(1 for t in low if t in text)

                routed_extra_seqs = {r["seq"] for r in routed_extra}
                candidates = [
                    r for r in b_prose
                    if r["seq"] not in exact_seqs and r["seq"] not in routed_extra_seqs
                ]
                seen: set[int] = set()
                deduped = []
                for r in candidates:
                    if r["seq"] in seen:
                        continue
                    seen.add(r["seq"])
                    r["broadened"] = True
                    deduped.append(r)
                deduped.sort(key=_term_hits, reverse=True)  # stable: keeps bm25 order within ties
                fused = fused + deduped[: int(k) - len(fused)]

        # Prose-ranked results (exact + broadened) own the k slots; routed
        # leads ride along AFTER them so they can never displace a prose hit.
        fused = fused[: int(k)] + routed_extra
        if overflow_note:
            target = next(
                (r for r in reversed(fused) if r.get("via") in ("headline", "both")),
                fused[-1] if fused else None,
            )
            if target is not None:
                target["_marker"] = (target.get("_marker") or "") + overflow_note
        for r in fused:
            r.pop("_prose", None)
            r.pop("_sid", None)
        self._record(select_cols, rows=len(fused))
        fused = _preview_hits(fused, chars)
        # Routed rows carry a provenance marker (`⟦via summary of S<n>: …⟧`,
        # plus the overflow-session list on the last one). Applied AFTER the
        # preview truncation so the marker — the one channel agents reliably
        # read (they print snippets) — can't be cut off.
        for r in fused:
            marker = r.pop("_marker", None)
            if marker:
                key = "snippet" if "snippet" in r else "content"
                r[key] = (r.get(key) or "") + marker
        return fused

    def expand(self, seqs, *, code: bool = False) -> list[dict]:
        """**Return** (does not print) the FULL untruncated text of the chosen
        turns — the path for reading detailed history before a decision. Print the
        rows yourself, and only the parts you need to read.

        ``seqs`` carries straight over from ``search``: pass its returned list of
        hits (``ms.expand(hits)``), a chosen subset
        (``ms.expand([h["seq"] for h in hits if …])``), a literal list
        (``ms.expand([175, 1887])``), or a single seq — all work. Returns the rows
        ``{seq, step_index, kind, role, name, headline, content}`` where ``content``
        is the full untruncated prose (code elided to ``‹code›`` markers unless
        ``code=True`` adds a ``code`` field) — keep them in a variable, never
        re-expand the same ``seq``. Rows come back in ``seq`` order; an empty
        selection returns ``[]``.
        """
        seqs = _to_seq_list(seqs)
        if not seqs:
            return []
        cols = "seq, step_index, kind, role, name, headline, prose AS content"
        if code:
            cols += ", code"
        ph = ",".join("?" * len(seqs))
        sql = (
            f"SELECT {cols} FROM hist.conversation_history "
            f"WHERE seq IN ({ph}) ORDER BY seq"
        )
        rows = [
            {kk: r[kk] for kk in r.keys()} for r in self._conn.execute(sql, seqs)
        ]
        self._record(sql, rows=len(rows))
        return rows

    def _fts_available(self) -> bool:
        """True iff the read-only history DB has the FTS5 index table."""
        if self._fts_ok is None:
            try:
                row = self._conn.execute(
                    "SELECT 1 FROM hist.sqlite_master WHERE type='table' "
                    "AND name='conversation_history_fts'"
                ).fetchone()
                self._fts_ok = row is not None
            except sqlite3.OperationalError:
                self._fts_ok = False  # no hist attached at all
        return self._fts_ok

    def _search_like(
        self, query, scope, kind, k, snippet=False, step_range=None, msg_range=None,
        steps=None, seqs=None, search_code=False, code_only=False,
    ):
        """LIKE fallback when FTS5 is unavailable. Returns (sql, params, rows).

        Without FTS5 there is no match-centred excerpt, so ``snippet=True``
        degrades to a head-of-prose slice — still compact, just not centred on
        the query term. Matches/returns the same prose/code columns as the FTS
        path so behaviour is consistent regardless of which route runs.
        """
        like = f"%{query}%"
        if code_only:
            where = ["code LIKE ?"]
            params: list = [like]
        elif search_code:
            where = ["(prose LIKE ? OR code LIKE ?)"]
            params = [like, like]
        else:
            where = ["prose LIKE ?"]
            params = [like]
        if scope == "session" and self._session_id:
            where.append("session_id = ?")
            params.append(self._session_id)
        elif scope == "task" and self._task_id:
            where.append("task_id = ?")
            params.append(self._task_id)
            if self._shared_run_ids:  # shared tier + own session (see search())
                ph = ",".join("?" * len(self._shared_run_ids))
                where.append(f"(run_id IN ({ph}) OR session_id = ?)")
                params.extend(self._shared_run_ids)
                params.append(self._session_id)
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if step_range is not None:
            where.append("step_index BETWEEN ? AND ?")
            params.extend([int(step_range[0]), int(step_range[1])])
        if msg_range is not None:
            where.append("msg_index BETWEEN ? AND ?")
            params.extend([int(msg_range[0]), int(msg_range[1])])
        if steps is not None:
            where.append(f"step_index IN ({','.join('?' * len(steps))})")
            params.extend(steps)
        if seqs is not None:
            where.append(f"seq IN ({','.join('?' * len(seqs))})")
            params.extend(seqs)
        preview_col = "code" if code_only else "prose"
        text_col = (
            f"substr({preview_col}, 1, {_SNIPPET_LIKE_CHARS}) AS snippet"
            if snippet
            else "prose AS content"
        )
        sql = (
            f"SELECT seq, step_index, kind, role, name, headline, {text_col}, "
            "(code IS NOT NULL AND code != '') AS has_code "
            "FROM hist.conversation_history "
            "WHERE " + " AND ".join(where) + " ORDER BY seq DESC LIMIT ?"
        )
        params.append(k)
        rows = [
            {kk: r[kk] for kk in r.keys()} for r in self._conn.execute(sql, params)
        ]
        return sql, params, rows

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __repr__(self) -> str:
        return f"<MemorySpace session={self._session_id!r} task={self._task_id!r}>"
