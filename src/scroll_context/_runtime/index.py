"""The in-context memory map: temporally-chunked levels over folded history.

Every unit of retired history — a live-eviction sweep, a prior session, a
closed turn — enters level 0 as one *block* (a contiguous ``seq`` span plus
its extractive line(s)). When a level fills to ``level_cap`` blocks, the
OLDEST ``level_cap - 1`` blocks fold into ONE block one level up — a *chunk*:
a single line whose endpoints are the first member's opening summary and the
last member's closing summary, carrying the member tag range, the unit count,
and the union span. Chunks of chunks propagate the outermost endpoints, so
depth k covers ~``(level_cap - 1)^k`` units with one line.

This replaces the earlier similarity-clustering carry (Jaccard partitions,
extractive cluster labels, render-budget elision). Chunking is purely
temporal: members are consecutive in fold order, so every block at every
level has a CONTIGUOUS seq span — the drill-down contract is uniform
(``seq BETWEEN lo AND hi`` recovers any block's underlying rows, and
``… AND headline IS NOT NULL`` lists its per-unit headlines). Titles are a
pluggable consolidation policy; v0 is deterministic endpoints (the actors'
own words, never generated) plus a *topic strip* — the words recurring across
member summaries (see :func:`_chunk_topics`) — so a chunk line carries scent
for its middle, not just its edges. A learned consolidator can replace
:func:`_chunk_title` later without touching the structure — it runs at fold
time and its output freezes, so the map stays system-maintained and the
model is never responsible for index correctness.

Invariants (see tests):
- per-level block count stays under ``level_cap`` after every fold;
- blocks read coarsest-level-first give ascending, disjoint spans that
  exactly partition everything ever folded (losslessness: any folded row is
  inside exactly one block's span);
- the rendered map is bounded by construction: at most ~``level_cap`` lines
  per level, levels growing logarithmically in folded units — no elision
  machinery needed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Max blocks a level holds before its oldest fold up. Must be >= 3: at cap 2
# every fold would chunk a single block (a rename, no compression) and the
# index would grow one level per fold.
_LEVEL_CAP = 5

# Shown for an eviction that carried no milestone headlines: the span is still
# tracked (and recallable) even though no step in it was flagged.
_NO_MILESTONE = "(no milestone)"

# --- chunk topic strips -------------------------------------------------------
# A chunk's endpoint title brackets its era but carries no scent for the
# members in between — measured on BEAM, models rarely drill a bare bracket, so
# mid-era sessions go undiscovered (temporal/knowledge-update regressions). The
# topic strip restores that scent for a few tokens: the words that recur across
# the member summaries, harvested extractively at fold time and frozen (same
# contract as the endpoint title — deterministic, the actors' own words).
_TOPICS_PER_CHUNK = 6
_TOPIC_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]{2,}")
# Function words plus conversation-generic verbs/nouns that carry no topical
# scent in summary lines. Deliberately light on nouns: err toward keeping a
# word that might be topical ("trip", "budget") over dropping it.
_TOPIC_STOPWORDS = frozenset("""
    the and for with that this you your our are was were has have had can could
    will would should may might must not but from into over under about after
    before between during out off all any each more most some such than then
    too very just also how what when where which who why its per via los una
    using use used get got getting make making made take took want wants wanted
    need needs needed try trying tried help helped helping work working worked
    discuss discussed discussing provide provided providing start started
    starting finish finished let lets looking look looked ask asked asking
    session sessions user assistant question questions answer answered
    specific certain various several way ways thing things good great best
    right sure certainly one two three like new old
    did does done doing many much earlier later based here there because
    i'm i've i'd you're you've it's that's here's there's what's let's
    conversation summary summarize summarized synthesis arc overall said
    created produced brought resolved identified achieved discussed order
    chronological
""".split())


def _topic_words(texts) -> set[str]:
    """The topical vocabulary of a bundle of summary lines."""
    words: set[str] = set()
    for t in texts:
        for w in _TOPIC_WORD_RE.findall((t or "").lower()):
            if w not in _TOPIC_STOPWORDS:
                words.add(w)
    return words


def _chunk_topics(members: list["Block"], head: str, tail: str) -> tuple:
    """Recurring member-summary words, minus what the endpoints already show.

    Ranked by how many MEMBERS mention a word (coverage — a chunk-representative
    term beats a one-member term), then by total mentions, then alphabetically
    for determinism. Words visible in the chunk's own ``head``/``tail`` are
    excluded: the strip is *added* scent, not a paraphrase of the title. Member
    chunks contribute their own topic strips too, so depth compounds coverage
    without unbounded text retention.
    """
    df: Counter = Counter()
    tf: Counter = Counter()
    for b in members:
        texts = [ln.head for ln in b.lines] + [ln.tail for ln in b.lines]
        texts.extend(b.topics)
        seen = set()
        for t in texts:
            for w in _TOPIC_WORD_RE.findall((t or "").lower()):
                if w in _TOPIC_STOPWORDS:
                    continue
                tf[w] += 1
                seen.add(w)
        for w in seen:
            df[w] += 1
    shown = _topic_words([head, tail])
    ranked = sorted(
        (w for w in df if w not in shown),
        key=lambda w: (-df[w], -tf[w], w),
    )
    # One slot per stem: "agent"/"agents" or "error"/"errors" carry the same
    # scent — crude singular/plural collapse keeps the strip's slots diverse.
    out: list[str] = []
    stems: set[str] = set()
    shown_stems = {w.rstrip("s") for w in shown}
    for w in ranked:
        stem = w.rstrip("s")
        if stem in stems or stem in shown_stems:
            continue
        stems.add(stem)
        out.append(w)
        if len(out) == _TOPICS_PER_CHUNK:
            break
    return tuple(out)


def _tag_str(tag) -> str:
    """Render a grouping tag: legacy int session -> "S<n>"; labels as-is."""
    return f"S{tag}" if isinstance(tag, int) else str(tag)


@dataclass(frozen=True)
class Leaf:
    """One evicted milestone step: its seq and its model-authored headline."""

    seq: int
    headline: str


@dataclass(frozen=True)
class Line:
    """One entry shown inside a block.

    ``seq_lo``/``seq_hi`` is the span the line stands for — a single step has
    ``lo == hi``; a chunk line carries the whole chunk's span. ``head`` is the
    opening summary of that span, ``tail`` the closing one; a raw leaf has
    ``head == tail``. ``sessions`` holds the grouping tag(s): historically
    prior-session ints (rendered "S<n>"), now generalized to string labels
    ("S61", "P2", "T3") so any granularity folds through the same machinery.
    """

    seq_lo: int
    seq_hi: int
    head: str
    tail: str
    sessions: tuple = ()

    @property
    def text(self) -> str:
        """Display text: a single headline, or ``first - last`` for a span."""
        return self.head if self.head == self.tail else f"{self.head} - {self.tail}"

    @property
    def span(self) -> str:
        """Locator text: seq span, prefixed by the tag when known."""
        seq = (
            f"seq {self.seq_lo}"
            if self.seq_lo == self.seq_hi
            else f"seq {self.seq_lo}–{self.seq_hi}"
        )
        if len(self.sessions) == 1:
            return f"{_tag_str(self.sessions[0])} {seq}"
        return seq


@dataclass
class Block:
    """A contiguous span of folded history at one level.

    A level-0 block is one folded UNIT (an eviction sweep with its leaf
    lines, or a single-line session/turn span). A higher-level block is a
    CHUNK: one line summarizing ``units`` consecutive units, formed by
    :meth:`EvictionIndex._fold`.
    """

    seq_lo: int
    seq_hi: int
    lines: list[Line]
    units: int = 1
    chunk: bool = False
    # Chunk-only: the recurring member-summary words (see _chunk_topics).
    topics: tuple = field(default=())

    @property
    def first(self) -> str:
        """Opening summary: the leftmost line's head."""
        if not self.lines:
            return _NO_MILESTONE
        return self.lines[0].head or self.lines[0].tail or _NO_MILESTONE

    @property
    def last(self) -> str:
        """Closing summary: the rightmost line's tail."""
        if not self.lines:
            return _NO_MILESTONE
        return self.lines[-1].tail or self.lines[-1].head or _NO_MILESTONE

    @property
    def tags(self) -> tuple:
        """Ordered distinct tags across the block's lines."""
        seen: dict = {}
        for ln in self.lines:
            for s in ln.sessions:
                seen.setdefault(s)
        return tuple(seen)


def _chunk_title(members: list["Block"]) -> tuple[str, str]:
    """The v0 consolidation policy: outermost endpoint summaries, verbatim.

    ``(head, tail)`` = the first member's opening summary and the last
    member's closing summary — the actors' own words, deterministic, honest
    about being a range ("A … B" brackets an era). This is the pluggable seam
    for a trained consolidation policy later: swap this function, keep the
    structure. Titles are computed at fold time and frozen thereafter (KV-
    stable, and cheap enough to be replaced by a slow/offline learned policy).
    """
    return members[0].first, members[-1].last


def _tag_range(tags: tuple) -> str:
    if not tags:
        return ""
    if len(tags) == 1:
        return _tag_str(tags[0])
    return f"{_tag_str(tags[0])}–{_tag_str(tags[-1])}"


def _unit_word(tags: tuple) -> str:
    """Human word for a chunk's units, from its tag prefixes."""
    prefixes = {_tag_str(t)[:1] for t in tags}
    if prefixes and prefixes <= {"S", "P"}:
        return "sessions"
    if prefixes == {"T"}:
        return "turns"
    return "spans"


class EvictionIndex:
    """The in-context spine: a stack of levels, each a list of blocks oldest-first."""

    def __init__(self, session_id: str, *, level_cap: int | None = None) -> None:
        # Resolve the default from the module global at call time (not as a
        # default arg) so tests that monkeypatch _LEVEL_CAP still take effect.
        cap = _LEVEL_CAP if level_cap is None else level_cap
        if cap < 3:
            raise ValueError(f"level_cap must be >= 3, got {cap}")
        self._session_id = session_id
        self._level_cap = cap
        self._levels: list[list[Block]] = []

    @property
    def is_empty(self) -> bool:
        return not any(self._levels)

    # -- the two entry moves --------------------------------------------------

    def add_eviction(self, leaves: list[Leaf], *, seq_lo: int, seq_hi: int) -> None:
        """Fold one live-eviction sweep onto L0.

        ``leaves`` are the evicted milestone steps (each a ``seq · headline``
        line); ``seq_lo``/``seq_hi`` is the FULL evicted span (tool results
        and unheadlined steps included) so a range query recovers everything.
        """
        lines = [Line(lf.seq, lf.seq, lf.headline, lf.headline) for lf in leaves]
        if not self._levels:
            self._levels.append([])
        self._levels[0].append(Block(seq_lo, seq_hi, lines))
        self._fold(0)

    def add_span(
        self,
        *,
        seq_lo: int,
        seq_hi: int,
        head: str,
        tail: str,
        session: int | None = None,
        tag: str | None = None,
    ) -> None:
        """Fold one already-reduced span (a prior session, a closed turn) onto L0.

        The block's single line spans ``seq_lo..seq_hi`` with distinct
        endpoint summaries (``head == tail`` collapses to one), tagged for
        locator rendering (``S<n>`` seeded sessions, ``P<n>`` own prior
        sessions, ``T<n>`` closed turns).
        """
        label = tag if tag is not None else session
        line = Line(
            seq_lo, seq_hi, head, tail,
            sessions=(label,) if label is not None else (),
        )
        if not self._levels:
            self._levels.append([])
        self._levels[0].append(Block(seq_lo, seq_hi, [line]))
        self._fold(0)

    # -- the one compaction move ----------------------------------------------

    def _fold(self, k: int) -> None:
        """Chunk the oldest blocks up while levels are full (iterative cascade).

        When level ``k`` reaches ``level_cap``, its oldest ``level_cap - 1``
        blocks — a temporally CONSECUTIVE run, so the chunk's span is the
        contiguous union — become one chunk block at ``k + 1``; the newest
        block stays. Every fold strictly shrinks the level, so per-level
        counts stay bounded and depth grows logarithmically in folded units.
        """
        while k < len(self._levels) and len(self._levels[k]) >= self._level_cap:
            members = self._levels[k][: self._level_cap - 1]
            self._levels[k] = self._levels[k][self._level_cap - 1:]
            head, tail = _chunk_title(members)
            tags = tuple(t for b in members for t in b.tags)
            chunk = Block(
                seq_lo=members[0].seq_lo,
                seq_hi=members[-1].seq_hi,
                lines=[
                    Line(
                        members[0].seq_lo, members[-1].seq_hi, head, tail,
                        sessions=tags,
                    )
                ],
                units=sum(b.units for b in members),
                chunk=True,
                topics=_chunk_topics(members, head, tail),
            )
            if k + 1 == len(self._levels):
                self._levels.append([])
            self._levels[k + 1].append(chunk)
            k += 1

    # -- rendering ------------------------------------------------------------

    def render(self, header: str | None = None) -> str:
        """The single placeholder message: the whole map + how to expand it.

        Levels print coarsest-first (highest level on top, L0 at the bottom),
        so the model reads oldest → newest, coarse → fine. Every line carries
        a contiguous seq span — expanding ANY of them, chunk or unit, is the
        same one-``BETWEEN`` recipe. Bounded by construction (≤ ~level_cap
        lines per level), so there is no elision machinery.
        """
        lines = [
            header or (
                "<system-info>[context compressed] Evicted history is durable; "
                "this is its index — newest units are listed per line at the "
                "bottom, older spans are chunked upward with endpoint summaries. "
                "Expand any span inside execute_python."
            ),
        ]
        for k in range(len(self._levels) - 1, -1, -1):
            for block in self._levels[k]:
                if block.chunk:
                    tr = _tag_range(block.tags)
                    where = f"{tr}  " if tr else ""
                    topics = (
                        f"  · topics: {', '.join(block.topics)}"
                        if block.topics else ""
                    )
                    lines.append(
                        f"[L{k}] {where}seq {block.seq_lo}–{block.seq_hi}  — "
                        f"{block.units} {_unit_word(block.tags)}  "
                        f"⟦ {block.lines[0].text} ⟧{topics}"
                    )
                else:
                    lines.append(f"[L{k}] seq {block.seq_lo}–{block.seq_hi}")
                    for ln in block.lines:
                        lines.append(f"  · {ln.span}  ⟦ {ln.text} ⟧")
        # `seq` is a table-wide PRIMARY KEY, so a span range identifies its rows
        # unambiguously across every session — no session_id filter (which would
        # exclude prior sessions whose id differs from this run's). `S<n>` lines
        # map to `step_index = n` on `kind='conversation'` rows.
        lines += [
            "Recall (inside execute_python):",
            "  • a span's per-unit headlines (works for ANY line above, chunk or "
            "single): ms.sql_query(\"SELECT seq, headline FROM "
            "hist.conversation_history WHERE seq BETWEEN <lo> AND <hi> AND "
            "headline IS NOT NULL ORDER BY seq\")",
            "  • a span's (or one turn's) full content: ms.sql_query(\"SELECT seq, "
            "kind, role, content FROM hist.conversation_history WHERE seq BETWEEN "
            "<lo> AND <hi> ORDER BY seq\")",
            "  • an `S<n>` session line by session number: "
            "ms.sql_query(\"SELECT seq, role, content FROM hist.conversation_history "
            "WHERE kind='conversation' AND step_index IN (<n>, ...) ORDER BY seq\")",
            "  • keyword search within any span above (chunk or single): "
            "ms.search(\"your terms\", seq_range=(<lo>, <hi>), scope='task') — "
            "ranked, sanitized, previewed; for code use code_only=True. "
            "ms.expand([seq,...]) pulls full untruncated turns.",
            "</system-info>",
        ]
        return "\n".join(lines)
