"""The eviction index — an in-context, level-capped tree of evicted turns.

The whole index lives in the prompt as ONE placeholder, so the model always
*sees the map* of what it evicted. The structure is a stack of levels — an
odometer:

    L0 (bottom)  the newest evictions; each block lists its turns in full.
    Lk (k ≥ 1)   older history, carried up and squeezed to span endpoints.

Each level holds at most ``_LEVEL_CAP`` blocks. Every eviction drops one new
block on L0 (``add_eviction``). When a level fills, it *carries*: keep the
newest block as-is, collapse the rest to one line each, and stack those lines as
a single new block one level up (``_carry`` → ``_collapse``). The carry cascades
upward, exactly like a digit rolling past 9. So recent history sits low and
detailed; old history rides up and is reduced to its endpoints.

Nothing is lost. Every line carries a ``seq`` span and the full turns stay in
``conversation_history``; a collapsed line is a *zoomed-out view* the model
re-expands with one ``ms.sql_query`` over its span.
"""
from __future__ import annotations

from dataclasses import dataclass


# Max blocks a level holds before it carries up. The carry keeps the newest
# block and folds the other (_LEVEL_CAP - 1) into one block a level higher — so
# each rolled-up block holds _LEVEL_CAP - 1 lines (4 at the default of 5).
# Must be >= 3: at cap 2 the carry would fold only one block (a rename, no
# compression) and the index would grow one level per eviction.
_LEVEL_CAP = 5

# Shown for an eviction that carried no milestone headlines: the span is still
# tracked (and recallable) even though no turn in it was flagged.
_NO_MILESTONE = "(no milestone)"


@dataclass(frozen=True)
class Leaf:
    """One evicted milestone turn: its durable ``seq`` and its ``headline``."""

    seq: int
    headline: str


@dataclass(frozen=True)
class Line:
    """One entry shown inside a block.

    ``seq_lo``/``seq_hi`` is the span the line stands for — a single turn has
    ``lo == hi``; a collapsed child block carries the child's whole span.
    ``head`` is the leftmost headline in that span, ``tail`` the rightmost; a raw
    leaf has ``head == tail``.
    """

    seq_lo: int
    seq_hi: int
    head: str
    tail: str

    @property
    def text(self) -> str:
        """Display text: a single headline, or ``first - last`` for a span."""
        return self.head if self.head == self.tail else f"{self.head} - {self.tail}"

    @property
    def span(self) -> str:
        return (
            f"seq {self.seq_lo}"
            if self.seq_lo == self.seq_hi
            else f"seq {self.seq_lo}–{self.seq_hi}"
        )


@dataclass
class Block:
    """A run of lines at one level; its ``seq`` span covers all of them."""

    seq_lo: int
    seq_hi: int
    lines: list[Line]

    @property
    def first(self) -> str:
        """Leftmost (oldest) headline anywhere in the block.

        An eviction with no milestone turns has no lines; its endpoint falls back
        to ``_NO_MILESTONE`` so the span still carries (and collapses) cleanly.
        """
        return self.lines[0].head if self.lines else _NO_MILESTONE

    @property
    def last(self) -> str:
        """Rightmost (newest) headline anywhere in the block."""
        return self.lines[-1].tail if self.lines else _NO_MILESTONE


def _collapse(blocks: list[Block]) -> Block:
    """Fold a run of blocks into ONE block: each input becomes a single line
    carrying that input's full span and its endpoint headlines (``first - last``).

    Self-similar: collapsing already-collapsed blocks just keeps the leftmost and
    rightmost headline of each, so a turn, a span, and a span-of-spans all reduce
    the same way — which is what lets the carry cascade to any depth losslessly.
    """
    return Block(
        seq_lo=min(b.seq_lo for b in blocks),
        seq_hi=max(b.seq_hi for b in blocks),
        lines=[Line(b.seq_lo, b.seq_hi, b.first, b.last) for b in blocks],
    )


class EvictionIndex:
    """The in-context spine: a stack of levels, each a list of blocks oldest-first."""

    def __init__(self, session_id: str, *, level_cap: int | None = None) -> None:
        # Resolve the default from the module global at call time (not as a
        # default arg) so tests that monkeypatch _LEVEL_CAP still take effect.
        cap = _LEVEL_CAP if level_cap is None else level_cap
        if cap < 3:
            # At cap 2 the carry folds only one block (a rename, no compression)
            # and the index would grow one level per eviction.
            raise ValueError(f"level_cap must be >= 3, got {cap}")
        self._session_id = session_id
        self._level_cap = cap
        self._levels: list[list[Block]] = []

    @property
    def is_empty(self) -> bool:
        return not any(self._levels)

    # -- the two moves -------------------------------------------------------

    def add_eviction(self, leaves: list[Leaf], *, seq_lo: int, seq_hi: int) -> None:
        """Drop one eviction onto L0 as a new block, then run the carry.

        ``leaves`` are the evicted milestone turns (each a ``seq · headline``
        line); ``seq_lo``/``seq_hi`` is the *full* evicted span (tool results and
        unheadlined turns included) so a range query recovers everything.
        """
        lines = [Line(lf.seq, lf.seq, lf.headline, lf.headline) for lf in leaves]
        if not self._levels:
            self._levels.append([])
        self._levels[0].append(Block(seq_lo, seq_hi, lines))
        self._carry(0)

    def add_span(self, *, seq_lo: int, seq_hi: int, head: str, tail: str) -> None:
        """Drop one already-reduced span onto L0 as a block, then run the carry.

        Unlike :meth:`add_eviction`'s point leaves, this block's single line
        spans ``seq_lo..seq_hi`` and carries distinct endpoint headlines, so it
        renders as ``head - tail`` at every level (``head == tail`` collapses to
        one). Used to seed a whole prior session represented by its first and
        last milestone headline.
        """
        if not self._levels:
            self._levels.append([])
        line = Line(seq_lo, seq_hi, head, tail)
        self._levels[0].append(Block(seq_lo, seq_hi, [line]))
        self._carry(0)

    def _carry(self, k: int) -> None:
        """Carry from level k upward while levels are full (iterative cascade).

        The carry is the only roll-up move and it always makes progress: a full
        level of ``_LEVEL_CAP`` blocks becomes one kept block plus a single new
        block one level higher, so the index can never exceed ``_LEVEL_CAP``
        blocks per level. Done as a loop (not recursion) so a long cascade can
        never hit the interpreter's recursion limit.
        """
        while len(self._levels[k]) >= self._level_cap:
            *older, newest = self._levels[k]
            self._levels[k] = [newest]
            if k + 1 == len(self._levels):
                self._levels.append([])
            self._levels[k + 1].append(_collapse(older))
            k += 1

    # -- rendering -----------------------------------------------------------

    def render(self, header: str | None = None, *, repl_name: str = "execute_python") -> str:
        """The single placeholder message: the whole map + how to expand it.

        Levels print coarsest-first (highest level on top, L0 at the bottom), so
        the model reads oldest → newest, top → bottom. ``header`` overrides the
        default ``[context compressed]`` intro line (must open ``<system-info>``)
        — used to render a seeded prior-sessions map with its own wording.
        ``repl_name`` is the host's name for the Python REPL tool, so the
        recovery instructions name a tool that actually exists (``scroll_repl``
        for OpenAI-format harnesses, ``execute_python`` for scroll_react).
        """
        lines = [
            header or (
                "<system-info>[context compressed] Evicted turns are durable; this "
                "is their index — newest evictions are listed per turn at the bottom, "
                "older spans are carried up and shown as endpoint pairs. Expand any "
                f"span inside {repl_name}."
            ),
        ]
        for k in range(len(self._levels) - 1, -1, -1):
            for block in self._levels[k]:
                lines.append(f"[L{k}] seq {block.seq_lo}–{block.seq_hi}")
                for ln in block.lines:
                    lines.append(f"  · {ln.span}  ⟦ {ln.text} ⟧")
        # `seq` is a table-wide PRIMARY KEY, so a span range identifies its rows
        # unambiguously across every session — no session_id filter (which would
        # exclude seeded sessions whose id differs from this run's).
        lines += [
            f"Recall (inside {repl_name}):",
            "  • expand a span to its per-turn headlines: ms.sql_query("
            "\"SELECT seq, headline FROM hist.conversation_history WHERE seq "
            "BETWEEN <lo> AND <hi> AND headline IS NOT NULL ORDER BY seq\")",
            "  • a span's (or one turn's) full content: ms.sql_query(\"SELECT seq, "
            "kind, role, content FROM hist.conversation_history WHERE seq BETWEEN "
            "<lo> AND <hi> ORDER BY seq\")",
            "  • keyword search (FTS5): ms.sql_query(\"SELECT seq, kind, content "
            "FROM hist.conversation_history WHERE seq IN (SELECT rowid FROM "
            "hist.conversation_history_fts('{prose} : (YOUR KEYWORDS)')) ORDER BY "
            "seq\") — prose is ranked apart from code; for code use "
            "ms.search(code_only=True). Prefer ms.search() for relevance ranking "
            "and ms.expand([seq,...]) to pull full untruncated turns.",
            "</system-info>",
        ]
        return "\n".join(lines)
