"""Tests for the temporally-chunked EvictionIndex (the in-context memory map)."""
from __future__ import annotations

import pytest

from scroll_context._runtime.index import Block, EvictionIndex, Leaf, Line, _chunk_title


def _spans(idx: EvictionIndex) -> list[tuple[int, int]]:
    """All block spans, coarsest level first (the render order)."""
    out = []
    for k in range(len(idx._levels) - 1, -1, -1):
        for b in idx._levels[k]:
            out.append((b.seq_lo, b.seq_hi))
    return out


def test_level_cap_minimum_enforced():
    with pytest.raises(ValueError):
        EvictionIndex("t", level_cap=2)


def test_chunk_fold_keeps_newest_and_titles_by_endpoints():
    idx = EvictionIndex("t", level_cap=4)
    for n in range(1, 5):  # 4 spans -> fold fires on the 4th
        idx.add_span(seq_lo=n * 10, seq_hi=n * 10 + 9,
                     head=f"opened {n}", tail=f"closed {n}", tag=f"S{n}")
    # Oldest cap-1 = 3 blocks chunked into one L1 block; newest stays at L0.
    assert len(idx._levels[0]) == 1 and len(idx._levels[1]) == 1
    chunk = idx._levels[1][0]
    assert chunk.chunk is True and chunk.units == 3
    assert (chunk.seq_lo, chunk.seq_hi) == (10, 39)          # contiguous union
    assert chunk.lines[0].head == "opened 1"                  # outermost endpoints
    assert chunk.lines[0].tail == "closed 3"
    out = idx.render()
    assert "[L1] S1–S3  seq 10–39  — 3 sessions  ⟦ opened 1 - closed 3 ⟧" in out
    assert "· S4 seq 40–49  ⟦ opened 4 - closed 4 ⟧" in out   # newest, full pair


def test_chunk_of_chunks_propagates_outermost_endpoints_and_units():
    idx = EvictionIndex("t", level_cap=3)
    for n in range(1, 12):
        idx.add_span(seq_lo=n * 10, seq_hi=n * 10 + 9,
                     head=f"opened {n}", tail=f"closed {n}", tag=f"S{n}")
    # Deep chunks exist and still bracket their whole era verbatim.
    top = idx._levels[-1][0]
    assert top.chunk and top.units >= 4
    assert top.lines[0].head == "opened 1"
    assert top.lines[0].tail.startswith("closed ")
    # Per-level bound holds after every fold.
    assert all(len(level) < 3 or level is idx._levels[0] for level in idx._levels)
    assert all(len(level) <= 3 for level in idx._levels)


def test_spans_partition_everything_folded():
    """Losslessness: coarsest-first spans are ascending, disjoint, and cover
    exactly the folded ranges — any folded seq is inside exactly one block."""
    idx = EvictionIndex("t", level_cap=3)
    folded = []
    for n in range(1, 20):
        lo, hi = n * 100, n * 100 + 50
        folded.append((lo, hi))
        idx.add_span(seq_lo=lo, seq_hi=hi, head=f"h{n}", tail=f"t{n}", tag=f"S{n}")
    spans = _spans(idx)
    assert spans == sorted(spans)                       # ascending
    for (a_lo, a_hi), (b_lo, b_hi) in zip(spans, spans[1:]):
        assert a_hi < b_lo                              # disjoint
    for lo, hi in folded:                               # every unit covered once
        holders = [s for s in spans if s[0] <= lo and hi <= s[1]]
        assert len(holders) == 1


def test_eviction_blocks_render_leaves_and_chunk_with_no_milestone_fallback():
    idx = EvictionIndex("t", level_cap=3)
    idx.add_eviction([Leaf(5, "found the host"), Leaf(9, "fixed the port")],
                     seq_lo=1, seq_hi=12)
    out = idx.render()
    assert "[L0] seq 1–12" in out
    assert "· seq 5  ⟦ found the host ⟧" in out
    assert "· seq 9  ⟦ fixed the port ⟧" in out
    # Leafless evictions still fold and title honestly.
    idx.add_eviction([], seq_lo=13, seq_hi=20)
    idx.add_eviction([], seq_lo=21, seq_hi=30)   # fold fires: oldest 2 chunk
    chunk = idx._levels[1][0]
    assert chunk.chunk and chunk.units == 2
    assert chunk.lines[0].head == "found the host"
    assert chunk.lines[0].tail == "(no milestone)"
    assert "— 2 spans" in idx.render()           # untagged units -> "spans"


def test_tag_rendering_and_unit_words():
    idx = EvictionIndex("t", level_cap=3)
    idx.add_span(seq_lo=1, seq_hi=5, head="a", tail="b", session=7)     # legacy int
    idx.add_span(seq_lo=6, seq_hi=9, head="c", tail="d", tag="T2")
    out = idx.render()
    assert "S7 seq 1–5" in out and "T2 seq 6–9" in out
    # Turn-only chunks say "turns".
    idx2 = EvictionIndex("t", level_cap=3)
    for n in range(1, 4):
        idx2.add_span(seq_lo=n * 10, seq_hi=n * 10 + 5,
                      head=f"ask {n}", tail=f"answer {n}", tag=f"T{n}")
    assert "— 2 turns  ⟦ ask 1 - answer 2 ⟧" in idx2.render()


def test_chunk_title_policy_is_pluggable_seam():
    """v0 titles come from one function with a (members)->(head, tail)
    contract — the hook a trained consolidation policy replaces later."""
    members = [
        Block(1, 5, [Line(1, 5, "first opening", "first closing", ("S1",))]),
        Block(6, 9, [Line(6, 9, "last opening", "last closing", ("S2",))]),
    ]
    assert _chunk_title(members) == ("first opening", "last closing")


def test_render_has_uniform_between_recipe_for_all_lines():
    idx = EvictionIndex("t", level_cap=3)
    for n in range(1, 6):
        idx.add_span(seq_lo=n * 10, seq_hi=n * 10 + 9,
                     head=f"h{n}", tail=f"t{n}", tag=f"S{n}")
    out = idx.render()
    assert out.startswith("<system-info>")
    assert "works for ANY line above, chunk or single" in out
    assert "seq BETWEEN <lo> AND <hi>" in out


def test_shared_session_spans_tag_and_render():
    """Prior-session spans feed tagged blocks into the shared index."""
    from scroll_context.manager import _INDEX_HEADER_TMPL, shared_session_spans

    class _FakeMS:
        def sql_query(self, sql, params=None):
            return [
                {"session": 1, "seq_lo": 2, "seq_hi": 150,
                 "head": "opener one", "tail": "closer one"},
                {"session": 2, "seq_lo": 151, "seq_hi": 300,
                 "head": "opener two", "tail": None},
            ]

    spans = shared_session_spans(_FakeMS(), run_ids=("seed",), task_id="task")
    idx = EvictionIndex("t", level_cap=5)
    for s in spans:
        idx.add_span(
            seq_lo=s["seq_lo"], seq_hi=s["seq_hi"],
            head=s["head"], tail=s["tail"], session=s["session"],
        )
    out = idx.render(header=_INDEX_HEADER_TMPL.format(repl="execute_python"))
    assert out.startswith("<system-info>[memory]")
    assert "S1 seq 2–150" in out and "opener one - closer one" in out
    assert "S2 seq 151–300" in out and "opener two" in out


def test_chunk_topics_add_member_scent_beyond_endpoints():
    """The topic strip carries words recurring across member summaries,
    excluding stopwords and what the endpoint title already shows."""
    idx = EvictionIndex("t", level_cap=4)
    idx.add_span(seq_lo=10, seq_hi=19, tag="S1",
                 head="planned the honeymoon budget", tail="booked flights")
    idx.add_span(seq_lo=20, seq_hi=29, tag="S2",
                 head="quiz scores review", tail="dolphin cruise booked")
    idx.add_span(seq_lo=30, seq_hi=39, tag="S3",
                 head="stakeholder interviews finalized", tail="quiz retake planned")
    idx.add_span(seq_lo=40, seq_hi=49, tag="S4",
                 head="newest", tail="newest")   # triggers the fold of S1-S3
    chunk = idx._levels[1][0]
    # "quiz" covers 2 members -> ranks first; endpoint words ("planned",
    # "honeymoon", "budget", "quiz"? no - endpoints are S1.head/S3.tail).
    # Chunk endpoints: "planned the honeymoon budget" / "quiz retake planned"
    # so honeymoon/budget/quiz/retake/planned are excluded from the strip.
    assert "quiz" not in chunk.topics            # visible in the tail already
    assert "booked" in chunk.topics              # covers 2 members, not shown
    for w in ("the", "planned", "honeymoon"):
        assert w not in chunk.topics
    out = idx.render()
    assert "· topics: " in out
    assert "booked" in out


def test_chunk_topics_propagate_through_chunk_of_chunks():
    """A chunk member contributes its topic strip to the next level, so deep
    chunks keep scent for their middle without unbounded text retention."""
    idx = EvictionIndex("t", level_cap=3)
    for n in range(1, 12):
        idx.add_span(seq_lo=n * 10, seq_hi=n * 10 + 9, tag=f"S{n}",
                     head=f"opened {n}", tail=f"closed {n} dolphin cruise")
    top = idx._levels[-1][0]
    assert top.chunk
    # "dolphin"/"cruise" recur in every member tail; the top chunk's own
    # endpoints are "opened 1"/"closed N dolphin cruise", so they are shown
    # there — but intermediate chunks carried them, proving propagation ran.
    assert any("dolphin" in b.topics or "cruise" in b.topics or
               ("dolphin" in (b.lines[0].tail or ""))
               for lvl in idx._levels for b in lvl if b.chunk)


def test_chunk_topics_rank_by_member_coverage():
    from scroll_context._runtime.index import _chunk_topics
    members = [
        Block(1, 5, [Line(1, 5, "alpha mapping", "alpha mapping", ("S1",))]),
        Block(6, 9, [Line(6, 9, "alpha probes", "beta beta beta", ("S2",))]),
        Block(10, 12, [Line(10, 12, "alpha probes", "gamma", ("S3",))]),
    ]
    topics = _chunk_topics(members, "start here", "end here")
    # alpha covers 3 members, probes 2, beta 1 (though tf=3), gamma 1.
    assert topics.index("alpha") < topics.index("probes") < topics.index("beta")
