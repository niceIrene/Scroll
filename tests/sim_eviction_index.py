"""Randomized simulation / stress test for the level-capped EvictionIndex.

No LLM and no DB: this drives :class:`EvictionIndex` directly with many synthetic
evictions, each given a *random* set of milestone headlines (sometimes none), and
checks the odometer invariants after every single eviction. It is the cheap way to
gain confidence that the carry/collapse algorithm stays correct over long runs and
deep cascades.

Run standalone (verbose):

    .venv/bin/python tests/sim_eviction_index.py

It is also a pytest (``test_eviction_index_simulation``) so it runs in CI.

Invariants checked after each eviction
--------------------------------------
1. Per-level cap     — no level ever holds more than ``_LEVEL_CAP`` blocks.
2. Stratified order  — blocks read coarsest-level-first give strictly ascending,
                       disjoint spans (older = higher level, newest = L0).
3. Losslessness      — those spans exactly partition the full evicted range; the
                       total covered length equals the sum of eviction lengths, so
                       no evicted span is ever dropped.
4. Endpoint fidelity — every block's first/last headline equals the first/last
                       eviction in its span (its milestone, or "(no milestone)"),
                       i.e. collapse really keeps the right endpoints at any depth.
5. Line sanity       — each block's lines are ordered and inside the block's span.
6. Render            — render() runs and emits exactly one header per block.
"""
from __future__ import annotations

import random

from scroll_context._runtime import index as index_mod
from scroll_context._runtime.index import (
    EvictionIndex,
    Leaf,
    _NO_MILESTONE,
)


def _gen_evictions(rng: random.Random, n: int) -> list[dict]:
    """Build ``n`` contiguous evictions, each with a random milestone subset.

    Each eviction is ``{lo, hi, miles}`` where ``miles`` is a (possibly empty)
    list of ``(seq, headline)`` for the turns that emitted a ⟦…⟧ headline. The
    empty case (no milestones in an evicted middle) is deliberately frequent.
    """
    evictions: list[dict] = []
    cursor = 1
    hl = 0
    for _ in range(n):
        span = rng.randint(1, 8)
        lo, hi = cursor, cursor + span - 1
        cursor = hi + 1
        # 0..span milestones — 0 is allowed (an eviction nobody flagged).
        k = rng.randint(0, span)
        seqs = sorted(rng.sample(range(lo, hi + 1), k))
        miles = []
        for s in seqs:
            hl += 1
            miles.append((s, f"h{hl}"))  # unique headline → unambiguous endpoint check
        evictions.append({"lo": lo, "hi": hi, "miles": miles})
    return evictions


def _blocks_render_order(idx: EvictionIndex) -> list:
    """All blocks, coarsest level first then oldest-first — render order."""
    out = []
    for k in range(len(idx._levels) - 1, -1, -1):
        out.extend(idx._levels[k])
    return out


def check_invariants(idx: EvictionIndex, added: list[dict], cap: int) -> None:
    levels = idx._levels

    # (1) per-level cap
    for k, level in enumerate(levels):
        assert len(level) <= cap, f"level L{k} has {len(level)} blocks > cap {cap}"

    blocks = _blocks_render_order(idx)
    if not blocks:
        return

    # (2) stratified, disjoint, ascending spans (older=higher level → newest=L0)
    for a, b in zip(blocks, blocks[1:]):
        assert a.seq_hi < b.seq_lo, f"spans overlap / out of order: {a.seq_hi} !< {b.seq_lo}"

    # (3) losslessness: spans partition the whole evicted range
    glo = min(e["lo"] for e in added)
    ghi = max(e["hi"] for e in added)
    assert blocks[0].seq_lo == glo, f"lowest span {blocks[0].seq_lo} != global lo {glo}"
    assert blocks[-1].seq_hi == ghi, f"highest span {blocks[-1].seq_hi} != global hi {ghi}"
    covered = sum(b.seq_hi - b.seq_lo + 1 for b in blocks)
    total = sum(e["hi"] - e["lo"] + 1 for e in added)
    assert covered == total, f"coverage {covered} != evicted total {total} (lost a span!)"

    # (4) endpoint fidelity — blocks align to eviction boundaries; first/last
    #     must match the first/last eviction in the block's span.
    by_lo = {e["lo"]: e for e in added}
    by_hi = {e["hi"]: e for e in added}
    for blk in blocks:
        e_first = by_lo.get(blk.seq_lo)
        e_last = by_hi.get(blk.seq_hi)
        assert e_first is not None, f"block lo {blk.seq_lo} not an eviction boundary"
        assert e_last is not None, f"block hi {blk.seq_hi} not an eviction boundary"
        exp_first = e_first["miles"][0][1] if e_first["miles"] else _NO_MILESTONE
        exp_last = e_last["miles"][-1][1] if e_last["miles"] else _NO_MILESTONE
        assert blk.first == exp_first, f"first {blk.first!r} != expected {exp_first!r}"
        assert blk.last == exp_last, f"last {blk.last!r} != expected {exp_last!r}"

    # (5) line sanity — ordered, inside the block span
    for blk in blocks:
        for ln in blk.lines:
            assert blk.seq_lo <= ln.seq_lo <= ln.seq_hi <= blk.seq_hi, "line escapes block span"
        for a, b in zip(blk.lines, blk.lines[1:]):
            assert a.seq_hi < b.seq_lo, "lines out of order within a block"

    # (6) render runs and has one header per block
    text = idx.render()
    assert text.count("[L") == len(blocks), "render header count != block count"


def run_scenario(seed: int, n_evictions: int, cap: int) -> dict:
    """Drive the index through ``n_evictions`` random evictions at the given cap,
    checking invariants after each. Returns summary stats."""
    rng = random.Random(seed)
    original_cap = index_mod._LEVEL_CAP
    index_mod._LEVEL_CAP = cap
    try:
        idx = EvictionIndex("sim:run")
        added: list[dict] = []
        empties = 0
        for e in _gen_evictions(rng, n_evictions):
            added.append(e)
            if not e["miles"]:
                empties += 1
            idx.add_eviction(
                [Leaf(s, h) for s, h in e["miles"]], seq_lo=e["lo"], seq_hi=e["hi"]
            )
            check_invariants(idx, added, cap)
        return {
            "seed": seed,
            "evictions": n_evictions,
            "cap": cap,
            "empty_evictions": empties,
            "levels": len(idx._levels),
            "blocks": sum(len(lvl) for lvl in idx._levels),
        }
    finally:
        index_mod._LEVEL_CAP = original_cap


# Scenarios across the valid cap range (must be >= 3; the shipped default is 5),
# two sizes, several seeds.
_SCENARIOS = [
    (seed, n, cap)
    for cap in (3, 5, 8)
    for n in (200, 1000)
    for seed in range(4)
]


def test_eviction_index_simulation():
    """Pytest entry: every scenario must satisfy all invariants throughout."""
    for seed, n, cap in _SCENARIOS:
        run_scenario(seed, n, cap)


# A realistic debugging-session narrative for the visual demo: each entry is one
# eviction's milestone headlines ([] = an evicted stretch nobody flagged).
_DEMO = [
    ['cloned repo (412 files)', 'entrypoint = app/main.py'],
    ['config.db_host = "prod-3"'],
    [],                                                  # ran ls/cat, nothing notable
    ['repro: 500 on /orders when cart empty', 'stack → orders.py:88'],
    ['root cause: cart_id None, not guarded'],
    ['patched orders.py:88 with a guard'],
    [],                                                  # re-ran, scrolled logs
    ['unit tests pass (142)', 'added test_empty_cart'],
    ['opened PR #4127'],
    ['CI red: flaky timeout in test_pay', 'retry gap = 47 days'],
    ['disabled flaky test, filed BUG-92'],
    ['CI green', 'merged PR #4127'],
]


def _index_only(idx: EvictionIndex) -> str:
    """Just the block map of render() — the header + recall footer stripped."""
    return "\n".join(
        ln for ln in idx.render().splitlines() if ln.startswith(("[L", "  · "))
    )


def demo_context_evolution(cap: int = 3) -> None:
    """Print the in-context index after each eviction, so you can watch the map
    morph: L0 blocks pile up, then carry to L1, then L2 — what the model sees."""
    original = index_mod._LEVEL_CAP
    index_mod._LEVEL_CAP = cap
    try:
        idx = EvictionIndex("run42:task7")
        seq = 10
        print(f"Context index evolution  (_LEVEL_CAP = {cap})")
        for step, miles in enumerate(_DEMO, 1):
            lo = seq
            leaves = []
            for h in miles:
                seq += 2
                leaves.append(Leaf(seq, h))
            seq += 2                                      # trailing non-milestone turns
            hi = seq
            seq += 1
            idx.add_eviction(leaves, seq_lo=lo, seq_hi=hi)
            tag = miles[0] if miles else "— no milestones —"
            print(f"\n── after eviction #{step:>2}  (seq {lo}–{hi}: {tag}) " + "─" * 6)
            print(_index_only(idx))
        print("\n\n========== the full placeholder the model actually sees ==========\n")
        print(idx.render())
    finally:
        index_mod._LEVEL_CAP = original


def main() -> None:
    print(f"Running {len(_SCENARIOS)} scenarios (cap × size × seed)…\n")
    worst_depth = 0
    for seed, n, cap in _SCENARIOS:
        stats = run_scenario(seed, n, cap)
        worst_depth = max(worst_depth, stats["levels"])
        print(
            f"  cap={stats['cap']}  evictions={stats['evictions']:>4}  "
            f"seed={stats['seed']}  →  levels={stats['levels']}  "
            f"blocks={stats['blocks']:>3}  empty={stats['empty_evictions']:>3}  OK"
        )
    print(
        f"\nAll {len(_SCENARIOS)} scenarios passed every invariant "
        f"(deepest index reached: {worst_depth} levels)."
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        cap = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        demo_context_evolution(cap)
    else:
        main()
