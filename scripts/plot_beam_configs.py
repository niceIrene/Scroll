#!/usr/bin/env python3
"""Plot BEAM accuracy + token cost across configurations on one sheet.

Compares runs of the 2x2 ablation grid (eviction/seed index on|off x thinking
on|off). Each run's configuration is AUTO-DETECTED from its artifacts, not
hardcoded: thinking = any step carries a ``reasoning`` block; index = the
system prompt advertises the ``headline`` column (the index-off ablation
strips it).

One figure, three aligned panels over the same x categories (one group per
probing-question type, plus Overall):

  1. accuracy      — mean judge score (per-chat category means averaged
                     across chats, same aggregation as summary.json)
  2. input tokens  — mean tokens_in per probe (retrieval/context cost)
  3. output tokens — mean tokens_out per probe (generation cost; thinking
                     runs pay reasoning tokens here)

Score/cost collection reuses scripts/beam_analysis.py. The full data table is
also printed to stdout (the accessible "table view" for the plot).

Usage::

    uv run --with matplotlib python scripts/plot_beam_configs.py
    uv run --with matplotlib python scripts/plot_beam_configs.py \
        runs/2026-08-07T04-09-12* runs/2026-08-07T06-00-28* --out beam_2x2.png
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

_HERE = Path(__file__).parent
_SPEC = importlib.util.spec_from_file_location("beam_analysis", _HERE / "beam_analysis.py")
beam_analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(beam_analysis)

# The four runs of the qwen3.8-max 2x2 ablation (index x thinking). Any other
# set of run dirs can be passed on the command line.
_DEFAULT_RUN_GLOBS = [
    "runs/2026-08-07T04-09-12*",
    "runs/2026-08-07T06-00-28*",
    "runs/2026-08-07T21-49-08*",
    "runs/2026-08-08T06-18-09*",
]

# A 2x2 factorial encoding carried by THREE redundant channels so no config
# rests on color alone: hue = the index axis (blue = index on, aqua = off),
# lightness = thinking (dark = on, light = off), and hatch = the index axis
# again (solid = index on, diagonal texture = off; inked tone-on-tone with a
# darker step of the fill's own hue). Colors validated with the dataviz
# palette checker (CVD worst adjacent dE 17.3, all checks pass; the two light
# steps are sub-3:1 on the surface — mitigated by the hatch, the direct value
# labels, and the stdout table view).
#   cfg -> (fill, hatch-ink, hatch pattern)
_CONFIG_STYLE = {
    (True, True): ("#2a78d6", "#104281", ""),      # index on  · thinking on
    (True, False): ("#6da7ec", "#1c5cab", ""),     # index on  · thinking off
    (False, True): ("#0e7d57", "#06402c", "//"),   # index off · thinking on
    (False, False): ("#1baf7a", "#0e7d57", "//"),  # index off · thinking off
}
# Fixed series order (also the bar order within each group).
_CONFIG_ORDER = [(True, True), (True, False), (False, True), (False, False)]

_SURFACE = "#fcfcfb"
_PAGE = "#f9f9f7"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"

_OVERALL = "Overall"


def _detect_config(run_dir: Path) -> tuple[bool, bool]:
    """(index_on, thinking_on), read from the run's own artifacts."""
    tj = sorted(run_dir.glob("tasks/*/probes/*/*/trajectory.json"))
    thinking = False
    if tj:
        steps = json.loads(tj[0].read_text()).get("steps", [])
        thinking = any(s.get("reasoning") for s in steps)
    cm = sorted(run_dir.glob("tasks/*/probes/*/*/call_messages.jsonl"))
    index_on = False
    if cm:
        with open(cm[0], encoding="utf-8") as f:
            sys_msg = json.loads(f.readline())["system"]["content"]
        if not isinstance(sys_msg, str):
            sys_msg = "".join(b.get("text", "") for b in sys_msg)
        index_on = "headline" in sys_msg
    return index_on, thinking


def _config_label(cfg: tuple[bool, bool]) -> str:
    idx, think = cfg
    return f"index {'on' if idx else 'off'} · thinking {'on' if think else 'off'}"


def _accuracy_by_category(run_dir: Path) -> dict[str, float]:
    """Per-category mean of per-chat means; Overall from manifest/summary."""
    rows, _scored, _skipped = beam_analysis._collect_scores(run_dir)
    out = {
        cat: d["sum"] / d["chats"] for cat, d in rows.items() if d["chats"]
    }
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        out[_OVERALL] = json.loads(manifest.read_text()).get("mean_reward")
    return out


def _cost_by_category(run_dir: Path) -> dict[str, tuple[float, float]]:
    """Per-category mean (tokens_in, tokens_out) per probe; plus Overall."""
    rows, probe_counts, _skipped = beam_analysis._collect_cost(run_dir)
    per_cat_in: dict[str, int] = defaultdict(int)
    per_cat_out: dict[str, int] = defaultdict(int)
    per_cat_n: dict[str, int] = defaultdict(int)
    for (chat, cat), vals in rows.items():
        per_cat_in[cat] += vals.get("tokens_in", 0)
        per_cat_out[cat] += vals.get("tokens_out", 0)
        per_cat_n[cat] += probe_counts[(chat, cat)]
    out = {
        cat: (per_cat_in[cat] / n, per_cat_out[cat] / n)
        for cat, n in per_cat_n.items() if n
    }
    total_n = sum(per_cat_n.values())
    if total_n:
        out[_OVERALL] = (
            sum(per_cat_in.values()) / total_n,
            sum(per_cat_out.values()) / total_n,
        )
    return out


def _short(cat: str) -> str:
    """Two-line tick labels: 'contradiction_resolution' -> 'contradiction\\nresolution'."""
    if cat == _OVERALL:
        return _OVERALL
    return cat.replace("_", "\n", 1).replace("_", " ")


def _style_axis(ax, ylabel: str) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_BASELINE)
    ax.tick_params(colors=_MUTED, labelcolor=_INK_2, length=0)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=_INK_2, fontsize=10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", default=_DEFAULT_RUN_GLOBS,
                    help="Run dirs (globs ok); default: the qwen3.8-max 2x2 grid.")
    ap.add_argument("--out", default="runs/beam_2x2_analysis.png", help="Output PNG path.")
    args = ap.parse_args()

    run_dirs: list[Path] = []
    for pat in args.runs:
        matches = sorted(glob.glob(str(pat)))
        if not matches:
            sys.exit(f"no run dir matches {pat!r}")
        run_dirs += [Path(m) for m in matches]

    # config -> data, ordered by the fixed series order
    series: dict[tuple[bool, bool], dict] = {}
    for rd in run_dirs:
        cfg = _detect_config(rd)
        if cfg in series:
            sys.exit(f"two runs detected as {_config_label(cfg)!r}: "
                     f"{series[cfg]['dir'].name} and {rd.name}")
        series[cfg] = {
            "dir": rd,
            "acc": _accuracy_by_category(rd),
            "cost": _cost_by_category(rd),
        }
    configs = [c for c in _CONFIG_ORDER if c in series] + [
        c for c in series if c not in _CONFIG_ORDER
    ]

    # Category order: fixed alphabetical, Overall last.
    cats = sorted({c for s in series.values() for c in s["acc"] if c != _OVERALL})
    cats.append(_OVERALL)

    # ---- table view (stdout) ------------------------------------------------
    print(f"{'category':<26}" + "".join(
        f"{_config_label(c):>28}" for c in configs))
    for cat in cats:
        line = f"{cat:<26}"
        for c in configs:
            acc = series[c]["acc"].get(cat)
            tin, tout = series[c]["cost"].get(cat, (float('nan'),) * 2)
            line += f"{acc:>10.3f} {tin/1000:>8.0f}k {tout/1000:>6.1f}k"
        print(line)

    # ---- figure: accuracy bars + annotated cost heatmap strips --------------
    # Accuracy keeps the grouped-bar form (comparison of magnitudes). Token
    # cost is a LOOKUP job — "what did config X pay on category Y?" — so it
    # gets two annotated heatmap strips (input, output) aligned to the same
    # category columns: rows are the four configs (same order as the bars,
    # swatch on the left), every cell carries its value. One hue per strip,
    # light→dark = cheap→expensive, normalized within the strip.
    from matplotlib.colors import LinearSegmentedColormap

    n_cfg = len(configs)
    width = 0.8 / n_cfg  # group span 0.8, equal bars with a whisker of air
    fig, (ax, ax_in, ax_out) = plt.subplots(
        3, 1, figsize=(16, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1, 1], "hspace": 0.14},
    )
    fig.patch.set_facecolor(_PAGE)

    xs = list(range(len(cats)))
    j = cats.index(_OVERALL)
    for i, cfg in enumerate(configs):
        offs = [x + (i - (n_cfg - 1) / 2) * width for x in xs]
        fill, ink, hatch = _CONFIG_STYLE.get(cfg, (_MUTED, _INK_2, ""))
        acc = [series[cfg]["acc"].get(c, 0) or 0 for c in cats]
        ax.bar(offs, acc, width * 0.9, color=fill, hatch=hatch, edgecolor=ink,
               linewidth=0.6, label=_config_label(cfg))
        # Selective direct labels: the Overall group only (relief for the
        # light series + the headline comparison, without numbering every bar).
        ax.text(offs[j], acc[j] + 0.012, f"{acc[j]:.2f}", ha="center",
                va="bottom", fontsize=8.5, color=_INK)

    _style_axis(ax, "mean judge score")
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axvline(len(cats) - 1.5, color=_GRID, linewidth=0.8)

    # Sequential ramps anchored on the palette's blue / aqua hues.
    cmap_in = LinearSegmentedColormap.from_list(
        "blues", ["#e7f1fc", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"])
    cmap_out = LinearSegmentedColormap.from_list(
        "aquas", ["#e4f7ef", "#8fdfc0", "#1baf7a", "#0e7d57", "#06402c"])

    def _heat_strip(axh, cmap, values, fmt, ylabel):
        """One annotated (config x category) strip; light→dark within strip."""
        vmax = max(v for row in values for v in row if v) or 1.0
        axh.set_facecolor(_SURFACE)
        axh.imshow(
            values, cmap=cmap, aspect="auto", vmin=0, vmax=vmax,
            extent=(-0.5, len(cats) - 0.5, n_cfg - 0.5, -0.5),
        )
        for r, row in enumerate(values):
            for c, v in enumerate(row):
                if not v:
                    continue
                # Ink flips to white on the dark end of the ramp.
                frac = v / vmax
                axh.text(c, r, fmt(v), ha="center", va="center", fontsize=8.3,
                         color="white" if frac > 0.55 else _INK)
        # White gridlines = the 2px surface gap between cells.
        for c in range(len(cats) - 1):
            axh.axvline(c + 0.5, color=_SURFACE,
                        linewidth=3 if c == len(cats) - 2 else 1.2)
        for r in range(n_cfg - 1):
            axh.axhline(r + 0.5, color=_SURFACE, linewidth=1.2)
        axh.set_yticks(range(n_cfg))
        axh.set_yticklabels(
            [_config_label(c).replace("thinking", "think") for c in configs],
            fontsize=8.5,
        )
        axh.tick_params(colors=_MUTED, labelcolor=_INK_2, length=0)
        for side in axh.spines.values():
            side.set_visible(False)
        axh.set_ylabel(ylabel, color=_INK_2, fontsize=10)

    tin_rows = [[series[c]["cost"].get(cat, (0, 0))[0] for cat in cats] for c in configs]
    tout_rows = [[series[c]["cost"].get(cat, (0, 0))[1] for cat in cats] for c in configs]
    _heat_strip(ax_in, cmap_in, tin_rows, lambda v: f"{v/1000:.0f}k",
                "input tokens / probe")
    _heat_strip(ax_out, cmap_out, tout_rows, lambda v: f"{v/1000:.1f}k",
                "output tokens / probe")

    ax_out.set_xticks(xs)
    ax_out.set_xticklabels([_short(c) for c in cats], fontsize=9)

    model = json.loads((series[configs[0]]["dir"] / "manifest.json").read_text())
    fig.subplots_adjust(top=0.86, left=0.13)
    fig.suptitle(
        f"BEAM 10M — accuracy and token cost by probing-question type "
        f"({model['model']['name']}; cost strips shade light→dark = cheap→expensive)",
        fontsize=13, color=_INK, y=0.965,
    )
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.03), ncol=4, frameon=False,
        fontsize=9.5, labelcolor=_INK_2, columnspacing=1.4, handlelength=1.6,
    )

    out = Path(args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_PAGE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
