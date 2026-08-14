from __future__ import annotations

import inspect
import re
from datetime import date
from typing import Any

from scroll_context._runtime.memoryspace import MemorySpace, or_terms


# Names ``populate()`` injects — excluded from the working-memory digest so it
# shows only the variables the *model* created, not the fixtures it was given.
# The last three exist only in var-context mode (``install_var_ops``).
_INJECTED = ("ms", "days_between", "or_terms", "pin", "note", "show")


def populate(
    ns: dict[str, Any],
    *,
    memoryspace: MemorySpace,
) -> None:
    """Inject the model-visible names into the runtime's REPL namespace.

    The model writes Python that resolves these names: `ms` for the read-only
    query window onto its durable cross-session history (ATTACHed as
    `hist.conversation_history`, with `ms.search(...)` / `ms.sql_query(...)` and
    `ms.session_id` / `ms.task_id` to scope to the current run or to the task
    across all runs); `days_between(d1, d2, inclusive=False)` for reliable
    calendar math; and `or_terms([...])` to build a safe FTS5 OR match
    expression (auto-quotes phrases/hyphenated terms). The model keeps its own
    working data in plain variables it assigns here — they persist across
    `execute_python` calls (see ``describe``).
    """
    ns["ms"] = memoryspace
    ns["days_between"] = _days_between
    ns["or_terms"] = or_terms


def _describe_value(val: Any) -> str:
    """One-line ``type, size`` descriptor for a persisted namespace variable."""
    if isinstance(val, str):
        return f"str, {len(val)} chars"
    if isinstance(val, (list, tuple, set, frozenset)):
        return f"{type(val).__name__}, {len(val)} items"
    if isinstance(val, dict):
        return f"dict, {len(val)} keys"
    if isinstance(val, (bool, int, float)):
        return f"{type(val).__name__} = {val!r}"
    if inspect.isfunction(val):
        return "function"
    return type(val).__name__


def describe(ns: dict[str, Any]) -> str:
    """Deterministic snapshot of the model-created variables in the namespace.

    The ``execute_python`` namespace persists across turns, so a variable the
    model stored survives even after the turn that printed it scrolls out of the
    window. Listing those variables each turn (name + type + size) keeps the
    model aware of its working set without re-querying. Injected fixtures
    (``ms``, helpers), imported modules, and dunder names are hidden so only the
    model's own data shows. Empty when nothing has been stored yet.
    """
    rows = []
    for name, val in ns.items():
        if name.startswith("__") or name in _INJECTED or inspect.ismodule(val):
            continue
        rows.append(f"  - {name} ({_describe_value(val)})")
    if not rows:
        return "vars: (empty)"
    return "vars:\n" + "\n".join(sorted(rows))


# --- var-context mode: typed variable views ----------------------------------
#
# In the SCROLL_VAR_CONTEXT ablation the variable store IS the model's curated
# context: each variable renders into the prompt as a typed metadata line (plus
# an auto- or model-chosen slice of its data), and namespace changes render as
# an append-only changelog. The helpers below supply the pieces: a schema
# preview (`_schema_of`), a per-variable metadata line (`var_meta_line`), a
# cheap namespace fingerprint + diff for the changelog (`fingerprint`/`diff`),
# a bounded value preview (`preview_value`), and the model-facing curation ops
# (`install_var_ops`: pin / note / show).

_PIN_LEVELS = ("meta", "head", "full")
_HEAD_ROWS = 3          # rows shown for a 'head' view (auto or pinned)
_AUTO_FULL_ITEMS = 5    # collections at or under this render fully by default
_PREVIEW_ROW_CHARS = 160  # per-row clip inside any rendered view
_PREVIEW_FULL_CAP = 2000  # hard cap on any single variable's rendered view


class VarViews:
    """Model-requested standing views (``pin``) and annotations (``note``)."""

    def __init__(self) -> None:
        self.pins: dict[str, str] = {}    # name -> level in _PIN_LEVELS
        self.notes: dict[str, str] = {}   # name -> free-text note

    def drop(self, name: str) -> None:
        self.pins.pop(name, None)
        self.notes.pop(name, None)


def _schema_of(val: Any) -> str:
    """Compact type/shape descriptor: ``list[dict{seq,date,…}], 40 items``."""
    if isinstance(val, str):
        return f"str, {len(val)} chars"
    if isinstance(val, dict):
        keys = list(val.keys())
        # Schema-like dicts (identifier keys) list their field names; DATA-keyed
        # dicts (seqs, dates, ids) would spray values into the prompt — render
        # their key→value types instead.
        if all(isinstance(k, str) and k.isidentifier() for k in keys[:6]):
            inner = ",".join(str(k) for k in keys[:6]) + (",…" if len(keys) > 6 else "")
            return f"dict{{{inner}}}, {len(val)} keys"
        ktype = type(keys[0]).__name__ if keys else "?"
        vtype = type(next(iter(val.values()))).__name__ if val else "?"
        return f"dict[{ktype}→{vtype}], {len(val)} keys"
    if isinstance(val, (list, tuple, set, frozenset)):
        # Internal list subclasses (e.g. the provenance-tagged ResultRows)
        # present as plain lists — the model should never see harness names.
        tname = "list" if isinstance(val, list) else type(val).__name__
        if len(val) == 0:
            return f"{tname}, empty"
        first = next(iter(val))
        if isinstance(first, dict):
            keys = list(first.keys())[:6]
            inner = ",".join(str(k) for k in keys) + (",…" if len(first) > 6 else "")
            return f"{tname}[dict{{{inner}}}], {len(val)} items"
        return f"{tname}[{type(first).__name__}], {len(val)} items"
    if isinstance(val, (bool, int, float)):
        return f"{type(val).__name__} = {val!r}"
    if inspect.isfunction(val):
        return "function"
    return type(val).__name__


def _row_seqs(val: Any) -> set[int] | None:
    """Row-identity set of a list-of-row-dicts, else None.

    Prefers ``seq``; falls back to ``msg_index`` (also unique per turn) so
    overlap detection still works when the model selected columns without
    ``seq``. Used for overlap comparison only — never rendered directly.
    """
    if not isinstance(val, list) or not val or not isinstance(val[0], dict):
        return None
    for key in ("seq", "msg_index"):
        out: set[int] = set()
        ok = True
        for r in val:
            if not isinstance(r, dict):
                return None
            s = r.get(key)
            if isinstance(s, int):
                out.add(s)
            else:
                ok = False
                break
        if ok and out:
            return out
    return None


def row_coverage(val: Any) -> str | None:
    """What a list of history rows COVERS: sessions, dates, seq span.

    The lineage query says what was *asked*; this says what came *back* — the
    fact that lets the model judge reuse ("does this already cover session
    13?") without re-reading the rows. None for anything that isn't a
    non-empty list of row dicts carrying the history schema keys.
    """
    if not isinstance(val, list) or not val or not all(isinstance(r, dict) for r in val):
        return None
    parts: list[str] = []
    sessions = sorted({r["step_index"] for r in val
                       if isinstance(r.get("step_index"), int)})
    if sessions:
        parts.append(
            "S" + ",S".join(str(s) for s in sessions) if len(sessions) <= 4
            else f"S{sessions[0]}–S{sessions[-1]} ({len(sessions)} sessions)"
        )
    dates: list[str] = []
    for r in val:
        d = r.get("date")
        if not isinstance(d, str) or not d:
            meta = r.get("metadata")
            d = meta.get("date") if isinstance(meta, dict) else None
        if isinstance(d, str) and d:
            dates.append(d)
    if dates:
        dates.sort()
        parts.append(dates[0] if dates[0] == dates[-1] else f"{dates[0]}…{dates[-1]}")
    # A raw seq span only as a last resort: sessions/dates already locate the
    # data for a human-and-model reader, and seq lists are prompt noise.
    if not parts:
        seqs = {s for r in val if isinstance(s := r.get("seq"), int)}
        if seqs:
            parts.append(
                f"seq {min(seqs)}" if len(seqs) == 1 else f"seqs {min(seqs)}…{max(seqs)}"
            )
    return ("covers " + ", ".join(parts)) if parts else None


def overlap_verdict(
    name: str, val: Any, candidates: dict[str, Any]
) -> tuple[str, str] | None:
    """Judge whether a freshly RETRIEVED variable duplicates prior holdings.

    The intended target is exactly one pathology: a retrieval op in a later
    step fetching rows the model already had. The caller therefore controls
    the two preconditions this function cannot see — it must be invoked only
    for variables created by a step that actually ran retrieval ops, and
    ``candidates`` must contain only variables that existed BEFORE that step
    (same-step siblings are derivation/reorganization by definition — the
    behavior the prompt teaches — and must never be compared).

    Returns ``("warn", text)`` when ≥ half of this variable's rows are already
    held by an at-least-as-large pre-existing variable; ``("upgrade", text)``
    instead when this variable came from ``ms.expand`` over a counterpart
    holding lossy ``snippets`` rows — that overlap is the intended
    snippet→full-text path and renders as lineage, not a warning. ``None``
    otherwise.
    """
    mine = _row_seqs(val)
    if not mine:
        return None
    # Only judge variables whose rows LOOK like history rows (they carry the
    # retrieval schema). A hand-built fact record ({'date': …, 'seq': 71})
    # created alongside a retrieval — the anchors idiom the prompt itself
    # teaches — must never be flagged for referencing rows the model holds.
    first = val[0] if isinstance(val, list) and val and isinstance(val[0], dict) else None
    if first is None or not ({"step_index", "msg_index", "content", "snippet", "kind"} & set(first)):
        return None
    best: tuple[str, Any] | None = None
    best_hit = 0
    for other_name, other_val in candidates.items():
        if other_name == name:
            continue
        theirs = _row_seqs(other_val)
        # Directional: only an at-least-as-large counterpart makes THIS
        # variable the redundant one — the superset a subset was sliced from
        # must not get flagged for being copied.
        if not theirs or len(theirs) < len(mine):
            continue
        hit = len(mine & theirs)
        if hit > best_hit:
            best, best_hit = (other_name, other_val), hit
    if best is None or best_hit * 2 < len(mine):
        return None
    other_name, other_val = best
    my_prov = getattr(val, "provenance", None) or ""
    other_prov = getattr(other_val, "provenance", None) or ""
    if my_prov.startswith("expand") and "snippets" in other_prov:
        return ("upgrade", f"↳ {name}: full-text upgrade of `{other_name}`")
    return ("warn", f"⚠ {name}: {best_hit}/{len(mine)} rows already in `{other_name}`")


def is_scratch(name: str, val: Any) -> bool:
    """Loop temporaries and imported classes — collapsed in digest/changelog.

    Conservative on purpose: single-character names (``h``, ``r``, ``c``) and
    bare classes (``Counter``) are near-certain noise; anything else the model
    named stays first-class, however badly named.
    """
    return len(name) == 1 or inspect.isclass(val)


def var_meta_line(
    name: str,
    val: Any,
    views: "VarViews | None" = None,
    *,
    intent: str | None = None,
    origin_seq: int | None = None,
    overlap: str | None = None,
) -> str:
    """The one-line typed metadata for a variable — the DIGEST's authority line.

    ``name (schema) — covers … ⚠|↳ overlap ← origin — note|while``: schema and
    coverage say what the data IS, ``overlap`` (a verdict precomputed at
    creation via :func:`overlap_verdict`, never recomputed live) says whether
    it duplicated prior holdings, the origin says how it was obtained —
    derived op facts (op, count, snippets vs full text, saturation) plus the
    ``seq`` of the producing step, whose durable row holds the exact query
    (``tool_input``) and raw output verbatim — and the trailing note
    (model-authored) or ``intent`` (the step's own distilled line) says what
    it was FOR.
    """
    parts = [f"{name} ({_schema_of(val)})"]
    cov = row_coverage(val)
    if cov:
        parts.append(f"— {cov}")
    if overlap:
        # Strip the leading "name: " the changelog form carries — the digest
        # line already names the variable.
        parts.append(overlap.replace(f" {name}:", "", 1))
    prov = getattr(val, "provenance", None)
    if prov and origin_seq is not None:
        parts.append(f"← {prov}, seq {origin_seq}")
    elif prov:
        parts.append(f"← {prov}")
    elif origin_seq is not None:
        parts.append(f"← seq {origin_seq}")
    note = views.notes.get(name) if views is not None else None
    if note:
        parts.append(f"— note: {note}")
    elif intent:
        parts.append(f'— while: "{intent}"')
    return "  ".join(parts)


def _clip_row(row: Any, chars: int = _PREVIEW_ROW_CHARS) -> str:
    text = " ".join(repr(row).split())
    return text if len(text) <= chars else text[:chars].rstrip() + "…"


def preview_value(val: Any, rows: int | None = None) -> str:
    """A bounded, printable slice of a value (``rows=None`` = full, still capped).

    Collections render one clipped line per item; dicts one per key; strings a
    head slice. Output never exceeds ``_PREVIEW_FULL_CAP`` chars — the model
    can always narrow further in code.
    """
    lines: list[str]
    if isinstance(val, str):
        lines = [val[: _PREVIEW_FULL_CAP // 2] + ("…" if len(val) > _PREVIEW_FULL_CAP // 2 else "")]
    elif isinstance(val, dict):
        items = list(val.items())
        shown = items if rows is None else items[:rows]
        lines = [f"{k!r}: {_clip_row(v)}" for k, v in shown]
        if len(items) > len(shown):
            lines.append(f"… +{len(items) - len(shown)} more keys")
    elif isinstance(val, (list, tuple, set, frozenset)):
        items = list(val)
        shown = items if rows is None else items[:rows]
        lines = [_clip_row(v) for v in shown]
        if len(items) > len(shown):
            lines.append(f"… +{len(items) - len(shown)} more items")
    else:
        lines = [_clip_row(val, _PREVIEW_FULL_CAP)]
    out = "\n".join("    " + ln for ln in lines)
    if len(out) > _PREVIEW_FULL_CAP:
        out = out[:_PREVIEW_FULL_CAP].rstrip() + "\n    …[view capped — slice in code]"
    return out


def model_vars(ns: dict[str, Any]) -> dict[str, Any]:
    """The model-created variables of a namespace (fixtures/modules hidden)."""
    return {
        name: val
        for name, val in ns.items()
        if not name.startswith("__") and name not in _INJECTED and not inspect.ismodule(val)
    }


def _sizeof(val: Any) -> int | None:
    try:
        return len(val)  # type: ignore[arg-type]
    except TypeError:
        return None


def fingerprint(ns: dict[str, Any]) -> dict[str, tuple]:
    """Cheap identity snapshot of the model's variables, for changelog diffing.

    ``name -> (id, size, scalar_repr)``: identity change = re-assignment; same
    identity with a size change = in-place mutation (append/update); scalar
    repr covers rebinding small immutables to a new equal-typed value.
    """
    out: dict[str, tuple] = {}
    for name, val in ns.items():
        if name.startswith("__") or name in _INJECTED or inspect.ismodule(val):
            continue
        scalar = (
            repr(val)
            if isinstance(val, (bool, int, float)) or (isinstance(val, str) and len(val) <= 80)
            else None
        )
        out[name] = (id(val), _sizeof(val), scalar)
    return out


def diff_names(
    before: dict[str, tuple], after_ns: dict[str, Any]
) -> dict[str, list[str]]:
    """Structured namespace diff for one REPL call.

    ``{"created": [...], "reassigned": [...], "mutated": [...], "deleted": [...]}``
    — reassigned = rebound to a new object, mutated = same object changed in
    place (append/update). Callers own the rendering (and any views cleanup).
    """
    after = fingerprint(after_ns)
    out: dict[str, list[str]] = {
        "created": [], "reassigned": [], "mutated": [], "deleted": []
    }
    for name in after:
        if name not in before:
            out["created"].append(name)
        elif before[name] != after[name]:
            key = "reassigned" if before[name][0] != after[name][0] else "mutated"
            out[key].append(name)
    out["deleted"] = [name for name in before if name not in after]
    return out


_CHANGE_MARK = {"created": "+", "reassigned": "~", "mutated": "±"}


def diff_changes(
    before: dict[str, tuple], after_ns: dict[str, Any], views: "VarViews | None" = None
) -> list[str]:
    """Changelog entries for one REPL call: created / re-assigned / mutated / deleted."""
    names = diff_names(before, after_ns)
    entries: list[str] = []
    for kind, mark in _CHANGE_MARK.items():
        for name in names[kind]:
            entries.append(f"{mark} {var_meta_line(name, after_ns[name], views)}")
    for name in names["deleted"]:
        if views is not None:
            views.drop(name)
        entries.append(f"- del {name}")
    return entries


def install_var_ops(ns: dict[str, Any], views: VarViews) -> None:
    """Inject the var-context curation ops (pin / note / show) into ``ns``.

    Only called in var-context mode; the names are in ``_INJECTED`` so they
    never show up as model data. Each op prints a confirmation — only printed
    text reaches the model.
    """

    def pin(name: str, level: str = "head", note: str | None = None) -> None:
        """Keep a standing view of variable ``name`` in every future digest.

        ``level``: 'meta' (one metadata line), 'head' (metadata + first rows),
        'full' (metadata + capped full view). ``note`` also attaches a
        description (same as ``note()``). ``pin(name, 'meta')`` on an already
        pinned variable demotes it; use ``del`` to drop the variable entirely.
        """
        if name not in ns:
            print(f"pin: no variable named {name!r}")
            return
        if level not in _PIN_LEVELS:
            print(f"pin: level must be one of {_PIN_LEVELS}")
            return
        views.pins[name] = level
        if note:
            views.notes[name] = str(note)
        print(f"pinned {name} at level {level!r}" + (f" — note: {note}" if note else ""))

    def note_(name: str, text: str) -> None:
        """Attach a one-line description to variable ``name`` (shown in its
        metadata everywhere: digest, changelog, pinned views)."""
        if name not in ns:
            print(f"note: no variable named {name!r}")
            return
        views.notes[name] = str(text)
        print(f"noted {name}: {text}")

    def show(val: Any, rows: int | None = 10) -> None:
        """Print a bounded preview of a value or variable name — the cheap way
        to look at data without dumping it (``rows=None`` = full, still capped)."""
        if isinstance(val, str) and val in ns:
            name, val = val, ns[val]
            print(var_meta_line(name, val, views))
        print(preview_value(val, rows))

    ns["pin"] = pin
    ns["note"] = note_
    ns["show"] = show


_DATE_RE = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")


def _parse_date(value: Any) -> date:
    """Pull a calendar date out of the messy formats dates actually arrive in.

    Accepts ISO (``2023-06-01``, ``2023-06-01T09:00``) *and* the LongMemEval
    style (``2023/06/01 (Thu) 09:00``) — anything with a leading
    ``YYYY[-/]M[-/]D``; the weekday/time tail is ignored.
    """
    m = _DATE_RE.search(str(value))
    if not m:
        raise ValueError(f"days_between: no YYYY-MM-DD / YYYY/MM/DD date in {value!r}")
    y, mo, d = (int(g) for g in m.groups())
    return date(y, mo, d)


def _days_between(d1: str, d2: str, inclusive: bool = False) -> int:
    """Absolute number of days between two dates.

    Parses both ISO (``2023-06-01``) and LongMemEval (``2023/06/01 (Thu) 09:00``)
    formats — pass ``question_date`` / ``session_date_iso`` straight in. Always
    non-negative; ``inclusive=True`` counts both endpoints (adds 1). LLM calendar
    math is unreliable past ~2-week deltas — compute it here instead.
    """
    n = abs((_parse_date(d2) - _parse_date(d1)).days)
    return n + 1 if inclusive else n
