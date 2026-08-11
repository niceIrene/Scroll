"""Generate per-turn LLM headlines for BEAM chat seeds.

For each BEAM task we walk its ``chat.json`` turns — a *turn* being one
user+assistant exchange — and ask the model for a one-line headline summarising
it. Results are written to ``<task_dir>/headlines.json``, a flat map keyed by
the assistant message's (globally-unique) ``id``:

    {"1": "Job-search advice for a 65-year-old re-entering the market", ...}

``ingest.build_seed_db`` reads that map and stamps each assistant row's
``headline`` column from it, so the live headline index over the seeded prior
sessions is a model-written table of contents rather than an extractive trim.
The file is a cache: re-running only fills in turns not already present (use
``--force`` to regenerate), so a 10M-scale task can be resumed after an
interruption without re-paying for finished turns.

By default every assistant turn gets a noun-phrase headline. With ``--endpoints``
only the first and last turn of each session (batch) are headlined, and the two
ends differ in kind: the *opening* turn gets the usual noun-phrase headline of
that exchange, while the *closing* turn gets a short summary of the whole session
(built from every message in it). The map is then sparse, and
``ingest.build_seed_db`` leaves every other assistant turn's ``headline`` empty
(it looks each id up in the map and gets nothing). So a session shows an opening
topic line and a closing "what happened" summary. A single-turn session has no
distinct closer and shows just its noun-phrase opener.

Usage (env / .env.local must carry ``OPENAI_BASE_URL``, model name, and key)::

    uv run python -m scroll_eval.evals.beam.headlines 100K-8       # one task
    uv run python -m scroll_eval.evals.beam.headlines 100K         # all 100K-* tasks
    uv run python -m scroll_eval.evals.beam.headlines all -w 16    # every task
    uv run python -m scroll_eval.evals.beam.headlines 100K-8 --endpoints  # ends only
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from scroll_eval.evals.beam.ingest import clean_content
from scroll_eval.evals.beam.judge.model import BeamJudgeModel
from scroll_eval.harness.runner import _LOCAL_TASKS_ROOT, _list_all_tasks, _load_dotenv

_DATASET = "beam"
# Keep prompts bounded — assistant replies run to several KB but the topic is
# carried by the opening; the tail rarely changes a one-line headline.
_CLIP = 1500
# A whole-session transcript can be large; clip it (keeping the head and tail,
# since the opening sets the topic and the closing carries the outcome) before
# asking for a summary.
_SESSION_CLIP = 16000

_PROMPT = (
    "You are building a searchable index over a long user/assistant chat "
    "history. Write ONE concise headline that captures the topic and outcome "
    "of the exchange below, so a later reader can locate this turn by skimming "
    "a list of such headlines.\n"
    "Rules: a noun phrase of at most 14 words; no quotes, no leading label, no "
    "trailing period; name the concrete subject (people, places, plans) rather "
    "than generic words like 'discussion' or 'advice'.\n\n"
    "User: {user}\n"
    "Assistant: {assistant}\n\n"
    "Headline:"
)

# Used for a session's *closing* turn (in --endpoints mode): summarise the whole
# session rather than headline that one exchange.
_SESSION_PROMPT = (
    "You are building a searchable index over a long user/assistant chat "
    "history. Below is the full transcript of one session — a continuous run of "
    "turns. Write a SHORT summary of what happened across the whole session: "
    "the topics covered and what was decided, resolved, or produced, so a later "
    "reader can tell at a glance what this session was about.\n"
    "Rules: one or two sentences, at most 35 words; no quotes, no leading label; "
    "name the concrete subjects (people, places, plans, artifacts) rather than "
    "generic words like 'discussion' or 'various topics'.\n\n"
    "Session transcript:\n{transcript}\n\n"
    "Summary:"
)


def iter_turn_pairs(chat: list[dict]) -> Iterator[dict]:
    """Yield one dict per user+assistant turn across all batches.

    A BEAM turn *group* holds 1+ exchanges ([u,a] or [u,a,u,a,...]); we pair
    each assistant message with the user message that immediately precedes it.
    Each yielded dict: ``{batch, assistant_id, user, assistant}``.
    """
    for batch in chat:
        batch_number = batch.get("batch_number")
        for group in batch.get("turns", []):
            last_user = ""
            for msg in group:
                role = msg.get("role")
                if role == "user":
                    last_user = clean_content(msg.get("content", ""))
                elif role == "assistant":
                    yield {
                        "batch": batch_number,
                        "assistant_id": msg.get("id"),
                        "user": last_user,
                        "assistant": clean_content(msg.get("content", "")),
                    }
                    last_user = ""


def _session_transcripts(chat: list[dict]) -> dict[object, str]:
    """Flatten each batch's messages into one ``User:/Assistant:`` transcript.

    Keyed by ``batch_number`` to match ``iter_turn_pairs``' ``batch`` field; used
    to summarise a session's closing turn against everything that happened in it.
    """
    out: dict[object, str] = {}
    for batch in chat:
        lines: list[str] = []
        for group in batch.get("turns", []):
            for msg in group:
                content = clean_content(msg.get("content", ""))
                if not content:
                    continue
                label = "User" if msg.get("role") == "user" else "Assistant"
                lines.append(f"{label}: {content}")
        out[batch.get("batch_number")] = "\n".join(lines)
    return out


def _clip_transcript(text: str, limit: int = _SESSION_CLIP) -> str:
    """Bound a transcript to ``limit`` chars, keeping its head and tail."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half].rstrip() + "\n…\n" + text[-half:].lstrip()


def _endpoint_pairs(pairs: list[dict], transcripts: dict[object, str]) -> list[dict]:
    """Keep only the first and last turn (assistant exchange) of each session.

    ``pairs`` are the ordered ``iter_turn_pairs`` dicts. We group by ``batch``
    and keep that group's first and last entry — so a single-turn session yields
    one pair, a multi-turn session its opener and closer. Order is preserved.

    Each kept pair is tagged with ``kind``: the opener is a ``"noun"`` headline
    (of that one exchange), the closer a ``"summary"`` of the whole session, with
    the session transcript attached for the summarising call. A single-turn
    session has no distinct closer, so its lone pair stays a noun headline.
    """
    by_batch: dict[object, list[dict]] = {}
    for p in pairs:
        by_batch.setdefault(p["batch"], []).append(p)
    picked: list[dict] = []
    for batch, group in by_batch.items():
        opener = group[0]
        opener["kind"] = "noun"
        picked.append(opener)
        if len(group) > 1:
            closer = group[-1]
            closer["kind"] = "summary"
            closer["transcript"] = transcripts.get(batch, "")
            picked.append(closer)
    return picked


def _clean_headline(raw: str) -> str:
    """Reduce a model reply to a single trimmed headline line.

    We keep the whole line — no character cap — so a headline is never cut
    mid-phrase with a trailing ``…``. The prompt already bounds it to a short
    noun phrase; honouring that in full (even slightly over the nominal budget)
    reads far better than a truncated sentence-stub.
    """
    line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    return line.strip().strip('"').strip("'").rstrip(".").strip()


def _clean_summary(raw: str) -> str:
    """Reduce a model reply to a one/two-sentence session summary.

    Like :func:`_clean_headline` this keeps the full text — no character cap —
    so a summary is never truncated mid-sentence; it just collapses to one line
    while keeping sentence punctuation.
    """
    text = " ".join(ln.strip() for ln in raw.splitlines() if ln.strip())
    return text.strip().strip('"').strip("'").strip()


def generate_headline(model: BeamJudgeModel, pair: dict) -> str:
    """Call the model for a turn's headline (empty string on failure).

    A pair tagged ``kind == "summary"`` (a session's closing turn) is summarised
    against the whole-session transcript; every other pair gets the noun-phrase
    headline of its own exchange.
    """
    if pair.get("kind") == "summary":
        prompt = _SESSION_PROMPT.format(
            transcript=_clip_transcript(pair.get("transcript", "")) or "(empty session)",
        )
        clean = _clean_summary
    else:
        prompt = _PROMPT.format(
            user=pair["user"][:_CLIP] or "(no user message)",
            assistant=pair["assistant"][:_CLIP] or "(no assistant message)",
        )
        clean = _clean_headline
    try:
        return clean(model.invoke(prompt).content)
    except Exception as exc:  # noqa: BLE001 — one bad turn shouldn't sink the task
        print(f"  ! turn {pair['assistant_id']}: {exc}", file=sys.stderr, flush=True)
        return ""


def generate_for_task(
    task_dir: str | Path,
    *,
    model: BeamJudgeModel,
    workers: int = 8,
    force: bool = False,
    endpoints: bool = False,
    position: int = 0,
    leave: bool = True,
) -> dict[str, str]:
    """Generate (and cache) headlines for one task's ``chat.json``.

    Returns the full ``{assistant_id: headline}`` map and writes it to
    ``<task_dir>/headlines.json``. Existing entries are reused unless ``force``.
    With ``endpoints`` only each session's first and last turn are headlined.
    ``position``/``leave`` place this task's progress bar when several tasks run
    concurrently (each gets a distinct row); ``tqdm.write`` is used for messages
    so they don't clobber the live bars.
    """
    task_dir = Path(task_dir)
    chat = json.loads((task_dir / "chat.json").read_text(encoding="utf-8"))
    out_path = task_dir / "headlines.json"

    headlines: dict[str, str] = {}
    if out_path.exists() and not force:
        headlines = json.loads(out_path.read_text(encoding="utf-8"))

    candidates = [p for p in iter_turn_pairs(chat) if p["assistant_id"] is not None]
    if endpoints:
        candidates = _endpoint_pairs(candidates, _session_transcripts(chat))
    pending = [p for p in candidates if not headlines.get(str(p["assistant_id"]))]
    total = len(candidates)
    if not pending:
        tqdm.write(f"  {task_dir.name}: {total} turns already cached, nothing to do")
        return headlines

    lock = threading.Lock()
    done = 0

    def _flush() -> None:
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(headlines, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)

    # A per-task bar; `initial` reflects turns already cached so the bar reads as
    # progress over the whole chat, not just this run's remaining turns.
    bar = tqdm(
        total=total,
        initial=total - len(pending),
        desc=task_dir.name,
        unit="turn",
        position=position,
        leave=leave,
    )
    with bar, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate_headline, model, p): p for p in pending}
        for fut in as_completed(futures):
            pair = futures[fut]
            headline = fut.result()
            with lock:
                if headline:
                    headlines[str(pair["assistant_id"])] = headline
                done += 1
                bar.update(1)
                # Checkpoint periodically so a long run is resumable.
                if done % 50 == 0:
                    _flush()

    _flush()
    tqdm.write(f"  {task_dir.name}: wrote {out_path} ({len(headlines)} headlines)")
    return headlines


def _resolve_tasks(selectors: list[str]) -> list[str]:
    """Expand selectors into task names.

    A selector is ``all`` (every task), an exact task name (``100K-8``), or a
    scale prefix (``100K`` / ``1M`` → all ``<prefix>-*`` tasks).
    """
    all_tasks = _list_all_tasks(_DATASET)
    picked: list[str] = []
    for sel in selectors:
        if sel == "all":
            picked = list(all_tasks)
            break
        if sel in all_tasks:
            picked.append(sel)
            continue
        prefixed = [t for t in all_tasks if t == sel or t.startswith(f"{sel}-")]
        if not prefixed:
            raise SystemExit(f"no beam task matches selector {sel!r}")
        picked.extend(prefixed)
    # de-dup, preserve order
    seen: set[str] = set()
    return [t for t in picked if not (t in seen or seen.add(t))]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scale",
        nargs="+",
        help="task name (100K-8), scale prefix (100K, 1M, 10M), or 'all'",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=8, help="concurrent requests per chat"
    )
    parser.add_argument(
        "-t",
        "--task-workers",
        type=int,
        default=1,
        help="chats generated in parallel (total in-flight requests ≈ workers × task-workers)",
    )
    parser.add_argument("--model", default=None, help="override model name")
    parser.add_argument("--force", action="store_true", help="regenerate cached headlines")
    parser.add_argument(
        "--endpoints",
        action="store_true",
        help=(
            "only index the first and last turn of each session (sparse map): a "
            "noun-phrase headline for the opener, a whole-session summary for the "
            "closer"
        ),
    )
    args = parser.parse_args(argv)

    _load_dotenv()
    model = BeamJudgeModel(model=args.model)
    if not model.model:
        raise SystemExit(
            "no model configured — set OPENAI_MODEL_NAME/SCROLL_MODEL (and "
            "OPENAI_BASE_URL + key) in the env or .env.local, or pass --model"
        )

    tasks = _resolve_tasks(args.scale)
    task_workers = max(1, min(args.task_workers, len(tasks)))
    print(
        f"Generating headlines for {len(tasks)} task(s) with model {model.model!r} "
        f"({task_workers} chat(s) in parallel, {args.workers} req/chat)"
    )

    def _run(name: str, position: int) -> None:
        generate_for_task(
            _LOCAL_TASKS_ROOT / _DATASET / name,
            model=model,
            workers=args.workers,
            force=args.force,
            endpoints=args.endpoints,
            position=position,
            # When chats run concurrently each owns a recycled bar row; leaving
            # them would pile up finished bars, so only keep the bar in the
            # single-chat case.
            leave=(task_workers == 1),
        )

    if task_workers == 1:
        for name in tasks:
            _run(name, position=0)
        return

    # Outer pool over chats. A queue of bar-row positions is recycled as tasks
    # finish so at most `task_workers` bars are on screen at once.
    slots: "queue.Queue[int]" = queue.Queue()
    for pos in range(task_workers):
        slots.put(pos)

    def _worker(name: str) -> None:
        pos = slots.get()
        try:
            _run(name, position=pos)
        except Exception as exc:  # noqa: BLE001 — isolate one chat's failure
            tqdm.write(f"  {name}: FAILED — {exc}")
        finally:
            slots.put(pos)

    with ThreadPoolExecutor(max_workers=task_workers) as pool:
        list(pool.map(_worker, tasks))


if __name__ == "__main__":
    main()
