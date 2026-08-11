"""Migrate a BEAM chat tier into ProjectX's local-tasks layout.

BEAM (https://github.com/.../beam, MIT) is a long-term-memory benchmark: each
conversation is a multi-session dialogue plus ~20 retrospective probing
questions graded by an LLM-as-judge against per-question rubrics. We do NOT run
it as a Harbor task (no code execution, the conversation is a *prior dialogue*
not a file to grep, grading is an LLM judge). Instead we adopt Harbor's
task-*directory* layout as a tidy on-disk dataset format and execute it with the
native beam runner (``scroll_eval.evals.beam.runner``).

For each source conversation ``<src>/<n>/`` this writes::

    <dest>/<scale>-<n>/
      task.toml                    # metadata + discovery anchor (runner="native")
      chat.json                    # the conversation, in canonical batch form
      questions.json               # AGENT-VISIBLE: [{id, type, question}] only
      solution/answers.json        # gold as judge-answers (oracle sanity check)
      tests/probing_questions.json # HELD OUT: full file (rubrics + gold)
      tests/test.sh                # thin shim -> shared judge module

``chat.json`` is copied verbatim when the source is already a flat batch list
(100K/500K/1M). The 10M tier nests batches inside per-plan wrappers
(``[{"plan-1": [batch, ...]}, ...]``) with batch numbers restarting at 1 in
every plan; migration flattens it here — NOT in the ingester — so every
downstream reader sees one canonical schema and the global session numbering
(load-bearing for event-ordering/temporal probes) is frozen on disk once.
Plan provenance is kept per batch (``plan``/``plan_batch_number``).

Read-only over the source; only writes under ``<dest>``.

Usage::

    # --scale picks the matching source tier automatically (case-insensitive):
    uv run python scripts/migrate_beam.py --scale 10M
    uv run python scripts/migrate_beam.py                 # default tier: 100K
    # override the source explicitly only for a non-standard layout:
    uv run python scripts/migrate_beam.py --src /path/to/chats/100K --scale 100K
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# The 10 probing-question types, in BEAM's canonical order.
QUESTION_TYPES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]

# Gold-answer field name varies per type; instruction_following /
# preference_following have no answer field (only a behavioral description).
# Pick the first present, non-empty field in this priority order. This is used
# ONLY for the oracle sanity check — grading uses `rubric`, never the gold.
GOLD_FIELDS = [
    "ideal_response",   # abstention
    "ideal_answer",     # contradiction_resolution
    "ideal_summary",    # summarization
    "answer",           # event_ordering, information_extraction, knowledge_update,
                        # multi_session_reasoning, temporal_reasoning
    "expected_compliance",  # instruction_following, preference_following (stand-in)
]

_TEST_SH = """\
#!/usr/bin/env bash
# Harbor-contract verifier shim: delegate to the shared BEAM judge module.
# Grades answers.json against this task's held-out rubrics and writes a reward
# in [0,1] to the reward file. Invoked by the native runner (or, if ever wired,
# by Harbor's verifier).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(dirname "$HERE")"
ANSWERS="${1:-$TASK_DIR/answers.json}"
OUT="${2:-$TASK_DIR/scores.json}"
REWARD="${3:-/logs/verifier/reward.txt}"
exec python -m scroll_eval.evals.beam.judge \\
    --questions "$HERE/probing_questions.json" \\
    --answers "$ANSWERS" \\
    --out "$OUT" \\
    --reward-file "$REWARD"
"""

def _is_plan_wrapped(chat: list) -> bool:
    """True for the 10M shape: a list of single-key ``{"plan-N": [batches]}`` dicts."""
    return bool(chat) and all(
        isinstance(el, dict)
        and len(el) == 1
        and next(iter(el)).startswith("plan-")
        and isinstance(next(iter(el.values())), list)
        for el in chat
    )


def normalize_chat(chat: list) -> list[dict]:
    """Return ``chat`` as a flat batch list, flattening 10M plan wrappers.

    Batch numbers restart at 1 inside each plan, so flattening renumbers them
    globally (1..N in plan order) — the ingester bakes ``batch_number`` into the
    ``[Session N]`` tags that ordering/temporal probes rely on. The original
    plan label and per-plan batch number are preserved on each batch.
    Flat inputs (100K/500K/1M) are returned unchanged.
    """
    if not _is_plan_wrapped(chat):
        return chat
    flat: list[dict] = []
    for wrapper in chat:
        plan, batches = next(iter(wrapper.items()))
        for batch in batches:
            out = dict(batch)
            out["plan"] = plan
            out["plan_batch_number"] = batch.get("batch_number")
            out["batch_number"] = len(flat) + 1
            flat.append(out)
    return flat


def _count_messages(chat: list[dict]) -> int:
    return sum(len(group) for batch in chat for group in batch.get("turns", []))


def _gold_for(question: dict) -> str:
    """Best-effort reference answer for the oracle check (not used in grading)."""
    for field in GOLD_FIELDS:
        value = question.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _task_toml(name: str, scale: str, conv_id: int, n_sessions: int, n_probes: int) -> str:
    return (
        'schema_version = "1.1"\n\n'
        "[task]\n"
        f'name = "{name}"\n'
        f'description = "BEAM long-term-memory probing over a {scale} multi-session conversation."\n\n'
        "[metadata]\n"
        'runner = "native"\n'        # marks this NOT a Harbor task; the native beam runner drives it
        'benchmark = "beam"\n'
        f'scale = "{scale}"\n'
        f"conversation_id = {conv_id}\n"
        f"n_sessions = {n_sessions}\n"
        f"n_probes = {n_probes}\n\n"
        "[verifier]\n"
        "timeout_sec = 1800.0\n"
    )


def migrate_one(src_dir: Path, dest_dir: Path, scale: str, conv_id: int) -> int:
    """Migrate one conversation directory. Returns the number of probes."""
    chat_path = src_dir / "chat.json"
    pq_path = src_dir / "probing_questions" / "probing_questions.json"
    if not chat_path.exists() or not pq_path.exists():
        raise FileNotFoundError(f"{src_dir} missing chat.json or probing_questions.json")

    raw = json.loads(chat_path.read_text(encoding="utf-8"))
    chat = normalize_chat(raw)
    # A chat the ingester would see as empty means an unrecognized schema —
    # fail here (cheap, offline) rather than inside an expensive LLM run.
    if _count_messages(chat) == 0:
        raise ValueError(f"{chat_path}: no turns after normalization (unknown schema?)")
    probing = json.loads(pq_path.read_text(encoding="utf-8"))

    (dest_dir / "solution").mkdir(parents=True, exist_ok=True)
    (dest_dir / "tests").mkdir(parents=True, exist_ok=True)

    # chat.json: verbatim copy when already canonical; re-serialized if flattened.
    if chat is raw:
        shutil.copyfile(chat_path, dest_dir / "chat.json")
    else:
        (dest_dir / "chat.json").write_text(
            json.dumps(chat, ensure_ascii=False), encoding="utf-8"
        )
    # Held-out probing questions: copied verbatim (full fidelity).
    shutil.copyfile(pq_path, dest_dir / "tests" / "probing_questions.json")

    # Agent-visible questions (id/type/question only) + oracle gold, in BEAM's
    # type order. id = "<type>-<index>".
    questions: list[dict] = []
    gold: dict[str, list[dict]] = {}
    n_probes = 0
    for qtype in QUESTION_TYPES:
        items = probing.get(qtype) or []
        gold[qtype] = []
        for index, q in enumerate(items):
            qid = f"{qtype}-{index}"
            text = q.get("question", "")
            questions.append({"id": qid, "type": qtype, "question": text})
            # solution/answers.json uses the judge's answer format so the oracle
            # check is simply `judge(solution/answers.json)`.
            gold[qtype].append({"id": qid, "question": text, "llm_response": _gold_for(q)})
            n_probes += 1

    (dest_dir / "questions.json").write_text(
        json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (dest_dir / "solution" / "answers.json").write_text(
        json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # No per-task instruction.md: the standing agent guidance is a single system
    # prompt in scroll_eval/evals/beam/prompts/system.md, and each probe's user
    # message is just its question.

    test_sh = dest_dir / "tests" / "test.sh"
    test_sh.write_text(_TEST_SH, encoding="utf-8")
    test_sh.chmod(0o755)

    name = f"beam/{scale}-{conv_id}"
    (dest_dir / "task.toml").write_text(
        _task_toml(name, scale, conv_id, n_sessions=len(chat), n_probes=n_probes),
        encoding="utf-8",
    )
    return n_probes


def migrate(src: Path, dest: Path, scale: str) -> list[str]:
    """Migrate every numbered conversation under ``src``. Returns task dir names."""
    if not src.exists():
        raise FileNotFoundError(f"source tier not found: {src}")
    conv_dirs = sorted(
        (d for d in src.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if not conv_dirs:
        raise FileNotFoundError(f"no numbered conversation dirs under {src}")

    written: list[str] = []
    for conv in conv_dirs:
        conv_id = int(conv.name)
        task_name = f"{scale}-{conv_id}"
        dest_dir = dest / task_name
        n_probes = migrate_one(conv, dest_dir, scale, conv_id)
        written.append(task_name)
        print(f"  {task_name}: {len(json.loads((dest_dir / 'chat.json').read_text()))} sessions, {n_probes} probes")
    return written


# Tier dirs shipped under external/beam/chats. --scale is matched
# case-insensitively against these (so "10m" and "10M" both work), which both
# picks the default --src AND fixes the task-name casing (beam CLI --scale
# expects "10M-*"). Resolved relative to the repo root so it works from any CWD.
_TIERS = ("100K", "500K", "1M", "10M")
_CHATS_ROOT = Path(__file__).resolve().parents[1] / "external" / "beam" / "chats"


def _canonical_scale(scale: str) -> str:
    """Canonical tier casing for a --scale value; passthrough for a custom label."""
    return next((t for t in _TIERS if t.lower() == scale.lower()), scale)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate a BEAM tier into local-tasks/.")
    parser.add_argument(
        "--scale", type=str, default="100K",
        help="Tier to migrate: 100K, 500K, 1M, or 10M (case-insensitive). "
             "Drives the default --src, so `--scale 10M` reads the 10M tier.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Source tier dir with numbered conversation subdirs. Defaults to "
             "external/beam/chats/<scale> derived from --scale; set this only to "
             "point at a non-standard location.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("local-tasks/beam"),
        help="Destination dataset dir (task dirs go under here).",
    )
    args = parser.parse_args()

    scale = _canonical_scale(args.scale)
    src = args.src.expanduser() if args.src is not None else _CHATS_ROOT / scale

    print(f"Migrating BEAM {scale} from {src} -> {args.dest}")
    written = migrate(src, args.dest, scale)
    print(f"Done: {len(written)} task(s) under {args.dest}")


if __name__ == "__main__":
    main()
