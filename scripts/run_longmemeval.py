#!/usr/bin/env python3
"""Multi-QA orchestrator for LongMemEval.

LongMemEval is structured as 500 independent QA items, each with its
own haystack of chat sessions. Scroll's :func:`run_single`
benchmarks ONE haystack per call, so this script wraps it: load the
dataset, optionally subset by question type or count, and run every
selected question_id as a separate (output_dir-isolated) sub-run
under a single config file.

Outputs land at::

    output/longmemeval/<policy>_<seed>_<config_hash8>/
        qa_<question_id>/                 # one per QA item
            config.json
            run_result.json
            probe_results.json
            conversation_log.jsonl
            ...
        hypotheses.jsonl                  # aggregate (eval-ready)
        summary.json                      # per-type accuracy

The aggregated ``hypotheses.jsonl`` is in the format expected by the
upstream LongMemEval evaluator (``{question_id, hypothesis}`` per
line) — for sanity-checking against published baselines you can run::

    python external/longmemeval/src/evaluation/evaluate_qa.py \\
        gpt-4o \\
        output/longmemeval/.../hypotheses.jsonl \\
        external/longmemeval/data/longmemeval_oracle.json

The judge fires inline during ``inject_probe`` though, so the
``probe_results.json`` ``score`` field is already populated when the
sub-run finishes. The shellout is only useful if you want to swap
judge models without re-running the agents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import agentscope

# Ensure src/ is on sys.path when run from repo root without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from Scroll.benchmark import run_single  # noqa: E402
from Scroll.core import AgentConfig, get_env  # noqa: E402
from Scroll.core._checkpoint import config_hash  # noqa: E402


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        cur = os.environ.get(key)
        if cur is None or (isinstance(cur, str) and not cur.strip()):
            os.environ[key] = val


def _load_dataset_qids(dataset_path: str) -> list[dict]:
    """Load just the QA metadata for orchestrator filtering/sampling.

    Memory optimization: prefer the tiny ``_index.json`` written by
    ``scripts/shard_lme_dataset.py`` (~50KB) over the full ~264MB
    dataset when only ``question_id`` + ``question_type`` are needed
    for QA selection. The orchestrator doesn't need haystack content.
    """
    p = Path(dataset_path)
    shard_index = Path(p.with_suffix("").as_posix() + "_shards/_index.json")
    if shard_index.exists():
        return json.loads(shard_index.read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))


def _run_one_qa(job: dict) -> dict:
    """Worker — runs ONE QA item to completion.

    Top-level so ``ProcessPoolExecutor`` can pickle it. ``job`` is a
    plain dict (everything picklable). Reads ``probe_results.json``
    after :func:`run_single` returns and packs the score + hypothesis
    into the result.

    Each call re-imports the framework and re-initializes
    AgentScope; that's a ~5s overhead per process worker, dwarfed by
    the per-QA work for ``_s_cleaned``.
    """
    # Imports inside the function so the parent process doesn't pay
    # for them when running sequentially through this same path.
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = _Path(__file__).resolve().parent.parent
    if str(_repo_root / "src") not in _sys.path:
        _sys.path.insert(0, str(_repo_root / "src"))

    # Silence the noisy "Task exception was never retrieved /
    # RuntimeError: Event loop is closed" tracebacks that fire from
    # httpx.AsyncClient.aclose() at subprocess shutdown. The LLM
    # calls themselves succeed; httpx's finalizer just runs after the
    # asyncio loop has already been closed. Cosmetic only — drop
    # asyncio logger records that match this pattern.
    import asyncio as _asyncio
    import logging as _logging
    class _SilenceLoopClosed(_logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return not (
                "Event loop is closed" in msg
                or "AsyncClient.aclose" in msg
                or "Task exception was never retrieved" in msg
            )
    _logging.getLogger("asyncio").addFilter(_SilenceLoopClosed())
    # asyncio also writes shutdown noise to stderr via
    # ``loop.call_exception_handler``. Override the default handler
    # to swallow the same patterns.
    def _silent_handler(loop, context):
        exc = context.get("exception")
        msg = str(context.get("message", ""))
        if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
            return
        if "AsyncClient.aclose" in msg:
            return
        # Anything else still goes to the default handler.
        loop.default_exception_handler(context)
    try:
        _asyncio.get_event_loop().set_exception_handler(_silent_handler)
    except RuntimeError:
        # No running loop yet — handler gets installed when the
        # agent's loop starts; falling back to logger filter is fine.
        pass

    # Per-subprocess tracing setup (must run before agentscope.init
    # and before any LLM call so OpenAIInstrumentor patches the
    # OpenAI SDK in this process). Each worker emits to the same
    # OTLP endpoint independently — Phoenix groups them by run-span
    # name (env_id/<output-folder-basename>).
    tracing_url = job.get("tracing_url")
    if tracing_url:
        from Scroll._tracing import setup as _setup_tracing
        from openinference.instrumentation.openai import OpenAIInstrumentor as _OAI
        _setup_tracing(tracing_url)
        _OAI().instrument()

    import agentscope as _agentscope
    _agentscope.init()
    if tracing_url:
        _agentscope._config.trace_enabled = True

    from Scroll.benchmark import run_single as _run_single
    from Scroll.core import AgentConfig as _AgentConfig, get_env as _get_env

    qid = job["qid"]
    qtype = job["qtype"]
    raw_cfg = job["raw_cfg"]

    per_qa_cfg = json.loads(json.dumps(raw_cfg))
    per_qa_cfg["simulation"]["question_id"] = qid
    per_qa_cfg["simulation"].pop("question_index", None)

    env_cfg = _get_env("longmemeval").parse_env_config(per_qa_cfg["simulation"])
    agent_cfg = _AgentConfig.from_dict(per_qa_cfg["agent"])
    data_cfg = per_qa_cfg.get("data_sources", {})

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(per_qa_cfg, indent=2, default=str), encoding="utf-8"
    )

    try:
        stats = _run_single(
            job["seed"],
            env_cfg,
            agent_cfg,
            data_cfg=data_cfg,
            fresh=job["fresh"],
            checkpoint=job["checkpoint"],
            output_dir=str(out_dir),
            env_id="longmemeval",
            keep_logs=job.get("keep_logs", True),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "question_id": qid,
            "question_type": qtype,
            "score": 0.0,
            "hypothesis": "",
            "active_sessions": 0,
            "error": repr(exc),
        }

    score = 0.0
    hypothesis = ""
    efficiency: dict = {}
    probe_path = out_dir / "probe_results.json"
    if probe_path.exists():
        d = json.loads(probe_path.read_text(encoding="utf-8"))
        probes = d.get("probes") or []
        if probes:
            score = float(probes[0].get("score", 0.0))
            hypothesis = str(probes[0].get("agent_answer", ""))
        efficiency = d.get("efficiency") or {}

    return {
        "question_id": qid,
        "question_type": qtype,
        "score": score,
        "hypothesis": hypothesis,
        "active_sessions": stats.active_turns,
        # Cost / behavior stats (per-QA) — token totals, LM-call count,
        # lessons-written count. Used for cross-policy comparison.
        "lm_calls": int(efficiency.get("lm_calls", 0) or 0),
        "prompt_tokens": int(efficiency.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(efficiency.get("completion_tokens", 0) or 0),
        "total_tokens": int(efficiency.get("total_tokens", 0) or 0),
        "lessons_count": int(efficiency.get("lessons_count", 0) or 0),
        "message_count": int(efficiency.get("message_count", 0) or 0),
        "wall_seconds": float(efficiency.get("wall_seconds", 0.0) or 0.0),
    }


def _select_qids(
    items: list[dict],
    *,
    question_ids: list[str] | None,
    question_types: list[str] | None,
    limit: int | None,
    per_type: int | None,
    seed_offset: int,
    include_abstention: bool = False,
) -> list[dict]:
    """Select a subset of QA items.

    Precedence:
      1. ``question_ids`` — explicit list (errors on missing ids).
      2. ``per_type`` — stratified sample: take N items from each
         distinct question_type (filtered by ``question_types`` if
         given). Output is interleaved by type so parallel workers
         start with a balanced mix.
      3. ``question_types`` + ``limit`` — flat top-K filter.

    ``seed_offset`` rotates the dataset before picking so different
    runs can sample different windows without re-seeding the dataset.
    """
    if question_ids:
        by_id = {it["question_id"]: it for it in items}
        missing = [q for q in question_ids if q not in by_id]
        if missing:
            raise SystemExit(f"question_ids missing from dataset: {missing}")
        return [by_id[q] for q in question_ids]

    pool = items
    if not include_abstention:
        pool = [it for it in pool if not it["question_id"].endswith("_abs")]
    if seed_offset > 0:
        pool = pool[seed_offset:] + pool[:seed_offset]
    if question_types:
        types = list(question_types)
        pool = [it for it in pool if it["question_type"] in set(types)]
    else:
        types = sorted({it["question_type"] for it in pool})

    if per_type is not None:
        # Group, take N per type, then interleave so parallel workers
        # pick a mix instead of all of one type first.
        groups: dict[str, list[dict]] = {qt: [] for qt in types}
        for it in pool:
            qt = it["question_type"]
            if qt in groups and len(groups[qt]) < per_type:
                groups[qt].append(it)
        out: list[dict] = []
        for i in range(per_type):
            for qt in types:
                if i < len(groups[qt]):
                    out.append(groups[qt][i])
        return out

    if limit is not None:
        pool = pool[:limit]
    return pool


def main() -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(description="Run LongMemEval across all selected QA items.")
    p.add_argument("--config", required=True, help="Path to a configs/longmemeval/*.json file.")
    p.add_argument(
        "--question-ids", nargs="+", default=None,
        help="Run only these question_ids (default: filter via --question-types / --limit).",
    )
    p.add_argument(
        "--question-types", nargs="+", default=None,
        help="Filter to these question_type values (e.g. multi-session knowledge-update).",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap on the number of QA items to run after filtering "
             "(top-K from filtered pool). Mutually exclusive with --per-type.",
    )
    p.add_argument(
        "--per-type", type=int, default=None,
        help="Stratified sample: take N items from each question_type "
             "(filtered by --question-types if given). Output is "
             "interleaved by type so parallel workers start with a mix.",
    )
    p.add_argument(
        "--seed-offset", type=int, default=0,
        help="Rotate the dataset by this many items before picking (for staggered seeds).",
    )
    p.add_argument(
        "--include-abstention", action="store_true",
        help="Include ``_abs``-suffixed abstention items (default: exclude — pure memory-recall test).",
    )
    p.add_argument(
        "--max-parallel", type=int, default=1,
        help="Run this many QA items concurrently (each in its own "
             "subprocess for clean isolation; default 1 = sequential). "
             "Each subprocess re-imports the framework, so the "
             "per-process startup overhead is ~5s — only useful when "
             "per-QA work dominates.",
    )
    p.add_argument(
        "--seed", type=int, default=1,
        help="Run seed (single-seed mode for now; matches the QA loop).",
    )
    p.add_argument(
        "--fresh", action=argparse.BooleanOptionalAction, default=True,
        help="Redo every QA, ignoring any existing per-QA "
             "probe_results.json (default: True). Pass --no-fresh to "
             "skip QAs that already have results (resume a partial run).",
    )
    p.add_argument(
        "--checkpoint", action="store_true",
        help="Keep per-QA conversation_log.jsonl + session_logs.jsonl + "
             "per-session checkpoint subdirs on disk. OFF by default — "
             "only probe_results.json + config.json + summary.json are "
             "written. Use --checkpoint when you need the full event log "
             "for debugging or offline ``Scroll rebuild-w``.",
    )
    p.add_argument(
        "--no-checkpoint", action="store_true",
        help="(no-op; this is the default behavior now — kept for "
             "backward compatibility with old scripts.)",
    )
    p.add_argument(
        "--keep-checkpoint", action="store_true",
        help="(Backward-compat alias for --checkpoint.)",
    )
    p.add_argument("--tracing-url", default=None, help="OTLP tracing endpoint.")
    p.add_argument(
        "--output-root", default="output/longmemeval",
        help="Root directory; per-run dirs are placed under <root>/<policy>_<seed>_<hash8>/qa_<qid>/.",
    )
    args = p.parse_args()

    if args.tracing_url:
        from Scroll._tracing import setup as setup_tracing
        from openinference.instrumentation.openai import OpenAIInstrumentor
        setup_tracing(args.tracing_url)
        OpenAIInstrumentor().instrument()
    agentscope.init()
    if args.tracing_url:
        agentscope._config.trace_enabled = True

    raw_cfg = json.loads(Path(args.config).read_text())
    env_id = raw_cfg.get("environment", "longmemeval")
    if env_id != "longmemeval":
        raise SystemExit(
            f"config {args.config} declares environment={env_id!r}; expected 'longmemeval'"
        )

    dataset_path = raw_cfg.get("simulation", {}).get("dataset_path")
    if not dataset_path:
        raise SystemExit("config simulation.dataset_path is required")
    if args.limit is not None and args.per_type is not None:
        raise SystemExit("--limit and --per-type are mutually exclusive")

    items = _load_dataset_qids(dataset_path)
    selected = _select_qids(
        items,
        question_ids=args.question_ids,
        question_types=args.question_types,
        limit=args.limit,
        per_type=args.per_type,
        seed_offset=args.seed_offset,
        include_abstention=args.include_abstention,
    )
    if not selected:
        raise SystemExit("no QA items selected — check --question-ids / --question-types / --limit / --per-type")

    cfg_hash = config_hash(raw_cfg)
    policy = raw_cfg["agent"]["policy"]
    run_root = Path(args.output_root) / f"{policy}_{args.seed}_{cfg_hash[:8]}"
    run_root.mkdir(parents=True, exist_ok=True)

    print(
        f"longmemeval orchestrator: policy={policy} seed={args.seed} "
        f"hash={cfg_hash[:8]} qa_items={len(selected)} dataset={dataset_path}"
    )

    per_qa_results: list[dict] = []
    qtype_correct: dict[str, list[float]] = defaultdict(list)
    started = time.time()

    def _absorb_skipped(skipped_list: list[dict]) -> None:
        """Read probe_results.json from skipped (already-done) QAs and
        merge them into the aggregate so the run-end summary covers
        the full selected set, not only the QAs that ran this invocation.
        """
        for s in skipped_list:
            probe_path = Path(s["out_dir"]) / "probe_results.json"
            try:
                d = json.loads(probe_path.read_text(encoding="utf-8"))
                probes = d.get("probes") or [{}]
                p0 = probes[0]
                score = float(p0.get("score", 0.0))
                hyp = str(p0.get("agent_answer", ""))
                eff = d.get("efficiency") or {}
            except Exception:  # noqa: BLE001
                score, hyp, eff = 0.0, "", {}
            r = {
                "question_id": s["qid"],
                "question_type": s["qtype"],
                "score": score,
                "hypothesis": hyp,
                "active_sessions": eff.get("total_sessions", 0),
                "efficiency": eff,
            }
            per_qa_results.append(r)
            qtype_correct[s["qtype"]].append(score)

    # Skip QAs whose ``probe_results.json`` already exists — resume
    # semantics. Without this filter the orchestrator re-runs every
    # selected QA from scratch, overwriting prior scores AND burning
    # API quota on work we already paid for. The check is the single
    # source of truth for "is this QA done"; ``run_single`` itself
    # does not gate on output presence. Skipped jobs' scores are
    # re-read from disk below so the summary still aggregates them.
    # ``--fresh`` bypasses the skip (caller explicitly wants a redo).
    jobs = []
    skipped: list[dict] = []
    for item in selected:
        qa_dir = run_root / f"qa_{item['question_id']}"
        probe_path = qa_dir / "probe_results.json"
        if probe_path.exists() and not args.fresh:
            skipped.append({"qid": item["question_id"], "qtype": item["question_type"], "out_dir": str(qa_dir)})
            continue
        # OFF by default — only probe_results.json + config.json land on
        # disk per QA. --checkpoint (or legacy --keep-checkpoint) opts
        # back into the full conversation_log.jsonl + session_logs.jsonl
        # + per-session checkpoint subdirs.
        keep_logs = bool(args.checkpoint or args.keep_checkpoint)
        jobs.append({
            "qid": item["question_id"],
            "qtype": item["question_type"],
            "raw_cfg": raw_cfg,
            "out_dir": str(qa_dir),
            "seed": args.seed,
            "fresh": args.fresh,
            "checkpoint": keep_logs,
            "keep_logs": keep_logs,
            "tracing_url": args.tracing_url,
        })
    if skipped:
        print(f"  resume: skipping {len(skipped)} QAs with existing probe_results.json; running {len(jobs)} new")
        _absorb_skipped(skipped)

    if args.max_parallel <= 1:
        # Sequential — share the parent process so we don't pay the
        # per-worker import overhead. ``_run_one_qa`` does its own
        # imports defensively but they're cached at module level.
        for idx, job in enumerate(jobs, start=1):
            progress = f"[{idx}/{len(jobs)} qid={job['qid']} type={job['qtype']}]"
            print(f"{progress} starting...", flush=True)
            r = _run_one_qa(job)
            per_qa_results.append(r)
            qtype_correct[r["question_type"]].append(r["score"])
            elapsed = time.time() - started
            err = f' error={r.get("error")}' if r.get("error") else ""
            print(
                f"{progress} score={r['score']:.2f} elapsed={elapsed:.1f}s{err}",
                flush=True,
            )
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(
            f"running {len(jobs)} QA items with max_parallel={args.max_parallel} "
            f"(each in its own subprocess)...",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
            future_to_job = {executor.submit(_run_one_qa, j): j for j in jobs}
            done_count = 0
            for fut in as_completed(future_to_job):
                done_count += 1
                job = future_to_job[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001
                    r = {
                        "question_id": job["qid"],
                        "question_type": job["qtype"],
                        "score": 0.0,
                        "hypothesis": "",
                        "active_sessions": 0,
                        "error": repr(exc),
                    }
                per_qa_results.append(r)
                qtype_correct[r["question_type"]].append(r["score"])
                elapsed = time.time() - started
                err = f' error={r.get("error")}' if r.get("error") else ""
                print(
                    f"[{done_count}/{len(jobs)} qid={r['question_id']} "
                    f"type={r['question_type']}] score={r['score']:.2f} "
                    f"elapsed={elapsed:.1f}s{err}",
                    flush=True,
                )

    # ------------- aggregate output --------------
    hyp_path = run_root / "hypotheses.jsonl"
    with hyp_path.open("w", encoding="utf-8") as f:
        for r in per_qa_results:
            f.write(json.dumps({"question_id": r["question_id"], "hypothesis": r["hypothesis"]}) + "\n")

    n = max(1, len(per_qa_results))
    walls = [r["wall_seconds"] for r in per_qa_results if r.get("wall_seconds")]
    walls_sorted = sorted(walls)
    median_wall = walls_sorted[len(walls_sorted) // 2] if walls_sorted else 0.0
    p95_wall = walls_sorted[int(0.95 * len(walls_sorted))] if walls_sorted else 0.0
    cost = {
        "avg_lm_calls": round(sum(r["lm_calls"] for r in per_qa_results) / n, 2),
        "avg_prompt_tokens": round(sum(r["prompt_tokens"] for r in per_qa_results) / n, 1),
        "avg_completion_tokens": round(sum(r["completion_tokens"] for r in per_qa_results) / n, 1),
        "avg_total_tokens": round(sum(r["total_tokens"] for r in per_qa_results) / n, 1),
        "avg_message_count": round(sum(r["message_count"] for r in per_qa_results) / n, 2),
        "avg_lessons_count": round(sum(r["lessons_count"] for r in per_qa_results) / n, 2),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in per_qa_results),
        "total_completion_tokens": sum(r["completion_tokens"] for r in per_qa_results),
        "total_lessons_written": sum(r["lessons_count"] for r in per_qa_results),
        # Wall-time stats per QA: avg / median / p95. Wall clock measured
        # from inside ``benchmark.py::run_single`` so it reflects true
        # per-QA cost (agent + judge), not orchestrator-side overhead.
        "avg_wall_seconds": round(sum(walls) / n, 1) if walls else 0.0,
        "median_wall_seconds": round(median_wall, 1),
        "p95_wall_seconds": round(p95_wall, 1),
        "total_wall_seconds": round(sum(walls), 1),
    }
    summary = {
        "policy": policy,
        "seed": args.seed,
        "config_hash": cfg_hash[:8],
        "dataset_path": dataset_path,
        "total_qa": len(per_qa_results),
        "avg_score": round(
            sum(r["score"] for r in per_qa_results) / n, 4
        ),
        "by_type": {
            qt: {
                "count": len(vals),
                "accuracy": round(sum(vals) / max(1, len(vals)), 4),
            }
            for qt, vals in qtype_correct.items()
        },
        "cost": cost,
        "results": per_qa_results,
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print()
    print(f"=== summary policy={policy} seed={args.seed} ===")
    print(f"total: {summary['total_qa']}, avg_score: {summary['avg_score']:.4f}")
    for qt, st in summary["by_type"].items():
        print(f"  {qt:>30s}: acc={st['accuracy']:.4f} n={st['count']}")
    print(f"  cost/QA: lm_calls={cost['avg_lm_calls']:.1f}  "
          f"tokens={cost['avg_total_tokens']:.0f} "
          f"(prompt={cost['avg_prompt_tokens']:.0f} + completion={cost['avg_completion_tokens']:.0f})")
    print(f"  wall/QA: avg={cost['avg_wall_seconds']:.1f}s  "
          f"median={cost['median_wall_seconds']:.1f}s  "
          f"p95={cost['p95_wall_seconds']:.1f}s  "
          f"(total serial-eq: {cost['total_wall_seconds']/60:.1f} min)")
    print(f"  lessons/QA: {cost['avg_lessons_count']:.2f}  "
          f"(total written across run: {cost['total_lessons_written']})")
    print(f"hypotheses → {hyp_path}")
    print(f"summary    → {run_root / 'summary.json'}")


if __name__ == "__main__":
    main()
