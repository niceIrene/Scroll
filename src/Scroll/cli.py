from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import agentscope

from Scroll.benchmark import aggregate, run_single
from Scroll.core import AgentConfig, get_env
from Scroll.core._checkpoint import config_hash


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


# ---------------------------------------------------------------------------
# rebuild-w — offline ``W = build(E)`` reconstruction.
#
# Loads a conversation_log.jsonl, instantiates a fresh memoryspace with
# the env's schema, runs the env's :class:`Ingestor` over the loaded
# entries, and dumps the resulting W to disk. The invariant test for
# the SCROLL pattern: running this against a jsonl produced by a normal
# run should produce a W functionally equivalent to the one the agent
# saw at runtime.
# ---------------------------------------------------------------------------


def _env_ingest_modules(env_id: str):
    """Return ``(ensure_schema, ingestor_cls)`` for an env id."""
    if env_id == "longmemeval":
        from Scroll.benchmarks.longmemeval.ingestor import (
            LMEIngestor, ensure_schema,
        )
        return ensure_schema, LMEIngestor
    if env_id == "vending":
        from Scroll.benchmarks.vending.ingestor import (
            VendingIngestor, ensure_schema,
        )
        return ensure_schema, VendingIngestor
    if env_id == "beam":
        from Scroll.benchmarks.beam.ingestor import (
            BeamIngestor, ensure_schema,
        )
        return ensure_schema, BeamIngestor
    raise SystemExit(
        f"rebuild-w: unknown env {env_id!r}. "
        f"Known: longmemeval, vending, beam."
    )


def _cmd_rebuild_w(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="Scroll rebuild-w",
        description="Rebuild W from a conversation_log.jsonl. "
                    "Exercises the W = build(E) invariant.",
    )
    parser.add_argument(
        "--log", required=True,
        help="Path to conversation_log.jsonl (the persisted E).",
    )
    parser.add_argument(
        "--env", required=True,
        choices=["longmemeval", "vending", "beam"],
        help="Which env's schema + ingestor to use.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Where to dump the rebuilt memoryspace "
             "(creates ``<dir>/memoryspace/...``).",
    )
    args = parser.parse_args(argv)

    from Scroll.log import ConversationLog
    from Scroll.tools.memoryspace import Memoryspace

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"rebuild-w: log not found: {log_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = ConversationLog(jsonl_path=None)
    log._jsonl_path = log_path  # type: ignore[attr-defined]
    log.load_from_jsonl()
    log._jsonl_path = None  # don't append on close

    ensure_schema, ingestor_cls = _env_ingest_modules(args.env)
    ms = Memoryspace()
    ensure_schema(ms)
    ingestor = ingestor_cls(ms)
    ingestor.consume(log.entries)
    ms.sqlite.commit()

    ms.dump_memoryspace(out_dir)
    summary = ms.schema_inspect()
    print(
        f"rebuild-w: env={args.env} entries={len(log.entries)} "
        f"tables={len(summary['tables'])} "
        f"vectors={summary['vector_count']} "
        f"json_keys={len(summary['json_keys'])} "
        f"-> {out_dir}/memoryspace/"
    )


def _output_dir(env_id: str, policy: str, seed: int, cfg_hash: str) -> Path:
    """Convention-based output directory using config hash for uniqueness.

    Layout: ``output/<env_id>/<policy>_<seed>_<hash8>/``. The env_id
    namespace prevents hash collisions across envs when the same
    ``policy`` runs on different domains (e.g. ``agentscope_qwen3``
    runs against both vending and tau-bench).
    """
    return Path("output") / env_id / f"{policy}_{seed}_{cfg_hash[:8]}"


def main() -> None:
    _load_dotenv()
    # Dispatch ``Scroll rebuild-w ...`` to the offline rebuild path
    # before the main argparse owner is constructed. This keeps the
    # benchmark CLI's top-level flags unchanged for the default mode
    # while letting ``rebuild-w`` have its own self-contained arg surface.
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "rebuild-w":
        _cmd_rebuild_w(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description="Run Scroll benchmark")
    parser.add_argument("--config", default="configs/vending/default.json", help="Path to JSON config")
    parser.add_argument(
        "--tracing-url",
        default=None,
        help="OTLP tracing endpoint URL (e.g. http://localhost:6006/v1/traces). "
             "Enables tracing and installs an OpenInference bridge: AgentScope "
             "natively emits OTel GenAI (`gen_ai.*`) conventions, which Phoenix "
             "renders as span_kind=UNKNOWN. This flag translates AgentScope's "
             "tool/agent/chain spans to OpenInference and adds "
             "OpenAIInstrumentor for proper LLM span rendering.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoints, start from session 1.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint writing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run only this seed (for single-run mode).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit output directory (overrides convention).",
    )
    args = parser.parse_args()

    if args.tracing_url:
        # Own the tracing setup so we can translate AgentScope's gen_ai.*
        # tool-span attrs into OpenInference conventions (Phoenix renders
        # OpenInference TOOL spans; it shows AgentScope's native spans as
        # span_kind=UNKNOWN). Also instrument the OpenAI SDK so LLM calls
        # (via OpenAIChatModel / Dashscope-compatible endpoint) surface as
        # proper GenAI chat spans.
        from Scroll._tracing import setup as setup_tracing
        from openinference.instrumentation.openai import OpenAIInstrumentor
        setup_tracing(args.tracing_url)
        OpenAIInstrumentor().instrument()
    agentscope.init()
    if args.tracing_url:
        # AgentScope's @trace decorators are gated on _config.trace_enabled,
        # which only flips true when tracing_url is passed to agentscope.init.
        # We manage tracing ourselves, so set the flag directly.
        agentscope._config.trace_enabled = True

    raw_config = json.loads(Path(args.config).read_text())
    env_id = raw_config.get("environment", "vending")
    env_cfg = get_env(env_id).parse_env_config(raw_config["simulation"])
    agent_cfg = AgentConfig.from_dict(raw_config["agent"])
    bench = raw_config["benchmark"]
    data_cfg = raw_config.get("data_sources", {})

    single_run = args.seed is not None
    seeds = [args.seed] if args.seed is not None else bench["seeds"]
    cfg_hash = config_hash(raw_config)

    all_results = []
    results = []
    try:
        for seed in seeds:
            if args.output_dir:
                out_dir = Path(args.output_dir)
            else:
                out_dir = _output_dir(env_id, agent_cfg.policy, seed, cfg_hash)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Write resolved config for reproducibility
            (out_dir / "config.json").write_text(
                json.dumps(raw_config, indent=2, default=str),
                encoding="utf-8",
            )

            run = run_single(
                seed,
                env_cfg,
                agent_cfg,
                data_cfg=data_cfg,
                fresh=args.fresh,
                checkpoint=not args.no_checkpoint,
                output_dir=str(out_dir),
                env_id=env_id,
            )
            results.append(run)
            all_results.append(run)
            # Per-run line: universal fields + each env_metrics key.
            env_metric_str = " ".join(
                f"{k}={v}" for k, v in run.env_metrics.items()
            )
            print(
                f"run policy={run.strategy:>16} seed={run.seed} "
                f"active_turns={run.active_turns:3d} "
                f"probe_score={run.probe_avg_score:.3f}"
                + (f" | {env_metric_str}" if env_metric_str else "")
            )

            # In single-run mode, write result for orchestrator collection
            if single_run:
                (out_dir / "run_result.json").write_text(
                    json.dumps(asdict(run), indent=2, default=str),
                    encoding="utf-8",
                )
    except KeyboardInterrupt:
        print("\nRun interrupted. Resume with the same command.")

    # Skip aggregation in single-run mode (orchestrator handles it)
    if single_run:
        return

    agg = aggregate(results)
    print("\n=== aggregate ===")
    for policy, vals in agg.items():
        parts = [f"{policy:>16}"] + [
            f"{k}={v}" for k, v in vals.items()
        ]
        print(" | ".join(parts))

    # Write aggregate results
    if results:
        agg_dir = Path("output")
        agg_dir.mkdir(parents=True, exist_ok=True)
        (agg_dir / "aggregate.json").write_text(
            json.dumps(agg, indent=2, default=str),
            encoding="utf-8",
        )
