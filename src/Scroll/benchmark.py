from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from opentelemetry import trace as otel_trace

from Scroll.core import (
    AgentConfig,
    BaseEnvConfig,
    RunStats,
    EnvSnapshot,
    ProbeResult,
    ProbeSpec,
    serialize_rng,
    restore_rng,
    get_env,
)
from Scroll.core._checkpoint import (
    config_hash,
    save_checkpoint,
    load_checkpoint,
)
from Scroll.core._evaluation import (
    inject_probe,
    write_probe_results,
)
from Scroll.log import ConversationLog

_tracer = otel_trace.get_tracer(__name__)


def _append_session_log(output_dir, session_idx, ds_notes, actions, env_logs, snapshot):
    """Append one session's full log record to session_logs.jsonl."""
    path = Path(output_dir) / "session_logs.jsonl"
    snap = asdict(snapshot)
    # ``snap["session_idx"]`` is redundant with the top-level field and ``logs``
    # is already in ``env_logs``; drop both from the snapshot block.
    snap.pop("session_idx", None)
    snap.pop("logs", None)
    record = {
        "session_idx": session_idx,
        "datasource_notes": ds_notes,
        "agent_actions": actions,
        "env_logs": env_logs,
        "snapshot": snap,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _truncate_session_log(output_dir, resume_session):
    """On resume, remove log entries past the checkpoint session to avoid duplicates."""
    path = Path(output_dir) / "session_logs.jsonl"
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["session_idx"] <= resume_session:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def _run_session_loop(
    env,
    agent,
    data,
    log,
    env_cfg: BaseEnvConfig,
    is_llm: bool,
    policy: str,
    seed: int,
    snapshots: list[EnvSnapshot],
    probe_results: list[ProbeResult],
    per_session_action_logs: list[list[str]],
    get_probes_for_session: Callable[[int], list[ProbeSpec]] = lambda s: [],
    checkpoint: bool = True,
    output_dir: str | None = None,
    cfg_hash: str = "",
) -> int:
    """Session-count based loop. Returns active_sessions."""
    active_sessions = 0
    try:
     for _ in range(env_cfg.num_sessions):
      with _tracer.start_as_current_span(f"session.{env.session_idx + 1}") as session_span:
        if is_llm:
            print(
                f"\r  [{policy}] seed={seed} session {env.session_idx + 1}/{env_cfg.num_sessions} ...",
                end="",
                flush=True,
            )
        ds_notes = data.begin_session(env.session_idx, env)
        agent.receive_context(env.session_idx, ds_notes)

        with _tracer.start_as_current_span("agent.run_session"):
            actions = agent.run_session(env)

        per_session_action_logs.append(list(actions))

        with _tracer.start_as_current_span("env.step_session") as env_span:
            session_res = env.step_session()
            env_span.set_attributes({
                "env.revenue": round(session_res.revenue, 2),
                "env.units_sold": session_res.sold_units,
            })
        logs = env.today_logs()
        agent.receive_outcomes(session_res.session_idx, logs)

        snapshots.append(env.build_snapshot())

        # Persist full per-session log for debugging / querying
        if output_dir:
            _append_session_log(
                output_dir, session_res.session_idx, ds_notes, actions, logs, snapshots[-1],
            )

        # Inject probes if scheduled for this session
        probes = get_probes_for_session(session_res.session_idx)
        if probes and is_llm:
            for probe in probes:
                result = inject_probe(agent, probe, snapshots, data)
                probe_results.append(result)

        if session_res.sold_units > 0:
            active_sessions += 1

        session_span.set_attributes({
            "session.number": session_res.session_idx,
            "session.revenue": round(session_res.revenue, 2),
            "session.units_sold": session_res.sold_units,
            "session.cash": round(session_res.cash, 2),
            "session.net_worth": round(env.net_worth(), 2),
        })

        # Save checkpoint after each session
        if checkpoint and output_dir:
            loop_state = {
                "active_sessions": active_sessions,
                "loop_type": "session",
            }
            save_checkpoint(
                output_dir, session_res.session_idx, env, data, agent, log,
                loop_state, policy, seed, cfg_hash,
            )

        if env.is_terminal():
            break
    except KeyboardInterrupt:
        print("\nInterrupted — checkpoint already saved for last completed session.")
        raise
    return active_sessions


def run_single(
    seed: int,
    env_cfg: BaseEnvConfig,
    agent_cfg: AgentConfig,
    data_cfg: dict | None = None,
    fresh: bool = False,
    checkpoint: bool = True,
    output_dir: str | None = None,
    env_id: str = "longmemeval",
    keep_logs: bool = True,
) -> RunStats:
    policy = agent_cfg.policy
    # Root span name mirrors the on-disk layout but always includes the
    # policy so traces are scannable regardless of how the env nests
    # its output dirs.
    if output_dir:
        basename = Path(output_dir).name
        if basename.startswith(f"{policy}_") or basename == policy:
            span_name = f"{env_id}/{basename}"
        else:
            span_name = f"{env_id}/{policy}/{basename}"
    else:
        span_name = f"{env_id}/{policy}.seed{seed}"
    with _tracer.start_as_current_span(span_name) as run_span:
        _t0 = time.perf_counter()
        stats = _run_single_inner(
            run_span, seed, env_cfg, agent_cfg, data_cfg,
            fresh, checkpoint, output_dir, env_id, keep_logs,
        )
        wall_seconds = round(time.perf_counter() - _t0, 2)
        # Surface end-to-end per-QA wall time as part of efficiency so
        # the orchestrator's summary aggregator can avg / sum it. The
        # value persisted to ``probe_results.json`` was written before
        # we knew the final number; patch it in-place if we wrote one.
        stats.efficiency["wall_seconds"] = wall_seconds
        if output_dir:
            _patch_wall_seconds_on_disk(output_dir, wall_seconds)
        return stats


def _patch_wall_seconds_on_disk(output_dir: str, wall_seconds: float) -> None:
    """Append ``efficiency.wall_seconds`` to the already-written
    ``probe_results.json``. Written outside of the main inner-run path
    because the wall-time can only be measured after the agent + judge
    are fully done.
    """
    pr_path = Path(output_dir) / "probe_results.json"
    if not pr_path.exists():
        return
    try:
        d = json.loads(pr_path.read_text())
        eff = d.setdefault("efficiency", {})
        eff["wall_seconds"] = wall_seconds
        pr_path.write_text(json.dumps(d, indent=2, default=str))
    except Exception:  # noqa: BLE001
        # Don't fail the run on a metric-bookkeeping error.
        pass


def _run_single_inner(
    run_span,
    seed: int,
    env_cfg: BaseEnvConfig,
    agent_cfg: AgentConfig,
    data_cfg: dict | None,
    fresh: bool,
    checkpoint: bool,
    output_dir: str | None,
    env_id: str,
    keep_logs: bool = True,
) -> RunStats:
    policy = agent_cfg.policy
    run_span.set_attributes({
        "run.policy": policy,
        "run.seed": seed,
        "run.env_id": env_id,
    })

    # Use registry to get env/datasource classes and probe functions
    entry = get_env(env_id)

    # Compute config hash for checkpoint validation
    cfg_dict = {
        "env_cfg": asdict(env_cfg) if hasattr(env_cfg, '__dataclass_fields__') else {},
        "agent_cfg": asdict(agent_cfg) if hasattr(agent_cfg, '__dataclass_fields__') else {},
        "data_cfg": data_cfg or {},
        "env_id": env_id,
    }
    cfg_hash = config_hash(cfg_dict)

    # ``conversation_log.jsonl`` is the full E (event log). It's only
    # needed for offline ``Scroll rebuild-w`` / trace inspection — at
    # 500-QA × 500-session-M scale it eats ~100GB+ of disk. Gate on
    # ``keep_logs`` (default True for backward compat; orchestrator
    # disables by default via --no-checkpoint).
    jsonl_path = (
        Path(output_dir) / "conversation_log.jsonl"
        if (output_dir and keep_logs) else None
    )

    # Try to resume from checkpoint
    resumed = False
    resume_active_sessions = 0

    if not fresh and output_dir:
        ckpt = load_checkpoint(output_dir, policy, seed, cfg_hash)
        if ckpt:
            resume_session = ckpt["_meta"]["session_idx"]
            print(f"  Resuming from checkpoint at session {resume_session}...")

            env = entry.env_cls.from_checkpoint(ckpt["env"], env_cfg)
            data = entry.datasource_cls(seed=seed, data_cfg=data_cfg)
            data.from_checkpoint(ckpt["data"])

            log = ConversationLog(jsonl_path=jsonl_path)
            entry_count = (ckpt.get("log") or {}).get("entry_count")
            log.load_from_jsonl(entry_count=entry_count)

            agent = entry.create_agent(policy, agent_cfg, log, data)
            agent.from_checkpoint(ckpt["agent"])

            loop_state = ckpt.get("loop_state", {})
            resume_active_sessions = loop_state.get(
                "active_sessions", loop_state.get("active_days", 0),
            )
            resumed = True

            # Truncate session log to checkpoint session to avoid duplicates on resume
            if output_dir:
                _truncate_session_log(output_dir, resume_session)

    if not resumed:
        env = entry.env_cls(env_cfg, seed)
        data = entry.datasource_cls(seed=seed, data_cfg=data_cfg)
        # Fresh run: clear any stale jsonl from a previous aborted attempt.
        if jsonl_path is not None and jsonl_path.exists():
            jsonl_path.unlink()
        log = ConversationLog(jsonl_path=jsonl_path)
        agent = entry.create_agent(policy, agent_cfg, log, data)

    snapshots: list[EnvSnapshot] = []
    probe_results: list[ProbeResult] = []
    per_session_action_logs: list[list[str]] = []
    is_llm = agent_cfg.policy not in ("heuristic",)

    try:
        # LongMemEval has a fixed-length session loop (N session-days + 1
        # probe day) governed by env.is_terminal(); always use the
        # session-count loop.
        # When ``keep_logs`` is False, hide ``output_dir`` from the
        # session loop so it skips writing ``session_logs.jsonl``
        # (and per-session checkpoint subdirs). ``probe_results.json``
        # is written outside this call, so it's unaffected.
        loop_output_dir = output_dir if keep_logs else None
        active_sessions = _run_session_loop(
            env, agent, data, log, env_cfg,
            is_llm, policy, seed, snapshots, probe_results,
            per_session_action_logs,
            get_probes_for_session=entry.get_probes_for_session,
            checkpoint=checkpoint and keep_logs,
            output_dir=loop_output_dir,
            cfg_hash=cfg_hash,
        )
    finally:
        log.close()

    # Add resumed active sessions
    active_sessions += resume_active_sessions

    if is_llm:
        print("\r" + " " * 80 + "\r", end="", flush=True)

    # Compute efficiency metrics
    efficiency = entry.compute_efficiency_metrics(per_session_action_logs)

    # Agent-level cost / behavior stats — independent of env, useful
    # for cross-policy comparison (token cost vs accuracy tradeoff,
    # procedural-memory accumulation rate).
    state = getattr(agent, "_tool_state", None)
    if state is not None:
        prompt_t = getattr(state, "_prompt_tokens", 0)
        completion_t = getattr(state, "_completion_tokens", 0)
        efficiency["lm_calls"] = getattr(state, "_lm_calls", 0)
        efficiency["prompt_tokens"] = prompt_t
        efficiency["completion_tokens"] = completion_t
        efficiency["total_tokens"] = prompt_t + completion_t
        efficiency["message_count"] = getattr(state, "_message_count", 0)

    try:
        agent.augment_efficiency(efficiency)
    except Exception:  # noqa: BLE001
        pass

    # Compute probe scores. ``avg_probe_score`` is over every probe;
    # any env-specific roll-ups (e.g. vending's A/B category split) come
    # from ``env.summarize_probes`` so core doesn't know about them.
    avg_probe_score = (
        round(sum(r.score for r in probe_results) / len(probe_results), 3)
        if probe_results else 0.0
    )
    try:
        extra_summary = env.summarize_probes(probe_results) or {}
    except Exception:  # noqa: BLE001
        extra_summary = {}

    # Env-specific run-end metrics (vending: net_worth/units_sold/bankrupt;
    # LME/BEAM: empty — their signal is probe_avg_score + token cost).
    try:
        env_metrics = env.report_run_metrics(agent) or {}
    except Exception:  # noqa: BLE001
        env_metrics = {}

    run_span.set_attributes({
        "run.active_sessions": active_sessions,
        "run.avg_probe_score": avg_probe_score,
    })
    for k, v in env_metrics.items():
        run_span.set_attribute(f"run.env.{k}", v)
    for k, v in extra_summary.items():
        run_span.set_attribute(f"run.{k}", v)
    for k, v in efficiency.items():
        run_span.set_attribute(f"run.efficiency.{k}", v)

    # Write probe results and efficiency to output dir. ``extra_summary``
    # gets merged into the summary block of probe_results.json.
    if output_dir and probe_results:
        write_probe_results(probe_results, efficiency, output_dir, extra_summary)

    return RunStats(
        strategy=policy,
        seed=seed,
        active_sessions=active_sessions,
        probe_avg_score=avg_probe_score,
        efficiency=efficiency,
        env_metrics=env_metrics,
    )


def aggregate(results: list[RunStats]) -> dict[str, dict[str, float]]:
    """Aggregate per-run stats grouped by policy.

    Universal keys: ``mean_active_sessions``, ``mean_probe_avg_score``,
    plus ``mean_<k>`` / ``min_<k>`` for every numeric key that appears
    in ``RunStats.env_metrics`` (e.g. vending's ``mean_net_worth`` /
    ``min_net_worth`` / ``mean_units_sold``). Non-numeric env_metrics
    values are skipped.
    """
    grouped: dict[str, list[RunStats]] = {}
    for r in results:
        grouped.setdefault(r.strategy, []).append(r)

    out: dict[str, dict[str, float]] = {}
    for policy, runs in grouped.items():
        stats: dict[str, float] = {
            "mean_active_sessions": round(
                statistics.fmean(r.active_sessions for r in runs), 2
            ),
            "mean_probe_avg_score": round(
                statistics.fmean(r.probe_avg_score for r in runs), 3
            ),
        }
        # Aggregate env-specific numeric metrics. Union all keys seen
        # across runs; if a run is missing one (e.g. mixed-env compare,
        # which isn't expected but tolerated), it's skipped from that
        # key's mean.
        env_keys: set[str] = set()
        for r in runs:
            env_keys.update(r.env_metrics.keys())
        for key in sorted(env_keys):
            values = [
                r.env_metrics[key] for r in runs
                if key in r.env_metrics
                and isinstance(r.env_metrics[key], (int, float))
                and not isinstance(r.env_metrics[key], bool)
            ]
            if not values:
                continue
            stats[f"mean_{key}"] = round(statistics.fmean(values), 2)
            stats[f"min_{key}"] = round(min(values), 2)
        out[policy] = stats
    return out
