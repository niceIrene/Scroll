#!/usr/bin/env python3
"""Orchestrate parallel Docker-based benchmark runs.

Usage:
    python scripts/run_parallel.py \
        --configs configs/longmemeval/scroll_s.json \
        --seeds 1 2 3 \
        --max-parallel 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _config_hash(config: dict) -> str:
    """Deterministic SHA256 of a config dict (mirrors Scroll.core._checkpoint.config_hash)."""
    raw = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _read_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _run_container(
    config_path: str,
    seed: int,
    image: str,
    output_root: str,
    env_vars: dict[str, str],
    fresh: bool,
    tracing_url: str | None = None,
) -> tuple[str, int, str]:
    """Launch a single Docker container for one (config, seed) run."""
    cfg = _read_config(config_path)
    policy = cfg["agent"]["policy"]
    env_id = cfg.get("environment", "vending")
    hash8 = _config_hash(cfg)[:8]
    name = f"{policy}_{seed}_{hash8}"
    out_dir = Path(output_root) / env_id / name
    out_dir.mkdir(parents=True, exist_ok=True)

    container_out = f"/app/output/{env_id}/{name}"
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{Path(config_path).resolve()}:/app/run_config.json:ro",
        "-v", f"{out_dir.resolve()}:{container_out}",
    ]
    for k, v in env_vars.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [
        image,
        "--config", "/app/run_config.json",
        "--seed", str(seed),
        "--output-dir", container_out,
    ]
    if fresh:
        cmd.append("--fresh")
    if tracing_url:
        cmd += ["--tracing-url", tracing_url]

    print(f"[START] {name}")
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    minutes, seconds = divmod(elapsed, 60)
    time_str = f"{int(minutes)}m{seconds:05.2f}s"
    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
        print(f"[FAIL]  {name} ({time_str}): {stderr_tail}", file=sys.stderr)
    else:
        print(f"[DONE]  {name} ({time_str})")
    return name, result.returncode, str(out_dir)


def _aggregate_results(output_dirs: list[str]) -> dict:
    """Collect run_result.json from each output dir and aggregate by policy."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for d in output_dirs:
        p = Path(d) / "run_result.json"
        if p.exists():
            r = json.loads(p.read_text())
            grouped[r["strategy"]].append(r)

    agg = {}
    for strategy, runs in grouped.items():
        net = [r["net_worth"] for r in runs]
        sold = [r["units_sold"] for r in runs]
        active = [r["active_sessions"] for r in runs]
        agg[strategy] = {
            "mean_net_worth": round(statistics.fmean(net), 2),
            "min_net_worth": round(min(net), 2),
            "mean_units_sold": round(statistics.fmean(sold), 2),
            "mean_active_sessions": round(statistics.fmean(active), 2),
            "runs": len(runs),
        }
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Scroll benchmark agents in parallel via Docker."
    )
    parser.add_argument(
        "--configs", nargs="+", required=True,
        help="Config JSON files (one per agent type).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1],
        help="Seeds to run for each config (default: [1]).",
    )
    parser.add_argument(
        "--image", default="Scroll:latest",
        help="Docker image name (default: Scroll:latest).",
    )
    parser.add_argument(
        "--output", default="output",
        help="Root output directory (default: output/).",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=4,
        help="Max containers to run simultaneously (default: 4).",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Pass --fresh to each container (ignore checkpoints).",
    )
    parser.add_argument(
        "--tracing-url",
        help="OTLP tracing endpoint URL (e.g. http://host.docker.internal:6006/v1/traces). "
             "Passed to each container via --tracing-url.",
    )
    args = parser.parse_args()

    # Load .env file if present (so API keys don't need manual export)
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

    # Collect API keys from host environment
    env_vars = {}
    for key in ["US_DASHSCOPE_API_KEY", "US_DASHSCOPE_BASE_URL"]:
        val = os.environ.get(key)
        if val:
            env_vars[key] = val

    # Build list of (config_path, seed) runs
    runs = []
    for config_path in args.configs:
        for seed in args.seeds:
            runs.append((config_path, seed))

    print(f"Launching {len(runs)} runs with max {args.max_parallel} parallel\n")

    output_dirs: list[str] = []
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {
            pool.submit(
                _run_container, cfg_path, seed,
                args.image, args.output, env_vars, args.fresh,
                args.tracing_url,
            ): (cfg_path, seed)
            for cfg_path, seed in runs
        }

        for future in as_completed(futures):
            name, returncode, out_dir = future.result()
            output_dirs.append(out_dir)
            if returncode != 0:
                failed.append(name)

    # Aggregate and display results
    print()
    agg = _aggregate_results(output_dirs)
    if agg:
        print("=== Aggregate Results ===")
        for strategy, vals in agg.items():
            print(
                f"  {strategy:>16} | mean_net_worth={vals['mean_net_worth']:8.2f} "
                f"| min_net_worth={vals['min_net_worth']:8.2f} "
                f"| runs={vals['runs']}"
            )

        agg_path = Path(args.output) / "aggregate.json"
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        agg_path.write_text(json.dumps(agg, indent=2))
        print(f"\nResults written to {agg_path}")

    if failed:
        print(f"\n{len(failed)} run(s) failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
