"""Checkpoint helpers: RNG serialization, save/load from disk."""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def serialize_rng(rng: random.Random) -> list:
    """Serialize a random.Random state to a JSON-safe list."""
    state = rng.getstate()
    # state is (version, internalstate_tuple, gauss_next)
    # internalstate_tuple contains ints; gauss_next is a float or None
    return [state[0], list(state[1]), state[2]]


def restore_rng(rng: random.Random, data: list) -> None:
    """Restore a random.Random state from a JSON-safe list."""
    rng.setstate((data[0], tuple(data[1]), data[2]))


def config_hash(config: dict) -> str:
    """Deterministic SHA256 of a config dict for checkpoint validation."""
    raw = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def save_checkpoint(
    output_dir: str | Path,
    turn_idx: int,
    env,
    data,
    agent,
    log,
    loop_state: dict,
    policy: str,
    seed: int,
    cfg_hash: str,
) -> None:
    """Save a checkpoint after a completed turn.

    On-disk layout keeps the legacy ``session_NNN`` directory names
    so existing checkpoint trees stay readable across the PR #2
    rename. ``meta`` records both ``turn_idx`` (new) and
    ``session_idx`` (back-compat alias).
    """
    base = Path(output_dir) / "checkpoints"
    turn_dir = base / f"session_{turn_idx:03d}"
    turn_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "env": env.to_checkpoint(),
        "data": data.to_checkpoint(),
        "agent": agent.to_checkpoint(),
        "log": log.to_checkpoint(),  # {"entry_count": N} pointer into the jsonl
        "loop_state": loop_state,
    }
    (turn_dir / "checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2, default=str),
        encoding="utf-8",
    )

    meta = {
        "version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "turn_idx": turn_idx,
        "session_idx": turn_idx,  # legacy alias; drop in PR #6
        "policy": policy,
        "seed": seed,
        "config_hash": cfg_hash,
    }
    (turn_dir / "_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    # Update latest symlink
    latest = base / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(f"session_{turn_idx:03d}")


def load_checkpoint(
    output_dir: str | Path,
    policy: str,
    seed: int,
    cfg_hash: str,
) -> dict | None:
    """Load the latest checkpoint if valid. Returns None if no valid checkpoint."""
    latest = Path(output_dir) / "checkpoints" / "latest"
    if not latest.exists():
        return None

    resolved = latest.resolve()
    meta_path = resolved / "_meta.json"
    ckpt_path = resolved / "checkpoint.json"

    if not meta_path.exists() or not ckpt_path.exists():
        return None

    meta = json.loads(meta_path.read_text())

    # Validate checkpoint matches current run. Accept legacy "strategy"
    # key for backward-compatibility with checkpoints written before the
    # strategies-removal refactor.
    ckpt_policy = meta.get("policy", meta.get("strategy"))
    if ckpt_policy != policy:
        return None
    if meta.get("seed") != seed:
        return None
    if meta.get("config_hash") != cfg_hash:
        return None

    checkpoint = json.loads(ckpt_path.read_text())
    checkpoint["_meta"] = meta
    return checkpoint
