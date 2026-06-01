"""Checkpoint helpers: RNG serialization, save/load from disk."""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


# Keep this many most-recent turn checkpoints on disk; older directories
# are pruned after each save. Resume only reads ``latest`` (a symlink),
# so this is purely about retaining a small debug window without
# letting the tree grow with ``num_turns``.
_KEEP_RECENT_CHECKPOINTS = 3

# Matches both the new ``turn_NNN`` directories and the legacy
# ``session_NNN`` ones written before this cleanup. Rotation prunes
# either flavor so old trees don't linger.
_TURN_DIR_RE = re.compile(r"^(?:turn|session)_(\d{3,})$")


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

    Each turn writes its own ``turn_NNN/`` directory; ``latest`` is a
    symlink to the most recent one (the only entry ``load_checkpoint``
    reads). Directories older than the last
    ``_KEEP_RECENT_CHECKPOINTS`` turns are pruned so the tree stays
    bounded in ``num_turns``. Legacy ``session_NNN/`` dirs from older
    runs are pruned by the same pass.
    """
    base = Path(output_dir) / "checkpoints"
    turn_dir = base / f"turn_{turn_idx:03d}"
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
    latest.symlink_to(turn_dir.name)

    _rotate_turn_dirs(base, keep=_KEEP_RECENT_CHECKPOINTS)


def _rotate_turn_dirs(base: Path, *, keep: int) -> None:
    """Delete all but the ``keep`` highest-numbered turn directories."""
    indexed: list[tuple[int, Path]] = []
    for child in base.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        m = _TURN_DIR_RE.match(child.name)
        if m:
            indexed.append((int(m.group(1)), child))
    indexed.sort(key=lambda x: x[0], reverse=True)
    for _, path in indexed[keep:]:
        shutil.rmtree(path, ignore_errors=True)


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

    # Validate checkpoint matches current run. ``strategy`` is the
    # legacy field name for what is now ``policy``; older on-disk
    # checkpoints still use it.
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
