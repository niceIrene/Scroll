"""BEAM env config.

A BEAM run answers all probing questions for ONE chat at the configured
scale. ``num_turns`` is set by the env to ``num_batches + 1`` once
the chat is loaded; the probe turn fires at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class BeamEnvConfig:
    # Upper bound on the turn count; the env overwrites this with the
    # loaded chat's actual batch count + 1 probe turn.
    num_turns: int = 50

    # Root of the BEAM repo's chats directory. Submoduled at external/beam/.
    dataset_root: str = "external/beam/chats"

    # Token-budget scale (one of "100K" / "500K" / "1M" / "10M").
    # NOTE: BEAM's README documents "128K" but the on-disk directory is
    # named "100K"; the smallest scale here is the on-disk path.
    scale: str = "100K"

    # Which chat to run. BEAM chats are numbered directories under
    # chats/<scale>/. ``chat_id`` matches one of those directory names
    # (cast to str). Required at construction time — there is no
    # "first chat" default (forces explicit selection).
    chat_id: str | None = None

    # LLM judge config — used by tasks/probes.py to score each rubric
    # item against the agent's answer.
    judge_model: str = "gpt-4o-2024-08-06"
    judge_api_key_env: str = "OPENAI_API_KEY"
    judge_api_base: str | None = None

    # SCROLL-pure vs. legacy ingestion path (mirrors
    # LongMemEvalEnvConfig). When ``False`` (default, SCROLL-pure):
    # the env bulk-loads every batch into ``E`` at task start via
    # :meth:`ingest_all`, ``cfg.num_turns`` is forced to ``0``, and
    # all M probing questions fire end-of-task via
    # :meth:`get_end_of_task_probes`. When ``True`` (legacy):
    # ``cfg.num_turns = num_batches + 1``, the agent's per-turn
    # ``run_turn`` mirrors one batch into ``E`` per iteration, and
    # probes fire on the ``+1`` turn via the per-turn registry.
    agent_during_ingestion: bool = False

    # How the M end-of-task probes are isolated from each other.
    # ``"shared"`` (default): all probes share one agent session —
    # probe N sees probes 1..N-1's exchange in ``_history``. Cheap.
    # ``"fresh"``: each probe (except the first) gets a fresh agent
    # session — answers come purely from ``W``. See
    # :meth:`BaseEnvironment.probe_isolation` for the full contract.
    probe_isolation: str = "shared"

    @classmethod
    def from_dict(cls, d: dict) -> "BeamEnvConfig":
        """Construct from a raw config dict."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
