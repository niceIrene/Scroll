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

    @classmethod
    def from_dict(cls, d: dict) -> "BeamEnvConfig":
        """Construct from a raw config dict.

        Accepts legacy ``"days"`` / ``"num_sessions"`` keys as synonyms
        for ``"num_turns"`` so older configs continue to parse.
        """
        known = {f.name for f in fields(cls)}
        normalized = dict(d)
        if "num_turns" not in normalized:
            if "num_sessions" in normalized:
                normalized["num_turns"] = normalized.pop("num_sessions")
            elif "days" in normalized:
                normalized["num_turns"] = normalized["days"]
        return cls(**{k: v for k, v in normalized.items() if k in known})
