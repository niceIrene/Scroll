"""Configuration loading and ablation expansion for benchmark runs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from Scroll.core._models import AgentConfig, BaseEnvConfig
from Scroll.core._registry import get_env


def load_config(path: Path) -> dict:
    """Load raw JSON config from file."""
    return json.loads(path.read_text())


def parse_env_config(raw: dict, env_id: str | None = None) -> BaseEnvConfig:
    """Parse environment config via the env package's ``parse_env_config``.

    The ``environment`` key in the raw config selects which env's parser
    to use; defaults to ``vending`` for backwards compatibility with
    pre-multi-env configs.
    """
    env_id = env_id or raw.get("environment", "vending")
    return get_env(env_id).parse_env_config(raw["simulation"])


def parse_agent_config(raw: dict) -> AgentConfig:
    """Parse agent config from raw dict."""
    return AgentConfig.from_dict(raw["agent"])


def apply_overrides(base_config: dict, overrides: dict) -> dict:
    """Apply dotted-path overrides onto a base config dict.

    Example:
        apply_overrides(config, {"agent.memory_strategy": "sliding_window"})
        -> config["agent"]["memory_strategy"] = "sliding_window"
    """
    config = deepcopy(base_config)
    for key, value in overrides.items():
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return config


def expand_ablations(raw_config: dict) -> list[tuple[str, dict]]:
    """Expand a config with ablations into (label, config) pairs.

    If no ablations are defined, returns a single ("baseline", config) pair.
    """
    bench = raw_config.get("benchmark", {})
    ablations = bench.get("ablations", None)

    if not ablations:
        return [("baseline", raw_config)]

    result = []
    for ablation in ablations:
        label = ablation.get("label", "unnamed")
        overrides = ablation.get("overrides", {})
        variant = apply_overrides(raw_config, overrides)
        result.append((label, variant))

    return result
