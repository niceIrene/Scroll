"""Environment registry — hand-written dispatch table.

Each benchmark lives at ``Scroll.benchmarks.<env_id>`` and must
export from its package ``__init__.py``:

    ENV_CLS:             BaseEnvironment subclass
    DATASOURCE_CLS:      BaseDataSource subclass
    create_agent(policy, cfg, log, data) -> BaseAgent
    parse_env_config(raw_simulation_dict) -> BaseEnvConfig

Plus, in the env's ``tasks`` submodule:

    PROBES:                                 list[ProbeSpec]
    get_probes_for_turn(turn_idx)           Callable
    compute_efficiency_metrics(action_logs) Callable

And optionally, in the env's ``tools`` submodule:

    register_env_tools(toolkit, state)      Callable

To add a new env: drop a subpackage under ``Scroll.benchmarks/`` and
add one row to :data:`_ENV_PACKAGES` below. Third-party envs that
live outside the ``benchmarks/`` tree can use :func:`register_env`
to install themselves directly.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, NamedTuple


class EnvEntry(NamedTuple):
    env_cls: type
    datasource_cls: type
    register_tools: Callable
    probes: list
    get_probes_for_turn: Callable
    compute_efficiency_metrics: Callable
    create_agent: Callable
    parse_env_config: Callable


# Hand-written env id → import path. Add a row to register a new env.
_ENV_PACKAGES: dict[str, str] = {
    "vending":     "Scroll.benchmarks.vending",
    "longmemeval": "Scroll.benchmarks.longmemeval",
    "beam":        "Scroll.benchmarks.beam",
}


_REGISTRY: dict[str, EnvEntry] = {}


def register_env(env_id: str, entry: EnvEntry) -> None:
    """Explicitly install an env in the registry.

    Use for third-party envs that don't live under
    :mod:`Scroll.benchmarks`. Built-in envs are dispatched via
    :data:`_ENV_PACKAGES` and don't need this.
    """
    _REGISTRY[env_id] = entry


def get_env(env_id: str) -> EnvEntry:
    """Retrieve an env's :class:`EnvEntry`, lazy-importing on first access."""
    if env_id in _REGISTRY:
        return _REGISTRY[env_id]
    if env_id not in _ENV_PACKAGES:
        known = sorted(set(_ENV_PACKAGES) | set(_REGISTRY))
        raise KeyError(
            f"unknown env_id {env_id!r}; known: {known}. "
            f"For a third-party env, call register_env() first."
        )
    pkg = _ENV_PACKAGES[env_id]
    mod = importlib.import_module(pkg)
    tasks = importlib.import_module(f"{pkg}.tasks")

    # Tools module is optional — envs without action tools (LME, BEAM)
    # don't ship one.
    try:
        tools_mod = importlib.import_module(f"{pkg}.tools")
        reg_tools = getattr(tools_mod, "register_env_tools", _noop_register)
    except ImportError:
        reg_tools = _noop_register

    register_env(env_id, EnvEntry(
        env_cls=mod.ENV_CLS,
        datasource_cls=mod.DATASOURCE_CLS,
        register_tools=reg_tools,
        probes=getattr(tasks, "PROBES", []),
        get_probes_for_turn=getattr(
            tasks, "get_probes_for_turn", lambda turn_idx: [],
        ),
        compute_efficiency_metrics=getattr(
            tasks, "compute_efficiency_metrics", lambda x: {},
        ),
        create_agent=mod.create_agent,
        parse_env_config=mod.parse_env_config,
    ))
    return _REGISTRY[env_id]


def _noop_register(toolkit: Any, state: Any) -> None:
    """Fallback when an env has no ``tools.register_env_tools``."""
    pass
