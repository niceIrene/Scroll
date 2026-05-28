"""Environment registry — discovers benchmark packages by convention.

Each benchmark lives at ``Scroll.benchmarks.<env_id>`` and must expose:
    ENV_ID: str
    ENV_CLS / <Env>: BaseEnvironment subclass
    DATASOURCE_CLS / DataSourceManager: BaseDataSource subclass
    create_agent(policy, cfg, log, data) -> BaseAgent
    parse_env_config(raw_simulation_dict) -> BaseEnvConfig
    register_env_tools(toolkit, state): optional tool registration function
    PROBES (via tasks subpackage): list[ProbeSpec]
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


_REGISTRY: dict[str, EnvEntry] = {}


def register_env(env_id: str, entry: EnvEntry) -> None:
    """Explicitly register an environment."""
    _REGISTRY[env_id] = entry


def get_env(env_id: str) -> EnvEntry:
    """Retrieve an environment entry, auto-discovering if needed.

    Benchmarks live under ``Scroll.benchmarks.<env_id>`` after the
    benchmarks-vs-framework split. The old flat ``Scroll.<env_id>``
    path is no longer supported by this registry — if you have a
    third-party env that used the old layout, move it under
    ``benchmarks/`` (or register it explicitly via :func:`register_env`).
    """
    if env_id not in _REGISTRY:
        pkg = f"Scroll.benchmarks.{env_id}"
        mod = importlib.import_module(pkg)
        tasks = importlib.import_module(f"{pkg}.tasks")

        env_cls = getattr(mod, "ENV_CLS", None) or _find_attr(mod, "Env", fallback_attr="VendingEnv")
        ds_cls = getattr(mod, "DATASOURCE_CLS", None) or _find_attr(mod, "DataSourceManager")

        create_agent = getattr(mod, "create_agent", None)
        if create_agent is None:
            raise AttributeError(
                f"Env package {pkg} must export create_agent(policy, cfg, log, data)"
            )

        parse_env_config = getattr(mod, "parse_env_config", None)
        if parse_env_config is None:
            raise AttributeError(
                f"Env package {pkg} must export parse_env_config(raw_simulation_dict)"
            )

        # Tools registration: look for register_env_tools in the env's tools module
        try:
            tools_mod = importlib.import_module(f"{pkg}.tools")
            reg_tools = getattr(tools_mod, "register_env_tools", _noop_register)
        except ImportError:
            reg_tools = _noop_register

        probes_fn = getattr(tasks, "get_probes_for_turn", lambda turn_idx: [])
        register_env(env_id, EnvEntry(
            env_cls=env_cls,
            datasource_cls=ds_cls,
            register_tools=reg_tools,
            probes=getattr(tasks, "PROBES", []),
            get_probes_for_turn=probes_fn,
            compute_efficiency_metrics=getattr(tasks, "compute_efficiency_metrics", lambda x: {}),
            create_agent=create_agent,
            parse_env_config=parse_env_config,
        ))
    return _REGISTRY[env_id]


def _find_attr(mod: Any, *names: str, fallback_attr: str | None = None) -> type:
    """Find first matching attribute by name."""
    for name in names:
        if hasattr(mod, name):
            return getattr(mod, name)
    if fallback_attr and hasattr(mod, fallback_attr):
        return getattr(mod, fallback_attr)
    raise AttributeError(f"Module {mod.__name__} has none of: {names}")


def _noop_register(toolkit: Any, state: Any) -> None:
    """Fallback when no env-specific tools exist."""
    pass
