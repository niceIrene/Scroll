"""SCROLL benchmark applications.

Each subpackage here is a benchmark env that plugs into the SCROLL
framework. The :func:`Scroll.core.get_env` registry resolves env IDs
to module names under this package — e.g. ``get_env("longmemeval")``
loads ``Scroll.benchmarks.longmemeval``.

A benchmark package must expose:
  - ``ENV_ID``, ``ENV_CLS``, ``DATASOURCE_CLS``
  - ``parse_env_config(raw_simulation_dict)``
  - ``create_agent(policy, cfg, log, data)``
  - a ``tasks`` submodule with ``PROBES`` and
    ``get_probes_for_turn(turn_idx)``

See :class:`Scroll.core.BaseEnvironment` for the runtime contract.
"""
