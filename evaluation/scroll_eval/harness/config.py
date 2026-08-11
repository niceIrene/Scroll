from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    endpoint: str
    name: str
    api_key_env: str
    # Override the LLM's reasoning/thinking mode. None = leave the provider
    # default (DashScope ships thinking OFF). True/False explicitly sets the
    # `enable_thinking` flag we send. Only honored on DashScope models.
    thinking: bool | None = None
    # Cap the reasoning tokens when thinking is on (DashScope `thinking_budget`).
    # None = no explicit cap (model default). Must be a positive int. Only takes
    # effect alongside thinking on a DashScope model.
    thinking_budget: int | None = None


@dataclass(frozen=True)
class BudgetSpec:
    max_tokens: int = 50_000
    wall_time_s: int = 600


@dataclass(frozen=True)
class MemorySpec:
    """Durable cross-session conversation store + in-context window knob.

    ``db_path`` is the file-backed SQLite holding ``conversation_history``,
    shared across runs/sessions so an agent can retrieve past sessions.
    ``history_max_tokens`` bounds the in-context ``history`` window an agent
    sends to the LLM — dial it down to stress small-context scenarios. The
    large default effectively disables eviction.
    """

    db_path: str = "~/.scroll/history.db"
    history_max_tokens: int = 1_000_000


@dataclass(frozen=True)
class VerifierSpec:
    """Raise the verifier (test) timeout for slow graders.

    Harbor computes the verifier timeout as ``min(base * multiplier, max)``,
    where ``base`` is each task's own ``verifier.timeout_sec``. Set
    ``timeout_multiplier`` to scale every task proportionally, or
    ``timeout_sec`` to override the base directly. Both default to None
    (leave Harbor's defaults untouched). Useful when the verifier itself is
    slow — e.g. it installs heavy deps under emulation.
    """

    timeout_multiplier: float | None = None
    timeout_sec: float | None = None


@dataclass(frozen=True)
class TraceSpec:
    run_id: str = "auto"
    phoenix_project: str = "default"


@dataclass(frozen=True)
class AgentSpec:
    type: str
    id: str


SandboxType = Literal["docker", "e2b"]


@dataclass(frozen=True)
class SandboxSpec:
    """Where tasks run and how many run at once.

    ``type`` is forwarded to Harbor's ``--env`` flag: ``docker`` (local, the
    default) or ``e2b`` (e2b.dev cloud sandboxes, which need ``E2B_API_KEY``).
    ``parallelism`` is the number of tasks (each its own ``harbor run``) executed
    concurrently; 1 is the original serial behaviour.
    """

    type: SandboxType = "docker"
    parallelism: int = 1


Tasks = list[str] | Literal["all"]
DatasetType = Literal["local", "harbor"]


@dataclass(frozen=True)
class DatasetSpec:
    """Task source for a run.

    ``type='local'`` preserves the original ProjectX behavior:
    ``local-tasks/<name>/<task>/``.

    ``type='harbor'`` uses Harbor's package/registry dataset resolution via
    ``harbor run --dataset <name>@<version>``. A pinned, non-latest version is
    required so reruns have a stable requested input; Harbor's lock.json records
    the resolved per-task sha256 digests after the run.
    """

    name: str = ""
    type: DatasetType = "local"
    version: str | None = None
    registry_url: str | None = None
    registry_path: str | None = None
    n_tasks: int | None = None


@dataclass(frozen=True)
class RunConfig:
    agent: AgentSpec
    model: ModelSpec
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    tasks: Tasks = field(default_factory=list)
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    memory: MemorySpec = field(default_factory=MemorySpec)
    verifier: VerifierSpec = field(default_factory=VerifierSpec)
    trace: TraceSpec = field(default_factory=TraceSpec)
    sandbox: SandboxSpec = field(default_factory=SandboxSpec)
    # Optional per-benchmark tool surface: a list of tool names registered in
    # `scroll_eval._tools_common.TOOLS`. None means each agent picks its own
    # default (preserving backwards-compat for every legacy config).
    tools: list[str] | None = None


def load(path: Path) -> RunConfig:
    raw_text = path.read_text(encoding="utf-8")
    # Expand ${VAR} placeholders against os.environ before parsing.
    # Unknown vars pass through unchanged (os.path.expandvars behaviour).
    expanded = os.path.expandvars(raw_text)
    raw = yaml.safe_load(expanded)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")
    if "loop" in raw:
        raise ConfigError(
            f"{path}: top-level `loop:` is no longer supported; "
            "use `agent: {type, id}` instead"
        )
    try:
        agent = AgentSpec(**raw["agent"])
        model = ModelSpec(**raw["model"])
        budget = BudgetSpec(**raw.get("budget", {}))
        memory = MemorySpec(**raw.get("memory", {}))
        verifier = VerifierSpec(**raw.get("verifier", {}))
        trace = TraceSpec(**raw.get("trace", {}))
        sandbox = SandboxSpec(**raw.get("sandbox", {}))
    except KeyError as exc:
        raise ConfigError(f"{path}: missing required field {exc.args[0]}") from exc
    except TypeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc

    if sandbox.type not in ("docker", "e2b"):
        raise ConfigError(
            f"{path}: sandbox.type must be 'docker' or 'e2b', got {sandbox.type!r}"
        )
    if not isinstance(sandbox.parallelism, int) or sandbox.parallelism < 1:
        raise ConfigError(
            f"{path}: sandbox.parallelism must be an integer >= 1, "
            f"got {sandbox.parallelism!r}"
        )

    if not isinstance(memory.history_max_tokens, int) or memory.history_max_tokens < 1:
        raise ConfigError(
            f"{path}: memory.history_max_tokens must be an integer >= 1, "
            f"got {memory.history_max_tokens!r}"
        )

    if verifier.timeout_multiplier is not None and (
        not isinstance(verifier.timeout_multiplier, (int, float))
        or verifier.timeout_multiplier <= 0
    ):
        raise ConfigError(
            f"{path}: verifier.timeout_multiplier must be a positive number, "
            f"got {verifier.timeout_multiplier!r}"
        )
    if verifier.timeout_sec is not None and (
        not isinstance(verifier.timeout_sec, (int, float)) or verifier.timeout_sec <= 0
    ):
        raise ConfigError(
            f"{path}: verifier.timeout_sec must be a positive number, "
            f"got {verifier.timeout_sec!r}"
        )

    dataset = _parse_dataset(path, raw.get("dataset", ""))
    tools = _parse_tools(path, raw.get("tools"))

    return RunConfig(
        agent=agent,
        model=model,
        dataset=dataset,
        tasks=raw.get("tasks", []),
        budget=budget,
        memory=memory,
        verifier=verifier,
        trace=trace,
        sandbox=sandbox,
        tools=tools,
    )


def _parse_tools(path: Path, raw_tools: object) -> list[str] | None:
    if raw_tools is None:
        return None
    if not isinstance(raw_tools, list) or not all(
        isinstance(name, str) for name in raw_tools
    ):
        raise ConfigError(
            f"{path}: tools must be a list of strings, got {raw_tools!r}"
        )
    if not raw_tools:
        raise ConfigError(f"{path}: tools list is empty; omit the field to use the default")
    # Validate names against the registry so a typo fails at config-load time,
    # not deep inside the agent's first tool call.
    from scroll_eval._tools_common import TOOLS  # local import avoids cycles
    unknown = [name for name in raw_tools if name not in TOOLS]
    if unknown:
        known = sorted(TOOLS.keys())
        raise ConfigError(
            f"{path}: tools references unknown name(s) {unknown!r}; "
            f"registered tools are {known!r}"
        )
    return list(raw_tools)


def _parse_dataset(path: Path, raw_dataset: object) -> DatasetSpec:
    if isinstance(raw_dataset, str):
        return DatasetSpec(name=raw_dataset, type="local")
    if not isinstance(raw_dataset, dict):
        raise ConfigError(
            f"{path}: dataset must be a string or mapping, got {type(raw_dataset).__name__}"
        )
    try:
        dataset = DatasetSpec(**raw_dataset)
    except TypeError as exc:
        raise ConfigError(f"{path}: dataset: {exc}") from exc

    if dataset.type not in ("local", "harbor"):
        raise ConfigError(
            f"{path}: dataset.type must be 'local' or 'harbor', got {dataset.type!r}"
        )
    if not dataset.name:
        raise ConfigError(f"{path}: dataset.name is required")
    if dataset.type == "local":
        invalid = {
            "version": dataset.version,
            "registry_url": dataset.registry_url,
            "registry_path": dataset.registry_path,
            "n_tasks": dataset.n_tasks,
        }
        present = [key for key, value in invalid.items() if value is not None]
        if present:
            raise ConfigError(
                f"{path}: local datasets do not support fields: {', '.join(present)}"
            )
    else:
        if not dataset.version or dataset.version == "latest":
            raise ConfigError(
                f"{path}: harbor datasets require a pinned non-'latest' version"
            )
        if "${" in dataset.version:
            raise ConfigError(
                f"{path}: dataset.version contains an unresolved ${{...}} placeholder"
            )
        if dataset.n_tasks is not None and dataset.n_tasks < 1:
            raise ConfigError(f"{path}: dataset.n_tasks must be >= 1")
    return dataset


def with_tasks(cfg: RunConfig, tasks: list[str]) -> RunConfig:
    return replace(cfg, tasks=tasks)


def with_all_tasks(cfg: RunConfig) -> RunConfig:
    return replace(cfg, tasks="all")
