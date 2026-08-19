"""Shared Harbor BaseAgent for all scroll-eval agent families.

Registered with Harbor via:
    harbor run --agent-import-path scroll_eval.runner:ScrollEvalAgent ...

Reads (AgentType, AgentID) from env vars (SCROLL_AGENT_TYPE / SCROLL_AGENT_ID),
imports `scroll_eval.<type>.<id>.agent`, and awaits its `run(task, ctx)`.

The (type, id) values are passed as env vars by `scroll_eval/harness/runner.py`
before invoking the harbor CLI; see configs/*.yaml for the YAML schema.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from openai import OpenAI

from scroll_eval.harness.config import BudgetSpec
from scroll_eval.tracing import otel
from scroll_eval.types import LoopContext, TaskSpec, Trajectory


_DEFAULT_FAMILY = "base_agents"
_DEFAULT_ID = "scroll_react"


def _budget_from_env() -> BudgetSpec | None:
    """Reconstruct the configured budget from env vars set by harness/runner.py.

    Returns None when neither limit is present (e.g. a bare harbor invocation
    that bypasses our outer runner), letting the agent fall back to defaults.
    """
    max_tokens = os.environ.get("SCROLL_MAX_TOKENS")
    wall_time_s = os.environ.get("SCROLL_WALL_TIME_S")
    if max_tokens is None and wall_time_s is None:
        return None
    kwargs: dict[str, int] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    if wall_time_s is not None:
        kwargs["wall_time_s"] = int(wall_time_s)
    return BudgetSpec(**kwargs)


def _tools_from_env() -> list[dict] | None:
    """Resolve SCROLL_TOOLS (csv of tool names) into OpenAI tool schemas.

    Returns None when the env var is absent, letting each agent fall back to
    its own default surface. Unknown names raise via select_tools — fail fast
    is better than silently dropping a tool the user expected to be there.
    """
    raw = os.environ.get("SCROLL_TOOLS")
    if not raw:
        return None
    from scroll_eval._tools_common import select_tools  # local import to avoid cycles
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        return None
    return select_tools(names)


class ScrollEvalAgent(BaseAgent):
    """The only Harbor BaseAgent scroll-eval ships.

    Dispatches by (SCROLL_AGENT_TYPE, SCROLL_AGENT_ID) to a self-contained
    agent module that exposes `async def run(task, ctx) -> Trajectory`.
    """

    SUPPORTS_ATIF: bool = True
    SUPPORTS_WINDOWS: bool = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        mcp_servers: Any = None,
        skills_dir: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            logs_dir=logs_dir, model_name=model_name, logger=logger,
            mcp_servers=mcp_servers, skills_dir=skills_dir, *args, **kwargs,
        )
        self.logs_dir = Path(logs_dir)
        self.model_name = model_name or os.environ.get("SCROLL_MODEL", "")
        self.logger = logger or logging.getLogger("scroll_eval.runner")

    @staticmethod
    def name() -> str:
        return "scroll-eval"

    def version(self) -> str:
        return "0.2.0"

    async def setup(self, environment: Any) -> None:
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        family = os.environ.get("SCROLL_AGENT_TYPE", _DEFAULT_FAMILY)
        agent_id = os.environ.get("SCROLL_AGENT_ID", _DEFAULT_ID)
        agent_mod = importlib.import_module(f"scroll_eval.{family}.{agent_id}.agent")
        if not hasattr(agent_mod, "run"):
            raise AttributeError(
                f"scroll_eval.{family}.{agent_id}.agent must export `async def run(task, ctx)`"
            )

        task = TaskSpec(
            task_id=os.environ.get("SCROLL_TASK_ID") or getattr(environment, "task_id", "unknown"),
            instruction=instruction,
        )
        ctx = self._build_loop_context(environment, family, agent_id)
        with otel.task_run(
            ctx.tracer,
            task_id=task.task_id,
            agent=f"{family}/{agent_id}",
            model=self.model_name,
            run_id="auto",
        ):
            trajectory: Trajectory = await agent_mod.run(task, ctx)

        context.n_input_tokens = int(trajectory.metrics.get("tokens_in", 0))
        context.n_output_tokens = int(trajectory.metrics.get("tokens_out", 0))
        context.cost_usd = 0.0
        context.metadata = {
            "family": family,
            "agent_id": agent_id,
            "terminated": trajectory.terminated.value,
            "step_count": len(trajectory.steps),
        }

        out = self.logs_dir / "trajectory.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_trajectory_json(trajectory), encoding="utf-8")

    def _build_loop_context(self, environment: Any, family: str, agent_id: str) -> LoopContext:
        bare_model = (
            self.model_name.split("/", 1)[-1]
            if self.model_name and "/" in self.model_name
            else self.model_name
        )
        llm_openai = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or "",
            base_url=os.environ.get("OPENAI_BASE_URL"),
            max_retries=int(os.environ.get("SCROLL_LLM_MAX_RETRIES", "6")),
            timeout=float(os.environ.get("SCROLL_LLM_TIMEOUT_S", "120")),
        )
        llm_agentscope = _build_agentscope_model(bare_model)
        max_tokens = os.environ.get("SCROLL_HISTORY_MAX_TOKENS")
        return LoopContext(
            llm_openai=llm_openai,
            llm_agentscope=llm_agentscope,
            model_name=bare_model,
            tracer=otel.init_for_phoenix(phoenix_project=os.environ.get("SCROLL_PHOENIX_PROJECT") or None),
            budget=_budget_from_env(),
            environment=environment,
            tools=_tools_from_env(),
            run_id=os.environ.get("SCROLL_RUN_ID"),
            history_db_path=os.environ.get("SCROLL_MEMORY_DB"),
            history_max_tokens=int(max_tokens) if max_tokens else None,
            logs_dir=str(self.logs_dir),
        )


def _build_agentscope_model(
    bare_model: str,
    thinking: bool | None = None,
    thinking_budget: int | None = None,
) -> Any:
    """Construct an AgentScope ChatModelBase for the configured endpoint.

    Selects DashScopeChatModel when OPENAI_BASE_URL points at DashScope,
    otherwise OpenAIChatModel. Returns None if AgentScope isn't installed.

    ``thinking`` overrides reasoning/thinking mode (DashScope only): None leaves
    the provider default (off), True/False sets the ``enable_thinking`` flag we
    send. ``thinking_budget`` caps the reasoning tokens (DashScope
    ``thinking_budget``). When the caller passes None we fall back to the
    ``SCROLL_ENABLE_THINKING`` / ``SCROLL_THINKING_BUDGET`` env vars so the
    out-of-process (Harbor subprocess) path can configure them too.
    """
    try:
        from agentscope.credential import DashScopeCredential, OpenAICredential  # type: ignore
        from agentscope.model import DashScopeChatModel, OpenAIChatModel  # type: ignore
    except ImportError:
        return None

    if thinking is None:
        env_val = os.environ.get("SCROLL_ENABLE_THINKING")
        if env_val is not None:
            thinking = env_val.strip().lower() in ("1", "true", "yes", "on")
    if thinking_budget is None:
        env_budget = os.environ.get("SCROLL_THINKING_BUDGET")
        if env_budget:
            try:
                thinking_budget = int(env_budget)
            except ValueError:
                thinking_budget = None

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
    base_url = os.environ.get("OPENAI_BASE_URL") or ""

    # Backstop against indefinitely stalled requests (openai.AsyncClient
    # timeout: total wait for non-streamed calls, max inter-chunk gap for
    # streamed ones). Default is deliberately generous — 600s, the per-probe
    # wall budget scale, NOT the 120s default the raw llm_openai client uses —
    # because large single calls (e.g. 50k-token summarization chunks) can
    # legitimately run for minutes; the point is to kill hangs, not police
    # latency. SCROLL_LLM_TIMEOUT_S overrides both clients when set.
    timeout_s = float(os.environ.get("SCROLL_LLM_TIMEOUT_S") or 600)
    client_kwargs = {"timeout": timeout_s}

    if "dashscope" in base_url.lower():
        cred = DashScopeCredential(api_key=api_key, base_url=base_url)
        params = None
        if thinking is not None or thinking_budget is not None:
            param_kwargs: dict[str, Any] = {}
            if thinking is not None:
                param_kwargs["thinking_enable"] = thinking
            if thinking_budget is not None:
                param_kwargs["thinking_budget"] = thinking_budget
            params = DashScopeChatModel.Parameters(**param_kwargs)
        # DashScope streams reasoning deltas, so thinking-on requires stream=True;
        # the agent loop collapses that stream back to one ChatResponse.
        return DashScopeChatModel(
            credential=cred, model=bare_model, parameters=params, stream=bool(thinking),
            client_kwargs=client_kwargs,
        )
    cred = OpenAICredential(api_key=api_key, base_url=base_url)
    return OpenAIChatModel(
        credential=cred, model=bare_model, stream=False, client_kwargs=client_kwargs
    )


def _trajectory_json(trajectory: Trajectory) -> str:
    return json.dumps(
        {
            "task_id": trajectory.task_id,
            "final_answer": trajectory.final_answer,
            "terminated": trajectory.terminated.value,
            "metrics": trajectory.metrics,
            "steps": [asdict(s) for s in trajectory.steps],
        },
        indent=2,
    )
