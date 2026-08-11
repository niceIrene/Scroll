"""Standalone EdgeBench entrypoint for scroll_agent_A.

EdgeBench (github.com/ByteDance-Seed/EdgeBench) launches agents as CLIs inside
its work container: the harness writes the task prompt to /tmp/agent_prompt.md
and executes the agent's ``run_cmd`` via ``/bin/bash -c`` in the task
workspace. This module adapts scroll_agent_A's in-process ``run(task, ctx)``
loop to that contract: prompt file in, agent loop against the local workspace,
trajectory artifacts out. Scoring is workspace-file based on the EdgeBench
side (the agent runs ``sforge-submit`` via its ``bash`` tool, per the task
prompt), so stdout here is purely for the run log.

Deliberately NOT imported: ``scroll_eval.runner`` (Harbor), ``scroll_eval.evals.beam``
(seed DBs, shared_run_ids tiers, BEAM prompt framing), Phoenix/OTLP init. The
loop's beam-only features stay inert: ``SCROLL_SEED_INDEX`` unset,
``shared_run_ids=()``, ``system_prompt=None``.

Env contract (the EdgeBench ``ScrollAgentA`` subclass sets these):
  OPENAI_API_KEY / OPENAI_BASE_URL  OpenAI-compatible endpoint; a DashScope
                                    base URL selects the DashScope client,
                                    mirroring scroll_eval.runner.
  SCROLL_MODEL                    bare model name (required).
  SCROLL_WALL_TIME_S              per-launch wall budget, seconds (optional).
  SCROLL_MAX_TOKENS               per-launch token budget (optional).
  SCROLL_HISTORY_MAX_TOKENS       in-context window bound (optional).
  SCROLL_MAX_STEPS                  loop step cap (read by agent.py).
  SCROLL_TASK_ID                    task id override (default: cwd basename).
  SCROLL_AGENT_HOME                 state root (default /tmp/scroll_agent) —
                                    kept OUT of the workspace so agent
                                    bookkeeping never lands in submit_paths.

On ``--resume`` the loop restarts with the same run_id (hence the same
durable session in ``hist.conversation_history``), so the resumed session can
recall everything the previous one did via ``ms.search`` — EdgeBench's
auto-resume keeps relaunching until the task timeout.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from opentelemetry import trace

from scroll_eval.base_agents.scroll_agent_A import agent as scroll_agent
from scroll_eval.types import TaskSpec, Trajectory


_RUN_ID = "edgebench"

_RESUME_PREAMBLE = (
    "You are RESUMING work on the task below. A previous session already "
    "worked on it in this same workspace: the workspace files reflect all "
    "work so far, and the full record of the previous session(s) is durable "
    "in hist.conversation_history — recall it with ms.search(...) / "
    "ms.expand([...]) inside execute_python. Reorient first (inspect the "
    "workspace, recall what was done, what worked, and what remained), then "
    "continue improving the solution. Do not redo finished work.\n"
    "\n---\n\n"
)


@dataclass
class Budget:
    """Duck-typed stand-in for scroll_eval.harness.config.BudgetSpec.

    Defined locally so the container install doesn't need pyyaml (which
    scroll_eval.harness.config imports). The agent loop reads both fields via
    getattr, treating None as "unbounded".
    """

    max_tokens: int | None = None
    wall_time_s: int | None = None


@dataclass
class _ExecResult:
    stdout: str
    stderr: str
    return_code: int


class LocalEnvironment:
    """Local-subprocess implementation of the ``bash`` tool's environment.

    Same one-method surface as Harbor's BaseEnvironment as consumed by
    ``scroll_eval._tools_common.run_bash``: ``await exec(command, timeout_sec=…)``
    returning ``.stdout/.stderr/.return_code``, raising a RuntimeError whose
    message contains "timed out" on timeout (run_bash maps that to exit 124).
    Commands run in the process cwd — EdgeBench starts this CLI in the task
    workspace.
    """

    async def exec(self, command: str, timeout_sec: int = 60) -> _ExecResult:
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash", "-lc", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # so timeout kill reaps the whole tree
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except (asyncio.TimeoutError, TimeoutError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            await proc.wait()
            raise RuntimeError(f"command timed out after {timeout_sec}s")
        return _ExecResult(
            stdout=out.decode(errors="replace"),
            stderr=err.decode(errors="replace"),
            return_code=proc.returncode if proc.returncode is not None else -1,
        )


def _build_agentscope_model(bare_model: str) -> Any:
    """Construct the AgentScope chat model for the configured endpoint.

    Trimmed copy of scroll_eval.runner._build_agentscope_model (which cannot be
    imported here: scroll_eval.runner imports harbor at module level, and harbor
    is not installed in the EdgeBench container). Same env contract: DashScope
    client when OPENAI_BASE_URL points at DashScope, OpenAI-compatible client
    otherwise; SCROLL_ENABLE_THINKING / SCROLL_THINKING_BUDGET knobs.
    """
    from agentscope.credential import DashScopeCredential, OpenAICredential
    from agentscope.model import DashScopeChatModel, OpenAIChatModel

    thinking: bool | None = None
    env_val = os.environ.get("SCROLL_ENABLE_THINKING")
    if env_val is not None:
        thinking = env_val.strip().lower() in ("1", "true", "yes", "on")
    thinking_budget: int | None = None
    env_budget = os.environ.get("SCROLL_THINKING_BUDGET")
    if env_budget:
        try:
            thinking_budget = int(env_budget)
        except ValueError:
            thinking_budget = None

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
    base_url = os.environ.get("OPENAI_BASE_URL") or ""

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
        return DashScopeChatModel(
            credential=cred, model=bare_model, parameters=params, stream=bool(thinking)
        )
    cred = OpenAICredential(api_key=api_key, base_url=base_url or None)
    return OpenAIChatModel(credential=cred, model=bare_model, stream=False)


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _next_attempt_dir(state_dir: Path) -> Path:
    """Per-launch logs dir (attempt-01, attempt-02, …) so a resume never
    overwrites the previous launch's call_messages.jsonl / trajectory.json."""
    logs_root = state_dir / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    n = 1 + sum(1 for p in logs_root.iterdir() if p.name.startswith("attempt-"))
    attempt = logs_root / f"attempt-{n:02d}"
    attempt.mkdir(parents=True, exist_ok=True)
    return attempt


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scroll-agent-a",
        description="Run scroll_agent_A against the current workspace (EdgeBench entrypoint).",
    )
    parser.add_argument("prompt_file", nargs="?", default=None, help="Path to the task prompt file")
    parser.add_argument(
        "--ask", metavar="QUESTION", default=None,
        help="One-shot Q&A mode: run the loop on QUESTION in a scratch "
        "workspace and print the final answer to stdout (status goes to "
        "stderr). Repeated asks under the same --task-id share the durable "
        "history, so the agent can recall earlier questions via ms.search.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue a previous session (same durable history session)",
    )
    parser.add_argument("--task-id", default=None, help="Task id override")
    args = parser.parse_args(argv)

    ask_mode = args.ask is not None
    if ask_mode == (args.prompt_file is not None):
        parser.error("provide exactly one of: prompt_file, or --ask QUESTION")
    if ask_mode and args.resume:
        parser.error("--resume only applies to prompt_file mode")

    # In ask mode stdout carries exactly the answer; status lines go to stderr.
    def info(msg: str) -> None:
        print(msg, file=sys.stderr if ask_mode else sys.stdout, flush=True)

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")):
        print("scroll-agent-a: OPENAI_API_KEY (or DASHSCOPE_API_KEY) is not set", file=sys.stderr)
        return 2
    model_name = os.environ.get("SCROLL_MODEL", "").strip()
    if not model_name:
        print("scroll-agent-a: SCROLL_MODEL is not set", file=sys.stderr)
        return 2

    if ask_mode:
        instruction = args.ask
        task_id = args.task_id or os.environ.get("SCROLL_TASK_ID") or "ask"
    else:
        instruction = Path(args.prompt_file).read_text(encoding="utf-8")
        if args.resume:
            instruction = _RESUME_PREAMBLE + instruction
        task_id = (
            args.task_id
            or os.environ.get("SCROLL_TASK_ID")
            or Path.cwd().name
            or "edgebench-task"
        )

    state_dir = Path(os.environ.get("SCROLL_AGENT_HOME") or "/tmp/scroll_agent") / task_id
    state_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = _next_attempt_dir(state_dir)

    if ask_mode:
        # The bash tool runs subprocesses in the process cwd. In EdgeBench that
        # is the task workspace; for ad-hoc questions, keep any file activity in
        # a scratch workspace instead of wherever the user happened to launch.
        workspace = state_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        os.chdir(workspace)

    # Local import: the class only matters for the type; construction is cheap.
    from scroll_eval.types import LoopContext

    ctx = LoopContext(
        llm_openai=None,  # scroll_agent_A uses the agentscope client only
        llm_agentscope=_build_agentscope_model(model_name),
        model_name=model_name,
        # No TracerProvider is registered (we never call init_for_phoenix), so
        # this tracer yields non-recording spans — all otel.* calls are no-ops.
        tracer=trace.get_tracer("scroll_eval.edgebench"),
        budget=Budget(
            max_tokens=_int_env("SCROLL_MAX_TOKENS"),
            wall_time_s=_int_env("SCROLL_WALL_TIME_S"),
        ),
        environment=LocalEnvironment(),
        tools=None,
        # Stable run_id: a resumed launch lands in the same durable session
        # (session_id = f"{run_id}:{task_id}"), so ms.search recalls prior work.
        run_id=_RUN_ID,
        history_db_path=str(state_dir / "history.db"),
        history_max_tokens=_int_env("SCROLL_HISTORY_MAX_TOKENS"),
        logs_dir=str(attempt_dir),
        system_prompt=None,
        shared_run_ids=(),
    )
    task = TaskSpec(task_id=task_id, instruction=instruction)

    info(
        f"[scroll-agent-a] task={task_id} model={model_name} resume={args.resume} "
        f"max_steps={os.environ.get('SCROLL_MAX_STEPS') or 20} "
        f"wall_s={ctx.budget.wall_time_s} state={state_dir}"
    )

    started = time.monotonic()
    try:
        trajectory = asyncio.run(scroll_agent.run(task, ctx))
    except Exception as exc:  # noqa: BLE001 - report, let EdgeBench resume us
        print(f"[scroll-agent-a] crashed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

    (attempt_dir / "trajectory.json").write_text(_trajectory_json(trajectory), encoding="utf-8")
    metrics = trajectory.metrics
    info(
        f"[scroll-agent-a] finished: terminated={trajectory.terminated.value} "
        f"steps={len(trajectory.steps)} tokens_in={metrics.get('tokens_in')} "
        f"tokens_out={metrics.get('tokens_out')} wall={time.monotonic() - started:.0f}s "
        f"trajectory={attempt_dir / 'trajectory.json'}"
    )
    if metrics.get("error"):
        print(f"[scroll-agent-a] loop error: {metrics['error']}", file=sys.stderr, flush=True)
        return 1
    if ask_mode:
        if not trajectory.final_answer:
            info("[scroll-agent-a] the agent ended without submitting an answer")
            return 1
        print(trajectory.final_answer, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
