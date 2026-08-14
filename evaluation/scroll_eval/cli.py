from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from scroll_eval.harness import config as cfg_mod
from scroll_eval.harness import compare as compare_mod
from scroll_eval.harness import runner
from scroll_eval.harness import summarize as summarize_mod


_RANCHER_DOCKER = r"C:\Program Files\Rancher Desktop\resources\resources\win32\bin\docker.exe"


def _docker_exe() -> str:
    """Resolve docker.exe — PATH first, then known Rancher Desktop location."""
    found = shutil.which("docker")
    if found:
        return found
    if sys.platform == "win32" and os.path.exists(_RANCHER_DOCKER):
        return _RANCHER_DOCKER
    raise RuntimeError(
        "docker not found on PATH. Install Docker Desktop or Rancher Desktop, "
        "then open a new shell (PATH is set at shell launch time)."
    )

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a dotenv file. Shell-set vars take precedence."""
    typer.echo(f'loading dotenv from file {str(path)}')
    if not path.exists():
        typer.echo(f'env file does not exist: {str(path)}')
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
        typer.echo(f'added new key with value to env, key:{key}')

@app.command()
def run(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    agent: str = typer.Option("scroll-eval", help="Harbor agent name."),
    dataset: str = typer.Option("", help="Override dataset name."),
    task: list[str] = typer.Option(
        None, "--task", help="Override config tasks; repeatable."
    ),
    all_tasks: bool = typer.Option(
        False, "--all-tasks", help="When true, ignore --task option, and run all the tasks of the dataset."
    ),
    agent_type: Optional[str] = typer.Option(
        None, "--agent-type", help="Override config agent.type."),
    agent_id: Optional[str] = typer.Option(
        None, "--agent-id", help="Override config agent.id."),
    sandbox_type: Optional[str] = typer.Option(
        None, "--sandbox-type", help="Override sandbox.type: docker | e2b."),
    parallelism: Optional[int] = typer.Option(
        None, "--parallelism", help="Override sandbox.parallelism."),
    runs_root: Path = typer.Option(Path("runs"), help="Output root directory."),
    env_file: Path = typer.Option(Path(".env.local"), help="Environment variable file path."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Stream harbor's stdout live to the terminal."
    ),
) -> None:
    """Execute one experiment (one agent over one task set)."""
    from dataclasses import replace

    from scroll_eval.harness.config import AgentSpec, SandboxSpec

    # Load the dotenv file BEFORE parsing the config so ${VAR} placeholders in
    # the YAML (e.g. ${OPENAI_BASE_URL}) expand against the dotenv values.
    # Without this, config.load() runs expandvars against an environment that
    # doesn't yet have the dotenv entries, leaving the placeholders literal and
    # breaking the model.api_key_env / endpoint resolution downstream.
    _load_dotenv(env_file)

    cfg = cfg_mod.load(config_path)

    if dataset and dataset != cfg.dataset.name:
        cfg = replace(cfg, dataset=replace(cfg.dataset, name=dataset))

    if cfg.dataset.name == "beam":
        raise typer.BadParameter(
            "the 'beam' dataset is a native memory eval, not a Harbor task. "
            "Run it with: scroll-eval beam <config>",
            param_hint="CONFIG_PATH",
        )

    if cfg.dataset.name == "longmemeval":
        raise typer.BadParameter(
            "the 'longmemeval' dataset is a native memory eval, not a Harbor task. "
            "Run it with: scroll-eval longmemeval <config>",
            param_hint="CONFIG_PATH",
        )

    if all_tasks and task:
        task.clear()
        typer.echo('--all-tasks is set. Run all tasks.')

    if all_tasks:
        typer.echo('--all-tasks is set. Run all tasks.')
        cfg = cfg_mod.with_all_tasks(cfg)
    elif task:
        cfg = cfg_mod.with_tasks(cfg, list(task))

    new_type = agent_type or cfg.agent.type
    new_id = agent_id or cfg.agent.id
    if new_type != cfg.agent.type or new_id != cfg.agent.id:
        cfg = replace(cfg, agent=AgentSpec(type=new_type, id=new_id))

    if sandbox_type is not None or parallelism is not None:
        new_sandbox_type = sandbox_type or cfg.sandbox.type
        new_parallelism = parallelism if parallelism is not None else cfg.sandbox.parallelism
        if new_sandbox_type not in ("docker", "e2b"):
            raise typer.BadParameter(
                "sandbox type must be 'docker' or 'e2b'", param_hint="--sandbox-type"
            )
        if new_parallelism < 1:
            raise typer.BadParameter(
                "parallelism must be an integer >= 1", param_hint="--parallelism"
            )
        cfg = replace(
            cfg,
            sandbox=SandboxSpec(type=new_sandbox_type, parallelism=new_parallelism),
        )

    # env_file is already loaded into os.environ above (before config parse), so
    # it isn't threaded into runner.run — its internal _load_dotenv() only fills
    # in any vars not already set.
    run_dir = runner.run(cfg, agent=agent, runs_root=runs_root, verbose=verbose)
    typer.echo(f"Run complete: {run_dir}")


# BEAM tiers, as used in migrated task names ("<scale>-<n>"). Prefix matching
# is unambiguous: "1M-" cannot match "10M-..." names.
_BEAM_SCALES = ("100K", "500K", "1M", "10M")


@app.command()
def beam(
    config_path: Optional[Path] = typer.Argument(
        None, exists=True, dir_okay=False,
        help="Run config (omit when using --grade-only).",
    ),
    task: list[str] = typer.Option(
        None, "--task", help="Override config tasks (e.g. 100K-1); repeatable."
    ),
    all_tasks: bool = typer.Option(
        False, "--all-tasks", help="Run every migrated beam task (ignores --task)."
    ),
    scale: Optional[str] = typer.Option(
        None, "--scale", help="Run every migrated task of one tier: 100k, 500k, 1m or 10m."
    ),
    concurrency: int = typer.Option(
        4, "--concurrency", help="Probes the AGENT answers concurrently within a task."
    ),
    judge_workers: int = typer.Option(
        8, "--judge-workers", help="Probes the JUDGE grades concurrently per task."
    ),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model",
        help="Grade with this model instead of the agent model (e.g. qwen3.6-flash). "
             "Also settable via SCROLL_JUDGE_MODEL in the env / .env.local.",
    ),
    grade_only: Optional[Path] = typer.Option(
        None, "--grade-only", exists=True, file_okay=False,
        help="Re-grade an existing run dir (skip the agent); rebuilds "
             "scores/summary/manifest. Pair with --judge-model to re-score.",
    ),
    runs_root: Path = typer.Option(Path("runs"), help="Output root directory."),
    index: Optional[bool] = typer.Option(
        None, "--index/--no-index",
        help="Enable BOTH the headline eviction index and the seed map "
             "(sets SCROLL_EVICTION_INDEX and SCROLL_SEED_INDEX). Use --no-index "
             "for the index-off ablation. Omit to use the env/defaults (both on).",
    ),
    var_context: Optional[bool] = typer.Option(
        None, "--var-context/--no-var-context",
        help="Curated Python-variable context instead of verbatim replay past "
             "the last N turns (sets SCROLL_VAR_CONTEXT). Omit to use the "
             "env/default (off).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output."),
) -> None:
    """Run the BEAM long-term-memory benchmark natively (no Harbor sandbox)."""
    from scroll_eval.evals.beam.runner import run as beam_run
    from scroll_eval.harness.runner import _list_all_tasks, _load_dotenv

    if concurrency < 1:
        raise typer.BadParameter("must be >= 1", param_hint="--concurrency")
    if judge_workers < 1:
        raise typer.BadParameter("must be >= 1", param_hint="--judge-workers")
    if scale is not None and (all_tasks or task):
        raise typer.BadParameter(
            "cannot be combined with --task/--all-tasks", param_hint="--scale"
        )

    _load_dotenv()
    # The judge subprocess inherits this env (env=os.environ.copy()), so setting
    # it here is all the plumbing the smaller-judge-model override needs. The
    # explicit flag wins over any SCROLL_JUDGE_MODEL from .env.local.
    if judge_model:
        os.environ["SCROLL_JUDGE_MODEL"] = judge_model

    # Grade-only: re-judge an existing run dir, no agent run, no config needed.
    if grade_only is not None:
        if config_path is not None:
            raise typer.BadParameter(
                "pass either a config or --grade-only <run_dir>, not both",
                param_hint="--grade-only",
            )
        from scroll_eval.evals.beam.runner import grade_run
        run_dir = grade_run(grade_only, judge_workers=judge_workers, verbose=verbose)
        typer.echo(f"BEAM re-grade complete: {run_dir}")
        return
    if config_path is None:
        raise typer.BadParameter("provide a run config, or --grade-only <run_dir>")

    # Index ablation toggle: when explicitly passed, --index/--no-index sets BOTH
    # index env vars (eviction index + seed map) so the agent prompt and runtime
    # match. Left as None it doesn't touch the env, preserving the defaults (the
    # runner setdefaults SCROLL_SEED_INDEX=1 and the agent defaults the eviction
    # index on).
    if index is not None:
        os.environ["SCROLL_EVICTION_INDEX"] = "1" if index else "0"
        os.environ["SCROLL_SEED_INDEX"] = "1" if index else "0"
    # Var-context toggle: same left-as-None-preserves-env/default pattern as
    # --index/--no-index above.
    if var_context is not None:
        os.environ["SCROLL_VAR_CONTEXT"] = "1" if var_context else "0"
    cfg = cfg_mod.load(config_path)
    if scale is not None:
        tier = scale.upper()
        if tier not in _BEAM_SCALES:
            raise typer.BadParameter(
                f"unknown scale {scale!r}; expected one of: "
                + ", ".join(s.lower() for s in _BEAM_SCALES),
                param_hint="--scale",
            )
        names = [n for n in _list_all_tasks("beam") if n.startswith(f"{tier}-")]
        if not names:
            raise typer.BadParameter(
                f"no migrated {tier} tasks under local-tasks/beam "
                "(migrate the tier with scripts/migrate_beam.py first)",
                param_hint="--scale",
            )
        cfg = cfg_mod.with_tasks(cfg, names)
    elif all_tasks:
        cfg = cfg_mod.with_all_tasks(cfg)
    elif task:
        cfg = cfg_mod.with_tasks(cfg, list(task))

    run_dir = beam_run(
        cfg, runs_root=runs_root, verbose=verbose,
        concurrency=concurrency, judge_workers=judge_workers,
    )
    typer.echo(f"BEAM run complete: {run_dir}")


@app.command()
def longmemeval(
    config_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Run config."
    ),
    task: list[str] = typer.Option(
        None, "--task", help="Override config tasks (a question_id); repeatable."
    ),
    all_tasks: bool = typer.Option(
        False, "--all-tasks", help="Run every generated longmemeval task (ignores --task)."
    ),
    concurrency: int = typer.Option(
        4, "--concurrency", help="Tasks answered concurrently (each carries one probe)."
    ),
    judge_workers: int = typer.Option(
        8, "--judge-workers", help="Probes the JUDGE grades concurrently per task."
    ),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model",
        help="Grade with this model instead of the agent model. "
             "Also settable via SCROLL_JUDGE_MODEL in the env / .env.local.",
    ),
    runs_root: Path = typer.Option(Path("runs"), help="Output root directory."),
    index: Optional[bool] = typer.Option(
        None, "--index/--no-index",
        help="Enable BOTH the headline eviction index and the seed map "
             "(sets SCROLL_EVICTION_INDEX and SCROLL_SEED_INDEX). Omit for defaults (both on).",
    ),
    var_context: Optional[bool] = typer.Option(
        None, "--var-context/--no-var-context",
        help="Curated Python-variable context instead of verbatim replay past "
             "the last N turns (sets SCROLL_VAR_CONTEXT). Omit to use the "
             "env/default (off).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output."),
) -> None:
    """Run the LongMemEval memory benchmark natively (no Harbor sandbox)."""
    from scroll_eval.evals.longmemeval.runner import run as lme_run
    from scroll_eval.harness.runner import _load_dotenv

    if concurrency < 1:
        raise typer.BadParameter("must be >= 1", param_hint="--concurrency")
    if judge_workers < 1:
        raise typer.BadParameter("must be >= 1", param_hint="--judge-workers")

    _load_dotenv()
    # The judge subprocess inherits this env (env=os.environ.copy()); the explicit
    # flag wins over any SCROLL_JUDGE_MODEL from .env.local.
    if judge_model:
        os.environ["SCROLL_JUDGE_MODEL"] = judge_model
    if index is not None:
        os.environ["SCROLL_EVICTION_INDEX"] = "1" if index else "0"
        os.environ["SCROLL_SEED_INDEX"] = "1" if index else "0"
    if var_context is not None:
        os.environ["SCROLL_VAR_CONTEXT"] = "1" if var_context else "0"

    cfg = cfg_mod.load(config_path)
    if all_tasks:
        cfg = cfg_mod.with_all_tasks(cfg)
    elif task:
        cfg = cfg_mod.with_tasks(cfg, list(task))

    run_dir = lme_run(
        cfg, runs_root=runs_root, verbose=verbose,
        concurrency=concurrency, judge_workers=judge_workers,
    )
    typer.echo(f"LongMemEval run complete: {run_dir}")


trace_app = typer.Typer(no_args_is_help=True, help="Manage the Phoenix trace server.")
app.add_typer(trace_app, name="trace-server")


@trace_app.command("up")
def trace_up() -> None:
    """Start the Phoenix container in the background."""
    subprocess.run([_docker_exe(), "compose", "up", "-d", "phoenix"], check=True)
    typer.echo("Phoenix: http://localhost:6006")


@trace_app.command("down")
def trace_down() -> None:
    """Stop the Phoenix container."""
    subprocess.run([_docker_exe(), "compose", "down"], check=True)

@app.command()
def summary(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    show: str = typer.Option(
        "all", "--show", help="Filter rows: all | passed | failed | errored"
    ),
    write_markdown: bool = typer.Option(
        True, "--write-markdown/--no-markdown", help="Write a .md report alongside."
    ),
) -> None:
    """Print aggregate + per-task summary for a single run directory."""
    s = summarize_mod.summarize(run_dir)
    filter_arg = None if show == "all" else show
    typer.echo(summarize_mod.render_table(s, show_only=filter_arg))
    if write_markdown:
        out = run_dir.parent / f"summary__{run_dir.name}.md"
        out.write_text(summarize_mod.render_markdown(s), encoding="utf-8")
        typer.echo(f"\nMarkdown: {out}")


@app.command()
def compare(
    a: Path = typer.Argument(..., exists=True, file_okay=False),
    b: Path = typer.Argument(..., exists=True, file_okay=False),
    write_markdown: bool = typer.Option(
        True, "--write-markdown/--no-markdown", help="Write a .md report alongside."
    ),
) -> None:
    """Diff two run directories side by side."""
    report = compare_mod.compare(a, b)
    typer.echo(compare_mod.render_table(report))
    if write_markdown:
        out = a.parent / f"compare__{a.name}__vs__{b.name}.md"
        out.write_text(compare_mod.render_markdown(report), encoding="utf-8")
        typer.echo(f"Markdown: {out}")


if __name__ == "__main__":
    app()
