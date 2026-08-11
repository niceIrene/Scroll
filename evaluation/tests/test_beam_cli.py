"""The beam CLI's --scale flag selects exactly one migrated tier."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import scroll_eval.cli as cli

_CONFIG = Path(__file__).parent.parent.parent / "configs" / "beam.yaml"
_RUNNER = CliRunner()


@pytest.fixture()
def captured(monkeypatch):
    """Stub the beam runner; capture the resolved cfg.tasks instead of running."""
    box: dict = {}

    def fake_run(cfg, **kwargs):
        box["tasks"] = cfg.tasks
        return Path("runs/fake")

    import scroll_eval.evals.beam.runner as beam_runner

    monkeypatch.setattr(beam_runner, "run", fake_run)
    return box


def _beam(*args: str):
    return _RUNNER.invoke(cli.app, ["beam", str(_CONFIG), *args])


def test_scale_selects_only_that_tier(captured) -> None:
    res = _beam("--scale", "100k")  # case-insensitive
    assert res.exit_code == 0, res.output
    tasks = captured["tasks"]
    assert tasks and all(t.startswith("100K-") for t in tasks)

    res = _beam("--scale", "10M")
    assert res.exit_code == 0, res.output
    assert captured["tasks"] and all(t.startswith("10M-") for t in captured["tasks"])
    # "1M-" prefixing must not leak 10M tasks (and vice versa).
    assert not any(t.startswith("1M-") for t in captured["tasks"])


def test_scale_rejects_unknown_and_unmigrated_and_mixed_flags(captured) -> None:
    assert _beam("--scale", "2m").exit_code != 0          # not a BEAM tier
    assert _beam("--scale", "100k", "--task", "100K-1").exit_code != 0
    assert _beam("--scale", "100k", "--all-tasks").exit_code != 0
    assert "tasks" not in captured  # runner never invoked on any error path
