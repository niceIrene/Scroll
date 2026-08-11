from typer.testing import CliRunner

import scroll_eval.cli as cli
from scroll_eval.cli import app


def test_cli_accepts_agent_type_and_id_flags(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "agent: {type: base_agents, id: base_agent_A}\n"
        "model: {endpoint: http://x, name: m, api_key_env: K}\n"
        "tasks: []\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run", str(p),
            "--agent", "scroll-eval",
            "--agent-type", "base_agents",
            "--agent-id", "base_agent_A",
            "--runs-root", str(tmp_path / "runs"),
        ],
    )
    # The CLI shouldn't crash on flag parsing; it may error later
    # (harbor not installed in this test env) — exit_code != 2.
    assert result.exit_code != 2, result.output


def test_cli_overrides_sandbox(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text(
        "agent: {type: base_agents, id: base_agent_A}\n"
        "model: {endpoint: http://x, name: m, api_key_env: K}\n"
        "dataset: terminal-bench-2.1\n"
        "tasks: [financial-document-processor]\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_run(cfg, agent, runs_root, verbose=False):
        captured["cfg"] = cfg
        return tmp_path / "runs" / "fake"

    monkeypatch.setattr(cli.runner, "run", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(p),
            "--sandbox-type",
            "e2b",
            "--parallelism",
            "4",
            "--runs-root",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["cfg"].sandbox.type == "e2b"
    assert captured["cfg"].sandbox.parallelism == 4
