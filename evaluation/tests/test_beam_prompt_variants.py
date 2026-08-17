"""BEAM system-prompt variant resolution: per-agent file when present, else default."""
from __future__ import annotations

from pathlib import Path

from scroll_eval.evals.beam import prompts as beam_prompts


def _resolve(agent_id: str) -> str:
    """Mirror of the runner's variant-selection expression (runner.py run())."""
    variant = Path(beam_prompts.__file__).parent / f"system_{agent_id}.md"
    return f"system_{agent_id}" if variant.exists() else "system"


def test_variant_files_exist_for_ablation_arms():
    assert _resolve("scroll_tools") == "system_scroll_tools"
    assert _resolve("longctx_baseline") == "system_longctx_baseline"
    assert _resolve("scroll_react") == "system"        # default agent falls through


def test_variants_keep_the_grounding_contract():
    default = beam_prompts.load("system")
    tools = beam_prompts.load("system_scroll_tools")
    longctx = beam_prompts.load("system_longctx_baseline")
    for text in (default, tools, longctx):
        assert "Grounding rule (strict)" in text
        assert "not enough information" in text
    # Arm framing matches each arm's actual surface.
    assert "search_history" in tools and "execute_python" not in tools
    assert "transcript" in longctx and "search_history" not in longctx
