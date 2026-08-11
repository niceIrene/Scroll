import pytest

from scroll_eval._tools_common import (
    OPENAI_TOOLS_SCHEMA,
    TOOLS,
    select_tools,
)


def test_openai_tools_schema_is_canonical_pair_unchanged() -> None:
    """Legacy import surface must remain byte-equivalent: bash + submit_answer."""
    names = [entry["function"]["name"] for entry in OPENAI_TOOLS_SCHEMA]
    assert names == ["bash", "submit_answer"]


def test_tools_registry_includes_execute_python() -> None:
    assert "execute_python" in TOOLS
    assert TOOLS["execute_python"]["function"]["name"] == "execute_python"


def test_select_tools_returns_requested_subset_in_order() -> None:
    subset = select_tools(["execute_python", "submit_answer"])
    names = [entry["function"]["name"] for entry in subset]
    assert names == ["execute_python", "submit_answer"]


def test_select_tools_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        select_tools(["bash", "not_a_real_tool"])
