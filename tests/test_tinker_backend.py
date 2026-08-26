"""Offline tests for the Tinker sampling backend.

Cover the two pure layers — message rendering input and completion parsing —
without the tinker SDK or network. The SDK-touching paths (client creation,
sampling) are exercised only for their error behavior.
"""
from __future__ import annotations

import json

import pytest

agentscope = pytest.importorskip("agentscope")

from agentscope.message import (  # noqa: E402
    AssistantMsg,
    SystemMsg,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultState,
    UserMsg,
)

from scroll_eval.tinker_backend import (  # noqa: E402
    TinkerChatModel,
    messages_to_template_dicts,
    parse_completion,
)


# --- messages_to_template_dicts ----------------------------------------------


def test_plain_roles_pass_through():
    msgs = [
        SystemMsg(name="system", content="be brief"),
        UserMsg(name="user", content="hello"),
    ]
    out = messages_to_template_dicts(msgs)
    assert out == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


def test_assistant_with_tool_call_and_result_splits():
    """The folded AssistantMsg (call + result blocks) must unfold into an
    assistant dict with parsed-dict arguments plus a role:'tool' result dict —
    the shape Qwen chat templates expect."""
    msgs = [
        AssistantMsg(
            name="assistant",
            content=[
                TextBlock(text="let me check"),
                ToolCallBlock(
                    id="tc1",
                    name="execute_python",
                    input=json.dumps({"code": "print(1)"}),
                ),
                ToolResultBlock(
                    id="tc1",
                    name="execute_python",
                    output="1",
                    state=ToolResultState.SUCCESS,
                ),
            ],
        ),
    ]
    out = messages_to_template_dicts(msgs)
    assert len(out) == 2
    assistant, tool = out
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "let me check"
    (tc,) = assistant["tool_calls"]
    assert tc["function"]["name"] == "execute_python"
    # arguments must be a dict (a JSON string would double-encode via tojson)
    assert tc["function"]["arguments"] == {"code": "print(1)"}
    assert tool == {"role": "tool", "name": "execute_python", "content": "1"}


def test_malformed_tool_call_arguments_survive():
    msgs = [
        AssistantMsg(
            name="assistant",
            content=[ToolCallBlock(id="x", name="bash", input="{not json")],
        )
    ]
    (assistant,) = messages_to_template_dicts(msgs)
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"_raw": "{not json"}


# --- parse_completion ---------------------------------------------------------


def test_parse_text_only():
    (block,) = parse_completion("The answer is 42.<|im_end|>")
    assert block.type == "text"
    assert block.text == "The answer is 42."


def test_parse_thinking_then_tool_call():
    text = (
        "<think>need to search the log</think>\n"
        '<tool_call>\n{"name": "execute_python", "arguments": {"code": "ms.search(\'x\')"}}\n</tool_call>'
        "<|im_end|>"
    )
    blocks = parse_completion(text)
    types = [b.type for b in blocks]
    assert types == ["thinking", "tool_call"]
    assert blocks[0].thinking == "need to search the log"
    assert blocks[1].name == "execute_python"
    assert json.loads(blocks[1].input) == {"code": "ms.search('x')"}


def test_parse_text_between_thinking_and_tool_call():
    text = (
        "<think>plan</think>Searching now.\n"
        '<tool_call>{"name": "submit_answer", "arguments": {"answer": "42"}}</tool_call>'
    )
    blocks = parse_completion(text)
    assert [b.type for b in blocks] == ["thinking", "text", "tool_call"]
    assert blocks[1].text == "Searching now."


def test_unclosed_think_swallows_remainder():
    """max_tokens truncation mid-thought: everything after <think> is thinking,
    and no tool calls are parsed out of the chain of thought."""
    text = '<think>half a thought <tool_call>{"name": "bash"}</tool_call>'
    blocks = parse_completion(text)
    assert [b.type for b in blocks] == ["thinking"]


def test_malformed_tool_call_json_stays_visible():
    text = "<tool_call>{broken json}</tool_call>"
    (block,) = parse_completion(text)
    assert block.type == "text"
    assert "broken json" in block.text


def test_multiple_tool_calls():
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"k": 1}}</tool_call>'
    )
    blocks = parse_completion(text)
    assert [b.name for b in blocks] == ["a", "b"]


def test_implicit_open_thinking_with_xml_tool_call():
    """The real Qwen3.8 layout: the generation prompt pre-opens <think>, so
    the completion starts mid-thought with only </think>, and the tool call is
    the Qwen3-Coder XML convention, not hermes JSON."""
    text = (
        "The user wants 17*23. Let's use execute_python.\n"
        "</think>\n\n"
        "<tool_call>\n<function=execute_python>\n<parameter=code>\n"
        "print(17 * 23)\n"
        "</parameter>\n</function>\n</tool_call><|im_end|>"
    )
    blocks = parse_completion(text, implicit_think=True)
    assert [b.type for b in blocks] == ["thinking", "tool_call"]
    assert "execute_python" == blocks[1].name
    assert json.loads(blocks[1].input) == {"code": "print(17 * 23)"}


def test_xml_tool_call_multiple_params_keep_indentation():
    text = (
        "<tool_call><function=bash><parameter=command>\n"
        "for f in *; do\n  echo $f\ndone\n"
        "</parameter><parameter=timeout>\n300\n</parameter></function></tool_call>"
    )
    (block,) = parse_completion(text)
    args = json.loads(block.input)
    assert args["command"] == "for f in *; do\n  echo $f\ndone"
    assert args["timeout"] == "300"  # strings by design; bash coerces its own


def test_truncated_implicit_thinking_is_thinking_not_text():
    text = "half a thought that never closes"
    (block,) = parse_completion(text, implicit_think=True)
    assert block.type == "thinking"
    # without the flag the same text is visible output (non-thinking models)
    (block,) = parse_completion(text)
    assert block.type == "text"


# --- TinkerChatModel construction --------------------------------------------


def test_model_requires_name():
    with pytest.raises(ValueError):
        TinkerChatModel("")


def test_sampling_defaults_follow_thinking(monkeypatch):
    for var in (
        "SCROLL_TINKER_TEMPERATURE",
        "SCROLL_TINKER_TOP_P",
        "SCROLL_TINKER_TOP_K",
        "SCROLL_TINKER_MAX_NEW_TOKENS",
    ):
        monkeypatch.delenv(var, raising=False)
    thinking = TinkerChatModel("Qwen/Qwen3.8-27B", thinking=True)
    plain = TinkerChatModel("Qwen/Qwen3.8-27B", thinking=False)
    assert thinking.temperature == 0.6 and thinking.top_p == 0.95
    assert plain.temperature == 0.7 and plain.top_p == 0.8


def test_env_overrides_win(monkeypatch):
    monkeypatch.setenv("SCROLL_TINKER_TEMPERATURE", "0.1")
    monkeypatch.setenv("SCROLL_TINKER_MAX_NEW_TOKENS", "1234")
    m = TinkerChatModel("Qwen/Qwen3.8-27B", thinking=True)
    assert m.temperature == 0.1
    assert m.max_new_tokens == 1234


# --- _render tokenizer-output shapes -----------------------------------------


class _FakeTokenizer:
    """apply_chat_template stub returning a configurable shape."""

    def __init__(self, rendered):
        self._rendered = rendered
        self.kwargs = None

    def apply_chat_template(self, conversation, **kwargs):
        self.kwargs = kwargs
        return self._rendered


class _BatchEncodingLike(dict):
    """Mapping shape transformers v5 returns (has .keys, subscriptable)."""


@pytest.mark.parametrize(
    "rendered",
    [
        [1, 2, 3],                                        # transformers v4: flat list
        [[1, 2, 3]],                                      # batched template: nested
        _BatchEncodingLike(input_ids=[1, 2, 3], attention_mask=[1, 1, 1]),  # v5
        _BatchEncodingLike(input_ids=[[1, 2, 3]]),        # v5 batched
    ],
)
def test_render_normalizes_tokenizer_output_shapes(rendered):
    m = TinkerChatModel("Qwen/Qwen3.8-27B", thinking=True)
    m._tokenizer = _FakeTokenizer(rendered)
    ids = m._render([UserMsg(name="user", content="hi")], tools=None)
    assert ids == [1, 2, 3]
    assert m._tokenizer.kwargs["add_generation_prompt"] is True
    assert m._tokenizer.kwargs["enable_thinking"] is True
