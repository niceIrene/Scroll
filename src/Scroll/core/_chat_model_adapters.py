"""ChatModel adapters for non-default provider surfaces.

Each adapter exposes the same callable shape AgentScope's
``OpenAIChatModel`` does — ``await model([{role,content}, ...])``
returns a ``ChatResponse`` — so ``CodeActAgent`` / ``rlm``
don't have to special-case the provider.

  - :class:`OpenAIResponsesChatModel` — GPT-5 ``-pro-`` / ``openai.*``
    models that only respond on the Responses API endpoint (posts a
    Responses-shaped body to ``/chat/completions`` via Dashscope's
    compatibility router).
  - :class:`AnthropicMessagesChatModel` — Claude models via the native
    Anthropic ``/v1/messages`` API.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Type

from agentscope.message import TextBlock, ThinkingBlock, ToolUseBlock
from agentscope.model import ChatResponse, OpenAIChatModel
from agentscope.model._model_base import ChatModelBase
from agentscope.model._model_usage import ChatUsage
from agentscope.tracing import trace_llm

_log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# OpenAI Responses API (GPT-5 pro / openai.*-pro)
# --------------------------------------------------------------------------- #


def _flatten_tool_schema(t: dict) -> dict:
    """Chat-Completions ``{type,function:{name,...}}`` → Responses ``{type,name,...}``."""
    if t.get("type") != "function":
        return t
    fn = t.get("function") or {}
    return {
        "type": "function",
        "name": fn.get("name") or t.get("name"),
        "description": fn.get("description") or t.get("description", ""),
        "parameters": fn.get("parameters") or t.get("parameters", {}),
    }


def _content_to_str(content) -> str:
    """Flatten Chat-Completions ``content`` (string or list[block]) → string.

    Responses-API ``input[].content`` accepts a plain string for message
    items, which is the simpler shape we emit downstream.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for blk in content:
            if isinstance(blk, dict):
                # OpenAI Chat shape: {"type": "text", "text": "..."}
                if "text" in blk:
                    parts.append(blk.get("text") or "")
                elif blk.get("type") in ("input_text", "output_text"):
                    parts.append(blk.get("text") or "")
            elif isinstance(blk, str):
                parts.append(blk)
        return "".join(parts)
    return str(content)


def _to_responses_input(messages: list[dict]) -> list[dict]:
    """Transform Chat-Completions ``messages`` into Responses-API ``input`` items.

    Chat Completions packs assistant tool-calls inside the assistant message
    (``message.tool_calls = [...]``). Responses API rejects that field —
    each tool call must be a separate top-level ``function_call`` item,
    and tool results become ``function_call_output`` items keyed by
    ``call_id``. System / user / assistant text messages keep the
    ``{role, content}`` shape.
    """
    out: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = m.get("role")
        content = m.get("content")
        tool_calls = m.get("tool_calls")
        tool_call_id = m.get("tool_call_id")

        if role == "tool":
            # Chat: {role:'tool', tool_call_id, content}
            # Responses: {type:'function_call_output', call_id, output}
            out.append({
                "type": "function_call_output",
                "call_id": tool_call_id or "",
                "output": _content_to_str(content),
            })
            continue

        if role == "assistant" and tool_calls:
            # Emit text message first (if non-empty), then one function_call per tool_call.
            text = _content_to_str(content)
            if text.strip():
                out.append({"role": "assistant", "content": text})
            for tc in tool_calls:
                fn = tc.get("function") or {}
                out.append({
                    "type": "function_call",
                    "call_id": tc.get("id") or tc.get("call_id") or "",
                    "name": fn.get("name") or tc.get("name", ""),
                    "arguments": fn.get("arguments") or tc.get("arguments", "{}"),
                })
            continue

        # Plain system / user / assistant-without-tool_calls message
        if role in ("system", "user", "assistant"):
            out.append({"role": role, "content": _content_to_str(content)})
            continue

        # Unknown role — pass through unchanged
        out.append(m)
    return out


class OpenAIResponsesChatModel(OpenAIChatModel):
    """``OpenAIChatModel`` variant that calls the Responses API.

    Drop-in replacement for ``OpenAIChatModel`` for GPT-5 pro / o-series
    models that only respond on the Responses endpoint. Inherits init
    (client, generate_kwargs, reasoning_effort) so ``_init_model`` only
    needs a class swap.

    Text + tools only — streaming / structured_output / vision not
    implemented (Responses API supports them; Scroll doesn't use
    them on this path).
    """

    async def __call__(  # type: ignore[override]
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        structured_model: Type | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        if structured_model is not None:
            raise NotImplementedError("structured_model not supported")
        if self.stream:
            raise NotImplementedError("streaming not supported")
        if not isinstance(messages, list):
            raise ValueError(
                f"`messages` must be a list, got {type(messages).__name__}",
            )

        gen = dict(self.generate_kwargs)
        if "max_completion_tokens" in gen:
            gen.setdefault("max_output_tokens", gen.pop("max_completion_tokens"))
        if "max_tokens" in gen:
            gen.setdefault("max_output_tokens", gen.pop("max_tokens"))
        for k in ("logprobs", "top_logprobs", "stream", "stream_options",
                  "response_format", "frequency_penalty", "presence_penalty"):
            gen.pop(k, None)

        req: dict[str, Any] = {
            "model": self.model_name,
            "input": _to_responses_input(messages),
            **gen,
            **kwargs,
        }
        if tools:
            req["tools"] = [_flatten_tool_schema(t) for t in tools]
        if tool_choice and tool_choice not in (None, "none"):
            req["tool_choice"] = tool_choice
        if self.reasoning_effort and "reasoning" not in req:
            req["reasoning"] = {"effort": self.reasoning_effort}

        start = datetime.now()
        # POST a Responses-shaped body to ``/chat/completions``: Dashscope's
        # compat router accepts ``input`` here but rejects ``/responses``
        # with "Unsupported model" for gpt-5-pro-*. ``client.post`` reuses
        # the configured api_key / base_url / instrumentation.
        from typing import Any as _Any, Dict as _Dict
        raw = await self.client.post(
            "/chat/completions",
            body=req,
            cast_to=_Dict[str, _Any],
        )
        resp = dict(raw) if not isinstance(raw, dict) else raw
        return self._parse_responses_payload(resp, start)

    @staticmethod
    def _parse_responses_payload(resp: dict, start: datetime) -> ChatResponse:
        blocks: list = []
        for item in resp.get("output", []) or []:
            itype = item.get("type")
            if itype == "reasoning":
                summary = item.get("summary") or []
                thinking_text = "\n".join(
                    s.get("text", "") if isinstance(s, dict) else str(s)
                    for s in summary
                ).strip()
                blocks.append(ThinkingBlock(type="thinking", thinking=thinking_text))
            elif itype == "message":
                for c in item.get("content", []) or []:
                    if c.get("type") == "output_text":
                        blocks.append(TextBlock(type="text", text=c.get("text", "")))
            elif itype == "function_call":
                args_raw = item.get("arguments") or "{}"
                try:
                    args_obj = json.loads(args_raw)
                except (TypeError, ValueError):
                    args_obj = {"_raw": args_raw}
                blocks.append(ToolUseBlock(
                    type="tool_use",
                    id=item.get("call_id") or item.get("id") or "",
                    name=item.get("name", ""),
                    input=args_obj,
                ))

        usage = None
        u = resp.get("usage") or {}
        if u:
            usage = ChatUsage(
                input_tokens=u.get("input_tokens", 0),
                output_tokens=u.get("output_tokens", 0),
                time=(datetime.now() - start).total_seconds(),
                metadata=u,
            )
        return ChatResponse(content=blocks, id=str(resp.get("id", "")), usage=usage)


# --------------------------------------------------------------------------- #
# Anthropic Messages API (claude-*)
# --------------------------------------------------------------------------- #


class AnthropicMessagesChatModel(ChatModelBase):
    """Adapter for Anthropic's native ``/v1/messages`` API.

    Wraps ``anthropic.AsyncAnthropic`` so callers get the same
    ``ChatResponse`` shape ``OpenAIChatModel`` returns. Hoists
    ``role:"system"`` entries to the SDK's top-level ``system``
    parameter — Anthropic Messages rejects ``system`` as an input
    role and Scroll's history always carries system at index 0.

    Inherits ``ChatModelBase`` and decorates ``__call__`` with
    ``@trace_llm`` so AgentScope's built-in tracer emits a
    ``chat <model_name>`` LLM span per call (with input messages,
    output messages, tool definitions, and token counts). This is
    the same path that gives the OpenAI/DashScope models their
    nicely-attributed Phoenix spans — without it, the Anthropic
    SDK call would only show up as the surrounding TOOL/CHAIN spans
    with no LLM-level detail.
    """

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str | None = None,
        generate_kwargs: dict[str, Any] | None = None,
        stream: bool = False,
        auth_mode: str = "api_key",
        default_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(model_name=model_name, stream=stream)
        from anthropic import AsyncAnthropic
        # The SDK falls back to ``$ANTHROPIC_API_KEY`` / ``$ANTHROPIC_AUTH_TOKEN``
        # whenever the respective kwarg is None, and attaches X-Api-Key
        # + Authorization: Bearer whenever each is set. Some OAuth-style
        # proxies (e.g. idealab) reject requests that carry both, so we
        # forbid the SDK's env fallback entirely: ignore the dev's
        # ``$ANTHROPIC_API_KEY`` even if it happens to be set, and let
        # the explicit ``auth_mode`` / ``api_key`` from config decide.
        kwargs: dict[str, Any] = {}
        if auth_mode == "bearer":
            kwargs["auth_token"] = api_key
        else:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if default_headers:
            kwargs["default_headers"] = dict(default_headers)
        self._client = AsyncAnthropic(**kwargs)
        if auth_mode == "bearer":
            self._client.api_key = None
        else:
            self._client.auth_token = None
        self.generate_kwargs = dict(generate_kwargs or {})

    @trace_llm
    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        structured_model: Type | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        if structured_model is not None:
            raise NotImplementedError("structured_model not supported")
        if self.stream:
            raise NotImplementedError("streaming not supported")
        if not isinstance(messages, list):
            raise ValueError(
                f"`messages` must be a list, got {type(messages).__name__}",
            )

        system_parts: list[str] = []
        chat_messages: list[dict] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                c = m.get("content", "")
                if isinstance(c, str):
                    if c:
                        system_parts.append(c)
                elif isinstance(c, list):
                    for blk in c:
                        if isinstance(blk, dict) and blk.get("type") == "text":
                            system_parts.append(blk.get("text", ""))
            elif role == "tool":
                # OpenAI tool-result: ``{role: "tool", tool_call_id, content}``
                # Anthropic puts tool results inside a USER message as
                # a ``tool_result`` content block. Merge into the
                # previous user message if it already holds tool results,
                # so multiple tool calls in one assistant turn pair to
                # one user turn (Anthropic API contract).
                tr = {
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": _stringify_tool_content(m.get("content", "")),
                }
                if chat_messages and chat_messages[-1].get("role") == "user" \
                        and isinstance(chat_messages[-1].get("content"), list):
                    chat_messages[-1]["content"].append(tr)
                else:
                    chat_messages.append({"role": "user", "content": [tr]})
            elif role == "assistant" and m.get("tool_calls"):
                # OpenAI assistant with tool_calls → Anthropic content
                # blocks (text + tool_use).
                blocks: list[dict] = []
                text = m.get("content")
                if isinstance(text, str) and text:
                    blocks.append({"type": "text", "text": text})
                elif isinstance(text, list):
                    for blk in text:
                        if isinstance(blk, dict) and blk.get("type") == "text":
                            blocks.append({"type": "text", "text": blk.get("text", "")})
                for tc in m["tool_calls"] or []:
                    fn = tc.get("function") or {}
                    raw = fn.get("arguments")
                    if isinstance(raw, str):
                        try:
                            inp = json.loads(raw) if raw else {}
                        except (TypeError, ValueError):
                            inp = {"_raw": raw}
                    elif isinstance(raw, dict):
                        inp = raw
                    else:
                        inp = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name") or tc.get("name", ""),
                        "input": inp,
                    })
                chat_messages.append({"role": "assistant", "content": blocks})
            else:
                chat_messages.append(m)

        gen = dict(self.generate_kwargs)
        if "max_completion_tokens" in gen:
            gen.setdefault("max_tokens", gen.pop("max_completion_tokens"))
        for k in ("logprobs", "top_logprobs", "stream", "stream_options",
                  "response_format", "frequency_penalty", "presence_penalty"):
            gen.pop(k, None)
        gen.setdefault("max_tokens", 4096)

        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": chat_messages,
            **gen,
            **kwargs,
        }
        if system_parts:
            params["system"] = "\n\n".join(p for p in system_parts if p)
        if tools:
            params["tools"] = [_to_anthropic_tool(t) for t in tools]
        if tool_choice and tool_choice not in (None, "none"):
            params["tool_choice"] = _to_anthropic_tool_choice(tool_choice)

        start = datetime.now()
        resp = await self._client.messages.create(**params)
        return _parse_anthropic_response(resp, start)


def _stringify_tool_content(c) -> str:
    """Coerce a tool-result ``content`` field to a string for Anthropic.

    OpenAI's ``role: "tool"`` content is sometimes a list of blocks
    (``[{"type": "text", "text": "..."}]``) and sometimes a plain
    string. Anthropic's ``tool_result.content`` accepts either, but
    string is simplest and round-trips.
    """
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for blk in c:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict):
                parts.append(blk.get("text", "") or "")
        return "".join(parts)
    return "" if c is None else str(c)


def _to_anthropic_tool(t: dict) -> dict:
    """Convert an OpenAI Chat-Completions tool schema into Anthropic shape.

    AgentScope's substrate emits tools in OpenAI Chat-Completions form:
    ``{"type": "function", "function": {"name", "description", "parameters"}}``.
    Anthropic's Messages API wants
    ``{"name", "description", "input_schema"}``. Pass-through if the
    input is already in Anthropic shape (has ``name`` + ``input_schema``).
    """
    if "input_schema" in t and "name" in t:
        return t
    if t.get("type") == "function" and isinstance(t.get("function"), dict):
        fn = t["function"]
        out = {
            "name": fn.get("name") or t.get("name", ""),
            "input_schema": fn.get("parameters") or t.get("parameters", {"type": "object", "properties": {}}),
        }
        desc = fn.get("description") or t.get("description")
        if desc:
            out["description"] = desc
        return out
    return {
        "name": t.get("name", ""),
        "description": t.get("description", ""),
        "input_schema": t.get("parameters") or {"type": "object", "properties": {}},
    }


def _to_anthropic_tool_choice(choice):
    """Normalize tool_choice to Anthropic's ``{"type": ...}`` shape."""
    if isinstance(choice, dict):
        return choice
    if choice == "auto":
        return {"type": "auto"}
    if choice == "any" or choice == "required":
        return {"type": "any"}
    if isinstance(choice, str):
        return {"type": "tool", "name": choice}
    return choice


def _parse_anthropic_response(resp, start: datetime) -> ChatResponse:
    blocks: list = []
    for c in getattr(resp, "content", []) or []:
        ctype = getattr(c, "type", None)
        if ctype == "text":
            blocks.append(TextBlock(type="text", text=getattr(c, "text", "")))
        elif ctype == "thinking":
            blocks.append(ThinkingBlock(
                type="thinking", thinking=getattr(c, "thinking", ""),
            ))
        elif ctype == "tool_use":
            ti = getattr(c, "input", None)
            if not isinstance(ti, dict):
                ti = {} if ti is None else {"_raw": ti}
            blocks.append(ToolUseBlock(
                type="tool_use",
                id=getattr(c, "id", "") or "",
                name=getattr(c, "name", "") or "",
                input=ti,
            ))

    usage = None
    u = getattr(resp, "usage", None)
    if u is not None:
        meta = u.model_dump() if hasattr(u, "model_dump") else dict(u)
        usage = ChatUsage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            time=(datetime.now() - start).total_seconds(),
            metadata=meta,
        )

    return ChatResponse(
        content=blocks,
        id=getattr(resp, "id", "") or "",
        usage=usage,
    )


__all__ = ["OpenAIResponsesChatModel", "AnthropicMessagesChatModel"]
