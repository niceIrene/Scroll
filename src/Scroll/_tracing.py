from __future__ import annotations

from typing import Sequence

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

# Long conversations produce many llm.input_messages.{i}.* attrs — easily
# 5-10 per message. OTel's default span-attribute cap (128) uses an LRU-ish
# bounded dict that evicts OLDEST inserts, so early messages (including the
# system prompt at index 0) drop off while late messages survive. Lift the
# cap well above what a full ReAct day can produce so Phoenix renders the
# whole conversation.
_MAX_SPAN_ATTRS = 10_000

# Providers whose Python SDKs are already wrapped by ``openinference-
# instrumentation-openai``. For these, the underlying ChatCompletion call
# emits its own OpenInference LLM span underneath AgentScope's @trace_llm
# wrapper — keeping both as LLM produces redundant siblings, so we demote
# AgentScope's wrapper to CHAIN. Anything not in this set (e.g. our
# Anthropic adapter) has NO openinference child and needs the AS span
# promoted to LLM with attribute remapping.
_OPENINFERENCE_PATCHED_PROVIDERS = {
    "openai",
    "dashscope",
    "deepseek",
    "moonshot",
    "azure_ai_openai",
    "aws_bedrock",
}


class _OpenInferenceRewriter(SpanExporter):
    """Translate AgentScope's gen_ai.* + custom spans into OpenInference
    conventions so Phoenix renders a clean GenAI trace tree.

    AgentScope on its own only emits gen_ai.* conventions (OpenTelemetry
    generic AI), which Phoenix surfaces as span_kind=UNKNOWN. We map each
    AgentScope span to the matching OpenInference kind (TOOL / AGENT /
    CHAIN) and drop duplicates of spans already covered by the OpenAI
    instrumentor (chat) or by a richer sibling (tool.* vs execute_tool).
    """

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        out = []
        for s in spans:
            action = self._classify(s)
            if action == "drop":
                continue
            if action is not None:
                self._apply(s, action)
            out.append(s)
        return self._inner.export(out)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)

    @staticmethod
    def _classify(span: ReadableSpan) -> str | None:
        attrs = span.attributes or {}
        op = attrs.get("gen_ai.operation.name")
        name = span.name
        # Noise: formatter spans (internal prompt formatting) carry no value.
        if op == "format":
            return "drop"
        # Duplicate: AgentScope's @trace_tool inner span repeats what
        # execute_tool already reports, with less data. Drop the duplicate.
        if name.startswith("tool.") and "tool.name" in attrs and op is None:
            return "drop"
        if op == "execute_tool":
            return "tool"
        if op == "invoke_agent":
            return "agent"
        # AgentScope emits a ``chat <model>`` span on every ChatModelBase
        # call. For OpenAI-family providers (openai / dashscope /
        # deepseek / ...) openinference's OpenAI patch ALSO emits a
        # ChatCompletion LLM child underneath — the AS span is just a
        # wrapper, so we demote it to CHAIN to avoid two LLM spans on
        # the same call. For providers without an openinference patch
        # (Anthropic etc.), the AS span IS the LLM event — promote it
        # to LLM and map its gen_ai.* attrs to llm.* so Phoenix renders
        # the prompt / output / token counts inline.
        if op == "chat":
            provider = str(attrs.get("gen_ai.provider.name") or "")
            if provider in _OPENINFERENCE_PATCHED_PROVIDERS:
                return "chain"
            return "llm_chat"
        if (
            name.startswith("session.") or name.startswith("day.")
            or name in ("env.step_session", "agent.run_session",
                        "env.step_day", "agent.run_day")
        ):
            return "chain"
        if name.startswith("probe."):
            return "chain"
        return None  # passthrough (e.g. ChatCompletion already LLM-kinded)

    @staticmethod
    def _apply(span: ReadableSpan, action: str) -> None:
        attrs = span.attributes or {}
        if action == "tool":
            _set(
                span,
                {
                    "openinference.span.kind": "TOOL",
                    "tool.name": attrs.get("gen_ai.tool.name"),
                    "tool.description": attrs.get("gen_ai.tool.description"),
                    "input.value": attrs.get("gen_ai.tool.call.arguments"),
                    "input.mime_type": "application/json",
                    "output.value": _str(attrs.get("gen_ai.tool.call.result")),
                },
            )
        elif action == "agent":
            _set(
                span,
                {
                    "openinference.span.kind": "AGENT",
                    "input.value": _str(attrs.get("gen_ai.input.messages")),
                    "input.mime_type": "application/json",
                    "output.value": _str(attrs.get("gen_ai.output.messages")),
                    "output.mime_type": "application/json",
                },
            )
        elif action == "chain":
            _set(span, {"openinference.span.kind": "CHAIN"})
        elif action == "llm_chat":
            in_tok = attrs.get("gen_ai.usage.input_tokens")
            out_tok = attrs.get("gen_ai.usage.output_tokens")
            total_tok = None
            if isinstance(in_tok, int) and isinstance(out_tok, int):
                total_tok = in_tok + out_tok
            _set(
                span,
                {
                    "openinference.span.kind": "LLM",
                    "llm.model_name": attrs.get("gen_ai.request.model"),
                    "llm.system": attrs.get("gen_ai.provider.name"),
                    "llm.token_count.prompt": in_tok,
                    "llm.token_count.completion": out_tok,
                    "llm.token_count.total": total_tok,
                    "input.value": _str(attrs.get("agentscope.function.input")),
                    "input.mime_type": "application/json",
                    "output.value": (
                        _str(attrs.get("gen_ai.output.messages"))
                        or _str(attrs.get("agentscope.function.output"))
                    ),
                    "output.mime_type": "application/json",
                },
            )


def _str(v: object) -> str | None:
    if v is None:
        return None
    return v if isinstance(v, str) else str(v)


def _set(span: ReadableSpan, new_attrs: dict) -> None:
    # ReadableSpan._attributes wraps a mutable OrderedDict. Direct mutation
    # is the only way to inject attrs on an already-ended span.
    bound = getattr(span, "_attributes", None)
    if bound is None:
        return
    backing = getattr(bound, "_dict", None)
    if backing is None:
        return
    for k, v in new_attrs.items():
        if v is None:
            continue
        backing[k] = v


def setup(endpoint: str) -> None:
    exporter = _OpenInferenceRewriter(OTLPSpanExporter(endpoint=endpoint))
    processor = BatchSpanProcessor(exporter)
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.add_span_processor(processor)
    else:
        limits = SpanLimits(max_span_attributes=_MAX_SPAN_ATTRS)
        provider = TracerProvider(span_limits=limits)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

    # AgentScope ships ``@trace_llm`` on every ``ChatModelBase`` subclass
    # — including our ``AnthropicMessagesChatModel`` — but the decorator
    # short-circuits unless ``_config.trace_enabled`` is True. Flip it
    # here so the same Phoenix-visible LLM spans the OpenAI path gets
    # (``chat <model_name>`` with input/output messages, tool defs, token
    # counts) also appear for the Anthropic path. The rewriter's
    # ``op == "chat" → CHAIN`` branch already handles co-existence with
    # openinference's ``ChatCompletion`` LLM span on the OpenAI side.
    try:
        from agentscope import _config as _as_config
        _as_config.trace_enabled = True
    except Exception:  # noqa: BLE001
        pass
