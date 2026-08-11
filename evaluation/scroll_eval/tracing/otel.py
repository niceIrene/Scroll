from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)

_DEFAULT_OTLP_ENDPOINT = "http://localhost:6006/v1/traces"
_PROBE_TIMEOUT_S = 0.25  # one-time TCP reachability check at init

_provider_initialized: bool = False


def _collector_reachable(endpoint: str) -> bool:
    """One quick TCP connect to the collector's host:port.

    The BatchSpanProcessor retries failed exports forever with backoff, spamming
    "Transient error ... Connection refused" warnings for the whole run when the
    Phoenix server simply isn't up. Probing once at init lets us skip attaching
    the exporter instead (spans become cheap no-ops) with a single clear notice.
    """
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=_PROBE_TIMEOUT_S):
            return True
    except OSError:
        return False


def _tracing_mode() -> str:
    """SCROLL_TRACING: 'on' (always export), 'off' (never), 'auto' (probe)."""
    raw = os.environ.get("SCROLL_TRACING", "auto").strip().lower()
    if raw in ("off", "0", "false", "none", "disabled"):
        return "off"
    if raw in ("on", "1", "true", "always"):
        return "on"
    return "auto"


def init_for_phoenix(
    *,
    phoenix_project: str | None = None,
    endpoint: str | None = None,
) -> trace.Tracer:
    """Initialize the global OTel tracer once per process; return a tracer.

    First call creates a TracerProvider with:
      - ``service.name`` = "scroll-eval" (stable across families)
      - ``openinference.project.name`` = ``phoenix_project`` (routes the trace
        tree to a per-config Phoenix project in the UI)

    The OTLP exporter is attached according to ``SCROLL_TRACING``:
    ``on`` = always, ``off`` = never, ``auto`` (default) = only when a quick
    TCP probe finds the collector listening — so a run without a Phoenix
    server degrades to no-op spans with one notice instead of per-batch
    "Connection refused" retry warnings for the entire run.

    Subsequent calls return a tracer from the already-registered provider.
    ``phoenix_project`` is honoured only on the first call; subsequent values
    are ignored.  In production scroll-eval, the first call happens in the
    runner's ``_build_loop_context`` for the very first task; all tasks in
    the same agent subprocess share the same provider.
    """
    global _provider_initialized
    if not _provider_initialized:
        resource_attrs: dict[str, str] = {"service.name": "scroll-eval"}
        if phoenix_project:
            resource_attrs["openinference.project.name"] = phoenix_project
        provider = TracerProvider(resource=Resource.create(resource_attrs))
        resolved = endpoint or os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", _DEFAULT_OTLP_ENDPOINT
        )
        mode = _tracing_mode()
        attach = mode == "on" or (mode == "auto" and _collector_reachable(resolved))
        if attach:
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=resolved)))
        elif mode == "auto":
            print(
                f"[tracing] no OTLP collector reachable at {resolved} — span export "
                "disabled for this process (start Phoenix and rerun, or set "
                "SCROLL_TRACING=on to force export attempts)."
            )
        trace.set_tracer_provider(provider)
        _provider_initialized = True
    return trace.get_tracer(__name__)


def inject_into_env(env: dict) -> None:
    """Write the current trace context into ``env`` (W3C ``traceparent``).

    Lets a child process continue the same trace: the parent injects before
    spawning, the child reconstructs the context with :func:`context_from_env`.
    """
    from opentelemetry.propagate import inject

    inject(env)


def context_from_env(env: "dict | None" = None):
    """Extract a parent trace context from ``env`` (default ``os.environ``).

    Returns an OTel ``Context`` to pass explicitly as a span's ``context=`` —
    explicit parenting (rather than the thread-local *current* context) is what
    keeps spans correctly nested when the work runs on a thread pool. Returns an
    empty context (spans become roots) when no ``traceparent`` is present.
    """
    from opentelemetry.propagate import extract

    return extract(os.environ if env is None else env)


def force_flush(timeout_millis: int = 30000) -> None:
    """Flush pending spans through the BatchSpanProcessor.

    A subprocess MUST call this before exiting, or batched spans never leave the
    process. No-op when the provider has no ``force_flush`` (API-only provider).
    """
    flush = getattr(trace.get_tracer_provider(), "force_flush", None)
    if callable(flush):
        try:
            flush(timeout_millis)
        except Exception:  # noqa: BLE001 — flushing must never crash a run
            pass


def init_for_test(processor: SpanProcessor) -> trace.Tracer:
    """Test-only: install a tracer with a caller-supplied processor.

    Returns a tracer obtained directly from a fresh local provider so each
    test gets an isolated exporter (the OTel global provider may only be set
    once per process).
    """
    provider = TracerProvider()
    provider.add_span_processor(processor)
    return provider.get_tracer(__name__)


@contextmanager
def task_run(
    tracer: trace.Tracer,
    *,
    task_id: str,
    agent: str,
    model: str,
    run_id: str,
) -> Iterator[trace.Span]:
    with tracer.start_as_current_span(
        "task.run",
        attributes={
            "task_id": task_id,
            "agent": agent,
            "model": model,
            "run_id": run_id,
        },
    ) as span:
        yield span


@contextmanager
def loop_step(tracer: trace.Tracer, *, step_index: int) -> Iterator[trace.Span]:
    with tracer.start_as_current_span(
        "loop.step", attributes={"step_index": step_index}
    ) as span:
        yield span


def _set_token_counts(
    span: trace.Span,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    reasoning_tokens: int | None = None,
) -> None:
    """Set OpenInference token-count attributes (Phoenix's canonical keys).

    ``reasoning_tokens`` (e.g. qwen/o1 chain-of-thought) is a *subset* of
    ``completion_tokens`` and is recorded separately so Phoenix shows the
    reasoning breakdown.
    """
    if prompt_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens)
    if completion_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens)
    if prompt_tokens is not None and completion_tokens is not None:
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_TOTAL, prompt_tokens + completion_tokens
        )
    if reasoning_tokens is not None:
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
            reasoning_tokens,
        )


def _set_messages(span: trace.Span, base_key: str, messages: list[dict]) -> None:
    """Flatten chat messages onto a span per OpenInference semconv.

    Each message is a dict with ``role`` (str), ``content`` (str, optional) and
    ``tool_calls`` (optional list of ``{"name", "arguments"}``). Phoenix renders
    these as the LLM input/output message panels.
    """
    for i, msg in enumerate(messages or []):
        prefix = f"{base_key}.{i}.{MessageAttributes.MESSAGE_ROLE}"
        span.set_attribute(prefix, str(msg.get("role", "")))
        content = msg.get("content")
        if content:
            span.set_attribute(
                f"{base_key}.{i}.{MessageAttributes.MESSAGE_CONTENT}", str(content)
            )
        for j, tc in enumerate(msg.get("tool_calls") or []):
            tc_prefix = f"{base_key}.{i}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{j}"
            span.set_attribute(
                f"{tc_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                str(tc.get("name", "")),
            )
            args = tc.get("arguments")
            if args is not None:
                span.set_attribute(
                    f"{tc_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                    str(args),
                )


def llm_span_attributes(model: str) -> dict:
    """Attributes that mark a span as an LLM call (for callers that build the
    span manually instead of via the llm_call contextmanager)."""
    return {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
        SpanAttributes.LLM_MODEL_NAME: model,
    }


def set_llm_input(span: trace.Span, messages: list[dict] | None) -> None:
    """Record an LLM call's input messages onto its span (for callers that
    obtain the prompt outside the llm_call contextmanager)."""
    if messages:
        _set_messages(span, SpanAttributes.LLM_INPUT_MESSAGES, messages)


def set_llm_output(
    span: trace.Span,
    *,
    output_messages: list[dict] | None = None,
    reasoning_content: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    latency_ms: int | None = None,
) -> None:
    """Record an LLM call's response onto its (already-open) llm.call span.

    Called after the completion returns, when the output text, tool calls and
    token usage are known. ``reasoning_content`` (a reasoning model's
    chain-of-thought) is attached to the first output message as a typed
    ``contents`` part so it renders distinctly from the answer.
    """
    if output_messages:
        _set_messages(span, SpanAttributes.LLM_OUTPUT_MESSAGES, output_messages)
    if reasoning_content:
        part = (
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
            f".{MessageAttributes.MESSAGE_CONTENTS}.0"
        )
        span.set_attribute(
            f"{part}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "reasoning"
        )
        span.set_attribute(
            f"{part}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
            reasoning_content,
        )
    _set_token_counts(span, prompt_tokens, completion_tokens, reasoning_tokens)
    if latency_ms is not None:
        span.set_attribute("latency_ms", latency_ms)


@contextmanager
def llm_call(
    tracer: trace.Tracer,
    *,
    model: str,
    input_messages: list[dict] | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    latency_ms: int | None = None,
) -> Iterator[trace.Span]:
    attrs: dict[str, int | str] = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
        SpanAttributes.LLM_MODEL_NAME: model,
    }
    if latency_ms is not None:
        attrs["latency_ms"] = latency_ms
    with tracer.start_as_current_span("llm.call", attributes=attrs) as span:
        if input_messages:
            _set_messages(span, SpanAttributes.LLM_INPUT_MESSAGES, input_messages)
        _set_token_counts(span, prompt_tokens, completion_tokens)
        yield span


def tool_span_attributes(tool_name: str) -> dict:
    """Attributes that mark a span as a tool call (for callers that build the
    span manually instead of via the tool_call contextmanager)."""
    return {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
        SpanAttributes.TOOL_NAME: tool_name,
        "tool_name": tool_name,
    }


def set_tool_io(
    span: trace.Span,
    *,
    input_value: str | None = None,
    output_value: str | None = None,
) -> None:
    """Record a tool call's input (arguments) and output (result) onto its
    span using OpenInference semconv, so Phoenix renders the TOOL node."""
    if input_value is not None:
        span.set_attribute(SpanAttributes.INPUT_VALUE, str(input_value))
    if output_value is not None:
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(output_value))


@contextmanager
def tool_call(
    tracer: trace.Tracer,
    *,
    tool_name: str,
    exit_code: int | None = None,
    duration_ms: int | None = None,
) -> Iterator[trace.Span]:
    attrs: dict[str, int | str] = tool_span_attributes(tool_name)
    if exit_code is not None:
        attrs["exit_code"] = exit_code
    if duration_ms is not None:
        attrs["duration_ms"] = duration_ms
    with tracer.start_as_current_span("tool.call", attributes=attrs) as span:
        yield span
