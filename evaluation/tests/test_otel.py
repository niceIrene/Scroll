from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from scroll_eval.tracing import otel


def test_emits_three_span_types() -> None:
    exporter = InMemorySpanExporter()
    tracer = otel.init_for_test(SimpleSpanProcessor(exporter))

    with otel.task_run(tracer, task_id="t", agent="scroll-eval", model="m", run_id="r"):
        with otel.loop_step(tracer, step_index=0):
            with otel.llm_call(tracer, model="m"):
                pass
            with otel.tool_call(tracer, tool_name="bash"):
                pass

    names = sorted(s.name for s in exporter.get_finished_spans())
    assert names == ["llm.call", "loop.step", "task.run", "tool.call"]


def test_llm_call_records_token_attrs() -> None:
    exporter = InMemorySpanExporter()
    tracer = otel.init_for_test(SimpleSpanProcessor(exporter))

    with otel.llm_call(
        tracer, model="m", prompt_tokens=10, completion_tokens=5, latency_ms=42
    ):
        pass

    span = exporter.get_finished_spans()[0]
    attrs = dict(span.attributes)
    assert attrs["llm.token_count.prompt"] == 10
    assert attrs["llm.token_count.completion"] == 5
    assert attrs["llm.token_count.total"] == 15
    assert attrs["latency_ms"] == 42


def test_set_llm_output_records_reasoning_tokens() -> None:
    exporter = InMemorySpanExporter()
    tracer = otel.init_for_test(SimpleSpanProcessor(exporter))

    with otel.llm_call(tracer, model="m") as span:
        otel.set_llm_output(
            span, prompt_tokens=10, completion_tokens=400, reasoning_tokens=395
        )

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["llm.token_count.completion"] == 400
    assert attrs["llm.token_count.completion_details.reasoning"] == 395


def test_set_llm_output_records_reasoning_content() -> None:
    exporter = InMemorySpanExporter()
    tracer = otel.init_for_test(SimpleSpanProcessor(exporter))

    with otel.llm_call(tracer, model="m") as span:
        otel.set_llm_output(
            span,
            output_messages=[{"role": "assistant", "content": "42"}],
            reasoning_content="let me work it out step by step",
        )

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["llm.output_messages.0.message.content"] == "42"
    assert (
        attrs["llm.output_messages.0.message.contents.0.message_content.type"]
        == "reasoning"
    )
    assert (
        attrs["llm.output_messages.0.message.contents.0.message_content.text"]
        == "let me work it out step by step"
    )


def test_tool_call_records_tool_kind_and_io() -> None:
    exporter = InMemorySpanExporter()
    tracer = otel.init_for_test(SimpleSpanProcessor(exporter))

    with otel.tool_call(tracer, tool_name="bash") as span:
        otel.set_tool_io(
            span, input_value='{"command": "ls"}', output_value="exit=0\nfile1"
        )

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["openinference.span.kind"] == "TOOL"
    assert attrs["tool.name"] == "bash"
    assert attrs["input.value"] == '{"command": "ls"}'
    assert attrs["output.value"] == "exit=0\nfile1"


def test_llm_call_records_message_content() -> None:
    exporter = InMemorySpanExporter()
    tracer = otel.init_for_test(SimpleSpanProcessor(exporter))

    with otel.llm_call(
        tracer,
        model="m",
        input_messages=[
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "list files"},
        ],
    ) as span:
        otel.set_llm_output(
            span,
            output_messages=[
                {
                    "role": "assistant",
                    "content": "running ls",
                    "tool_calls": [{"name": "bash", "arguments": '{"command": "ls"}'}],
                }
            ],
            prompt_tokens=3,
            completion_tokens=4,
        )

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["openinference.span.kind"] == "LLM"
    assert attrs["llm.model_name"] == "m"
    assert attrs["llm.input_messages.0.message.role"] == "system"
    assert attrs["llm.input_messages.1.message.content"] == "list files"
    assert attrs["llm.output_messages.0.message.content"] == "running ls"
    assert (
        attrs["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"]
        == "bash"
    )
