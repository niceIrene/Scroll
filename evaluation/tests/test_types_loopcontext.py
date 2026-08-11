from scroll_eval.types import LoopContext


def test_loopcontext_has_polymorphic_llm_slots():
    ctx = LoopContext(
        llm_openai=object(),
        llm_agentscope=object(),
        model_name="qwen3.7-max",
        tracer=None,
        budget=None,
        environment=None,
    )
    assert ctx.model_name == "qwen3.7-max"
    assert ctx.llm_openai is not None
    assert ctx.llm_agentscope is not None
