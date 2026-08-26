"""Tinker sampling backend for the scroll eval harness.

Runs the agent's LLM calls through Tinker (thinkingmachines.ai) instead of a
chat-completions provider, so untrained base models — and, later, LoRA
checkpoints produced by post-training — are evaluated through the *same*
harness and prompts as the DashScope/OpenAI paths. This is the baseline-eval
entry point of the post-training plan: the untrained model's scores here are
the number every later SFT/RL claim is measured against.

Why not Tinker's OpenAI-compatible endpoint: it applies the HF chat template
server-side but documents no tool/function calling, and the scroll agents are
tool-driven. So this module goes through the raw ``tinker`` SDK instead:

  1. render the OpenAI-dict conversation + tool schemas to token ids with the
     model's own HF chat template (Qwen3-family templates emit hermes-style
     ``<tools>``/``<tool_call>`` markup natively),
  2. ``SamplingClient.sample`` on those ids,
  3. parse ``<think>``/``<tool_call>`` spans from the decoded completion back
     into AgentScope blocks.

:class:`TinkerChatModel` duck-types the AgentScope ``ChatModelBase`` call
surface the agents already consume — ``await model(messages, tools=...)`` →
``ChatResponse`` — so no agent code changes. Selection is by config:
``model.endpoint: tinker`` routes ``_build_agentscope_model`` here (see
``scroll_eval.runner``). ``model.name`` is the Tinker base-model id
(e.g. ``Qwen/Qwen3.8-27B``) or a ``tinker://…/sampler_weights/…`` checkpoint
path for evaluating trained weights.

Env knobs (all optional):
  SCROLL_TINKER_MAX_NEW_TOKENS  per-call completion cap (default 8192)
  SCROLL_TINKER_TEMPERATURE     override sampling temperature
  SCROLL_TINKER_TOP_P           override nucleus p
  SCROLL_TINKER_TOP_K           override top-k
  SCROLL_TINKER_TOKENIZER       HF tokenizer id when the SDK can't provide one
  TINKER_API_KEY                read by the tinker SDK itself
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import uuid
from typing import Any

from agentscope.message import Msg, TextBlock, ThinkingBlock, ToolCallBlock
from agentscope.model._model_response import ChatResponse
from agentscope.model._model_usage import ChatUsage

# Qwen recommends different sampling for thinking vs non-thinking decoding.
_THINKING_DEFAULTS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
_PLAIN_DEFAULTS = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}

_DEFAULT_MAX_NEW_TOKENS = 8192

# ChatML end-of-turn plus eos; sent as stop sequences and stripped from the
# decoded tail (providers differ on whether the stop text is included).
_STOP_STRINGS = ("<|im_end|>", "<|endoftext|>")

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Qwen3.8 renders tool calls in the Qwen3-Coder XML convention rather than
# hermes JSON: <function=NAME> wrapping <parameter=KEY>VALUE</parameter> pairs.
_XML_FN_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)


# --- message conversion -------------------------------------------------------


def messages_to_template_dicts(messages: list[Msg]) -> list[dict]:
    """AgentScope ``Msg`` list → OpenAI-style dicts for ``apply_chat_template``.

    The agent folds each assistant turn and its tool results into ONE
    ``AssistantMsg`` whose content carries ``ToolCallBlock``s and
    ``ToolResultBlock``s (see scroll_react ``_to_agentscope``). The chat
    template wants them back apart: an assistant dict with ``tool_calls``
    followed by ``role:"tool"`` result dicts. Tool-call arguments are parsed
    to dicts because Qwen templates ``tojson`` them — a pre-encoded string
    would double-encode.
    """
    out: list[dict] = []
    for m in messages:
        role = getattr(m, "role", None) or "user"
        content = getattr(m, "content", None)
        if isinstance(content, str) or content is None:
            out.append({"role": role, "content": content or ""})
            continue
        # Block content: split into text / tool_calls / trailing tool results.
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for block in content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_call":
                raw = getattr(block, "input", "{}")
                if isinstance(raw, str):
                    try:
                        args = json.loads(raw) if raw else {}
                    except json.JSONDecodeError:
                        args = {"_raw": raw}
                else:
                    args = raw if isinstance(raw, dict) else {}
                tool_calls.append(
                    {
                        "id": getattr(block, "id", "") or uuid.uuid4().hex,
                        "type": "function",
                        "function": {
                            "name": getattr(block, "name", "") or "",
                            "arguments": args,
                        },
                    }
                )
            elif btype == "tool_result":
                output = getattr(block, "output", "")
                tool_results.append(
                    {
                        "role": "tool",
                        "name": getattr(block, "name", "") or "",
                        "content": output if isinstance(output, str) else str(output),
                    }
                )
            # thinking blocks are never re-sent (the loop drops reasoning from
            # history); anything else is ignored rather than crashing the call.
        msg: dict = {"role": role, "content": "\n".join(p for p in text_parts if p)}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        out.append(msg)
        out.extend(tool_results)
    return out


# --- completion parsing -------------------------------------------------------


def _strip_stops(text: str) -> str:
    text = text.rstrip()
    for stop in _STOP_STRINGS:
        if text.endswith(stop):
            text = text[: -len(stop)].rstrip()
    return text


def _parse_tool_call_span(raw: str) -> tuple[str, dict] | None:
    """One <tool_call> span's inner content → (name, args) or None.

    Accepts both formats seen from Qwen chat templates: hermes JSON
    (``{"name": ..., "arguments": {...}}``) and the Qwen3-Coder XML style
    (``<function=NAME><parameter=KEY>VALUE</parameter>...</function>``), which
    is what Qwen3.8's template renders. XML parameter values stay strings —
    every scroll tool accepts string-typed args (bash coerces its own timeout),
    and guessing scalar types would corrupt code-valued parameters.
    """
    try:
        call = json.loads(raw)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(call, dict) and call.get("name"):
            args = call.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {"_raw": args}
            return str(call["name"]), args if isinstance(args, dict) else {}
        return None
    fn = _XML_FN_RE.search(raw)
    if fn is None:
        return None
    name = fn.group(1)
    # Template convention puts the value on its own line; trim only the framing
    # newlines so code parameters keep their internal indentation.
    args = {
        key: value.strip("\n")
        for key, value in _XML_PARAM_RE.findall(fn.group(2))
    }
    return name, args


def parse_completion(
    text: str, *, implicit_think: bool = False
) -> list[TextBlock | ThinkingBlock | ToolCallBlock]:
    """Decoded completion → AgentScope blocks.

    Handles the Qwen3-family layout: a thinking span first — either explicit
    ``<think>…</think>`` or *implicitly opened* (thinking-mode generation
    prompts end with an already-open ``<think>``, so the completion starts
    mid-thought and only ``</think>`` appears) — then visible text and zero or
    more ``<tool_call>…</tool_call>`` spans in JSON or XML form. A thinking
    span left unclosed (max_tokens truncation) swallows the remainder. A
    tool-call span that parses as neither format is surfaced as visible text
    so the loop (and the trajectory log) still show what the model did.
    """
    blocks: list[TextBlock | ThinkingBlock | ToolCallBlock] = []
    text = _strip_stops(text)

    first_close = text.find(_THINK_CLOSE)
    first_open = text.find(_THINK_OPEN)
    if implicit_think and first_close == -1 and first_open == -1:
        # Thinking mode with a pre-opened <think> and no close in sight:
        # the whole (truncated) completion is chain of thought.
        if text.strip():
            blocks.append(ThinkingBlock(thinking=text.strip()))
        return blocks
    if first_close != -1 and (first_open == -1 or first_open > first_close):
        # Implicit open: the prompt's trailing <think> means everything up to
        # the first </think> is chain of thought.
        thinking, text = text[:first_close], text[first_close + len(_THINK_CLOSE):]
        if thinking.strip():
            blocks.append(ThinkingBlock(thinking=thinking.strip()))
    elif first_open != -1:
        pre, _, rest = text.partition(_THINK_OPEN)
        thinking, closed, after = rest.partition(_THINK_CLOSE)
        if thinking.strip():
            blocks.append(ThinkingBlock(thinking=thinking.strip()))
        text = (pre + after) if closed else pre

    remainder_parts: list[str] = []
    cursor = 0
    for match in _TOOL_CALL_RE.finditer(text):
        remainder_parts.append(text[cursor : match.start()])
        cursor = match.end()
        parsed = _parse_tool_call_span(match.group(1))
        if parsed is None:
            remainder_parts.append(match.group(0))  # keep it visible, not silent
            continue
        name, args = parsed
        blocks.append(
            ToolCallBlock(id=uuid.uuid4().hex, name=name, input=json.dumps(args))
        )
    remainder_parts.append(text[cursor:])

    visible = "".join(remainder_parts).strip()
    if visible:
        # After thinking, before tool calls; the agent scans blocks by type so
        # ordering is cosmetic (it matches provider block order for the logs).
        blocks.insert(_first_non_thinking(blocks), TextBlock(text=visible))
    return blocks


def _first_non_thinking(blocks: list) -> int:
    for i, b in enumerate(blocks):
        if getattr(b, "type", None) != "thinking":
            return i
    return len(blocks)


# --- the model ----------------------------------------------------------------


class TinkerChatModel:
    """AgentScope-compatible chat model over Tinker's SamplingClient.

    Duck-types ``ChatModelBase.__call__``: ``await model(messages, tools=...)``
    returns a single non-streamed ``ChatResponse``. The blocking SDK calls run
    in a worker thread so concurrent probes (the eval runners' semaphore) still
    overlap. One ``ServiceClient``/``SamplingClient``/tokenizer triple is built
    lazily on first call and shared across the run.
    """

    def __init__(
        self,
        model: str,
        *,
        thinking: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        if not model:
            raise ValueError("TinkerChatModel needs a model id (Tinker base model or tinker:// path)")
        self.model = model
        self.thinking = thinking
        defaults = _PLAIN_DEFAULTS if thinking is False else _THINKING_DEFAULTS
        self.temperature = float(os.environ.get("SCROLL_TINKER_TEMPERATURE") or defaults["temperature"])
        self.top_p = float(os.environ.get("SCROLL_TINKER_TOP_P") or defaults["top_p"])
        self.top_k = int(os.environ.get("SCROLL_TINKER_TOP_K") or defaults["top_k"])
        self.max_new_tokens = int(
            max_new_tokens
            or os.environ.get("SCROLL_TINKER_MAX_NEW_TOKENS")
            or _DEFAULT_MAX_NEW_TOKENS
        )
        self._lock = threading.Lock()
        self._sampling_client: Any = None
        self._tokenizer: Any = None

    # -- lazy SDK setup --------------------------------------------------------

    def _ensure_clients(self) -> None:
        with self._lock:
            if self._sampling_client is not None:
                return
            try:
                import tinker  # noqa: PLC0415 - optional heavy dep, imported on use
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "the `tinker` SDK is not installed; "
                    "install with `uv sync --extra tinker` (evaluation workspace)"
                ) from exc
            service = tinker.ServiceClient()
            if self.model.startswith("tinker://"):
                self._sampling_client = service.create_sampling_client(model_path=self.model)
            else:
                self._sampling_client = service.create_sampling_client(base_model=self.model)
            self._tokenizer = self._load_tokenizer()

    def _load_tokenizer(self) -> Any:
        get_tok = getattr(self._sampling_client, "get_tokenizer", None)
        if callable(get_tok):
            try:
                return get_tok()
            except Exception:  # noqa: BLE001 - fall through to HF
                pass
        from transformers import AutoTokenizer  # noqa: PLC0415

        name = os.environ.get("SCROLL_TINKER_TOKENIZER") or self.model
        if name.startswith("tinker://"):
            raise RuntimeError(
                "cannot infer a HF tokenizer from a tinker:// checkpoint path; "
                "set SCROLL_TINKER_TOKENIZER to the base model's HF id"
            )
        return AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    # -- render / sample -------------------------------------------------------

    def _render(self, messages: list[Msg], tools: list[dict] | None) -> list[int]:
        template_kwargs: dict[str, Any] = {}
        if self.thinking is not None:
            # Qwen3 hybrid-thinking switch; unknown template vars are inert on
            # templates that don't read them.
            template_kwargs["enable_thinking"] = self.thinking
        rendered = self._tokenizer.apply_chat_template(
            messages_to_template_dicts(messages),
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=True,
            **template_kwargs,
        )
        # transformers v4 returns list[int]; v5 returns a BatchEncoding mapping
        # (and batched templates nest one level). ModelInput.from_ints needs the
        # flat id list.
        if hasattr(rendered, "keys"):
            rendered = rendered["input_ids"]
        if rendered and isinstance(rendered[0], list):
            rendered = rendered[0]
        return list(rendered)

    def _sample_blocking(self, prompt_ids: list[int]) -> tuple[str, int]:
        import tinker  # noqa: PLC0415

        types = tinker.types
        params = types.SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop=list(_STOP_STRINGS),
        )
        result = self._sampling_client.sample(
            prompt=types.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=params,
        )
        if hasattr(result, "result"):  # future-style return
            result = result.result()
        samples = getattr(result, "samples", None) or getattr(result, "sequences", None)
        if not samples:
            raise RuntimeError(f"tinker sample returned no samples for {self.model}")
        tokens = list(samples[0].tokens)
        text = self._tokenizer.decode(tokens, skip_special_tokens=False)
        return text, len(tokens)

    # -- AgentScope call surface ----------------------------------------------

    async def __call__(
        self,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice: Any = None,  # accepted for interface parity; unenforceable
        **kwargs: Any,
    ) -> ChatResponse:
        t0 = time.monotonic()
        await asyncio.to_thread(self._ensure_clients)
        prompt_ids = self._render(messages, tools)
        text, n_out = await asyncio.to_thread(self._sample_blocking, prompt_ids)
        blocks = parse_completion(text, implicit_think=self.thinking is True)
        return ChatResponse(
            content=blocks,
            is_last=True,
            usage=ChatUsage(
                input_tokens=len(prompt_ids),
                output_tokens=n_out,
                time=time.monotonic() - t0,
            ),
        )
