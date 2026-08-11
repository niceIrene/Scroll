"""Thin judge-model shim mimicking BEAM's langchain ``ChatOpenAI`` interface.

BEAM's metrics call ``model.invoke(x).content`` where ``x`` is either a prompt
string (the rubric judge) or a list of message dicts (the equivalence
classifier). We reproduce that surface over the ``openai`` client so the ported
metrics run unchanged. Defaults read the same env the agent uses, so the judge
runs on the same model (qwen3.7-max / DashScope) by default.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

# Transient errors worth retrying with backoff. The judge fans out up to
# (grade workers x within-probe workers) concurrent calls, which can trip a
# provider *burst* limiter (HTTP 429 limit_burst_rate) even when average
# throughput is fine. A single un-retried 429 propagates through ex.map and
# aborts a whole task's grading, so we retry with exponential backoff + jitter
# to de-synchronise the fan-out and ride out the burst.
_RETRY_EXC = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
_MAX_ATTEMPTS = 6
_BACKOFF_BASE_S = 2.0
_BACKOFF_CAP_S = 60.0


@dataclass
class _Response:
    content: str


def _retry_after_seconds(exc: Exception) -> float | None:
    """Honour a provider ``Retry-After`` header (seconds) when present."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


class BeamJudgeModel:
    """``invoke(prompt|messages) -> obj.content`` over the OpenAI client."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.environ.get("SCROLL_MODEL") or os.environ.get("OPENAI_MODEL_NAME") or ""
        self.temperature = temperature
        bare = self.model.split("/", 1)[-1] if "/" in self.model else self.model
        self.model = bare
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or "",
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            max_retries=max_retries,
            timeout=timeout,
        )

    def invoke(self, prompt_or_messages: Any) -> _Response:
        if isinstance(prompt_or_messages, str):
            messages = [{"role": "user", "content": prompt_or_messages}]
        else:
            messages = list(prompt_or_messages)
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                return _Response(content=resp.choices[0].message.content or "")
            except _RETRY_EXC as exc:  # transient: rate limit / timeout / 5xx
                last_exc = exc
                if attempt == _MAX_ATTEMPTS - 1:
                    break
                # Equal jitter: half a growing cap (so we genuinely back off the
                # burst limiter) plus random half (so the fan-out de-correlates).
                capped = min(_BACKOFF_BASE_S * (2 ** attempt), _BACKOFF_CAP_S)
                hinted = _retry_after_seconds(exc)
                delay = hinted if hinted is not None else capped / 2 + random.uniform(0, capped / 2)
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc
