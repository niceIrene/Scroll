"""LongMemEval env config.

A single LongMemEval *run* answers one QA item — its haystack of chat
sessions is streamed session-by-session to the agent, and a final probe
with the QA's question fires after the last session.

``num_sessions`` is set automatically by the env to
``len(haystack_sessions) + 1`` (the +1 is the probe-only session) once
the QA item is loaded; the value in the JSON config is treated as a
safety upper bound (the session-loop also exits via
:meth:`LongMemEvalEnv.is_terminal`).
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class LongMemEvalEnvConfig:
    # Upper bound on the session count; the env overwrites this with the
    # loaded item's actual session count + 1.
    num_sessions: int = 100

    # Path to the LongMemEval JSON file. Three official variants:
    #   - longmemeval_oracle.json     (only evidence sessions; smoke-test)
    #   - longmemeval_s_cleaned.json  (~40 sessions / ~115K tokens; default)
    #   - longmemeval_m_cleaned.json  (~500 sessions; full benchmark)
    dataset_path: str = "external/longmemeval/data/longmemeval_s_cleaned.json"

    # Pick exactly one of these to select the QA item:
    #   - question_id: match by ``question_id`` field (most readable)
    #   - question_index: zero-based index into the loaded list (fallback)
    # If neither is set, the env errors at construction time.
    question_id: str | None = None
    question_index: int | None = None

    # GPT-4o-style judge config. Uses the OpenAI Python client (not the
    # Dashscope-compatible endpoint), so CN_DASHSCOPE_API_KEY is required for
    # the default judge_model. Set ``judge_api_base`` to point at a
    # local vLLM server hosting llama-3.1-70b-instruct (LongMemEval's
    # other supported judge) or any OpenAI-compatible endpoint.
    judge_model: str = "gpt-4o-2024-08-06"
    judge_api_key_env: str = "CN_DASHSCOPE_API_KEY"
    judge_api_base: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "LongMemEvalEnvConfig":
        """Construct from a raw config dict.

        Accepts the legacy ``"days"`` key as a synonym for
        ``"num_sessions"`` so older configs continue to parse.
        """
        known = {f.name for f in fields(cls)}
        normalized = dict(d)
        if "num_sessions" not in normalized and "days" in normalized:
            normalized["num_sessions"] = normalized["days"]
        return cls(**{k: v for k, v in normalized.items() if k in known})
