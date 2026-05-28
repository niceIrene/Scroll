"""LongMemEval environment package — long-horizon chat-memory benchmark.

Each "run" answers one LongMemEval QA item. Each turn in the
turn-loop delivers one chat session from the item's haystack via
:meth:`LongMemEvalDataSource.begin_turn`. After every haystack session
has been streamed, a single end-of-history probe fires asking the
question; the agent's free-text reply is scored by an LLM judge.
"""

from Scroll.benchmarks.longmemeval.env import LongMemEvalEnv, LongMemEvalEnvConfig
from Scroll.benchmarks.longmemeval.datasource import LongMemEvalDataSource
from Scroll.benchmarks.longmemeval.agents import create_agent

ENV_ID = "longmemeval"
ENV_CLS = LongMemEvalEnv
DATASOURCE_CLS = LongMemEvalDataSource


def parse_env_config(raw: dict) -> LongMemEvalEnvConfig:
    return LongMemEvalEnvConfig.from_dict(raw)
