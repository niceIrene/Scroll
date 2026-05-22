"""BEAM environment package.

BEAM = "Beyond a Million Tokens: Benchmarking and Enhancing Long-Term
Memory in LLMs" (Tavakoli, ICLR 2026, arXiv:2510.27246).

Each task is one long conversation (100K / 500K / 1M / 10M tokens)
split into time-anchored batches. After the agent observes the full
conversation, 10 probing-question categories fire (~20 questions per
chat), each scored by an LLM judge against a per-question rubric.

The integration follows the LongMemEval pattern: each BEAM batch is
served day-by-day like an LME session, and the agent answers all
probing questions on the probe day.
"""

from Scroll.benchmarks.beam.env import BeamEnv
from Scroll.benchmarks.beam.datasource import BeamDataSource
from Scroll.benchmarks.beam.catalog import BeamEnvConfig
from Scroll.benchmarks.beam.agents import create_agent

ENV_ID = "beam"
ENV_CLS = BeamEnv
DATASOURCE_CLS = BeamDataSource


def parse_env_config(raw: dict) -> BeamEnvConfig:
    return BeamEnvConfig.from_dict(raw)
