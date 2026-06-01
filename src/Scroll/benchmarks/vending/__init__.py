"""Vending machine environment package."""

from Scroll.benchmarks.vending.env import VendingEnv, default_catalog, Product, EnvConfig
from Scroll.benchmarks.vending.datasource import DataSourceManager
from Scroll.benchmarks.vending.agents import create_agent

ENV_ID = "vending"
ENV_CLS = VendingEnv
DATASOURCE_CLS = DataSourceManager


def parse_env_config(raw: dict) -> EnvConfig:
    """Parse vending-specific simulation config from a raw dict."""
    return EnvConfig.from_dict(raw)
