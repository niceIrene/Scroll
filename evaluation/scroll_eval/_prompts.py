"""Shared prompts loader for all agent families.

Each agent's `prompts/__init__.py` calls `make_loader(__file__)` to get a
`load(name)` function that reads `<name>.md` from its own folder.
"""
from pathlib import Path
from typing import Callable


def make_loader(caller_file: str) -> Callable[[str], str]:
    base = Path(caller_file).parent

    def load(name: str) -> str:
        return (base / f"{name}.md").read_text(encoding="utf-8")

    return load
