"""Integration smoke test for the SCROLL dspy.RLM wrapper.

Builds a minimal ``AgentConfig`` + empty ``ConversationLog``, constructs
the ``rlm`` callable via ``Scroll.core._rlm.make_dspy_rlm``, hits the
real Dashscope endpoint through ``dspy.RLM`` (which spins up its own
Deno/Pyodide Python interpreter), and verifies:

  1. ``await rlm(query=..., context=...)`` returns a string answer
     containing "3" (counting "apple" occurrences in evidence).
  2. The log got mirrored ``kind="rlm_call"`` / ``kind="rlm_result"``
     entries with the right ``arguments.query`` / ``arguments.context``.

Requirements:
  - ``US_DASHSCOPE_API_KEY`` (or whatever ``cfg.qwen_api_key_env`` points to)
    set in ``.env`` or the shell environment.
  - ``deno`` on PATH (``brew install deno``) — dspy.RLM's default
    ``PythonInterpreter`` runs Deno/Pyodide subprocesses.

Run:
    python scripts/test_rlm.py
    python scripts/test_rlm.py \\
        --config configs/longmemeval/scroll_qwen37max.json --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

from Scroll.core._models import AgentConfig  # noqa: E402
from Scroll.log import ConversationLog  # noqa: E402
from Scroll.tools._rlm import make_dspy_rlm  # noqa: E402


QUERY = "evidence 里出现了几次 apple? 只返回数字。"
CONTEXT = "apple banana apple cherry apple"
EXPECTED_SUBSTRING = "3"


async def _run(cfg: AgentConfig, *, verbose: bool) -> tuple[str, ConversationLog]:
    log = ConversationLog()
    rlm = make_dspy_rlm(cfg, log=log, day_provider=lambda: 0)
    if verbose:
        print(f"calling rlm(query={QUERY!r}, context={CONTEXT!r})…")
    answer = await rlm(query=QUERY, context=CONTEXT)
    return answer, log


def _verify_log_mirror(log: ConversationLog) -> tuple[bool, list[str]]:
    """Return (ok, failures). Checks the rlm_call/rlm_result pair shape."""
    failures: list[str] = []
    calls = [e for e in log.entries if (e.metadata or {}).get("kind") == "rlm_call"]
    results = [e for e in log.entries if (e.metadata or {}).get("kind") == "rlm_result"]
    if len(calls) != 1:
        failures.append(f"expected 1 rlm_call entry, got {len(calls)}")
    if len(results) != 1:
        failures.append(f"expected 1 rlm_result entry, got {len(results)}")
    if calls:
        args = (calls[0].tool_call or {}).get("arguments") or {}
        if args.get("query") != QUERY:
            failures.append(f"rlm_call.arguments.query mismatch: {args.get('query')!r}")
        if args.get("context") != CONTEXT:
            failures.append(f"rlm_call.arguments.context mismatch: {args.get('context')!r}")
    if results:
        output = (results[0].tool_result or {}).get("output") or ""
        if not output:
            failures.append("rlm_result.tool_result.output is empty")
    return (not failures), failures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", default="configs/longmemeval/scroll_s.json",
        help="JSON config to read agent settings from (uses .agent subtree).",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if shutil.which("deno") is None:
        print(
            "ERROR: 'deno' not found on PATH. dspy.RLM's default "
            "PythonInterpreter requires Deno (brew install deno).",
            file=sys.stderr,
        )
        return 2

    raw = json.loads((ROOT / args.config).read_text())
    cfg = AgentConfig.from_dict(raw["agent"])

    api_key_env = cfg.qwen_api_key_env
    if not os.environ.get(api_key_env):
        print(f"ERROR: env var {api_key_env} is not set.", file=sys.stderr)
        return 2

    print(f"Model:    {cfg.qwen_model_name}")
    print(f"Config:   {args.config}")
    print(f"Query:    {QUERY}")
    print(f"Context:  {CONTEXT}  (3 apples)")
    print()

    try:
        answer, log = asyncio.run(_run(cfg, verbose=args.verbose))
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: rlm call raised: {exc!r}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    print("=" * 60)
    print("ANSWER:")
    print(answer)
    print("=" * 60)

    if args.verbose:
        print(f"\nlog.entries count: {len(log.entries)}")
        for e in log.entries:
            kind = (e.metadata or {}).get("kind")
            print(f"  kind={kind} role={e.role} turn_idx={e.turn_idx}")

    failures: list[str] = []
    if EXPECTED_SUBSTRING not in answer:
        failures.append(f"expected {EXPECTED_SUBSTRING!r} in answer, got: {answer!r}")
    ok, mirror_failures = _verify_log_mirror(log)
    failures.extend(mirror_failures)

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nPASS — answer contains '3', log mirror shape OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
