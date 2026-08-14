from __future__ import annotations

import ast
import asyncio
import builtins
import io
import textwrap
import traceback
import types
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from scroll_context._runtime.types import ExecutionResult

# Per-call stdout budget returned to the model. Past this, the captured output is
# a flood risk for the context window, so we DON'T dump it: keep a short head and
# append an actionable notice telling the model to print less. The namespace
# persists across calls, so its retrieved data is still in variables — the retry
# is a cheaper re-print (a slice / count / aggregate), not a re-fetch.
_DEFAULT_MAX_STDOUT_CHARS = 32_000  # ceiling: most a triage view usefully needs
_MIN_MAX_STDOUT_CHARS = 2_000       # floor: keep a minimal overview in tight configs
_STDOUT_HEAD_CHARS = 1_500          # how much of the over-budget output we still show
# Stable prefix of the overflow notice. Public so downstream context managers can
# recognize an already-executor-bounded output and not stub it a second time.
OVERFLOW_MARKER = "[output too long:"
_OVERFLOW_NOTE = (
    OVERFLOW_MARKER + " {n} chars printed, over the {limit}-char limit — the rest "
    "is hidden to protect your context window. Your variables persist, so re-run "
    "printing LESS: a count or list of seqs, snippet=True for a bounded triage "
    "view, or aggregate in a variable and print only the result — not whole rows.]"
)


def stdout_cap_for(history_max_tokens: int | None) -> int:
    """Per-call stdout char cap, scaled to the in-context budget.

    A single tool output past ~1/16 of the window crowds everything else and
    hastens eviction; at ~4 chars/token that is ``history_max_tokens // 4`` chars.
    Clamp to ``[_MIN_MAX_STDOUT_CHARS, _DEFAULT_MAX_STDOUT_CHARS]`` — the ceiling
    is the most a triage view usefully needs even in a huge window, the floor
    keeps a minimal overview possible in tight stress configs. An unknown/unbounded
    budget falls back to the ceiling.
    """
    if not history_max_tokens or history_max_tokens <= 0:
        return _DEFAULT_MAX_STDOUT_CHARS
    return max(_MIN_MAX_STDOUT_CHARS,
               min(_DEFAULT_MAX_STDOUT_CHARS, history_max_tokens // 4))


class Executor:
    """Persistent Python execution against a shared namespace.

    Compiles each cell with ``PyCF_ALLOW_TOP_LEVEL_AWAIT`` so the model can
    write ``result = await bash("ls")`` at the top level without wrapping
    every cell in an ``async def`` block. The compiled code is bound via
    ``types.FunctionType(code, ns)`` and called — when the source contains
    a top-level await, calling returns a coroutine we await; otherwise it
    returns None and the assignments have already happened in ``ns``.

    Each call captures stdout/stderr via ``redirect_stdout`` /
    ``redirect_stderr``. Any exception is formatted and returned in
    ``stderr``/``error``. Timeouts are enforced via ``asyncio.wait_for`` —
    fine for the I/O-heavy use case (bash hops, ms queries) but cannot
    interrupt a pure-Python tight loop.
    """

    def __init__(
        self,
        namespace: dict,
        *,
        timeout_s: float = 30.0,
        max_stdout_chars: int = _DEFAULT_MAX_STDOUT_CHARS,
    ) -> None:
        self._ns = namespace
        # Seed __builtins__ so user code can call print, len, range, etc.
        # without us having to pre-populate them by hand.
        self._ns.setdefault("__builtins__", builtins.__dict__)
        self._timeout_s = timeout_s
        self._max_stdout_chars = max_stdout_chars

    def _bounded_stdout(self, text: str) -> str:
        """Trim over-budget stdout to a head plus an actionable overflow notice."""
        if len(text) <= self._max_stdout_chars:
            return text
        return (
            text[:_STDOUT_HEAD_CHARS].rstrip()
            + "\n\n"
            + _OVERFLOW_NOTE.format(n=len(text), limit=self._max_stdout_chars)
        )

    async def execute(self, source: str) -> ExecutionResult:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            code_obj = compile(
                textwrap.dedent(source),
                "<execute_python>",
                "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )
        except SyntaxError:
            return ExecutionResult(
                stdout="",
                stderr=traceback.format_exc(),
                error="SyntaxError",
            )

        func = types.FunctionType(code_obj, self._ns)

        async def _run() -> None:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                result = func()
                if asyncio.iscoroutine(result):
                    await result

        try:
            await asyncio.wait_for(_run(), timeout=self._timeout_s)
        except asyncio.TimeoutError:
            return ExecutionResult(
                stdout=self._bounded_stdout(stdout_buf.getvalue()),
                stderr=stderr_buf.getvalue(),
                error=f"execute_python timed out after {self._timeout_s}s",
            )
        except Exception:
            return ExecutionResult(
                stdout=self._bounded_stdout(stdout_buf.getvalue()),
                stderr=stderr_buf.getvalue() + traceback.format_exc(),
                error="execution error",
            )

        return ExecutionResult(
            stdout=self._bounded_stdout(stdout_buf.getvalue()),
            stderr=stderr_buf.getvalue(),
        )

    @property
    def namespace(self) -> dict[str, Any]:
        return self._ns
