"""Persistent Python REPL substrate for CodeAct/RLM-style agents.

The agent writes Python code in fenced ```python``` blocks; the runtime
compiles each cell with ``PyCF_ALLOW_TOP_LEVEL_AWAIT`` so async tools
(``rlm``, async tool calls) can be awaited at top level.
Variables persist in ``self.globals`` across cells within a session.

Design note: this is the "Algorithm 1" substrate from the RLM paper
(Zhang/Kraska/Khattab 2026). The conversation log E is loaded into
the REPL as a Python variable (``log``), never piped into the LM
context as text. The LM emits Python that operates on the variable
symbolically.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import difflib
import inspect
import io
import re
import traceback
from dataclasses import dataclass, field


# Compile flag that makes ``await`` valid at module top-level. The
# compiled code object then carries CO_COROUTINE; ``eval`` returns a
# coroutine you can await.
_PyCF_ALLOW_TOP_LEVEL_AWAIT = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT


class EndOfSession(Exception):
    """Raised from inside the REPL to end the current session early.

    The runtime catches it inside ``execute_cell`` and surfaces a
    ``CellResult(session_ended=True)`` instead of treating it as an error.
    Kept as a defensive primitive — the REPL does not bind any helper
    that raises it; the session-loop's documented termination signal is
    "emit a response with no code block" (see ``CodeActAgent``).
    """


@dataclass
class CellResult:
    """Output of a single ``execute_cell`` call."""

    stdout: str = ""
    stderr: str = ""
    exception: str | None = None  # formatted traceback string
    exc: BaseException | None = None  # original exception object, for span.record_exception()
    session_ended: bool = False
    # Bookkeeping (not surfaced to the agent — for tracing):
    code_chars: int = 0

    # Back-compat property for callers that still read ``day_ended``.
    @property
    def day_ended(self) -> bool:
        return self.session_ended

    @day_ended.setter
    def day_ended(self, value: bool) -> None:
        self.session_ended = bool(value)

    def to_user_message(self) -> str:
        """Render this result as the next user-turn content for the LM.

        Format mirrors what the agent would see if it had run the code
        in a real REPL: stdout first, stderr second, exception trace
        last. Truncated lightly to keep token usage in check; the full
        output still lands in ``conversation_log.jsonl``.
        """
        parts: list[str] = []
        if self.stdout:
            parts.append(_truncate(self.stdout.rstrip(), 4000, label="stdout"))
        if self.stderr:
            parts.append("[stderr]\n" + _truncate(self.stderr.rstrip(), 1000, label="stderr"))
        if self.exception:
            parts.append("[exception]\n" + _truncate(self.exception.rstrip(), 1500, label="exception"))
        if self.session_ended:
            parts.append("[session ended]")
        if not parts:
            parts.append("(no output)")
        return "\n\n".join(parts)


def _truncate(text: str, limit: int, label: str = "output") -> str:
    """Trim ``text`` to ~``limit`` chars, keeping both ends.

    Head-only truncation silently dropped trailing prints, which
    matters when an agent prints multiple things in one cell (e.g.
    inspecting state and then ``next_user_message()`` in the same
    cell) — the most recent print is usually the one the model needs
    to act on next.

    We keep ~75% from the head and ~25% from the tail with a
    marker in between, so both the start of the cell's output (often
    the data being inspected) and the most recent prints (often what
    the LM needs to act on next) survive.
    """
    if len(text) <= limit:
        return text
    head_len = (limit * 3) // 4
    tail_len = limit - head_len
    head = text[:head_len]
    tail = text[-tail_len:]
    omitted = len(text) - len(head) - len(tail)
    return (
        f"{head}\n... ({label} truncated: {omitted} chars omitted, "
        f"{len(text)} total) ...\n{tail}"
    )


class CellRuntime:
    """A persistent async-aware Python REPL.

    One ``CellRuntime`` per agent. Globals survive across ``execute_cell``
    calls; reset by re-creating the runtime (e.g. at session boundaries
    when ``clear_namespace_each_session=True``).

    The runtime does NOT sandbox — agent-emitted code runs with full
    Python privileges in the current process, same trust model as
    today's tool callbacks. Re-evaluate if/when this is exposed to
    untrusted models.
    """

    def __init__(self, initial_globals: dict | None = None) -> None:
        self.globals: dict = {"__builtins__": __builtins__}
        if initial_globals:
            self.globals.update(initial_globals)

    def update_globals(self, **kwargs) -> None:
        """Merge ``kwargs`` into the persistent namespace."""
        self.globals.update(kwargs)

    def reset(self, initial_globals: dict | None = None) -> None:
        """Wipe per-session state; re-bind the namespace from a fresh dict.

        Persistent state objects (``log``, ``memoryspace``, ``rlm``)
        that the caller wants to survive must be re-passed in
        ``initial_globals``.
        """
        self.globals = {"__builtins__": __builtins__}
        if initial_globals:
            self.globals.update(initial_globals)

    async def execute_cell(self, code: str) -> CellResult:
        """Compile and run ``code`` in the persistent namespace.

        Awaits top-level awaits if any. Catches ``EndOfSession`` and
        surfaces it as ``session_ended=True``. Other exceptions are
        captured into ``result.exception`` as a formatted traceback so
        the agent can read what went wrong on its next turn.

        Jupyter-style auto-print: if the cell's last statement is a
        bare expression (e.g. ``today_session()`` or ``some_var`` on
        a line by itself), wrap it in ``print(...)`` so the value
        appears in stdout. Models trained on Jupyter notebooks
        habitually use this idiom; without auto-print they think their
        retrieval returned nothing and start retry-spiral cells.
        """
        result = CellResult(code_chars=len(code))
        if not code.strip():
            return result

        try:
            compiled = _compile_with_auto_print(code)
        except SyntaxError as e:
            result.exception = f"SyntaxError: {e}"
            result.exc = e
            return result

        is_async = bool(compiled.co_flags & inspect.CO_COROUTINE)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            try:
                if is_async:
                    # eval() on a CO_COROUTINE code object returns a
                    # coroutine that, when awaited, runs the cell and
                    # writes assignments into ``self.globals``.
                    coro = eval(compiled, self.globals)
                    if asyncio.iscoroutine(coro):
                        await coro
                else:
                    exec(compiled, self.globals)
            except EndOfSession:
                result.session_ended = True
            except Exception as exc:
                # Strip the runtime's own frames from the traceback so
                # the agent sees just its code's frames — same shape as
                # a real REPL error. We also keep the original exception
                # object on ``result.exc`` so the caller can attach it
                # to its OTel span via ``span.record_exception(...)``.
                result.exception = _format_user_traceback(self.globals)
                result.exc = exc

        result.stdout = stdout_buf.getvalue()
        result.stderr = stderr_buf.getvalue()
        return result


def _compile_with_auto_print(code: str):
    """Compile ``code`` with Jupyter-style auto-print of the last bare
    expression. Returns a code object compiled with
    ``PyCF_ALLOW_TOP_LEVEL_AWAIT``.

    If the cell's last statement is an ``ast.Expr`` (a bare expression
    on its own line, like ``today_session()`` or ``my_var``), we wrap
    it in ``print(repr(value))`` so the value is visible in stdout.
    Other statement kinds (assignments, calls, awaits with assignment,
    function defs) are left untouched.

    Doesn't print None — many tool calls return None and printing
    "None" would be noise.

    Raises ``SyntaxError`` (caller handles).
    """
    tree = ast.parse(code, mode="exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        # Skip wrapping if the expression is already a print() call —
        # double-wrapping would print twice.
        is_print_call = (
            isinstance(last.value, ast.Call)
            and isinstance(last.value.func, ast.Name)
            and last.value.func.id == "print"
        )
        if not is_print_call:
            # Wrap: _repr = <expr>; if _repr is not None: print(repr(_repr))
            # Use a temp var so awaits work (await is an Expr in tree).
            tmp_name = "_Scroll_autoprint_value"
            assign_to_tmp = ast.Assign(
                targets=[ast.Name(id=tmp_name, ctx=ast.Store())],
                value=last.value,
            )
            print_if_not_none = ast.If(
                test=ast.Compare(
                    left=ast.Name(id=tmp_name, ctx=ast.Load()),
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                ),
                body=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id="print", ctx=ast.Load()),
                            args=[ast.Name(id=tmp_name, ctx=ast.Load())],
                            keywords=[],
                        )
                    )
                ],
                orelse=[],
            )
            tree.body[-1] = assign_to_tmp
            tree.body.append(print_if_not_none)
            ast.fix_missing_locations(tree)
    return compile(tree, "<cell>", "exec", flags=_PyCF_ALLOW_TOP_LEVEL_AWAIT)


def _format_user_traceback(globals_dict: dict | None = None) -> str:
    """Format the current exception, hiding runtime internals.

    Walks the traceback and drops frames whose filename contains
    ``_codeact_runtime.py`` so the agent sees only its own ``<cell>``
    frames plus any tool/library code it called into. Appends a
    "did you mean?" hint for the common name-resolution errors that
    indicate the model is hallucinating callables / modules.
    """
    exc_type, exc_value, exc_tb = _sys_exc_info()
    if exc_value is None:
        return ""
    # Walk past our own frames at the bottom.
    while exc_tb is not None and "_codeact_runtime" in exc_tb.tb_frame.f_code.co_filename:
        exc_tb = exc_tb.tb_next
    lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    base = "".join(lines)
    hint = _format_namespace_hint(exc_type, exc_value, globals_dict or {})
    return base + (("\n" + hint) if hint else "")


def _sys_exc_info():
    import sys
    return sys.exc_info()


def _public_namespace_names(globals_dict: dict) -> list[str]:
    """Return user-callable / user-relevant names from the REPL globals.

    Filters out dunders, ``__builtins__``, and Python module references
    pulled in by user code. Includes both callables and known stateful
    objects (``log``, ``memoryspace``, ``ms``, ``rlm``, ``rlm``).
    """
    keep_objects = {"log", "memoryspace", "ms", "rlm", "rlm", "today"}
    out: list[str] = []
    for name, val in globals_dict.items():
        if name.startswith("_"):
            continue
        if name == "__builtins__":
            continue
        if callable(val) or name in keep_objects:
            out.append(name)
    return sorted(out)


def _format_namespace_hint(
    exc_type: type | None,
    exc_value: BaseException | None,
    globals_dict: dict,
) -> str:
    """Return a short hint when the model misnamed a callable/module.

    Triggers on:
      - ``NameError``           — bare name not in REPL globals
      - ``ModuleNotFoundError`` / ``ImportError`` — agent tried to
        ``import`` something; tools are bare globals, no import needed
      - ``AttributeError``      — attribute access on namespace object;
        only fires when the LHS object is itself a namespace name we
        recognise (best-effort)

    Stays silent on every other exception type so we don't spam noise
    onto unrelated tracebacks.
    """
    if exc_type is None or exc_value is None:
        return ""
    available = _public_namespace_names(globals_dict)
    if not available:
        return ""
    listing = ", ".join(available)

    if exc_type is NameError:
        missing = getattr(exc_value, "name", None)
        if not missing:
            m = re.search(r"name '([^']+)' is not defined", str(exc_value))
            missing = m.group(1) if m else None
        suggest = (
            ", ".join(difflib.get_close_matches(missing, available, n=3, cutoff=0.5))
            if missing
            else ""
        )
        suggest_line = f"\nClosest match: {suggest}" if suggest else ""
        return (
            f"[hint] Tools live as bare names in the REPL globals — no "
            f"`import` needed. Available: {listing}.{suggest_line}"
        )

    if exc_type in (ModuleNotFoundError, ImportError):
        missing = getattr(exc_value, "name", None) or ""
        suggest = (
            ", ".join(difflib.get_close_matches(missing, available, n=3, cutoff=0.5))
            if missing
            else ""
        )
        suggest_line = f"\nClosest match: {suggest}" if suggest else ""
        return (
            f"[hint] The REPL has no module named that. Tools are bound "
            f"as bare names — call them directly, no `import` needed. "
            f"Available: {listing}.{suggest_line}"
        )

    return ""


__all__ = [
    "CellResult",
    "CellRuntime",
    "EndOfSession",
]
