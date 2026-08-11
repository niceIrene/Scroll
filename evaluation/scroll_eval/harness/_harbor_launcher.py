"""Thin wrapper around the Harbor CLI that caps the E2B sandbox timeout.

Harbor's E2B backend hardcodes a 24h sandbox timeout
(`harbor/environments/e2b.py`: ``AsyncSandbox.create(..., timeout=86_400)``),
which the E2B API rejects on plans whose per-sandbox lifetime is capped below
that (e.g. ``400: Timeout cannot be greater than 1 hours``). We can't configure
Harbor's value, so before handing off to the Harbor CLI we clamp
``e2b.AsyncSandbox.create``'s ``timeout`` to ``E2B_SANDBOX_TIMEOUT`` (seconds,
default 3600 = 1h, the common free-tier cap).

The runner invokes this instead of the bare ``harbor`` console script (see
``scroll_eval.harness.runner._harbor_cmd``). For the Docker backend the patch is a
no-op — ``AsyncSandbox.create`` is never called. Tasks that legitimately need a
longer sandbox can raise ``E2B_SANDBOX_TIMEOUT`` (requires an E2B plan that
allows it).
"""
from __future__ import annotations

import os
import sys

_DEFAULT_E2B_TIMEOUT_S = 3600


def _e2b_timeout_cap() -> int:
    raw = os.environ.get("E2B_SANDBOX_TIMEOUT", "").strip()
    if not raw:
        return _DEFAULT_E2B_TIMEOUT_S
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_E2B_TIMEOUT_S
    return value if value > 0 else _DEFAULT_E2B_TIMEOUT_S


def apply_e2b_timeout_cap() -> None:
    """Clamp ``e2b.AsyncSandbox.create``'s timeout to the configured cap.

    Safe to call unconditionally: a no-op if the e2b SDK isn't importable (e.g.
    a Docker-only environment) and harmless if E2B is never used at runtime.
    """
    cap = _e2b_timeout_cap()
    try:
        from e2b import AsyncSandbox
    except Exception:
        return

    original_create = AsyncSandbox.create  # bound classmethod

    async def _capped_create(*args, timeout: int | None = None, **kwargs):
        if timeout is None or timeout > cap:
            timeout = cap
        return await original_create(*args, timeout=timeout, **kwargs)

    AsyncSandbox.create = staticmethod(_capped_create)


def main() -> None:
    apply_e2b_timeout_cap()
    # Hand off to Harbor's Typer app; it reads sys.argv[1:] and calls sys.exit.
    from harbor.cli.main import app

    app()


if __name__ == "__main__":
    sys.exit(main())
