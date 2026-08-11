import asyncio
import sys
import types

from scroll_eval.harness import _harbor_launcher as launcher


def test_timeout_cap_defaults_to_3600(monkeypatch) -> None:
    monkeypatch.delenv("E2B_SANDBOX_TIMEOUT", raising=False)
    assert launcher._e2b_timeout_cap() == 3600


def test_timeout_cap_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("E2B_SANDBOX_TIMEOUT", "7200")
    assert launcher._e2b_timeout_cap() == 7200


def test_timeout_cap_ignores_garbage_and_non_positive(monkeypatch) -> None:
    monkeypatch.setenv("E2B_SANDBOX_TIMEOUT", "not-a-number")
    assert launcher._e2b_timeout_cap() == 3600
    monkeypatch.setenv("E2B_SANDBOX_TIMEOUT", "0")
    assert launcher._e2b_timeout_cap() == 3600


def _install_fake_e2b(monkeypatch, recorder: dict) -> None:
    """Inject a fake ``e2b`` module exposing AsyncSandbox.create."""

    class FakeAsyncSandbox:
        @classmethod
        async def create(cls, *, timeout=None, **kwargs):
            recorder["timeout"] = timeout
            return "sandbox"

    fake_e2b = types.ModuleType("e2b")
    fake_e2b.AsyncSandbox = FakeAsyncSandbox
    monkeypatch.setitem(sys.modules, "e2b", fake_e2b)
    return FakeAsyncSandbox


def test_cap_clamps_oversized_timeout(monkeypatch) -> None:
    recorder: dict = {}
    sandbox_cls = _install_fake_e2b(monkeypatch, recorder)
    monkeypatch.delenv("E2B_SANDBOX_TIMEOUT", raising=False)

    launcher.apply_e2b_timeout_cap()
    # Harbor calls create with the hardcoded 24h value.
    asyncio.run(sandbox_cls.create(timeout=86_400, template="t"))
    assert recorder["timeout"] == 3600


def test_cap_preserves_small_timeout(monkeypatch) -> None:
    recorder: dict = {}
    sandbox_cls = _install_fake_e2b(monkeypatch, recorder)
    monkeypatch.setenv("E2B_SANDBOX_TIMEOUT", "1800")

    launcher.apply_e2b_timeout_cap()
    asyncio.run(sandbox_cls.create(timeout=600, template="t"))
    assert recorder["timeout"] == 600  # below cap → untouched


def test_apply_is_noop_without_e2b(monkeypatch) -> None:
    # Simulate e2b not installed: importing it raises.
    monkeypatch.setitem(sys.modules, "e2b", None)
    launcher.apply_e2b_timeout_cap()  # must not raise
