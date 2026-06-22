"""Unit tests for optional Langfuse tracing (#1052).

Covers the two contracts that matter: a true no-op when keys are unset, and a
correctly-shaped generation observation when they are (SDK mocked so the test
needs no network / real Langfuse).
"""

from __future__ import annotations

import pytest

from podcast_scraper.utils import langfuse_tracing as lt

_PUB = "LANGFUSE_PUBLIC_KEY"
_SEC = "LANGFUSE_SECRET_KEY"
_URL = "LANGFUSE_BASE_URL"


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch):
    # Clear any ambient creds + drop the cached client so each test re-inits.
    for key in (_PUB, _SEC, _URL, "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)
    lt._reset_for_tests()
    yield
    lt._reset_for_tests()


class _FakeObs:
    def __init__(self, store: dict):
        self._store = store

    def end(self) -> None:
        self._store["ended"] = True


class _FakeLangfuse:
    last: "dict" = {}

    def __init__(self, **kwargs):
        _FakeLangfuse.last = {"init_kwargs": kwargs}

    def create_trace_id(self, *, seed=None):
        _FakeLangfuse.last["seed"] = seed
        return f"trace-for-{seed}"

    def start_observation(self, **kwargs):
        _FakeLangfuse.last["observation"] = kwargs
        return _FakeObs(_FakeLangfuse.last)

    def flush(self):
        _FakeLangfuse.last["flushed"] = True


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    import langfuse

    monkeypatch.setattr(langfuse, "Langfuse", _FakeLangfuse)


# ── no-op path ────────────────────────────────────────────────────────────


def test_disabled_when_keys_unset():
    assert lt.langfuse_enabled() is False
    assert lt.get_langfuse_client() is None


def test_emit_is_silent_noop_when_disabled():
    # Must not raise and must not touch any SDK when keys are unset.
    lt.emit_langfuse_span(
        provider="anthropic", capability="summarization", model="claude", cost=0.01
    )


def test_enabled_requires_both_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(_PUB, "pk-test")
    assert lt.langfuse_enabled() is False  # secret still missing
    monkeypatch.setenv(_SEC, "sk-test")
    assert lt.langfuse_enabled() is True


# ── enabled path (SDK mocked) ─────────────────────────────────────────────


def test_emit_generation_shape_when_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(_PUB, "pk-test")
    monkeypatch.setenv(_SEC, "sk-test")
    monkeypatch.setenv(_URL, "http://langfuse.local:3000")
    _install_fake_sdk(monkeypatch)

    lt.emit_langfuse_span(
        provider="anthropic",
        capability="summarization",
        model="claude-opus",
        cost=0.0123,
        prompt_tokens=100,
        completion_tokens=42,
        run_seed="/app/output/run-1",
        feed_id="https://feeds/x.xml",
        triggered_guardrail=True,
        env="prod",
    )

    last = _FakeLangfuse.last
    assert last["init_kwargs"]["base_url"] == "http://langfuse.local:3000"
    # Trace grouped deterministically by the run seed.
    assert last["seed"] == "/app/output/run-1"
    obs = last["observation"]
    assert obs["trace_context"] == {"trace_id": "trace-for-/app/output/run-1"}
    assert obs["as_type"] == "generation"
    assert obs["model"] == "claude-opus"
    assert obs["usage_details"] == {"input": 100, "output": 42}
    assert obs["cost_details"] == {"total": 0.0123}
    assert obs["metadata"]["provider"] == "anthropic"
    assert obs["metadata"]["stage"] == "summarization"
    assert obs["metadata"]["triggered_guardrail"] is True
    assert last["ended"] is True  # observation was ended


def test_emit_never_raises_on_sdk_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(_PUB, "pk-test")
    monkeypatch.setenv(_SEC, "sk-test")

    class _Boom:
        def __init__(self, **kwargs):
            pass

        def create_trace_id(self, *, seed=None):
            raise RuntimeError("boom")

        def start_observation(self, **kwargs):
            raise RuntimeError("boom")

        def flush(self):
            pass

    import langfuse

    monkeypatch.setattr(langfuse, "Langfuse", _Boom)
    # A misbehaving SDK must never propagate into the pipeline.
    lt.emit_langfuse_span(provider="openai", capability="gi", model="gpt", cost=0.5, run_seed="r")
