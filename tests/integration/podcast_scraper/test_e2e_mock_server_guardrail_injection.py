"""Unit-level tests for the E2E mock-server guardrail-injection registry
(#999 / ADR-099).

Pure class-state tests — exercise the ``inject_violation`` /
``_pop_injected_violation`` / ``clear_violations`` classmethods directly,
without standing up an HTTP server. Full HTTP-layer injection is covered
in the E2E test, but the registry semantics are pure logic worth pinning
at the unit tier so a future refactor of the mock server doesn't silently
break test injections.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="mock-server import indirectly requires FastAPI")

from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_violations_around_each_test():
    """Class-state is shared across all instances; reset before AND after each
    test so failures don't leak the injection into a sibling test."""
    E2EHTTPRequestHandler.clear_violations()
    yield
    E2EHTTPRequestHandler.clear_violations()


def test_inject_then_pop_returns_violation_type():
    E2EHTTPRequestHandler.inject_violation("/v1/audio/transcriptions", "transcription:length_floor")
    popped = E2EHTTPRequestHandler._pop_injected_violation("/v1/audio/transcriptions")
    assert popped == "transcription:length_floor"


def test_pop_is_one_shot():
    """An injected violation fires exactly once. Subsequent calls to
    the same route return a normal mock response."""
    E2EHTTPRequestHandler.inject_violation("/v1/diarize", "diarize:empty_segments")
    first = E2EHTTPRequestHandler._pop_injected_violation("/v1/diarize")
    second = E2EHTTPRequestHandler._pop_injected_violation("/v1/diarize")
    assert first == "diarize:empty_segments"
    assert second is None  # already popped


def test_no_injection_returns_none():
    assert E2EHTTPRequestHandler._pop_injected_violation("/v1/diarize") is None


def test_multiple_routes_independent():
    E2EHTTPRequestHandler.inject_violation("/v1/audio/transcriptions", "transcription:empty")
    E2EHTTPRequestHandler.inject_violation("/v1/diarize", "diarize:empty_segments")
    # Pop one — the other stays.
    assert (
        E2EHTTPRequestHandler._pop_injected_violation("/v1/audio/transcriptions")
        == "transcription:empty"
    )
    assert E2EHTTPRequestHandler._pop_injected_violation("/v1/diarize") == "diarize:empty_segments"


def test_clear_violations_drops_all():
    E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:empty_content")
    E2EHTTPRequestHandler.inject_violation("/v1/diarize", "diarize:empty_segments")
    E2EHTTPRequestHandler.clear_violations()
    assert E2EHTTPRequestHandler._pop_injected_violation("/v1/chat/completions") is None
    assert E2EHTTPRequestHandler._pop_injected_violation("/v1/diarize") is None


def test_repeated_inject_overwrites():
    """Re-injecting the same route swaps the violation_type; doesn't queue."""
    E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:empty_content")
    E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:thinking_prose")
    popped = E2EHTTPRequestHandler._pop_injected_violation("/v1/chat/completions")
    assert popped == "chat:thinking_prose"  # the second injection wins
