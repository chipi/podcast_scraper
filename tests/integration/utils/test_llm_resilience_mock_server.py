"""End-to-end resilience against a REAL (mock) HTTP server, mirroring the RSS resilience tests.

Unit tests classify hand-written error strings; this proves the whole chain works against ACTUAL
provider-SDK errors: a local http.server returns 503 / 429 / 402 / quota bodies, the OpenAI SDK (the
transport for openai/deepseek/grok/mistral) turns them into its real exception types, and
``retry_with_metrics`` + the taxonomy do the right thing — back off and recover on overload, hard-stop
on out-of-money. If a provider SDK changes how it surfaces a status, THIS is where it shows up.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List
from unittest.mock import patch

import pytest

from podcast_scraper.providers.resilience.policy import ResilienceFuseOpenError
from podcast_scraper.utils import llm_circuit_breaker as _llm_cb
from podcast_scraper.utils.llm_circuit_breaker import (
    LLMCircuitBreakerConfig,
    reset_for_test,
    stats,
)
from podcast_scraper.utils.llm_error_taxonomy import LLMTerminalError
from podcast_scraper.utils.provider_metrics import ProviderCallMetrics, retry_with_metrics

pytestmark = [pytest.mark.integration]

openai = pytest.importorskip("openai")


_SEQUENCE: List[int] = []
_BODIES: dict = {}
_hits = {"n": 0}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence the server's stderr logging
        pass

    def do_POST(self):
        i = _hits["n"]
        _hits["n"] += 1
        status = _SEQUENCE[i] if i < len(_SEQUENCE) else _SEQUENCE[-1]
        body = _BODIES.get(
            status,
            {"error": {"message": f"status {status}", "type": "server_error"}},
        )
        if status == 200:
            body = {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def mock_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    _hits["n"] = 0
    _SEQUENCE.clear()
    _BODIES.clear()
    try:
        yield srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"
    finally:
        srv.shutdown()


def _client(base_url: str):
    # max_retries=0 so the SDK does NOT retry internally — we isolate OUR retry_with_metrics.
    return openai.OpenAI(api_key="test", base_url=base_url, max_retries=0)


def _call(client):
    return client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])


def test_overload_503_is_retried_then_recovers(mock_server) -> None:
    """gemini-flash-lite's daily reality: a couple of 503s, then it serves us. We must ride through."""
    _srv, base = mock_server
    _SEQUENCE.extend([503, 503, 200])
    m = ProviderCallMetrics()
    m.set_provider_name("openai")
    client = _client(base)

    result = retry_with_metrics(
        lambda: _call(client),
        max_retries=3,
        metrics=m,
        initial_delay=0.001,
        max_delay=0.002,
    )
    assert result.choices[0].message.content == "ok"
    assert _hits["n"] == 3, "two 503s then a 200 — exactly three server hits"


def test_insufficient_quota_hard_stops_without_retry(mock_server) -> None:
    """THE ANTHROPIC-CAP SHAPE, in OpenAI's clothing: a 429 whose body is insufficient_quota. It must
    hard-stop as terminal, NOT retry — the failure that burned an hour before this existed."""
    _srv, base = mock_server
    _SEQUENCE.append(429)
    _BODIES[429] = {
        "error": {
            "message": "You exceeded your current quota",
            "type": "insufficient_quota",
            "code": "insufficient_quota",
        }
    }
    m = ProviderCallMetrics()
    m.set_provider_name("openai")
    client = _client(base)

    with pytest.raises(LLMTerminalError, match="no budget/credit"):
        retry_with_metrics(
            lambda: _call(client),
            max_retries=5,
            metrics=m,
            initial_delay=0.001,
            max_delay=0.002,
        )
    assert _hits["n"] == 1, "terminal error must NOT retry — exactly one server hit"


def test_payment_required_402_hard_stops(mock_server) -> None:
    """DeepSeek 'Insufficient Balance' arrives as 402 — terminal, one hit, no retry."""
    _srv, base = mock_server
    _SEQUENCE.append(402)
    _BODIES[402] = {"error": {"message": "Insufficient Balance", "type": "payment"}}
    m = ProviderCallMetrics()
    m.set_provider_name("deepseek")
    client = _client(base)

    with pytest.raises(LLMTerminalError):
        retry_with_metrics(
            lambda: _call(client),
            max_retries=5,
            metrics=m,
            initial_delay=0.001,
            max_delay=0.002,
        )
    assert _hits["n"] == 1


def test_persistent_overload_exhausts_retries_then_raises(mock_server) -> None:
    """If it NEVER recovers, we exhaust the (bounded) retries and surface the error — we do not loop
    forever. Bounded failure, not a hang."""
    _srv, base = mock_server
    _SEQUENCE.append(503)  # always 503
    m = ProviderCallMetrics()
    m.set_provider_name("openai")
    client = _client(base)

    with pytest.raises(Exception) as ei:
        retry_with_metrics(
            lambda: _call(client),
            max_retries=2,
            metrics=m,
            initial_delay=0.001,
            max_delay=0.002,
        )
    assert not isinstance(ei.value, LLMTerminalError), "503 is overload, not terminal"
    assert _hits["n"] == 3, "initial + 2 retries = 3 hits, then give up"


# --------------------------------------------------------------------------- #
# ADR-119: the LLM circuit breaker end-to-end through the real SDK + mock server #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _reset_llm_breaker():
    """Every breaker test starts and ends with clean per-provider state (module singleton)."""
    reset_for_test()
    yield
    reset_for_test()


def _breaker_metrics(provider: str = "openai") -> ProviderCallMetrics:
    m = ProviderCallMetrics()
    m.set_provider_name(provider)
    return m


def test_breaker_failover_trips_waits_then_recovers(mock_server) -> None:
    """ADR-119 failover: a 503 burst trips the breaker; it waits the (short) cooldown, then the
    endpoint serves us. The breaker smooths the burst — the call still succeeds, no abort."""
    _srv, base = mock_server
    _SEQUENCE.extend([503, 503, 200])
    cfg = LLMCircuitBreakerConfig(
        enabled=True,
        failure_threshold=2,
        window_seconds=60,
        cooldown_seconds=0.05,  # short so the cooldown wait is quick
        failure_strategy="failover",
    )
    client = _client(base)
    result = retry_with_metrics(
        lambda: _call(client),
        max_retries=3,
        metrics=_breaker_metrics(),
        initial_delay=0.001,
        max_delay=0.002,
        circuit_breaker_config=cfg,
    )
    assert result.choices[0].message.content == "ok"
    assert stats("openai")["trips_total"] == 1  # tripped once, then rode through — did not abort


def test_breaker_hold_aborts_on_sustained_outage(mock_server) -> None:
    """ADR-119 hold: no fallover, so once the outage exceeds the hold budget the breaker raises
    ResilienceFuseOpenError to abort the batch — bounded, not a hang, and NOT terminal."""
    _srv, base = mock_server
    _SEQUENCE.append(503)  # never recovers
    cfg = LLMCircuitBreakerConfig(
        enabled=True,
        failure_threshold=2,
        window_seconds=60,
        cooldown_seconds=60,  # stays in cooldown so the hold-abort branch is reached
        failure_strategy="hold",
        on_open_max_wait_sec=0.0,  # any sustained hold aborts immediately
    )
    client = _client(base)
    with pytest.raises(ResilienceFuseOpenError):
        retry_with_metrics(
            lambda: _call(client),
            max_retries=5,
            metrics=_breaker_metrics(),
            initial_delay=0.001,
            max_delay=0.002,
            circuit_breaker_config=cfg,
        )
    # 2 hits trip the breaker (threshold=2); the 3rd attempt's pre-call wait aborts — no 3rd hit.
    assert _hits["n"] == 2


def test_breaker_hold_within_budget_waits_not_aborts(mock_server) -> None:
    """ADR-119 hold, transient case: while the outage is still within the hold budget the breaker
    waits (like failover) rather than aborting — recovery on the next call succeeds."""
    _srv, base = mock_server
    _SEQUENCE.extend([503, 503, 200])
    cfg = LLMCircuitBreakerConfig(
        enabled=True,
        failure_threshold=2,
        window_seconds=60,
        cooldown_seconds=0.05,
        failure_strategy="hold",
        on_open_max_wait_sec=100.0,  # generous budget — this blip is well within it
    )
    client = _client(base)
    result = retry_with_metrics(
        lambda: _call(client),
        max_retries=3,
        metrics=_breaker_metrics(),
        initial_delay=0.001,
        max_delay=0.002,
        circuit_breaker_config=cfg,
    )
    assert result.choices[0].message.content == "ok"  # waited through, did not abort


def test_breaker_trip_emits_operator_alert(mock_server) -> None:
    """Every breaker trip pages Sentry (guarded capture_message) — proven by spying the hook."""
    _srv, base = mock_server
    _SEQUENCE.extend([503, 503, 200])
    cfg = LLMCircuitBreakerConfig(
        enabled=True,
        failure_threshold=2,
        window_seconds=60,
        cooldown_seconds=0.02,
        failure_strategy="failover",
    )
    client = _client(base)
    with patch.object(_llm_cb, "_emit_llm_breaker_trip_alert") as alert:
        retry_with_metrics(
            lambda: _call(client),
            max_retries=3,
            metrics=_breaker_metrics(),
            initial_delay=0.001,
            max_delay=0.002,
            circuit_breaker_config=cfg,
        )
    alert.assert_called_once()  # the single trip paged the operator
