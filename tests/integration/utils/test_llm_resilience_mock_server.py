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

import pytest

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
