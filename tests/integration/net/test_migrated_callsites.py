"""End-to-end: migrated call sites route through the outbound factory (#1129 / #1130).

The unit tests for each migrated module were preserved (they mock at their own
layer). This module proves the SEAM: when the registry is configured, a real
call from a migrated production subsystem traverses the mock proxy / mock TLS
server. If the migration accidentally bypassed the factory, these tests fail.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from podcast_scraper.net.outbound_config import OutboundConfig, ProxyConfig, TlsConfig
from podcast_scraper.net.outbound_registry import _reset_registry_for_tests, get_registry

pytestmark = [pytest.mark.integration, pytest.mark.integration_http]

# Placeholder credential — the mock target does not authenticate; SDKs just require
# a non-empty string so their init passes. Kept separate from any `api_key=` shape
# so pre-commit secret scanners don't flag the literal.
_SDK_INIT_PLACEHOLDER = "fixture-value-not-a-key"


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    _reset_registry_for_tests()
    yield
    _reset_registry_for_tests()


def test_custom_ca_bundle_reaches_migrated_callsite(mock_tls_server, self_signed_ca) -> None:
    """A CA-bundle registry lets `audio_bridge._head_request` verify a self-signed target."""
    from podcast_scraper.server.app_audio_bridge import _head_request

    get_registry().swap(
        OutboundConfig(tls=TlsConfig(verify=True, ca_bundle=str(self_signed_ca.ca_pem)))
    )
    status, _final, _headers = _head_request(mock_tls_server.base_url + "/audio.mp3", 5.0)
    assert status == 200


# --- chunk 3e seams ---------------------------------------------------------


def test_openai_sdk_http_client_routes_through_proxy(mock_http_proxy, mock_target_server) -> None:
    """The OpenAI SDK's `http_client=sdk_http_client(subsystem="llm_openai")`
    actually threads through the outbound factory (#1142 chunk 3e).

    Chunk 3e's SDK migrations went through `sdk_http_client` — a thin
    wrapper over `create_client` — passed as `http_client=` to the OpenAI
    constructor. This test proves the SDK's transport actually uses that
    client by pointing an OpenAI-shape client at the mock target and
    asserting the mock proxy sees the hop. The OpenAI SDK will fail to
    parse the mock target's plain-text response — that's fine: we only
    care about the proxy hop record, which is enough to prove the seam.
    """
    from openai import OpenAI

    from podcast_scraper.net import sdk_http_client

    get_registry().swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url=mock_http_proxy.url)))
    http_client = sdk_http_client(subsystem="llm_openai", timeout=5.0)
    client = OpenAI(
        api_key="sk-test-not-real",
        base_url=mock_target_server.base_url + "/v1",
        http_client=http_client,
    )
    try:
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    except Exception:
        # Mock target returns "target-ok" text — SDK parse fails. Fine.
        pass
    assert any(
        hop["subsystem"] == "llm_openai"
        and "/v1/chat/completions" in hop["url"]
        and hop["method"] == "POST"
        for hop in mock_http_proxy.seen
    )


def test_anthropic_sdk_http_client_routes_through_proxy(
    mock_http_proxy, mock_target_server
) -> None:
    """Anthropic SDK honors `http_client=sdk_http_client(...)`."""
    from anthropic import Anthropic

    from podcast_scraper.net import sdk_http_client

    get_registry().swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url=mock_http_proxy.url)))
    http_client = sdk_http_client(subsystem="llm_anthropic", timeout=5.0)
    client = Anthropic(
        api_key=_SDK_INIT_PLACEHOLDER,
        base_url=mock_target_server.base_url,
        http_client=http_client,
    )
    try:
        client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
    except Exception:
        # Mock target returns "target-ok" — SDK parse fails. Seam is what we test.
        pass
    assert any(
        hop["subsystem"] == "llm_anthropic"
        and "/v1/messages" in hop["url"]
        and hop["method"] == "POST"
        for hop in mock_http_proxy.seen
    )


def test_deepseek_sdk_http_client_routes_through_proxy(mock_http_proxy, mock_target_server) -> None:
    """DeepSeek uses OpenAI SDK shape — its ``http_client=`` must route through the proxy."""
    from openai import OpenAI

    from podcast_scraper.net import sdk_http_client

    get_registry().swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url=mock_http_proxy.url)))
    http_client = sdk_http_client(subsystem="llm_deepseek", timeout=5.0)
    client = OpenAI(
        api_key=_SDK_INIT_PLACEHOLDER,
        base_url=mock_target_server.base_url + "/v1",
        http_client=http_client,
    )
    try:
        client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "hi"}],
        )
    except Exception:
        pass
    assert any(
        hop["subsystem"] == "llm_deepseek"
        and "/v1/chat/completions" in hop["url"]
        and hop["method"] == "POST"
        for hop in mock_http_proxy.seen
    )


def test_grok_sdk_http_client_routes_through_proxy(mock_http_proxy, mock_target_server) -> None:
    """Grok uses OpenAI SDK shape — its ``http_client=`` must route through the proxy."""
    from openai import OpenAI

    from podcast_scraper.net import sdk_http_client

    get_registry().swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url=mock_http_proxy.url)))
    http_client = sdk_http_client(subsystem="llm_grok", timeout=5.0)
    client = OpenAI(
        api_key=_SDK_INIT_PLACEHOLDER,
        base_url=mock_target_server.base_url + "/v1",
        http_client=http_client,
    )
    try:
        client.chat.completions.create(
            model="grok-2",
            messages=[{"role": "user", "content": "hi"}],
        )
    except Exception:
        pass
    assert any(
        hop["subsystem"] == "llm_grok"
        and "/v1/chat/completions" in hop["url"]
        and hop["method"] == "POST"
        for hop in mock_http_proxy.seen
    )


def test_ollama_sdk_http_client_routes_through_proxy(mock_http_proxy, mock_target_server) -> None:
    """Ollama LLM provider uses OpenAI SDK shape — its ``http_client=`` must route through the proxy."""
    from openai import OpenAI

    from podcast_scraper.net import sdk_http_client

    get_registry().swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url=mock_http_proxy.url)))
    http_client = sdk_http_client(subsystem="llm_ollama", timeout=5.0)
    client = OpenAI(
        api_key=_SDK_INIT_PLACEHOLDER,
        base_url=mock_target_server.base_url + "/v1",
        http_client=http_client,
    )
    try:
        client.chat.completions.create(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
        )
    except Exception:
        pass
    assert any(
        hop["subsystem"] == "llm_ollama"
        and "/v1/chat/completions" in hop["url"]
        and hop["method"] == "POST"
        for hop in mock_http_proxy.seen
    )


def test_mistral_sdk_http_client_routes_through_proxy(mock_http_proxy, mock_target_server) -> None:
    """Mistral SDK honors the factory client under either the 2.x ``client=`` kwarg
    or the 1.x fallback path (both exercised by ``_build_mistral_client``).

    Uses the provider's own ``_load_mistral_sdk`` shim so the seam test tracks
    whichever mistralai layout is installed (2.x under ``mistralai.client.sdk``,
    1.x under ``mistralai``).
    """
    from podcast_scraper.net import sdk_http_client
    from podcast_scraper.providers.mistral.mistral_provider import Mistral

    if Mistral is None:
        pytest.skip("mistralai SDK not installed in this environment")

    get_registry().swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url=mock_http_proxy.url)))
    http_client = sdk_http_client(subsystem="llm_mistral", timeout=5.0)
    server_url = mock_target_server.base_url
    try:
        client = Mistral(api_key=_SDK_INIT_PLACEHOLDER, server_url=server_url, client=http_client)
    except TypeError as exc:
        if "client" not in str(exc):
            raise
        pytest.skip("Mistral SDK 1.x does not accept a factory http client")
    try:
        client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "hi"}],
        )
    except Exception:
        pass
    assert any(
        hop["subsystem"] == "llm_mistral" and hop["method"] == "POST"
        for hop in mock_http_proxy.seen
    )


# --- hardened_http_client seams (2026-07-09 hardening pass) ----------------
#
# ``hardened_http_client`` used to bypass the outbound registry entirely,
# leaving DGX Whisper + diarize multipart POSTs on plain httpx — no proxy,
# no CA bundle, no mTLS, no ``verify=False``. Both seams below prove it now
# delegates through ``create_client``.


def test_hardened_http_client_routes_through_configured_proxy(
    mock_http_proxy, mock_target_server
) -> None:
    """DGX Whisper / diarize multipart POSTs honor UI-configured proxy."""
    from podcast_scraper.providers.resilience.sockets import hardened_http_client

    get_registry().swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url=mock_http_proxy.url)))
    with hardened_http_client(10.0, subsystem="dgx_whisper") as client:
        resp = client.post(
            mock_target_server.base_url + "/v1/audio/transcriptions",
            data={"model": "faster-whisper-medium"},
        )
    assert resp.status_code == 200
    assert any(
        hop["subsystem"] == "dgx_whisper"
        and "/v1/audio/transcriptions" in hop["url"]
        and hop["method"] == "POST"
        for hop in mock_http_proxy.seen
    )


def test_hardened_http_client_uses_custom_ca_bundle(mock_tls_server, self_signed_ca) -> None:
    """DGX Whisper / diarize multipart POSTs verify a self-signed cert via UI-configured CA."""
    from podcast_scraper.providers.resilience.sockets import hardened_http_client

    get_registry().swap(
        OutboundConfig(tls=TlsConfig(verify=True, ca_bundle=str(self_signed_ca.ca_pem)))
    )
    with hardened_http_client(10.0, subsystem="dgx_diarize") as client:
        resp = client.get(mock_tls_server.base_url + "/audio.mp3")
    assert resp.status_code == 200


def test_hardened_http_client_honors_verify_false(mock_tls_server) -> None:
    """A ``tls.verify=False`` registry lets hardened multipart POSTs hit a self-signed target.

    This is the case env-mirror does NOT cover — before task 21's fix the
    DGX multipart POST would silently reject the self-signed cert even
    though the operator had explicitly disabled verification.
    """
    from podcast_scraper.providers.resilience.sockets import hardened_http_client

    get_registry().swap(OutboundConfig(tls=TlsConfig(verify=False)))
    with hardened_http_client(10.0, subsystem="dgx_diarize") as client:
        resp = client.get(mock_tls_server.base_url + "/audio.mp3")
    assert resp.status_code == 200
