"""End-to-end proxy behavior for #1129 — real sockets, real factory, mock proxy.

Covers what unit tests can't: proxy traversal, no_proxy bypass, basic-auth
challenge/response, and hot-apply mid-session. Uses the localhost mock proxy
+ target from ``conftest.py``; all bindings are ephemeral so parallel workers
never collide.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from podcast_scraper.net import create_client
from podcast_scraper.net.outbound_config import OutboundConfig, ProxyConfig
from podcast_scraper.net.outbound_registry import _reset_registry_for_tests, get_registry

pytestmark = [pytest.mark.integration, pytest.mark.integration_http]


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    _reset_registry_for_tests()
    yield
    _reset_registry_for_tests()


def test_request_traverses_configured_proxy(mock_http_proxy, mock_target_server) -> None:
    get_registry().swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url=mock_http_proxy.url)))
    with create_client(subsystem="test") as client:
        res = client.get(mock_target_server.base_url + "/via-proxy")
    assert res.status_code == 200
    # The proxy saw the request as an absolute-URI proxy call.
    assert len(mock_http_proxy.seen) == 1
    hop = mock_http_proxy.seen[0]
    assert hop["method"] == "GET"
    assert hop["url"] == mock_target_server.base_url + "/via-proxy"
    assert hop["subsystem"] == "test"
    # The origin saw the forwarded request (proxy's own forward).
    assert any(r["path"] == "/via-proxy" for r in mock_target_server.seen)


def test_no_proxy_pattern_bypasses_proxy(mock_http_proxy, mock_target_server) -> None:
    # `127.0.0.1` in no_proxy should cause the client to go direct.
    get_registry().swap(
        OutboundConfig(
            proxy=ProxyConfig(enabled=True, url=mock_http_proxy.url, no_proxy=("127.0.0.1",))
        )
    )
    with create_client(subsystem="test") as client:
        res = client.get(mock_target_server.base_url + "/direct")
    assert res.status_code == 200
    assert mock_http_proxy.seen == []
    assert any(r["path"] == "/direct" for r in mock_target_server.seen)


def test_disabled_proxy_bypasses_even_when_url_set(mock_http_proxy, mock_target_server) -> None:
    get_registry().swap(OutboundConfig(proxy=ProxyConfig(enabled=False, url=mock_http_proxy.url)))
    with create_client(subsystem="test") as client:
        res = client.get(mock_target_server.base_url + "/off")
    assert res.status_code == 200
    assert mock_http_proxy.seen == []


def test_basic_auth_407_without_creds(mock_http_proxy_with_auth, mock_target_server) -> None:
    # Point at proxy but *without* embedding creds in url → expect 407.
    unauthed_url = f"http://{mock_http_proxy_with_auth.host}:{mock_http_proxy_with_auth.port}"
    get_registry().swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url=unauthed_url)))
    with create_client(subsystem="test") as client:
        res = client.get(mock_target_server.base_url + "/needs-auth")
    assert res.status_code == 407


def test_basic_auth_succeeds_with_userinfo(mock_http_proxy_with_auth, mock_target_server) -> None:
    # URL carries user:pass — httpx forwards it as Proxy-Authorization.
    get_registry().swap(
        OutboundConfig(proxy=ProxyConfig(enabled=True, url=mock_http_proxy_with_auth.url))
    )
    with create_client(subsystem="test") as client:
        res = client.get(mock_target_server.base_url + "/authed")
    assert res.status_code == 200
    assert len(mock_http_proxy_with_auth.seen) == 1


def test_hot_apply_new_client_uses_new_proxy(mock_http_proxy, mock_target_server) -> None:
    # Start with no proxy → direct hit.
    with create_client(subsystem="test") as client:
        client.get(mock_target_server.base_url + "/before")
    assert mock_http_proxy.seen == []
    # Swap in a proxy at runtime — next factory-built client picks it up.
    get_registry().swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url=mock_http_proxy.url)))
    with create_client(subsystem="test") as client:
        client.get(mock_target_server.base_url + "/after")
    assert any(r["url"].endswith("/after") for r in mock_http_proxy.seen)
    # And the "before" request is NOT in the proxy log (proves the swap boundary).
    assert not any(r["url"].endswith("/before") for r in mock_http_proxy.seen)


def test_explicit_proxy_kwarg_would_lose_no_proxy_semantics_so_factory_uses_mounts(
    mock_http_proxy, mock_target_server
) -> None:
    # Regression guard for the design decision documented in outbound_http.py:
    # `httpx.Client(proxy=...)` alone routes everything through the proxy and
    # silently ignores no_proxy. The factory uses `mounts=` instead, so this
    # test asserts that no_proxy bypass works even when a proxy is set.
    get_registry().swap(
        OutboundConfig(
            proxy=ProxyConfig(enabled=True, url=mock_http_proxy.url, no_proxy=("127.0.0.1",))
        )
    )
    with create_client(subsystem="test") as client:
        client.get(mock_target_server.base_url + "/bypass-me")
    assert mock_http_proxy.seen == []
