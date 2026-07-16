"""End-to-end TLS behavior for #1130 — real sockets, real factory, mock TLS server.

Covers what unit tests can't: custom CA trust, verify=false override, mTLS
client-cert exchange. All bindings ephemeral; certs minted per session by the
``self_signed_ca`` fixture.
"""

from __future__ import annotations

from typing import Iterator

import httpx
import pytest

from podcast_scraper.net import create_client
from podcast_scraper.net.outbound_config import OutboundConfig, TlsConfig
from podcast_scraper.net.outbound_registry import _reset_registry_for_tests, get_registry

pytestmark = [pytest.mark.integration, pytest.mark.integration_http]


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    _reset_registry_for_tests()
    yield
    _reset_registry_for_tests()


def test_default_trust_rejects_self_signed(mock_tls_server) -> None:
    # Registry defaults: verify=True, system CA store.
    with create_client(subsystem="test") as client:
        with pytest.raises(httpx.ConnectError):
            client.get(mock_tls_server.base_url + "/x", timeout=3.0)


def test_custom_ca_bundle_accepts_self_signed(mock_tls_server, self_signed_ca) -> None:
    get_registry().swap(
        OutboundConfig(tls=TlsConfig(verify=True, ca_bundle=str(self_signed_ca.ca_pem)))
    )
    with create_client(subsystem="test") as client:
        res = client.get(mock_tls_server.base_url + "/via-ca", timeout=3.0)
    assert res.status_code == 200
    assert any(r["path"] == "/via-ca" for r in mock_tls_server.seen)


def test_verify_false_reaches_self_signed(mock_tls_server) -> None:
    get_registry().swap(OutboundConfig(tls=TlsConfig(verify=False)))
    with create_client(subsystem="test") as client:
        res = client.get(mock_tls_server.base_url + "/insecure", timeout=3.0)
    assert res.status_code == 200


def test_mtls_requires_client_cert(mock_mtls_server, self_signed_ca) -> None:
    # No client cert but trusting the server CA → server closes on TLS handshake.
    get_registry().swap(
        OutboundConfig(tls=TlsConfig(verify=True, ca_bundle=str(self_signed_ca.ca_pem)))
    )
    with create_client(subsystem="test") as client:
        # A misconfigured client (no cert) triggers ConnectError before the app
        # ever sees the request.
        with pytest.raises((httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)):
            client.get(mock_mtls_server.base_url + "/mtls", timeout=3.0)
    # And with client cert + key configured → the request lands.
    get_registry().swap(
        OutboundConfig(
            tls=TlsConfig(
                verify=True,
                ca_bundle=str(self_signed_ca.ca_pem),
                client_cert=str(self_signed_ca.client_cert),
                client_key=str(self_signed_ca.client_key),
            )
        )
    )
    with create_client(subsystem="test") as client:
        res = client.get(mock_mtls_server.base_url + "/mtls", timeout=3.0)
    assert res.status_code == 200
    assert any(r["path"] == "/mtls" for r in mock_mtls_server.seen)
