"""Localhost mock servers for outbound-network integration tests (#1129 / #1130).

- ``mock_target_server`` — a plain HTTP echo server; returns 200 with a JSON body
  describing what it saw. Doubles as the "origin" for proxy tests and the plain
  end for baseline requests.

- ``mock_http_proxy`` — a threaded HTTP forward proxy that records every request
  observed. Supports optional ``Proxy-Authorization: Basic`` challenge (returns
  407 without credentials). Handles absolute-URI GET/HEAD/POST (plain HTTP
  through proxy). CONNECT is NOT implemented — we don't need HTTPS-through-proxy
  for these tests (they exercise config threading, not tunneling).

- ``self_signed_ca`` — a session-scoped fixture that mints a self-signed CA +
  server leaf using stdlib ``ssl`` + a small helper that shells out to
  ``openssl`` (present on macOS + all Linux CI images). Cert/key/CA paths are
  returned as ``pathlib.Path`` so :class:`TlsConfig` can point at them.

- ``mock_tls_server`` — wraps the target server in an ``ssl.SSLContext``
  presenting the fixture's server cert. Optional ``require_client_cert=True``
  for mTLS tests.

All servers bind ``127.0.0.1:0`` (ephemeral) so multiple workers don't collide.
Each fixture tears down cleanly at test end.
"""

from __future__ import annotations

import base64
import http.server
import socket
import ssl
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pytest

# --- target server ------------------------------------------------------------


@dataclass
class MockTargetServer:
    host: str
    port: int
    scheme: str = "http"
    seen: list[dict[str, str]] = field(default_factory=list)
    shutdown: threading.Event = field(default_factory=threading.Event)

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"


def _make_target_handler(seen: list[dict[str, str]]) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        # Silence stdout noise during tests.
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

        def _record(self) -> None:
            seen.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "host": self.headers.get("Host", ""),
                    "subsystem": self.headers.get("X-Outbound-Subsystem", ""),
                }
            )

        def do_GET(self) -> None:  # noqa: N802
            self._record()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"target-ok")

        def do_HEAD(self) -> None:  # noqa: N802
            self._record()
            self.send_response(200)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            self._record()
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length:
                self.rfile.read(length)
            self.send_response(200)
            self.end_headers()

    return Handler


def _start_target(
    ssl_ctx: ssl.SSLContext | None = None,
) -> tuple[MockTargetServer, http.server.ThreadingHTTPServer, threading.Thread]:
    seen: list[dict[str, str]] = []
    handler = _make_target_handler(seen)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    if ssl_ctx is not None:
        server.socket = ssl_ctx.wrap_socket(server.socket, server_side=True)
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = MockTargetServer(
        host=str(host),
        port=int(port),
        scheme=("https" if ssl_ctx is not None else "http"),
        seen=seen,
    )
    return target, server, thread


@pytest.fixture
def mock_target_server() -> Iterator[MockTargetServer]:
    target, server, _thread = _start_target()
    try:
        yield target
    finally:
        server.shutdown()
        server.server_close()


# --- forward HTTP proxy -------------------------------------------------------


@dataclass
class MockHttpProxy:
    host: str
    port: int
    seen: list[dict[str, str]] = field(default_factory=list)
    require_auth: str | None = None  # e.g. "alice:hunter2"

    @property
    def url(self) -> str:
        creds = f"{self.require_auth}@" if self.require_auth else ""
        return f"http://{creds}{self.host}:{self.port}"


def _make_proxy_handler(seen: list[dict[str, str]], require_auth: str | None):
    class ProxyHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

        def _needs_auth(self) -> bool:
            if not require_auth:
                return False
            got = self.headers.get("Proxy-Authorization", "")
            expected = "Basic " + base64.b64encode(require_auth.encode()).decode()
            return got != expected

        def _forward(self) -> None:
            if self._needs_auth():
                self.send_response(407)
                self.send_header("Proxy-Authenticate", 'Basic realm="test"')
                self.end_headers()
                return
            # Absolute URI path from client (proxy request semantics).
            target_url = self.path
            if not target_url.startswith("http://"):
                self.send_error(400, "proxy expected absolute URI")
                return
            seen.append(
                {
                    "method": self.command,
                    "url": target_url,
                    "subsystem": self.headers.get("X-Outbound-Subsystem", ""),
                }
            )
            # Extract host:port + path from absolute URI.
            after_scheme = target_url[len("http://") :]
            slash = after_scheme.find("/")
            if slash == -1:
                netloc, rel_path = after_scheme, "/"
            else:
                netloc, rel_path = after_scheme[:slash], after_scheme[slash:]
            host, _, port_str = netloc.partition(":")
            port = int(port_str) if port_str else 80
            body = b""
            if self.command == "POST":
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = self.rfile.read(length) if length else b""
            with socket.create_connection((host, port), timeout=5) as sock:
                headers_out = [
                    f"{self.command} {rel_path} HTTP/1.1",
                    f"Host: {netloc}",
                    "Connection: close",
                ]
                if body:
                    headers_out.append(f"Content-Length: {len(body)}")
                sock.sendall(("\r\n".join(headers_out) + "\r\n\r\n").encode())
                if body:
                    sock.sendall(body)
                resp = b""
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    resp += chunk
            self.wfile.write(resp)

        do_GET = _forward  # type: ignore[assignment]
        do_HEAD = _forward  # type: ignore[assignment]
        do_POST = _forward  # type: ignore[assignment]

    return ProxyHandler


@pytest.fixture
def mock_http_proxy() -> Iterator[MockHttpProxy]:
    seen: list[dict[str, str]] = []
    handler = _make_proxy_handler(seen, require_auth=None)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    proxy = MockHttpProxy(host=str(host), port=int(port), seen=seen)
    try:
        yield proxy
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def mock_http_proxy_with_auth() -> Iterator[MockHttpProxy]:
    creds = "alice:hunter2"
    seen: list[dict[str, str]] = []
    handler = _make_proxy_handler(seen, require_auth=creds)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    proxy = MockHttpProxy(host=str(host), port=int(port), seen=seen, require_auth=creds)
    try:
        yield proxy
    finally:
        server.shutdown()
        server.server_close()


# --- TLS fixtures -------------------------------------------------------------


@dataclass
class SelfSignedTls:
    ca_pem: Path
    server_cert: Path
    server_key: Path
    client_cert: Path
    client_key: Path


def _openssl(*args: str, cwd: Path) -> None:
    subprocess.run(["openssl", *args], cwd=cwd, check=True, capture_output=True)


def _mint_self_signed(tmp: Path) -> SelfSignedTls:
    # CA
    _openssl("genrsa", "-out", "ca.key", "2048", cwd=tmp)
    _openssl(
        "req",
        "-x509",
        "-new",
        "-nodes",
        "-key",
        "ca.key",
        "-sha256",
        "-days",
        "1",
        "-subj",
        "/CN=test-ca",
        "-out",
        "ca.pem",
        cwd=tmp,
    )
    # Server
    _openssl("genrsa", "-out", "server.key", "2048", cwd=tmp)
    _openssl(
        "req",
        "-new",
        "-key",
        "server.key",
        "-subj",
        "/CN=127.0.0.1",
        "-out",
        "server.csr",
        cwd=tmp,
    )
    ext = tmp / "server.ext"
    ext.write_text("subjectAltName=IP:127.0.0.1,DNS:localhost\n", encoding="utf-8")
    _openssl(
        "x509",
        "-req",
        "-in",
        "server.csr",
        "-CA",
        "ca.pem",
        "-CAkey",
        "ca.key",
        "-CAcreateserial",
        "-out",
        "server.pem",
        "-days",
        "1",
        "-sha256",
        "-extfile",
        "server.ext",
        cwd=tmp,
    )
    # Client
    _openssl("genrsa", "-out", "client.key", "2048", cwd=tmp)
    _openssl(
        "req",
        "-new",
        "-key",
        "client.key",
        "-subj",
        "/CN=test-client",
        "-out",
        "client.csr",
        cwd=tmp,
    )
    _openssl(
        "x509",
        "-req",
        "-in",
        "client.csr",
        "-CA",
        "ca.pem",
        "-CAkey",
        "ca.key",
        "-CAcreateserial",
        "-out",
        "client.pem",
        "-days",
        "1",
        "-sha256",
        cwd=tmp,
    )
    return SelfSignedTls(
        ca_pem=tmp / "ca.pem",
        server_cert=tmp / "server.pem",
        server_key=tmp / "server.key",
        client_cert=tmp / "client.pem",
        client_key=tmp / "client.key",
    )


@pytest.fixture(scope="session")
def self_signed_ca(tmp_path_factory: pytest.TempPathFactory) -> SelfSignedTls:
    if (
        subprocess.call(
            ["openssl", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        != 0
    ):
        pytest.skip("openssl not available on PATH")
    tmp = tmp_path_factory.mktemp("tls-fixtures")
    return _mint_self_signed(tmp)


@pytest.fixture
def mock_tls_server(self_signed_ca: SelfSignedTls) -> Iterator[MockTargetServer]:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # Reject TLS 1.0/1.1 in the mock server — CodeQL py/insecure-protocol.
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile=self_signed_ca.server_cert, keyfile=self_signed_ca.server_key)
    target, server, _thread = _start_target(ssl_ctx=ctx)
    try:
        yield target
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def mock_mtls_server(self_signed_ca: SelfSignedTls) -> Iterator[MockTargetServer]:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # Reject TLS 1.0/1.1 in the mock server — CodeQL py/insecure-protocol.
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile=self_signed_ca.server_cert, keyfile=self_signed_ca.server_key)
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_verify_locations(cafile=self_signed_ca.ca_pem)
    target, server, _thread = _start_target(ssl_ctx=ctx)
    try:
        yield target
    finally:
        server.shutdown()
        server.server_close()
