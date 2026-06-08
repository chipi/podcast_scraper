"""Tiny 503-on-demand HTTP server for #814 Stage B chaos tests.

Binds a local port and answers everything with HTTP 503. Used by the Stage B
chaos make targets to simulate "DGX down" or "DGX + cloud both down" without
touching the real services.

Usage from a shell:

    python scripts/tools/chaos_proxy.py --port 18443
    # Then, in another terminal:
    curl http://127.0.0.1:18443/anything    # → 503

How the chaos tests use it:

- ``make preprod-chaos-dgx-down`` points ``dgx_tailnet_host`` at this server
  (overriding DGX_TAILNET_FQDN). The transcribe provider's health check hits
  ``GET /v1/models`` → 503 → falls back to OpenAI cloud Whisper. Episode
  completes; ``dgx_fallback_active`` breadcrumb fires.
- ``make preprod-chaos-both-down`` additionally points ``OPENAI_BASE_URL`` at
  this server. Cloud Whisper also gets 503. Pipeline aborts with operator-
  visible error and non-zero exit. No half-baked output.

Why not a real forward proxy: the chaos tests only need "block this endpoint",
not "block one path but forward another." A forward proxy adds Host/CONNECT
tunneling complexity (HTTPS) we don't need. If a future test wants partial
denial, replace this with a real proxy then.

This script makes no outbound calls and writes no files. Safe to leave running.
"""

from __future__ import annotations

import argparse
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger("chaos_proxy")


class _ChaosHandler(BaseHTTPRequestHandler):
    """Answers every request with 503."""

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003 — stdlib override
        logger.info("%s %s → 503", self.command, self.path)

    def _deny(self) -> None:
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Chaos-Proxy", "denied")
        self.end_headers()
        # Body shape mimics what the OpenAI / Ollama clients tolerate gracefully.
        self.wfile.write(
            b'{"error": {"message": "chaos proxy denial", ' b'"type": "service_unavailable"}}'
        )

    def do_GET(self) -> None:
        self._deny()

    def do_POST(self) -> None:
        self._deny()

    def do_PUT(self) -> None:
        self._deny()

    def do_DELETE(self) -> None:
        self._deny()

    def do_HEAD(self) -> None:
        self._deny()

    def do_OPTIONS(self) -> None:
        self._deny()

    def do_PATCH(self) -> None:
        self._deny()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tiny 503-everything HTTP server for chaos testing (#814 Stage B)."
    )
    parser.add_argument(
        "--port", type=int, default=18443, help="local port to bind (default 18443)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("chaos_proxy denying everything on 127.0.0.1:%d", args.port)

    try:
        server = HTTPServer(("127.0.0.1", args.port), _ChaosHandler)
    except OSError as exc:
        logger.error("Failed to bind 127.0.0.1:%d: %s", args.port, exc)
        return 1

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down chaos_proxy")
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
