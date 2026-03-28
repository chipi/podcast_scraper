"""E2E mock: HEAD on fixture URLs (downloader probes size without full GET)."""

from __future__ import annotations

import os
import sys

import httpx
import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper.providers.ollama.ollama_provider import _ollama_native_api_root
from tests.e2e.fixtures.e2e_http_server import E2EHTTPServer


@pytest.mark.integration
@pytest.mark.critical_path
def test_e2e_server_head_audio_returns_200_and_content_length() -> None:
    """HEAD /audio/... must succeed so RSS downloader can read Content-Length."""
    server = E2EHTTPServer()
    server.start()
    try:
        base = server.base_url
        assert base is not None
        url = f"{base.rstrip('/')}/audio/p01_e01.mp3"
        response = httpx.head(url, timeout=5.0)
        response.raise_for_status()
        assert response.headers.get("Content-Length") is not None
        assert response.content == b""
    finally:
        server.stop()


@pytest.mark.integration
@pytest.mark.critical_path
def test_e2e_server_head_transcript_returns_200() -> None:
    server = E2EHTTPServer()
    server.start()
    try:
        base = server.base_url
        assert base is not None
        url = f"{base.rstrip('/')}/transcripts/p01_e01.txt"
        response = httpx.head(url, timeout=5.0)
        response.raise_for_status()
        assert response.content == b""
    finally:
        server.stop()


@pytest.mark.integration
@pytest.mark.critical_path
def test_e2e_server_head_ollama_api_version() -> None:
    server = E2EHTTPServer()
    server.start()
    try:
        assert server.urls is not None
        ollama_base = _ollama_native_api_root(server.urls.ollama_api_base())
        url = f"{ollama_base}/api/version"
        response = httpx.head(url, timeout=5.0)
        response.raise_for_status()
        assert response.headers.get("Content-Length") is not None
        assert response.content == b""
    finally:
        server.stop()
