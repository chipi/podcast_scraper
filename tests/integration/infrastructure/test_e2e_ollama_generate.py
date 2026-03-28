"""E2E mock: Ollama native /api/generate for model warm-up (acceptance + OllamaProvider)."""

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
def test_e2e_server_post_api_generate_returns_200_for_warmup() -> None:
    """OllamaProvider.warmup POSTs to /api/generate; mock must not 404."""
    server = E2EHTTPServer()
    server.start()
    try:
        assert server.urls is not None
        ollama_openai_base = server.urls.ollama_api_base()
        generate_url = _ollama_native_api_root(ollama_openai_base) + "/api/generate"
        response = httpx.post(
            generate_url,
            json={
                "model": "llama3.1:8b",
                "prompt": "ping",
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=5.0,
        )
        response.raise_for_status()
        data = response.json()
        assert data.get("done") is True
        assert data.get("model") == "llama3.1:8b"
        assert "response" in data
    finally:
        server.stop()
