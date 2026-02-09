"""Fake Gemini SDK client for E2E testing.

This module provides a fake Gemini client that routes SDK calls to the E2E
mock server via HTTP, similar to how OpenAI tests work. This allows Gemini
tests to use real HTTP requests to mock endpoints instead of Python-level mocking.

Usage:
    In E2E tests, monkeypatch the Gemini SDK to use this fake client:

    ```python
    from tests.fixtures.mock_server.gemini_mock_client import FakeGeminiClient

    # In conftest.py or test setup
    monkeypatch.setattr("google.genai.GenerativeModel", FakeGeminiClient)
    ```
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class FakeGeminiResponse:
    """Fake response object that mimics Gemini SDK response structure."""

    def __init__(self, text: str):
        """Initialize fake response.

        Args:
            text: Response text content
        """
        self.text = text


class FakeGenerativeModel:
    """Fake GenerativeModel that routes calls to E2E mock server.

    This class mimics the interface of google.genai.GenerativeModel
    but makes HTTP requests to the E2E mock server instead of Google's API.
    """

    def __init__(
        self,
        model_name: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize fake GenerativeModel.

        Args:
            model_name: Model name (e.g., "gemini-2.0-flash")
            api_key: API key (required for compatibility, but not used with mock server)
            base_url: Base URL for E2E mock server (e.g., "http://127.0.0.1:8000/v1beta")
            system_instruction: System instruction/prompt
                (accepted for compatibility, stored but not used)
            **kwargs: Additional arguments (ignored for mock)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url or "http://127.0.0.1:8000/v1beta"
        self.system_instruction = system_instruction
        logger.debug("FakeGenerativeModel initialized: model=%s, base_url=%s", model_name, base_url)

    def generate_content(
        self,
        contents: Any,
        *,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> FakeGeminiResponse:
        """Generate content by calling E2E mock server.

        Args:
            contents: Content to send (can be text, audio, or multimodal)
            generation_config: Generation configuration (temperature, max_tokens, etc.)
            **kwargs: Additional arguments (ignored for mock)

        Returns:
            FakeGeminiResponse with text content from mock server
        """
        # Build request payload
        request_data: Dict[str, Any] = {
            "contents": self._normalize_contents(contents),
        }

        if generation_config:
            request_data["generationConfig"] = generation_config

        # Build URL
        url = f"{self.base_url}/models/{self.model_name}:generateContent"

        logger.debug("FakeGenerativeModel.generate_content: POST %s", url)

        # Make HTTP request to E2E mock server
        try:
            response = requests.post(
                url,
                json=request_data,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            # Parse response
            data = response.json()

            # Extract text from response (mimics Gemini SDK response structure)
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and "text" in parts[0]:
                        text = parts[0]["text"]
                        return FakeGeminiResponse(text)

            # Fallback: return empty response
            logger.warning("Unexpected response format from E2E server: %s", data)
            return FakeGeminiResponse("")

        except requests.exceptions.RequestException as e:
            logger.error("Failed to call E2E mock server: %s", e)
            raise RuntimeError(f"E2E mock server request failed: {e}") from e

    def _normalize_contents(self, contents: Any) -> List[Dict[str, Any]]:
        """Normalize contents to API format.

        Args:
            contents: Contents in various formats (string, list, dict, etc.)

        Returns:
            Normalized contents list
        """
        if isinstance(contents, str):
            return [{"parts": [{"text": contents}]}]
        elif isinstance(contents, list):
            normalized = []
            for item in contents:
                if isinstance(item, str):
                    normalized.append({"parts": [{"text": item}]})
                elif isinstance(item, dict):
                    normalized.append(item)
                elif isinstance(item, list):
                    # Handle nested lists (multimodal)
                    parts = []
                    for part in item:
                        if isinstance(part, str):
                            parts.append({"text": part})
                        elif isinstance(part, dict):
                            parts.append(part)
                    normalized.append({"parts": parts})
            return normalized
        elif isinstance(contents, dict):
            return [contents]
        else:
            # Fallback: convert to string
            return [{"parts": [{"text": str(contents)}]}]

    def _has_audio_content(self, contents: Any) -> bool:
        """Check if contents contain audio data.

        Args:
            contents: Contents to check

        Returns:
            True if audio content is detected
        """
        if isinstance(contents, list):
            for item in contents:
                if isinstance(item, list):
                    for part in item:
                        if isinstance(part, dict) and part.get("mime_type", "").startswith(
                            "audio/"
                        ):
                            return True
                elif isinstance(item, dict):
                    if item.get("mime_type", "").startswith("audio/"):
                        return True
        elif isinstance(contents, dict):
            if contents.get("mime_type", "").startswith("audio/"):
                return True
        return False


def create_fake_gemini_client(base_url: str) -> type:
    """Create a fake Gemini client class bound to a specific base URL.

    Args:
        base_url: Base URL for E2E mock server

    Returns:
        Fake client class that can be used to replace GenerativeModel
    """

    class FakeGenerativeModelWithBase(FakeGenerativeModel):
        """Fake GenerativeModel with pre-configured base URL."""

        def __init__(self, model_name: str, **kwargs: Any):
            """Initialize with pre-configured base URL."""
            # Remove base_url from kwargs if present (use the one from closure)
            kwargs.pop("base_url", None)
            super().__init__(model_name, base_url=base_url, **kwargs)

    return FakeGenerativeModelWithBase
