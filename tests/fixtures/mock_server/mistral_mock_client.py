"""Fake Mistral SDK client for E2E and integration testing.

This module provides a fake Mistral client that routes SDK calls to the E2E
mock server via HTTP, similar to how Gemini tests work. This allows Mistral
tests to use real HTTP requests to mock endpoints instead of Python-level mocking.

Usage:
    In E2E or integration tests, monkeypatch the Mistral SDK to use this fake client:

    ```python
    from tests.fixtures.mock_server.mistral_mock_client import create_fake_mistral_client

    # In conftest.py or test setup
    FakeMistral = create_fake_mistral_client(base_url)
    monkeypatch.setattr("mistralai.Mistral", FakeMistral)
    ```
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class FakeMistralTranscriptionResponse:
    """Fake transcription response that mimics Mistral SDK response structure."""

    def __init__(self, text: str):
        """Initialize fake transcription response.

        Args:
            text: Transcribed text content
        """
        self.text = text


class FakeMistralChatResponse:
    """Fake chat response that mimics Mistral SDK response structure."""

    def __init__(self, content: str, usage: Optional[Dict[str, Any]] = None):
        """Initialize fake chat response.

        Args:
            content: Response text content
            usage: Token usage information (optional)
        """
        # Create fake choice object
        fake_message = type("obj", (object,), {"content": content})()
        fake_choice = type("obj", (object,), {"message": fake_message})()
        self.choices = [fake_choice]
        self.usage = type("obj", (object,), usage or {})() if usage else None


class FakeMistralAudioTranscriptions:
    """Fake audio.transcriptions object that routes to E2E server."""

    def __init__(self, base_url: str):
        """Initialize fake audio transcriptions.

        Args:
            base_url: Base URL for E2E mock server
        """
        self.base_url = base_url

    def complete(
        self,
        model: str,
        file: Any,
        *,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> FakeMistralTranscriptionResponse:
        """Complete transcription by calling E2E mock server.

        Args:
            model: Model name (e.g., "voxtral-mini-latest")
            file: File object with file_name and content
            language: Optional language code
            **kwargs: Additional arguments (ignored for mock)

        Returns:
            FakeMistralTranscriptionResponse with transcribed text
        """
        # Build URL
        url = f"{self.base_url}/audio/transcriptions"

        # Build request payload
        # Mistral uses multipart/form-data for file uploads (same as OpenAI)
        file_content = file.content if hasattr(file, "content") else b""
        file_name = file.file_name if hasattr(file, "file_name") else "audio.mp3"

        logger.debug("FakeMistralAudioTranscriptions.complete: POST %s", url)

        # Make HTTP request to E2E mock server with multipart/form-data
        try:
            # Use files parameter for multipart/form-data
            files = {
                "file": (file_name, file_content, "audio/mpeg"),
            }
            data: Dict[str, Any] = {
                "model": model,
            }
            if language:
                data["language"] = language

            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=30,
            )
            response.raise_for_status()

            # Parse response - E2E server returns text directly when response_format="text"
            # or JSON with {"text": "..."} otherwise
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                data = response.json()
                text = data.get("text", "")
            else:
                # Plain text response
                text = response.text

            return FakeMistralTranscriptionResponse(text)

        except requests.exceptions.RequestException as e:
            logger.error("Failed to call E2E mock server for transcription: %s", e)
            raise RuntimeError(f"E2E mock server request failed: {e}") from e


class FakeMistralChat:
    """Fake chat object that routes to E2E server."""

    def __init__(self, base_url: str):
        """Initialize fake chat.

        Args:
            base_url: Base URL for E2E mock server
        """
        self.base_url = base_url

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> FakeMistralChatResponse:
        """Complete chat by calling E2E mock server.

        Args:
            model: Model name (e.g., "mistral-large-latest")
            messages: List of message dicts with "role" and "content"
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments (ignored for mock)

        Returns:
            FakeMistralChatResponse with response content
        """
        # Build URL
        url = f"{self.base_url}/chat/completions"

        # Build request payload
        request_data: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if temperature is not None:
            request_data["temperature"] = temperature
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens

        logger.debug("FakeMistralChat.complete: POST %s", url)

        # Make HTTP request to E2E mock server
        try:
            response = requests.post(
                url,
                json=request_data,
                timeout=30,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            # Parse response
            data = response.json()

            # Extract content and usage from response
            content = ""
            usage = None

            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]

            if "usage" in data:
                usage = data["usage"]

            return FakeMistralChatResponse(content, usage)

        except requests.exceptions.RequestException as e:
            logger.error("Failed to call E2E mock server for chat: %s", e)
            raise RuntimeError(f"E2E mock server request failed: {e}") from e


class FakeMistral:
    """Fake Mistral client that routes calls to E2E mock server.

    This class mimics the interface of mistralai.Mistral but makes HTTP requests
    to the E2E mock server instead of Mistral's API.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        server: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize fake Mistral client.

        Args:
            api_key: API key (required for compatibility, but not used with mock server)
            server: Server URL for E2E mock server (e.g., "http://127.0.0.1:8000/v1")
            **kwargs: Additional arguments (ignored)
        """
        self.api_key = api_key
        # Use server URL if provided, otherwise default to localhost
        if server:
            # Remove trailing slash if present
            self.base_url = server.rstrip("/")
        else:
            self.base_url = "http://127.0.0.1:8000/v1"

        # Create fake sub-objects
        self.audio = type("obj", (object,), {})()
        self.audio.transcriptions = FakeMistralAudioTranscriptions(self.base_url)
        self.chat = FakeMistralChat(self.base_url)

        logger.debug("FakeMistral initialized: base_url=%s", self.base_url)


def create_fake_mistral_client(base_url: str) -> type:
    """Create a fake Mistral client class bound to a specific base URL.

    Args:
        base_url: Base URL for E2E mock server

    Returns:
        Fake client class that can be used to replace Mistral
    """

    class FakeMistralWithBase(FakeMistral):
        """Fake Mistral with pre-configured base URL."""

        def __init__(self, **kwargs: Any):
            """Initialize with pre-configured base URL."""
            # Remove server from kwargs if present (use the one from closure)
            kwargs.pop("server", None)
            super().__init__(server=base_url, **kwargs)

    return FakeMistralWithBase
