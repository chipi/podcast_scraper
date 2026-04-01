#!/usr/bin/env python3
"""Tests for retryable error classification utilities.

These tests verify that errors are correctly classified as retryable or non-retryable.
"""

import unittest

from podcast_scraper.utils.retryable_errors import (
    get_retry_reason,
    is_non_retryable_http_error,
    is_retryable_error,
)


class _MiniHttpResponse:
    """Minimal ``response`` with a real ``int`` status (not ``httpx.Response``).

    ``tests/unit/.../test_ollama_provider.py`` replaces ``sys.modules['httpx']`` with a
    ``MagicMock`` at import time. Importing real ``httpx`` in this module would yield a
    mock ``Response`` whose ``status_code`` is not an ``int``, breaking
    ``_http_status_as_int`` and this test under full unit collection (CI / ``pytest -n 0``).
    """

    __slots__ = ("status_code",)

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _SdkStyle500Error(Exception):
    """Mimic OpenAI SDK errors with ``response.status_code`` without importing ``openai``.

    The ``openai`` package is optional in some CI unit jobs; production code still
    handles this shape via ``http_code`` when ``ImportError`` or mocked SDK types.
    """

    def __init__(self, message: str, *, response: object | None = None, body: object = None):
        super().__init__(message)
        self.response = response
        self.body = body


class TestIsRetryableError(unittest.TestCase):
    """Test is_retryable_error function."""

    def test_rate_limit_429_retryable(self):
        """Test that 429 rate limit errors are retryable."""
        error = Exception("429 Rate limit exceeded")
        self.assertTrue(is_retryable_error(error))

    def test_rate_limit_text_retryable(self):
        """Test that rate limit text errors are retryable."""
        error = Exception("Rate limit exceeded")
        self.assertTrue(is_retryable_error(error))

    def test_quota_error_retryable(self):
        """Test that quota errors are retryable."""
        error = Exception("Quota exceeded")
        self.assertTrue(is_retryable_error(error))

    def test_server_error_500_retryable(self):
        """Test that 500 server errors are retryable."""
        error = Exception("500 Internal Server Error")
        self.assertTrue(is_retryable_error(error))

    def test_server_error_502_retryable(self):
        """Test that 502 Bad Gateway errors are retryable."""
        error = Exception("502 Bad Gateway")
        self.assertTrue(is_retryable_error(error))

    def test_server_error_503_retryable(self):
        """Test that 503 Service Unavailable errors are retryable."""
        error = Exception("503 Service Unavailable")
        self.assertTrue(is_retryable_error(error))

    def test_connection_error_retryable(self):
        """Test that connection errors are retryable."""
        error = Exception("Connection refused")
        self.assertTrue(is_retryable_error(error))

    def test_timeout_error_retryable(self):
        """Test that timeout errors are retryable."""
        error = Exception("Request timeout")
        self.assertTrue(is_retryable_error(error))

    def test_auth_error_401_not_retryable(self):
        """Test that 401 authentication errors are not retryable."""
        error = Exception("401 Unauthorized")
        self.assertFalse(is_retryable_error(error))

    def test_auth_error_403_not_retryable(self):
        """Test that 403 forbidden errors are not retryable."""
        error = Exception("403 Forbidden")
        self.assertFalse(is_retryable_error(error))

    def test_validation_error_400_not_retryable(self):
        """Test that 400 validation errors are not retryable."""
        error = Exception("400 Bad Request")
        self.assertFalse(is_retryable_error(error))

    def test_not_found_404_not_retryable(self):
        """Test that 404 not found errors are not retryable."""
        error = Exception("404 Not Found")
        self.assertFalse(is_retryable_error(error))

    def test_unknown_error_defaults_to_retryable(self):
        """Test that unknown errors default to retryable (conservative)."""
        error = Exception("Some unknown error")
        # Should default to retryable for safety
        self.assertTrue(is_retryable_error(error))

    def test_ollama_local_internal_server_error_not_retryable(self):
        """Local Ollama: SDK-shaped HTTP 500 (response.status_code) is not retried."""
        err = _SdkStyle500Error("server error", response=_MiniHttpResponse(500), body=None)
        self.assertFalse(is_retryable_error(err, error_context="ollama_local"))
        self.assertTrue(is_retryable_error(err, error_context="default"))

    def test_ollama_local_500_message_not_retryable(self):
        """Local Ollama: plain exception mentioning HTTP 500 fails fast."""
        err = Exception("Error code: 500 - internal error")
        self.assertFalse(is_retryable_error(err, error_context="ollama_local"))
        self.assertTrue(is_retryable_error(err, error_context="default"))

    def test_ollama_local_internalservererror_typename(self):
        """Ollama: exception class name InternalServerError fails fast without digits in message."""

        class InternalServerError(Exception):
            pass

        err = InternalServerError("model OOM")
        self.assertFalse(is_retryable_error(err, error_context="ollama_local"))

    def test_default_context_structured_5xx_without_text_digits(self):
        """Remote API: response.status_code 5xx retries even when str(exception) is vague."""

        class OpaqueError(Exception):
            pass

        err = OpaqueError("upstream failure")
        err.response = _MiniHttpResponse(503)
        self.assertTrue(is_retryable_error(err, error_context="default"))

    def test_builtin_connectionerror_retryable_by_type_name(self):
        """ConnectionError matches retryable type-name heuristics."""
        err = ConnectionError("reset by peer")
        self.assertTrue(is_retryable_error(err))


class TestIsNonRetryableHttpError(unittest.TestCase):
    """Test is_non_retryable_http_error function."""

    def test_401_unauthorized(self):
        """Test that 401 is non-retryable."""
        error = Exception("401 Unauthorized")
        self.assertTrue(is_non_retryable_http_error(error))

    def test_403_forbidden(self):
        """Test that 403 is non-retryable."""
        error = Exception("403 Forbidden")
        self.assertTrue(is_non_retryable_http_error(error))

    def test_400_bad_request(self):
        """Test that 400 is non-retryable."""
        error = Exception("400 Bad Request")
        self.assertTrue(is_non_retryable_http_error(error))

    def test_404_not_found(self):
        """Test that 404 is non-retryable."""
        error = Exception("404 Not Found")
        self.assertTrue(is_non_retryable_http_error(error))

    def test_429_not_in_non_retryable(self):
        """Test that 429 is not in non-retryable list (it's retryable)."""
        error = Exception("429 Too Many Requests")
        self.assertFalse(is_non_retryable_http_error(error))

    def test_500_not_in_non_retryable(self):
        """Test that 500 is not in non-retryable list (it's retryable)."""
        error = Exception("500 Internal Server Error")
        self.assertFalse(is_non_retryable_http_error(error))

    def test_405_method_not_allowed(self):
        self.assertTrue(is_non_retryable_http_error(Exception("405 Method Not Allowed")))

    def test_409_conflict(self):
        self.assertTrue(is_non_retryable_http_error(Exception("409 Conflict")))

    def test_413_payload_too_large(self):
        self.assertTrue(is_non_retryable_http_error(Exception("413 Payload Too Large")))

    def test_415_unsupported_media(self):
        self.assertTrue(is_non_retryable_http_error(Exception("415 Unsupported Media Type")))

    def test_422_unprocessable(self):
        self.assertTrue(is_non_retryable_http_error(Exception("422 Unprocessable Entity")))


class TestGetRetryReason(unittest.TestCase):
    """Test get_retry_reason function."""

    def test_429_reason(self):
        """Test that 429 errors return '429' as reason."""
        error = Exception("429 Rate limit exceeded")
        self.assertEqual(get_retry_reason(error), "429")

    def test_500_reason(self):
        """Test that 500 errors return '500' as reason."""
        error = Exception("500 Internal Server Error")
        self.assertEqual(get_retry_reason(error), "500")

    def test_502_reason(self):
        """Test that 502 errors return '502' as reason."""
        error = Exception("502 Bad Gateway")
        self.assertEqual(get_retry_reason(error), "502")

    def test_503_reason(self):
        self.assertEqual(get_retry_reason(Exception("503 Service Unavailable")), "503")

    def test_504_reason(self):
        self.assertEqual(get_retry_reason(Exception("504 Gateway Timeout")), "504")

    def test_timeout_reason(self):
        """Test that timeout errors return 'timeout' as reason."""
        error = Exception("Request timeout")
        self.assertEqual(get_retry_reason(error), "timeout")

    def test_connection_reason(self):
        """Test that connection errors return 'connection_error' as reason."""
        error = Exception("Connection refused")
        self.assertEqual(get_retry_reason(error), "connection_error")

    def test_default_reason(self):
        """Test that unknown errors return exception type name."""
        error = ValueError("Some error")
        self.assertEqual(get_retry_reason(error), "ValueError")


if __name__ == "__main__":
    unittest.main()
