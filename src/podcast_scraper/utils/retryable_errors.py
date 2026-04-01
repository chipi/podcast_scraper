"""Error classification utilities for retry logic.

This module provides functions to classify exceptions as retryable or non-retryable,
helping to determine when retries should be attempted.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def _http_status_as_int(value: object) -> int | None:
    """Return HTTP status only for a real int.

    Ignores mocks: ``int(MagicMock)`` is misleadingly truthy.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def is_retryable_error(error: Exception, *, error_context: str = "default") -> bool:
    """Determine if an error is retryable.

    Retryable errors are transient failures that may succeed on retry:
    - Rate limits (429)
    - Server errors (5xx)
    - Connection errors (network issues)
    - Timeout errors

    Non-retryable errors are permanent failures that won't succeed on retry:
    - Client errors (4xx except 429)
    - Authentication errors (401, 403)
    - Validation errors (400)
    - Not found errors (404)

    Args:
        error: Exception to classify
        error_context: When ``\"ollama_local\"``, HTTP 500 / InternalServerError from the
            local Ollama server is treated as non-retryable. Retrying the same request
            rarely fixes OOM, context-limit, or model-load crashes and makes runs feel
            stuck (multiple long timeouts). Remote APIs keep default 5xx retry behavior.

    Returns:
        True if error is retryable, False otherwise
    """
    error_str = str(error).lower()
    error_type_name = type(error).__name__.lower()

    resp_obj = getattr(error, "response", None)
    raw_status = getattr(resp_obj, "status_code", None) if resp_obj is not None else None
    http_code = _http_status_as_int(raw_status)
    if http_code is None:
        http_code = _http_status_as_int(getattr(error, "status_code", None))

    if error_context == "ollama_local":
        try:
            from openai import APIStatusError, InternalServerError

            # Tests may patch ``openai`` so these names are not real types; avoid TypeError.
            if isinstance(InternalServerError, type) and isinstance(error, InternalServerError):
                return False
            if isinstance(APIStatusError, type) and isinstance(error, APIStatusError):
                if getattr(error, "status_code", None) == 500:
                    return False
        except ImportError:
            pass
        # Duck-type HTTP 500 when SDK classes are mocked but ``response.status_code`` is a real int.
        if http_code == 500:
            return False
        if re.search(r"\b500\b", str(error)) or "internal server error" in error_str:
            return False
        # SDK may stringify without status digits; ``InternalServerError`` is 5xx-class.
        if error_type_name == "internalservererror":
            return False

    # Structured 5xx (e.g. OpenAI SDK) — retry for remote APIs even if ``str(error)`` is unhelpful.
    if error_context != "ollama_local" and http_code is not None and 500 <= http_code <= 599:
        return True

    # Rate limit errors (always retryable)
    if (
        "429" in str(error)
        or "rate limit" in error_str
        or "rate_limit" in error_str
        or "quota" in error_str
        or "resource exhausted" in error_str
        or "too many requests" in error_str
    ):
        return True

    # Server errors (5xx) - retryable
    if (
        "500" in str(error)
        or "501" in str(error)
        or "502" in str(error)
        or "503" in str(error)
        or "504" in str(error)
        or "505" in str(error)
        or "internal server error" in error_str
        or (error_context != "ollama_local" and "server error" in error_str)
        or "bad gateway" in error_str
        or "service unavailable" in error_str
        or "gateway timeout" in error_str
    ):
        return True

    # Connection errors - retryable
    connection_error_indicators = [
        "connection",
        "connect",
        "network",
        "socket",
        "dns",
        "timeout",
        "timed out",
        "connection reset",
        "connection refused",
        "connection aborted",
        "broken pipe",
        "errno",
    ]
    if any(indicator in error_str for indicator in connection_error_indicators):
        return True

    # Timeout errors - retryable
    timeout_indicators = [
        "timeout",
        "timed out",
        "read timeout",
        "connect timeout",
        "request timeout",
    ]
    if any(indicator in error_str for indicator in timeout_indicators):
        return True

    # Check exception type names for common retryable patterns
    retryable_type_patterns = [
        "connectionerror",
        "timeouterror",
        "timeout",
        "connecterror",
        "networkerror",
        "httperror",  # Some HTTP errors are retryable (5xx)
    ]
    if any(pattern in error_type_name for pattern in retryable_type_patterns):
        # But exclude non-retryable HTTP errors
        if not is_non_retryable_http_error(error):
            return True

    # Non-retryable errors (4xx except 429)
    if is_non_retryable_http_error(error):
        return False

    # Default: if we can't determine, assume retryable (conservative)
    # This allows retries for unknown errors, which is safer than failing immediately
    logger.debug(f"Unknown error type {type(error).__name__}, assuming retryable: {error}")
    return True


def is_non_retryable_http_error(error: Exception) -> bool:
    """Check if error is a non-retryable HTTP error (4xx except 429).

    Args:
        error: Exception to check

    Returns:
        True if error is a non-retryable HTTP error, False otherwise
    """
    error_str = str(error).lower()

    # Authentication/authorization errors (401, 403) - not retryable
    if (
        "401" in str(error)
        or "403" in str(error)
        or "unauthorized" in error_str
        or "forbidden" in error_str
        or "authentication" in error_str
        or "authorization" in error_str
    ):
        return True

    # Validation errors (400) - not retryable
    if (
        "400" in str(error)
        or "bad request" in error_str
        or "validation" in error_str
        or "invalid" in error_str
    ):
        return True

    # Not found errors (404) - not retryable
    if "404" in str(error) or "not found" in error_str:
        return True

    # Method not allowed (405) - not retryable
    if "405" in str(error) or "method not allowed" in error_str:
        return True

    # Conflict (409) - usually not retryable (but could be in some cases)
    # We'll treat it as non-retryable to be safe
    if "409" in str(error) or "conflict" in error_str:
        return True

    # Payload too large (413) - not retryable
    if "413" in str(error) or "payload too large" in error_str:
        return True

    # Unsupported media type (415) - not retryable
    if "415" in str(error) or "unsupported media type" in error_str:
        return True

    # Unprocessable entity (422) - not retryable
    if "422" in str(error) or "unprocessable entity" in error_str:
        return True

    # Too many requests (429) - retryable, so not in this list
    # We handle 429 separately in is_retryable_error()

    return False


def get_retry_reason(error: Exception) -> str:
    """Get a human-readable reason for retry.

    Args:
        error: Exception that triggered retry

    Returns:
        String describing why retry is happening (e.g., "429", "500", "timeout")
    """
    error_str = str(error).lower()

    # Rate limit
    if "429" in str(error) or "rate limit" in error_str:
        return "429"

    # Server errors
    if "500" in str(error):
        return "500"
    if "502" in str(error):
        return "502"
    if "503" in str(error):
        return "503"
    if "504" in str(error):
        return "504"

    # Connection/timeout errors
    if "timeout" in error_str or "timed out" in error_str:
        return "timeout"
    if "connection" in error_str:
        return "connection_error"

    # Default to exception type name
    return type(error).__name__
