"""Secret redaction utilities for configuration serialization.

This module provides functions to redact sensitive information (API keys, tokens,
passwords, etc.) from configuration dictionaries before serialization to prevent
secrets from being persisted in output files.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Set

# Denylist of key patterns that should be redacted
# Keys matching these patterns (case-insensitive) will have their values redacted
SECRET_KEY_PATTERNS: Set[str] = {
    "api_key",
    "apikey",
    "api-key",
    "token",
    "authorization",
    "auth",
    "password",
    "passwd",
    "secret",
    "credential",
    "private_key",
    "private-key",
    "access_token",
    "access-token",
    "refresh_token",
    "refresh-token",
    "session_id",
    "session-id",
    "bearer",
}

# Pattern to detect values that look like API keys or tokens
# Matches strings that look like:
# - sk-... (OpenAI-style keys)
# - Bearer tokens
# - Long alphanumeric strings (potential tokens)
API_KEY_PATTERN = re.compile(r"^(sk-[a-zA-Z0-9]{20,}|Bearer\s+[a-zA-Z0-9]{20,}|[a-zA-Z0-9]{32,})$")


def _is_secret_key(key: str) -> bool:
    """Check if a key matches secret key patterns.

    Args:
        key: Key name to check

    Returns:
        True if key should be redacted, False otherwise
    """
    key_lower = key.lower()
    # Check if key contains any secret pattern
    return any(pattern in key_lower for pattern in SECRET_KEY_PATTERNS)


def _looks_like_secret(value: Any) -> bool:
    """Check if a value looks like a secret (API key, token, etc.).

    Args:
        value: Value to check

    Returns:
        True if value looks like a secret, False otherwise
    """
    if not isinstance(value, str):
        return False
    # Check if value matches API key pattern
    return bool(API_KEY_PATTERN.match(value.strip()))


def redact_secrets(data: Any, redact_patterns: bool = True) -> Any:
    """Recursively redact secrets from a data structure.

    This function traverses dictionaries and lists recursively, redacting values
    for keys that match secret patterns or values that look like secrets.

    Args:
        data: Data structure to redact (dict, list, or primitive)
        redact_patterns: If True, also redact values that look like secrets
                        (pattern-based detection)

    Returns:
        Data structure with secrets redacted (replaced with "__redacted__")
    """
    if isinstance(data, dict):
        redacted: Dict[str, Any] = {}
        for key, value in data.items():
            if _is_secret_key(key):
                # Redact based on key pattern
                redacted[key] = "__redacted__"
            elif redact_patterns and _looks_like_secret(value):
                # Redact based on value pattern (second layer)
                redacted[key] = "__redacted__"
            else:
                # Recursively redact nested structures
                redacted[key] = redact_secrets(value, redact_patterns=redact_patterns)
        return redacted
    elif isinstance(data, list):
        # Recursively redact list items
        return [redact_secrets(item, redact_patterns=redact_patterns) for item in data]
    else:
        # For primitive types, check if value looks like a secret
        if redact_patterns and _looks_like_secret(data):
            return "__redacted__"
        return data
