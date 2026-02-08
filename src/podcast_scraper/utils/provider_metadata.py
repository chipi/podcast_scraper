"""Provider metadata logging utilities.

This module provides utilities for logging non-sensitive provider metadata
(account, project, region, endpoint) for debugging and troubleshooting.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def log_provider_metadata(
    provider_name: str,
    account: Optional[str] = None,
    project: Optional[str] = None,
    region: Optional[str] = None,
    endpoint: Optional[str] = None,
    organization: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """Log non-sensitive provider metadata for debugging.

    This function logs provider metadata that is useful for debugging
    but does not expose sensitive information like API keys.

    Args:
        provider_name: Name of the provider (e.g., "OpenAI", "Gemini")
        account: Account identifier (non-sensitive)
        project: Project identifier (non-sensitive)
        region: Region identifier (e.g., "us-east-1")
        endpoint: API endpoint URL (without credentials)
        organization: Organization identifier (non-sensitive)
        base_url: Base URL for API (without credentials)
    """
    metadata: Dict[str, Any] = {}
    if account:
        metadata["account"] = account
    if project:
        metadata["project"] = project
    if region:
        metadata["region"] = region
    if endpoint:
        metadata["endpoint"] = endpoint
    if organization:
        metadata["organization"] = organization
    if base_url:
        metadata["base_url"] = base_url

    if metadata:
        logger.debug(
            "%s provider metadata: %s",
            provider_name,
            ", ".join(f"{k}={v}" for k, v in metadata.items()),
        )


def extract_region_from_endpoint(endpoint: Optional[str]) -> Optional[str]:
    """Extract region from endpoint URL if possible.

    Args:
        endpoint: API endpoint URL

    Returns:
        Region identifier if extractable, None otherwise
    """
    if not endpoint:
        return None

    # Common patterns:
    # - us-east-1.api.openai.com -> us-east-1
    # - api.us-east-1.anthropic.com -> us-east-1
    # - gemini.googleapis.com (no region in URL)
    import re

    patterns = [
        r"([a-z]+-[a-z]+-\d+)\.api\.",
        r"api\.([a-z]+-[a-z]+-\d+)\.",
        r"([a-z]+-[a-z]+-\d+)\.",
    ]

    for pattern in patterns:
        match = re.search(pattern, endpoint.lower())
        if match:
            return match.group(1)

    return None


def validate_api_key_format(
    api_key: Optional[str],
    provider_name: str,
    expected_prefixes: Optional[list[str]] = None,
) -> tuple[bool, Optional[str]]:
    """Validate API key format (without exposing the key).

    Args:
        api_key: API key to validate
        provider_name: Name of provider for error messages
        expected_prefixes: Optional list of expected prefixes (e.g., ["sk-", "sk-proj-"])

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not api_key:
        return False, f"{provider_name} API key is missing"

    if len(api_key) < 10:
        return False, f"{provider_name} API key appears to be too short (may be invalid)"

    if expected_prefixes:
        key_lower = api_key.lower()
        if not any(key_lower.startswith(prefix.lower()) for prefix in expected_prefixes):
            return (
                False,
                f"{provider_name} API key does not match expected format "
                f"(expected prefix: {', '.join(expected_prefixes)})",
            )

    return True, None
