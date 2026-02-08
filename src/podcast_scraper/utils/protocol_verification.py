"""Protocol verification utilities.

This module provides runtime verification of protocol compliance for providers.
Verification is only enabled in __debug__ mode to ensure zero production overhead.
"""

from __future__ import annotations

import logging
from typing import Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def verify_protocol_compliance(
    provider: object,
    protocol: type[Protocol],  # type: ignore[valid-type]
    protocol_name: str,
) -> bool:
    """Verify that a provider implements the given protocol.

    This function uses runtime protocol checking (via @runtime_checkable) to verify
    that a provider correctly implements all required protocol methods.

    Args:
        provider: The provider instance to verify
        protocol: The Protocol class to check against
        protocol_name: Human-readable name of the protocol (for error messages)

    Returns:
        True if provider implements the protocol, False otherwise

    Note:
        This function only performs verification in __debug__ mode. In production
        (when Python is run with -O or -OO), this function returns True immediately
        without any checks, ensuring zero overhead.
    """
    if not __debug__:
        # Production mode: skip verification for zero overhead
        return True

    # Development mode: perform runtime verification
    if not isinstance(provider, protocol):
        missing_methods = []
        # Check which methods are missing
        for attr_name in dir(protocol):
            if attr_name.startswith("_"):
                continue
            attr = getattr(protocol, attr_name, None)
            if callable(attr) and not hasattr(provider, attr_name):
                missing_methods.append(attr_name)

        if missing_methods:
            logger.warning(
                "Provider %s does not fully implement %s protocol. " "Missing methods: %s",
                type(provider).__name__,
                protocol_name,
                ", ".join(missing_methods),
            )
            return False

    return True
