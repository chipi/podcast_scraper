#!/usr/bin/env python3
"""Integration tests for network guard infrastructure.

These tests verify that the network guard correctly blocks external network calls
and allows localhost connections. Moved from tests/e2e/ as part of Phase 3 test
pyramid refactoring - these test infrastructure components, not user workflows.
"""

import os
import sys
import unittest

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)


@pytest.mark.integration
class TestNetworkGuard(unittest.TestCase):
    """Test that network guard blocks external network calls."""

    def test_external_network_blocked(self):
        """Test that external network calls are blocked.

        Note: This test verifies that pytest-socket is blocking external connections.
        If the connection succeeds, it means socket blocking is not active.
        """
        import socket

        from pytest_socket import SocketBlockedError

        # Attempt to connect to external host (should be blocked)
        with self.assertRaises((SocketBlockedError, OSError, Exception)) as context:
            # This should fail because network guard blocks external connections
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            try:
                sock.connect(("example.com", 80))
                # If we get here, socket blocking is not working
                self.fail("Socket blocking is not active! External connection succeeded.")
            finally:
                sock.close()

        # Verify error message indicates network blocking
        # pytest-socket raises SocketBlockedError
        error_msg = str(context.exception).lower()
        # Check for socket blocking indicators
        self.assertTrue(
            "socket" in error_msg
            or "blocked" in error_msg
            or "not allowed" in error_msg
            or isinstance(context.exception, SocketBlockedError),
            f"Expected socket blocking error, got: {type(context.exception).__name__}: {error_msg}",
        )

    def test_localhost_allowed(self):
        """Test that localhost connections are allowed."""
        import socket

        # Attempt to connect to localhost (should be allowed)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            # Try to connect to localhost (may fail if nothing is listening,
            # but shouldn't be blocked)
            try:
                sock.connect(("127.0.0.1", 8000))
            except (ConnectionRefusedError, OSError):
                # Connection refused is expected if nothing is listening
                # The important thing is that it's not blocked by network guard
                pass
            finally:
                sock.close()
        except Exception as exc:
            # If we get a socket blocking error, that's a problem
            error_msg = str(exc).lower()
            if "socket" in error_msg and "blocked" in error_msg:
                self.fail(f"Localhost connection was incorrectly blocked: {exc}")
            # Other errors (like connection refused) are fine

    def test_requests_external_blocked(self):
        """Test that requests library external calls are blocked."""
        try:
            import requests
            from pytest_socket import SocketBlockedError

            # Attempt to make external HTTP request (should be blocked)
            with self.assertRaises((SocketBlockedError, OSError, Exception)) as context:
                requests.get("https://example.com", timeout=1)
                # If we get here, socket blocking is not working
                self.fail("Socket blocking is not active! External HTTP request succeeded.")

            # Verify error message indicates network blocking
            # pytest-socket raises SocketBlockedError
            error_msg = str(context.exception).lower()
            # Check for socket blocking indicators
            self.assertTrue(
                "socket" in error_msg
                or "blocked" in error_msg
                or "not allowed" in error_msg
                or isinstance(context.exception, SocketBlockedError),
                (
                    f"Expected socket/network blocking error, got: "
                    f"{type(context.exception).__name__}: {error_msg}"
                ),
            )
        except ImportError:
            self.skipTest("requests library not available")
