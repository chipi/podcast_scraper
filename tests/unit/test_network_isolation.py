#!/usr/bin/env python3
"""Test that network isolation is enforced in unit tests.

This test verifies that the network blocker in tests/unit/conftest.py
correctly prevents network calls in unit tests.
"""

import unittest


class TestNetworkIsolation(unittest.TestCase):
    """Test that network calls are blocked in unit tests."""

    def test_requests_get_blocked(self):
        """Test that requests.get() is blocked."""
        import requests

        with self.assertRaises(Exception) as context:
            requests.get(
                "https://example.com", timeout=1
            )  # nosec B113 - intentional: testing network blocking

        # Verify it's our NetworkCallDetectedError
        self.assertIn("Network call detected", str(context.exception))
        self.assertIn("requests", str(context.exception))

    def test_requests_post_blocked(self):
        """Test that requests.post() is blocked."""
        import requests

        with self.assertRaises(Exception) as context:
            requests.post(
                "https://example.com", data={"test": "data"}, timeout=1
            )  # nosec B113 - intentional: testing network blocking

        self.assertIn("Network call detected", str(context.exception))

    def test_requests_session_blocked(self):
        """Test that requests.Session() methods are blocked."""
        import requests

        session = requests.Session()
        with self.assertRaises(Exception) as context:
            session.get("https://example.com")

        self.assertIn("Network call detected", str(context.exception))
        self.assertIn("requests.Session", str(context.exception))

    def test_urllib_request_blocked(self):
        """Test that urllib.request.urlopen() is blocked."""
        import urllib.request

        with self.assertRaises(Exception) as context:
            urllib.request.urlopen(
                "https://example.com"
            )  # nosec B310 - intentional: testing network blocking

        self.assertIn("Network call detected", str(context.exception))
        self.assertIn("urllib.request", str(context.exception))

    def test_socket_create_connection_blocked(self):
        """Test that socket.create_connection() is blocked."""
        import socket

        with self.assertRaises(Exception) as context:
            socket.create_connection(("example.com", 80))

        self.assertIn("Network call detected", str(context.exception))
        self.assertIn("socket", str(context.exception))
