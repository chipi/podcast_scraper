"""Mock SDK clients for E2E and integration testing.

This module provides fake SDK clients that route SDK calls to the E2E mock server
via HTTP. These are needed for SDKs that don't support custom base URLs (Gemini, Mistral).

Usage:
    from tests.fixtures.mock_server.gemini_mock_client import create_fake_gemini_client
    from tests.fixtures.mock_server.mistral_mock_client import create_fake_mistral_client
"""
