#!/usr/bin/env python3
"""Security tests for summarization module.

Tests critical security functions:
- _validate_model_source() - Model source validation

Note: TestRevisionPinning and TestPruneCacheSecurity have been moved to
tests/integration/test_summarizer_security_integration.py because they require
filesystem I/O (tempfile.mkdtemp() in setUp/tearDown).
"""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Try to import summarizer, skip tests if dependencies not available
try:
    from podcast_scraper.providers.ml import summarizer

    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False
    summarizer = types.ModuleType("summarizer")  # type: ignore[assignment]


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestValidateModelSource(unittest.TestCase):
    """Test _validate_model_source() security function."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger_patcher = patch("podcast_scraper.summarizer.logger")
        self.mock_logger = self.logger_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.logger_patcher.stop()

    def test_trusted_source_facebook(self):
        """Test that facebook models are recognized as trusted."""
        summarizer._validate_model_source("facebook/bart-large-cnn")
        self.mock_logger.debug.assert_called_once_with("Loading model from verified trusted source")
        self.mock_logger.warning.assert_not_called()

    def test_trusted_source_google(self):
        """Test that google models are recognized as trusted."""
        summarizer._validate_model_source("google/pegasus-large")
        self.mock_logger.debug.assert_called_once_with("Loading model from verified trusted source")
        self.mock_logger.warning.assert_not_called()

    def test_trusted_source_sshleifer(self):
        """Test that sshleifer models are recognized as trusted."""
        summarizer._validate_model_source("sshleifer/distilbart-cnn-12-6")
        self.mock_logger.debug.assert_called_once_with("Loading model from verified trusted source")
        self.mock_logger.warning.assert_not_called()

    def test_trusted_source_allenai(self):
        """Test that allenai models are recognized as trusted."""
        summarizer._validate_model_source("allenai/led-large-16384")
        self.mock_logger.debug.assert_called_once_with("Loading model from verified trusted source")
        self.mock_logger.warning.assert_not_called()

    def test_untrusted_source_warns(self):
        """Test that untrusted sources trigger security warning."""
        summarizer._validate_model_source("malicious-user/suspicious-model")
        self.mock_logger.debug.assert_not_called()
        self.mock_logger.warning.assert_called_once()
        warning_message = str(self.mock_logger.warning.call_args[0][0])
        self.assertIn("SECURITY NOTICE", warning_message)
        self.assertIn("custom (untrusted) source", warning_message)

    def test_local_model_warns(self):
        """Test that local/non-standard model identifiers trigger warning."""
        summarizer._validate_model_source("local-model")
        self.mock_logger.debug.assert_not_called()
        self.mock_logger.warning.assert_called_once()
        warning_message = str(self.mock_logger.warning.call_args[0][0])
        self.assertIn("SECURITY NOTICE", warning_message)

    def test_no_sensitive_info_logged(self):
        """Test that model names are not logged in clear text."""
        model_name = "malicious-user/suspicious-model"
        summarizer._validate_model_source(model_name)
        # Check that model name is not in any log calls
        all_logs = []
        for call in self.mock_logger.debug.call_args_list:
            all_logs.extend(call[0])
        for call in self.mock_logger.warning.call_args_list:
            all_logs.extend(call[0])
        log_text = " ".join(str(msg) for msg in all_logs)
        # Model name should not appear in logs (security: avoid logging untrusted input)
        self.assertNotIn("malicious-user", log_text)
        self.assertNotIn("suspicious-model", log_text)


# TestRevisionPinning and TestPruneCacheSecurity moved to
# tests/integration/test_summarizer_security_integration.py
# because they require filesystem I/O (tempfile.mkdtemp() in setUp/tearDown)


if __name__ == "__main__":
    unittest.main()
