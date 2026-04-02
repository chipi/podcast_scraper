"""Tests for log_redaction utilities."""

import unittest

from podcast_scraper.utils.log_redaction import format_exception_for_log, redact_for_log


class TestRedactForLog(unittest.TestCase):
    """Test redact_for_log."""

    def test_none_and_empty(self):
        self.assertEqual(redact_for_log(None), "")
        self.assertEqual(redact_for_log(""), "")

    def test_bearer_redacted(self):
        raw = "HTTP 401: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.x.y"
        out = redact_for_log(raw)
        self.assertIn("[REDACTED]", out)
        self.assertNotIn("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", out)

    def test_sk_openai_redacted(self):
        raw = "Invalid key sk-12345678901234567890123456789012 for user"
        out = redact_for_log(raw)
        self.assertIn("sk-[REDACTED]", out)
        self.assertNotIn("sk-12345678901234567890123456789012", out)

    def test_sk_ant_redacted(self):
        raw = "bad sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890ABCD"
        out = redact_for_log(raw)
        self.assertIn("sk-ant-[REDACTED]", out)

    def test_truncation(self):
        long_text = "a" * 5000
        out = redact_for_log(long_text, max_len=100)
        self.assertLessEqual(len(out), 120)
        self.assertTrue(out.endswith("…[truncated]"))

    def test_format_exception_for_log(self):
        exc = ValueError("Auth failed: Bearer secret-token-here")
        out = format_exception_for_log(exc)
        self.assertIn("[REDACTED]", out)
        self.assertNotIn("secret-token-here", out)


if __name__ == "__main__":
    unittest.main()
