"""Regression guards for ``parse_rss_items`` integrity checks.

ADR-100 parallel: the RSS parser was using ``logger.debug`` to silently
swallow ``DefusedXMLParseError`` and HTML-disguised-as-RSS payloads, hiding
real failures from the operator. The 2026-06-15 close-out promoted the
logging to WARNING and added an HTML-body detector. These tests guard the
new behavior so future refactors don't quietly regress to silent
degradation.
"""

from __future__ import annotations

import logging

import pytest

from podcast_scraper.rss.parser import parse_rss_items


class TestParseRssItemsWarnings:
    def test_empty_bytes_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="podcast_scraper.rss.parser"):
            title, authors, items = parse_rss_items(b"")
        assert title == ""
        assert authors == []
        assert items == []
        warn_messages = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("empty xml_bytes" in r.getMessage() for r in warn_messages)

    def test_html_body_disguised_as_rss_warns(self, caplog):
        html_payload = (
            b"<!doctype html><html><head><title>503 Service Unavailable</title>"
            b"</head><body>Sorry, the origin server is down.</body></html>"
        )
        with caplog.at_level(logging.WARNING, logger="podcast_scraper.rss.parser"):
            title, authors, items = parse_rss_items(html_payload)
        assert items == []
        warn_messages = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("looks like HTML, not RSS XML" in r.getMessage() for r in warn_messages)

    def test_html_uppercase_DOCTYPE_also_detected(self, caplog):
        html_payload = b"<!DOCTYPE html PUBLIC><html><body>x</body></html>"
        with caplog.at_level(logging.WARNING, logger="podcast_scraper.rss.parser"):
            parse_rss_items(html_payload)
        warn_messages = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("looks like HTML, not RSS XML" in r.getMessage() for r in warn_messages)

    def test_malformed_xml_warns_at_warning_level(self, caplog):
        # Truncated XML — DefusedXML raises ParseError. Should now be WARNING,
        # not the prior DEBUG-level swallow.
        truncated = b"<?xml version='1.0'?><rss><channel><title>Broken"
        with caplog.at_level(logging.WARNING, logger="podcast_scraper.rss.parser"):
            title, authors, items = parse_rss_items(truncated)
        assert title == ""
        assert items == []
        warn_messages = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("defusedxml ParseError" in r.getMessage() for r in warn_messages)

    def test_valid_minimal_rss_no_warnings(self, caplog):
        valid = (
            b"<?xml version='1.0'?>"
            b"<rss><channel><title>Test Feed</title>"
            b"<item><title>One</title></item>"
            b"</channel></rss>"
        )
        with caplog.at_level(logging.WARNING, logger="podcast_scraper.rss.parser"):
            title, authors, items = parse_rss_items(valid)
        assert title == "Test Feed"
        assert len(items) == 1
        # No WARNING-level log lines on the happy path.
        warn_messages = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warn_messages == []


@pytest.mark.parametrize(
    "html_prefix",
    [
        b"<!doctype html>",
        b'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0">',
        b"<html>",
        b'<HTML lang="en">',
        b"  \n\t  <html>",  # leading whitespace
    ],
)
def test_html_detection_covers_common_shapes(html_prefix, caplog):
    payload = html_prefix + b"<body>oops</body></html>"
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.rss.parser"):
        parse_rss_items(payload)
    assert any(
        "looks like HTML, not RSS XML" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )
