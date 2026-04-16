#!/usr/bin/env python3
"""Unit tests for WebVTT / SRT cue parsing (issue #544)."""

import pytest

from podcast_scraper.transcript_formats.cues import parse_srt, parse_webvtt

pytestmark = [pytest.mark.unit]


def test_parse_webvtt_two_cues_plain_concat_alignment() -> None:
    body = """WEBVTT

00:00:00.000 --> 00:00:01.000
Hello

00:00:01.000 --> 00:00:02.000
 world
"""
    plain, segs = parse_webvtt(body)
    assert plain == "Hello world"
    assert len(segs) == 2
    assert segs[0] == {"start": 0.0, "end": 1.0, "text": "Hello"}
    assert segs[1] == {"start": 1.0, "end": 2.0, "text": " world"}
    assert "".join(s["text"] for s in segs) == plain


def test_parse_webvtt_strips_html_tags() -> None:
    body = """WEBVTT

00:00:00.000 --> 00:00:01.000
Hi <c.color>there</c>
"""
    plain, segs = parse_webvtt(body)
    assert plain == "Hi there"
    assert segs[0]["text"] == "Hi there"


def test_parse_webvtt_skips_note() -> None:
    body = """WEBVTT

NOTE
comment line

00:00:00.000 --> 00:00:01.000
Only
"""
    plain, segs = parse_webvtt(body)
    assert plain == "Only"


def test_parse_webvtt_cue_identifier_line() -> None:
    body = """WEBVTT

1
00:00:00.000 --> 00:00:01.000
One
"""
    plain, segs = parse_webvtt(body)
    assert plain == "One"


def test_parse_webvtt_no_header_returns_empty() -> None:
    plain, segs = parse_webvtt("just text\n")
    assert plain == ""
    assert segs == []


def test_parse_srt_basic() -> None:
    body = """1
00:00:00,000 --> 00:00:01,000
Hello

2
00:00:01,000 --> 00:00:02,000
world
"""
    plain, segs = parse_srt(body)
    assert plain == "Helloworld"
    assert len(segs) == 2
    assert segs[0]["start"] == 0.0
    assert segs[1]["end"] == 2.0


def test_parse_srt_multiline_cue() -> None:
    body = """1
00:00:00,000 --> 00:00:02,000
Line one
Line two
"""
    plain, segs = parse_srt(body)
    assert len(segs) == 1
    assert segs[0]["text"] == "Line one Line two"
    assert plain == segs[0]["text"]


def test_parse_srt_two_cues_concat_space() -> None:
    body = """1
00:00:00,000 --> 00:00:01,000
Hello

2
00:00:01,000 --> 00:00:02,000
 world
"""
    plain, segs = parse_srt(body)
    assert plain == "Hello world"


def test_parse_srt_garbage_returns_empty() -> None:
    plain, segs = parse_srt("not srt at all")
    assert plain == ""
    assert segs == []
