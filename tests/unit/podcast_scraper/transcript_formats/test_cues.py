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


class TestWebVTTVoiceSpans:
    """`<v Speaker 3>` says who is talking, and we were deleting it (#2099).

    `_normalize_cue_text` strips it as an HTML-like tag, so a publisher transcript that names every
    turn arrived as ONE undifferentiated voice and the episode could never carry host/guest
    attribution however good naming became. Measured on the production corpus: every episode that
    used a publisher transcript ended with a single voice — 128 of 128.
    """

    def test_the_voice_span_becomes_the_segment_speaker(self) -> None:
        from podcast_scraper.transcript_formats.cues import parse_webvtt

        vtt = (
            "WEBVTT\n\n"
            "0:00:18.370 --> 0:00:21.670\n"
            "<v Speaker 3>Hello and welcome to another episode.\n\n"
            "0:00:23.090 --> 0:00:24.090\n"
            "<v Speaker 4>And I'm Tracy Alloway.\n"
        )
        _plain, segments = parse_webvtt(vtt)
        assert [s.get("speaker") for s in segments] == ["Speaker 3", "Speaker 4"]

    def test_the_speaker_tag_is_not_left_in_the_text(self) -> None:
        from podcast_scraper.transcript_formats.cues import parse_webvtt

        vtt = "WEBVTT\n\n0:00:01.000 --> 0:00:02.000\n<v Joe Wiesenthal>I'm Joe Wiesenthal.\n"
        plain, segments = parse_webvtt(vtt)
        assert "<v" not in plain
        assert segments[0]["text"].strip() == "I'm Joe Wiesenthal."
        assert segments[0]["speaker"] == "Joe Wiesenthal"

    def test_a_classed_voice_span_keeps_only_the_name(self) -> None:
        # `<v.loud Mark>` — the class is part of the TAG, not the speaker.
        from podcast_scraper.transcript_formats.cues import parse_webvtt

        vtt = "WEBVTT\n\n0:00:01.000 --> 0:00:02.000\n<v.loud Mark Galeotti>Hello there.\n"
        _plain, segments = parse_webvtt(vtt)
        assert segments[0]["speaker"] == "Mark Galeotti"

    def test_a_cue_with_no_voice_span_has_no_speaker_key(self) -> None:
        # Unchanged shape for the transcripts that carry no speaker information — the key is
        # ABSENT rather than None, so "did the file say?" stays answerable downstream.
        from podcast_scraper.transcript_formats.cues import parse_webvtt

        vtt = "WEBVTT\n\n0:00:01.000 --> 0:00:02.000\nJust some narration.\n"
        _plain, segments = parse_webvtt(vtt)
        assert "speaker" not in segments[0]

    def test_other_inline_tags_are_still_stripped_and_do_not_become_speakers(self) -> None:
        from podcast_scraper.transcript_formats.cues import parse_webvtt

        vtt = "WEBVTT\n\n0:00:01.000 --> 0:00:02.000\n<i>emphasis</i> and <b>bold</b>.\n"
        _plain, segments = parse_webvtt(vtt)
        assert "speaker" not in segments[0]
        assert segments[0]["text"].strip() == "emphasis and bold."

    def test_timings_and_text_are_unchanged_by_the_capture(self) -> None:
        from podcast_scraper.transcript_formats.cues import parse_webvtt

        vtt = "WEBVTT\n\n" "0:00:02.730 --> 0:00:05.600\n" "<v Speaker 1>Bloomberg Audio Studios.\n"
        plain, segments = parse_webvtt(vtt)
        assert segments[0]["start"] == 2.730
        assert segments[0]["end"] == 5.600
        assert plain == segments[0]["text"]
