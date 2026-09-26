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
    # Two cues cut between words are two words (#2075): "Helloworld" was the bug.
    assert plain == "Hello world"
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


class TestCuesCutBetweenWordsStaySeparate:
    """41% of cue boundaries on the production snapshot's publisher transcripts glued two words
    ("PresidentJeff Schmidt"), hiding spoken names from every reader (#2075)."""

    VTT = (
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v Speaker 3>speaking with the Fed President\n\n"
        "00:00:02.000 --> 00:00:03.000\n<v Speaker 3>Jeff Schmidt. Thank you.\n"
    )

    def test_a_space_separates_the_cues(self) -> None:
        plain, _segs = parse_webvtt(self.VTT)
        assert "President Jeff Schmidt" in plain

    def test_plain_text_is_still_exactly_the_segments_joined(self) -> None:
        plain, segs = parse_webvtt(self.VTT)
        assert "".join(s["text"] for s in segs) == plain

    def test_an_existing_space_is_not_doubled(self) -> None:
        body = (
            "1\n00:00:00,000 --> 00:00:01,000\nHello \n\n2\n00:00:01,000 --> 00:00:02,000\nworld\n"
        )
        plain, _segs = parse_srt(body)
        assert plain == "Hello world"


class TestCueEscapesAreDecoded:
    """A cue file that says ``&amp;`` means ``&`` — and the stored transcript must say ``&``.

    The WebVTT grammar REQUIRES ``&amp;`` for a literal ampersand in cue text (likewise ``&lt;``,
    ``&gt;``, ``&nbsp;``), so every conforming publisher transcript carries them. They were being
    stored raw, which is not cosmetic: the transcript is the processing base, so ``Johnson &amp;
    Johnson`` reached summaries, GI quotes, entity names, the KG and the search index, where it
    cannot match a query for ``Johnson & Johnson``.
    """

    def _cue(self, body: str) -> str:
        return f"WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n<v Joe>{body}</v>\n"

    def test_an_escaped_ampersand_is_decoded(self) -> None:
        plain, _ = parse_webvtt(self._cue("We saw this at Spoke &amp; Wrench."))
        assert plain.strip() == "We saw this at Spoke & Wrench."

    def test_escaped_angle_brackets_are_decoded(self) -> None:
        plain, _ = parse_webvtt(self._cue("5 &lt; 7 &gt; 3"))
        assert plain.strip() == "5 < 7 > 3"

    def test_markup_the_publisher_escaped_survives_as_text(self) -> None:
        """Tags are stripped BEFORE escapes are decoded. The other order would strip ``<b>`` —
        markup the publisher deliberately wrote as text."""
        plain, _ = parse_webvtt(self._cue("he wrote &lt;b&gt;bold&lt;/b&gt;"))
        assert plain.strip() == "he wrote <b>bold</b>"

    def test_a_legacy_html_entity_is_left_alone(self) -> None:
        """Why this is a fixed table and not ``html.unescape``: that decodes the HTML5 legacy set,
        turning an ampersand followed by a word into a character the publisher never wrote."""
        plain, _ = parse_webvtt(self._cue("&copy 2026 Acme"))
        assert plain.strip() == "&copy 2026 Acme"

    def test_a_bare_ampersand_is_untouched(self) -> None:
        plain, _ = parse_webvtt(self._cue("AT&T earnings"))
        assert plain.strip() == "AT&T earnings"

    def test_nbsp_becomes_a_real_space(self) -> None:
        """``html.unescape`` yields ``\\xa0``, which the horizontal-whitespace collapse does not
        treat as a space, so it would survive into the stored text and every quote span."""
        plain, _ = parse_webvtt(self._cue("a&nbsp;b"))
        assert plain.strip() == "a b"

    def test_srt_gets_the_same_decoding(self) -> None:
        plain, _ = parse_srt("1\n00:00:00,000 --> 00:00:05,000\nTea &amp; biscuits\n")
        assert plain.strip() == "Tea & biscuits"

    def test_decoding_keeps_plain_text_and_segments_in_one_coordinate_space(self) -> None:
        """#545: quote char offsets come from ``plain``. Decoding SHORTENS the text, so doing it on
        the joined string instead of per segment would desynchronise every offset."""
        vtt = (
            "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n<v A>Spoke &amp; Wrench\n\n"
            "00:00:02.000 --> 00:00:04.000\n<v B> and 5 &lt; 7\n"
        )
        plain, segs = parse_webvtt(vtt)
        assert "".join(s["text"] for s in segs) == plain
        assert "&amp;" not in plain and "&lt;" not in plain

    def test_the_voice_span_still_names_the_speaker(self) -> None:
        _plain, segs = parse_webvtt(self._cue("Spoke &amp; Wrench"))
        assert [s.get("speaker") for s in segs] == ["Joe"]


class TestWebVttSpeakerWrittenAsCueText:
    """A WebVTT can name its turns as cue TEXT, not only as a `<v>` tag (#2096).

    Measured on prod 2026-09-26. Six Odd Lots episodes published a roster of two hosts and NO guest
    while every title named one. Their WebVTT writes each turn as `Speaker 1: …` instead of
    `<v Speaker 1>`, and `parse_webvtt` only read the tag — so the label stayed embedded in the
    prose: 280 literal `Speaker 1` strings in one 81 KB transcript, `speaker=None` on all 1,290
    segments. That wrecks two consumers at once: attribution matches markers at LINE START (and the
    labels are mid-line), and the chunker finds no sentence boundary. 51 corpus episodes carry the
    shape.

    `parse_srt` already handled it. Identical input produced opposite results from the two parsers,
    which is the asymmetry these pin.
    """

    _TEXT_PREFIX = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:04.000\n"
        "Speaker 1: Hello, Odd Lots listeners. I am Joe Weisenthal.\n\n"
        "00:00:04.000 --> 00:00:07.000\n"
        "Speaker 2: And I am Tracy Alloway.\n"
    )

    def test_the_speaker_is_read_from_the_cue_text(self) -> None:
        _plain, segs = parse_webvtt(self._TEXT_PREFIX)

        assert [s.get("speaker") for s in segs] == ["Speaker 1", "Speaker 2"]

    def test_the_label_does_not_survive_into_the_prose(self) -> None:
        """Left in the text it becomes a mid-line marker attribution cannot use and chunking trips
        over. Stripping it matches what ``parse_srt`` has always done."""
        plain, _segs = parse_webvtt(self._TEXT_PREFIX)

        assert "Speaker 1:" not in plain
        assert "Speaker 2:" not in plain
        assert plain.startswith("Hello, Odd Lots listeners.")

    def test_a_voice_tag_still_wins_when_both_are_present(self) -> None:
        """Explicit markup beats a text convention."""
        doc = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:04.000\n"
            "<v Joe Weisenthal>Speaker 9: the tag must win.\n"
        )

        _plain, segs = parse_webvtt(doc)

        assert [s.get("speaker") for s in segs] == ["Joe Weisenthal"]

    def test_a_voice_tagged_file_is_unchanged(self) -> None:
        """Guard: the 781-voice-span path must not regress."""
        doc = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:04.000\n"
            "<v Joe Weisenthal>Hello there.\n\n"
            "00:00:04.000 --> 00:00:07.000\n"
            "<v Tracy Alloway>And hello from me.\n"
        )

        plain, segs = parse_webvtt(doc)

        assert [s.get("speaker") for s in segs] == ["Joe Weisenthal", "Tracy Alloway"]
        assert plain == "Hello there. And hello from me."

    def test_prose_that_merely_contains_a_colon_is_not_a_speaker(self) -> None:
        """Why the pattern stays narrow to ``Speaker N``: a free ``Name:`` is indistinguishable
        from prose, and treating it as a label would invent speakers."""
        doc = "WEBVTT\n\n" "00:00:00.000 --> 00:00:04.000\n" "Note: this is not a speaker label.\n"

        plain, segs = parse_webvtt(doc)

        assert [s.get("speaker") for s in segs] == [None]
        assert plain.startswith("Note:")

    def test_webvtt_and_srt_now_agree_on_the_same_input(self) -> None:
        """The asymmetry itself, as a property."""
        srt = (
            "1\n00:00:00,000 --> 00:00:04,000\n"
            "Speaker 1: Hello, Odd Lots listeners. I am Joe Weisenthal.\n\n"
            "2\n00:00:04,000 --> 00:00:07,000\n"
            "Speaker 2: And I am Tracy Alloway.\n"
        )

        vtt_plain, vtt_segs = parse_webvtt(self._TEXT_PREFIX)
        srt_plain, srt_segs = parse_srt(srt)

        assert [s.get("speaker") for s in vtt_segs] == [s.get("speaker") for s in srt_segs]
        assert vtt_plain == srt_plain
