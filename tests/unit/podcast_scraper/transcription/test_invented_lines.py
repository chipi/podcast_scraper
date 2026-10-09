"""Lines Whisper writes that nobody said are removed, and only those (#2187).

WHY THIS EXISTS. On the V.6b real feeds (2026-10-08) the French episode opened with nine segments
of "Sous-titrage Société Radio-Canada" while the diarizer heard the guest talking, and the Spanish
one ended on "Gracias por ver el video." The prod snapshot of 2026-09-20 has the same lines inside
English episodes ("Untertitelung im Auftrag des ZDF,", "Sous-titrage Société Radio-Canada ."). They
pass the loop and confidence checks, so nothing removed them. The real-speech lines below are the
near-miss shapes found on that snapshot (written fresh, not copied): they contain the words and
must stay.
"""

from __future__ import annotations

import pytest

from podcast_scraper.transcription.invented_lines import drop_invented_lines, is_invented_line

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text",
    [
        " Sous-titrage Société Radio-Canada",  # V.6b French, nine times
        "Sous-titrage ST' 501",  # V.6b French
        " Gracias por ver el video.",  # V.6b Spanish, the last segment
        "Sous-titrage Société Radio-Canada .",  # prod, inside an English episode
        "Untertitelung im Auftrag des ZDF,",  # prod, inside an English episode
        "Untertitelung des ZDF, 2020",
        "Untertitel im Auftrag des ZDF für funk, 2017",
        "SOUS-TITRAGE SOCIETE RADIO-CANADA",
        "Sottotitoli creati dalla comunità Amara.org",
        "Thank you for watching.",
    ],
)
def test_invented_lines_are_recognised(text: str) -> None:
    assert is_invented_line(text)


@pytest.mark.parametrize(
    "text",
    [
        " und dann kam die Frage nach den Untertiteln",  # a sentence that mentions subtitles
        "Speaker 2: Thanks for watching and for listening, everyone.",  # sign-off inside a turn
        "Thanks for watching the show today.",  # a longer sign-off
        "and that is the whole method cool thanks for watching",  # one whole Whisper segment
        "with subtitles in English",
        "they only added the subtitles later.",
        "",
        None,
    ],
)
def test_speech_that_contains_the_words_is_not(text) -> None:
    assert not is_invented_line(text)


class TestDropInventedLines:
    RESULT = {
        "text": "Sous-titrage Société Radio-Canada Bonjour à tous. Et voilà.",
        "segments": [
            {"start": 0.0, "end": 30.0, "text": " Sous-titrage Société Radio-Canada"},
            {"start": 30.0, "end": 32.0, "text": " Bonjour à tous."},
            {"start": 32.0, "end": 33.5, "text": " Et voilà."},
        ],
        "language": "fr",
    }

    def test_the_line_is_removed_recorded_and_the_text_rebuilt(self) -> None:
        out = drop_invented_lines(self.RESULT)
        assert [s["text"] for s in out["segments"]] == [" Bonjour à tous.", " Et voilà."]
        assert out["text"] == "Bonjour à tous. Et voilà."
        assert out["asr_invented_lines"] == [
            {"start": 0.0, "end": 30.0, "text": "Sous-titrage Société Radio-Canada"}
        ]
        assert out["language"] == "fr"

    def test_the_input_is_not_mutated(self) -> None:
        drop_invented_lines(self.RESULT)
        assert len(self.RESULT["segments"]) == 3
        assert "asr_invented_lines" not in self.RESULT

    def test_nothing_to_remove_returns_the_same_object(self) -> None:
        clean = {"text": "Bonjour.", "segments": [{"start": 0, "end": 1, "text": "Bonjour."}]}
        assert drop_invented_lines(clean) is clean

    def test_a_result_without_segments_is_left_alone(self) -> None:
        bare = {"text": "Sous-titrage Société Radio-Canada"}
        assert drop_invented_lines(bare) is bare


@pytest.mark.parametrize(
    "text",
    [
        # Rádio Novelo Apresenta (pt-BR, 2026-10-09): twice over the closing music, kept because
        # of the stray leading "A".
        " A Sous-titrage Société Radio-Canada",
        "Sous-titrage Société Radio-Canada ok",
    ],
)
def test_a_stray_short_token_beside_the_line_does_not_hide_it(text: str) -> None:
    assert is_invented_line(text)


@pytest.mark.parametrize(
    "text",
    [
        "Eu li Sous-titrage Société Radio-Canada no final do vídeo",
        "Thanks for watching, see you next week",
    ],
)
def test_a_real_sentence_around_the_words_is_still_speech(text: str) -> None:
    assert not is_invented_line(text)
