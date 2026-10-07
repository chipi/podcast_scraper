"""A transcript that does not read as its declared language fails the episode (#2187).

THE FAILURE IS SILENT WITHOUT THIS. Whisper honours the language it is sent: Spanish audio
requested as ``en`` came back as correct Spanish text reporting ``language: en`` (p10_e01,
2026-10-07). The service echoes the request, so ``asr_language_mismatch`` cannot fire, and a feed
with a wrong ``<language>`` tag would put source-language text through every English stage.

The texts are the v3 fixture transcripts — authored prose in all six tier-1 languages — so the
check is measured on real language, not on a word list fed back to itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from podcast_scraper import config as config_mod
from podcast_scraper.languages_guard import (
    transcript_language_contradiction,
    TRANSCRIPT_LANGUAGE_MIN_WORDS,
)

pytestmark = pytest.mark.unit

V3 = Path(__file__).resolve().parents[2] / "fixtures" / "transcripts" / "v3"
LANGUAGE_BY_FEED = {"p10": "es", "p11": "it", "p12": "fr", "p13": "de", "p14": "pt"}
LANGUAGES = ["en", *LANGUAGE_BY_FEED.values()]


def _fixtures() -> List[tuple]:
    out = []
    for path in sorted(V3.glob("p??_e??.txt")):
        text = path.read_text(encoding="utf-8")
        if len(text.split()) >= TRANSCRIPT_LANGUAGE_MIN_WORDS:
            out.append((path.stem, LANGUAGE_BY_FEED.get(path.stem[:3], "en"), text))
    return out


FIXTURES = _fixtures()


def test_the_fixtures_cover_every_tier_one_language() -> None:
    assert {lang for _, lang, _ in FIXTURES} == set(LANGUAGES)


@pytest.mark.parametrize("stem,lang,text", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_a_correctly_declared_transcript_passes(stem: str, lang: str, text: str) -> None:
    """A false refusal loses a good episode, so every fixture must pass under its own tag —
    Portuguese included, whose word list is the thinnest (0.051 share at its lowest)."""
    assert transcript_language_contradiction(text, lang) is None


@pytest.mark.parametrize("stem,lang,text", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_a_wrongly_declared_transcript_is_refused_under_every_other_tag(
    stem: str, lang: str, text: str
) -> None:
    for declared in LANGUAGES:
        if declared != lang:
            assert (
                transcript_language_contradiction(text, declared) is not None
            ), f"{stem} ({lang}) passed as {declared!r}"


def test_regional_tags_are_judged_by_their_primary_language() -> None:
    _, _, spanish = next(f for f in FIXTURES if f[1] == "es")
    assert transcript_language_contradiction(spanish, "es-MX") is None
    assert transcript_language_contradiction(spanish, "en-US") is not None


def test_the_reason_names_both_languages_and_the_remedy() -> None:
    _, _, spanish = next(f for f in FIXTURES if f[1] == "es")
    reason = transcript_language_contradiction(spanish, "en") or ""
    assert "reads as 'es'" in reason
    assert "language is 'en'" in reason
    assert "operator override" in reason


class TestWhatItDoesNotJudge:
    def test_a_short_transcript_passes(self) -> None:
        _, _, spanish = next(f for f in FIXTURES if f[1] == "es")
        short = " ".join(spanish.split()[: TRANSCRIPT_LANGUAGE_MIN_WORDS - 50])
        assert transcript_language_contradiction(short, "en") is None

    @pytest.mark.parametrize("declared", [None, "", "sr", "ja"])
    def test_a_language_with_no_word_list_passes(self, declared: Any) -> None:
        """No list, no opinion: refusing would stop a language the text check cannot read."""
        _, _, spanish = next(f for f in FIXTURES if f[1] == "es")
        assert transcript_language_contradiction(spanish, declared) is None


class _ReachedDiarization(Exception):
    pass


class TestThePipelineRefusesTheEpisode:
    """Behaviour through ``transcribe_media_to_text``: refused after ASR, before diarization,
    with nothing written, and countable like every other refusal."""

    def _run(self, monkeypatch, text: str, requested: str) -> Dict[str, Any]:
        from podcast_scraper.workflow import episode_processor as ep

        seen: Dict[str, Any] = {"recorded": [], "incidents": [], "saved": False}
        result = {"text": text, "segments": [], "language_requested": requested}
        monkeypatch.setattr(ep, "_bind_episode_correlation", lambda job, cfg: None)
        monkeypatch.setattr(ep, "_maybe_dispatch_reprocess_stage", lambda *a, **k: None)
        monkeypatch.setattr(ep, "_check_and_reuse_existing_transcript", lambda *a, **k: None)
        monkeypatch.setattr(ep, "_check_transcript_cache", lambda *a, **k: None)
        monkeypatch.setattr(ep, "_preprocess_audio_if_needed", lambda job, cfg, m, pm: m)
        monkeypatch.setattr(ep, "_cleanup_temp_media", lambda *a, **k: None)
        monkeypatch.setattr(
            ep, "_transcribe_with_segments_maybe_chunked", lambda *a, **k: (dict(result), 1.0)
        )

        def _saved(*a: Any, **k: Any) -> str:
            seen["saved"] = True
            return "transcripts/x.txt"

        monkeypatch.setattr(ep, "_save_transcript_file", _saved)
        monkeypatch.setattr(
            ep,
            "_record_unresolved_transcript",
            lambda job, cfg, pm, stage, *, error_type="", detail=None: seen["recorded"].append(
                {"stage": stage, "error_type": error_type, "detail": detail}
            ),
        )
        monkeypatch.setattr(
            ep,
            "_append_transcription_incident",
            lambda cfg, job, *, category="", message="", exception_type="": seen[
                "incidents"
            ].append({"category": category, "exception_type": exception_type}),
        )

        def _diarization(*a: Any, **k: Any) -> Any:
            # Not an exception the pipeline catches, so reaching diarization ends the run here.
            seen["diarized"] = True
            raise _ReachedDiarization

        seen["diarized"] = False
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.diarization.pipeline.apply_diarization_to_result",
            _diarization,
        )
        cfg = config_mod.Config(rss="https://example.com/f.xml").model_copy(
            update={"feed_declared_language": requested, "diarize": True}
        )
        job = MagicMock()
        job.idx = 1
        job.temp_media = None
        try:
            seen["return"] = ep.transcribe_media_to_text(
                job, cfg, None, None, "/tmp/out", MagicMock(), None
            )
        except _ReachedDiarization:
            seen["return"] = None
        return seen

    def test_spanish_text_under_an_english_tag_fails_before_anything_is_written(
        self, monkeypatch
    ) -> None:
        _, _, spanish = next(f for f in FIXTURES if f[1] == "es")
        seen = self._run(monkeypatch, spanish, "en")
        assert seen["return"] == (False, None, 0)
        assert seen["diarized"] is False
        assert seen["saved"] is False
        assert [r["error_type"] for r in seen["recorded"]] == ["TranscriptLanguageMismatch"]
        assert "reads as 'es'" in (seen["recorded"][0]["detail"] or "")
        assert seen["incidents"] == [
            {"category": "hard", "exception_type": "TranscriptLanguageMismatch"}
        ]

    def test_a_correctly_declared_episode_is_not_refused(self, monkeypatch) -> None:
        """Mutation partner of the above: the same harness, the right tag, no refusal — so the
        refusal is the guard's doing, not the harness's."""
        _, _, english = next(f for f in FIXTURES if f[1] == "en")
        seen = self._run(monkeypatch, english, "en")
        assert seen["diarized"] is True
        assert seen["recorded"] == []
        assert seen["incidents"] == []
