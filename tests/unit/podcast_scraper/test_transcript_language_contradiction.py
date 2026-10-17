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


def _run_pipeline(monkeypatch, text: str, requested: str, **overrides: Any) -> Dict[str, Any]:
    from podcast_scraper.workflow import episode_processor as ep

    seen: Dict[str, Any] = {"recorded": [], "decided": [], "incidents": [], "saved": False}
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
        "_mark_reprocess_nothing_to_do",
        lambda episode, cfg, pm, stage, *, error_type, detail, satisfied=False, refused=False: seen[
            "decided"
        ].append({"stage": stage, "error_type": error_type, "refused": refused}),
    )
    monkeypatch.setattr(
        ep,
        "_append_transcription_incident",
        lambda cfg, job, *, category="", message="", exception_type="": seen["incidents"].append(
            {"category": category, "exception_type": exception_type}
        ),
    )

    def _diarization(*a: Any, **k: Any) -> Any:
        # Not an exception the pipeline catches, so reaching diarization ends the run here.
        seen["diarized"] = True
        seen["transcribe_clip"] = k.get("transcribe_clip")
        seen["diarized_result"], seen["diarized_cfg"] = a[0], a[2]
        raise _ReachedDiarization

    seen["diarized"] = False
    monkeypatch.setattr(
        "podcast_scraper.providers.ml.diarization.pipeline.apply_diarization_to_result",
        _diarization,
    )
    monkeypatch.setattr(
        ep,
        "_record_language_refusal",
        lambda episode, cfg, detail, *, detected=None: seen.setdefault("refusals", []).append(
            {"detail": detail, "detected": detected}
        ),
    )
    # #2375: a refusal withdraws the episode's served runs; recorded here, never a real corpus scan.
    monkeypatch.setattr(
        ep,
        "_withdraw_served_runs",
        lambda episode, cfg, detected: seen.setdefault("withdrawn", []).append(detected) or 0,
    )
    if "identified" in overrides:
        identified = overrides.pop("identified")
        monkeypatch.setattr(ep, "_identify_episode_language", lambda *a, **k: identified)
    cfg = config_mod.Config(rss="https://example.com/f.xml").model_copy(
        update={"feed_declared_language": requested, "diarize": True, **overrides}
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


class TestThePipelineRefusesTheEpisode:
    """Behaviour through ``transcribe_media_to_text``: refused after ASR, before diarization,
    with nothing written, and countable like every other refusal."""

    def _run(self, monkeypatch, text: str, requested: str, **overrides: Any) -> Dict[str, Any]:
        return _run_pipeline(monkeypatch, text, requested, **overrides)

    def test_spanish_text_under_an_english_tag_fails_before_anything_is_written(
        self, monkeypatch
    ) -> None:
        _, _, spanish = next(f for f in FIXTURES if f[1] == "es")
        seen = self._run(monkeypatch, spanish, "en")
        assert seen["return"] == (False, None, 0)
        assert seen["diarized"] is False
        assert seen["saved"] is False
        # #2375: refusing is the guard working, so the episode is a decision, not a failure: a
        # `skipped` ledger row and a work-list request met, and a `policy` incident, not `hard`.
        assert seen["recorded"] == []
        assert seen["decided"] == [
            {"stage": "transcription", "error_type": "TranscriptLanguageMismatch", "refused": True}
        ]
        assert "reads as 'es'" in seen["refusals"][0]["detail"]
        assert seen["incidents"] == [
            {"category": "policy", "exception_type": "TranscriptLanguageMismatch"}
        ]

    def test_a_correctly_declared_episode_is_not_refused(self, monkeypatch) -> None:
        """Mutation partner of the above: the same harness, the right tag, no refusal — so the
        refusal is the guard's doing, not the harness's."""
        _, _, english = next(f for f in FIXTURES if f[1] == "en")
        seen = self._run(monkeypatch, english, "en")
        assert seen["diarized"] is True
        assert seen["recorded"] == [] and seen["decided"] == []
        assert seen["incidents"] == []
        # #2187 A2: the fresh-transcription path hands diarization a way to re-transcribe gaps.
        assert callable(seen["transcribe_clip"])

    def test_recovery_switched_off_hands_diarization_no_transcriber(self, monkeypatch) -> None:
        """#2187 A2 off-switch: gaps are still detected (diarization runs), never re-transcribed."""
        _, _, english = next(f for f in FIXTURES if f[1] == "en")
        seen = self._run(
            monkeypatch, english, "en", transcription_recover_untranscribed_speech=False
        )
        assert seen["diarized"] is True
        assert seen["transcribe_clip"] is None


class TestAnEpisodeInAnotherLanguageThanItsShow:
    """#2369 / RFC-130: a flagged episode the ASR identifies as an enabled language continues in
    that language; anything else is refused as before, with what was identified recorded."""

    def _run(self, monkeypatch, text: str, requested: str, **overrides: Any) -> Dict[str, Any]:
        return _run_pipeline(monkeypatch, text, requested, **overrides)

    def test_an_enabled_language_continues_in_that_language(self, monkeypatch) -> None:
        _, _, spanish = next(f for f in FIXTURES if f[1] == "es")
        seen = self._run(monkeypatch, spanish, "en", identified="es")
        assert seen["diarized"] is True
        assert seen["recorded"] == [] and seen["incidents"] == []
        assert "withdrawn" not in seen, "a continuing episode withdrew its runs"
        assert seen["diarized_result"]["language_detected"] == "es"
        assert seen["diarized_cfg"].language_override == "es"

    def test_a_language_that_is_not_enabled_is_refused_with_it_recorded(self, monkeypatch) -> None:
        _, _, spanish = next(f for f in FIXTURES if f[1] == "es")
        seen = self._run(monkeypatch, spanish, "en", identified="tr")
        assert seen["diarized"] is False and seen["saved"] is False
        assert [d["error_type"] for d in seen["decided"]] == ["TranscriptLanguageMismatch"]
        assert seen["refusals"][0]["detected"] == "tr"
        assert "identified as 'tr'" in seen["refusals"][0]["detail"]
        assert seen["withdrawn"] == ["tr"], "the refused episode's served runs were not withdrawn"

    def test_no_identification_is_refused_as_before(self, monkeypatch) -> None:
        _, _, spanish = next(f for f in FIXTURES if f[1] == "es")
        seen = self._run(monkeypatch, spanish, "en", identified=None)
        assert seen["diarized"] is False
        assert seen["refusals"][0]["detected"] is None
        assert "identified" not in seen["refusals"][0]["detail"]

    def test_an_episode_in_its_shows_language_is_never_identified(self, monkeypatch) -> None:
        from podcast_scraper.workflow import episode_processor as ep

        def _must_not_run(*a: Any, **k: Any) -> Any:
            raise AssertionError("identification ran on an episode in its show's language")

        monkeypatch.setattr(ep, "_identify_episode_language", _must_not_run)
        _, _, english = next(f for f in FIXTURES if f[1] == "en")
        seen = self._run(monkeypatch, english, "en")
        assert seen["diarized"] is True
