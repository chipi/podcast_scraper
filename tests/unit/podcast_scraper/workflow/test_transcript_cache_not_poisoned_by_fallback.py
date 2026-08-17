"""The transcript cache must not store a transcript under a key that lies about its audio (#35).

THE DEFECT
The transcript cache key includes ``preprocessing_fingerprint(cfg)``, whose own docstring calls it
"identity of the audio the transcriber will actually see (#1173)". It is computed from CONFIG:
``pp=off`` when preprocessing is disabled, otherwise ``pp=on|sr=…|silrm=…|…``.

But preprocessing can be enabled and still produce nothing — that is exactly the #18/#558 damage,
where a 300 s flat budget killed preprocessing on long episodes and the pipeline fell back to
sending the ORIGINAL file to the provider. The transcript that came back was built from RAW audio
and got cached under a key claiming ``pp=on``. The function violates its own stated invariant.

WHY THIS BLOCKS THE REPAIR SPECIFICALLY
The runbook's step 6 re-transcribes the episodes damaged by #18. The cache is keyed on the hash of
the ORIGINAL media, which has not changed, plus the same ``pp=on`` fingerprint, which also has not
changed — because the config was always ``pp=on``; it was the RUN that failed. So the repair run
scores a cache hit and re-serves the very transcript it was launched to replace, and every gate
downstream of it goes green. A false green on the one run whose entire purpose is fixing that
transcript.

THE FIX, at cause: do not WRITE a cache entry whose fingerprint the run did not honour. A run that
asked for preprocessing and fell back to raw audio produces a transcript that is still perfectly
usable for THIS run — it just must not be replayed under a key that misdescribes it.

Deliberately NOT fixed by putting the preprocessing outcome in the key: the outcome is unknown
until after the work the key exists to skip, and it would silently miss every entry already
written.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.workflow import episode_processor

pytestmark = [pytest.mark.unit]


@pytest.fixture
def media(tmp_path):
    """A raw media file standing in for the downloaded episode audio."""
    path = tmp_path / "episode_raw.mp3"
    path.write_bytes(b"raw audio bytes")
    return str(path)


@pytest.fixture
def preprocessed(tmp_path):
    """The file preprocessing WOULD have produced — a different path, which is the signal."""
    path = tmp_path / "episode_preprocessed.mp3"
    path.write_bytes(b"preprocessed audio bytes")
    return str(path)


def _cfg(**overrides: Any) -> config.Config:
    base: Dict[str, Any] = {
        "rss_url": "https://example.com/feed.xml",
        "transcript_cache_enabled": True,
        "preprocessing_enabled": True,
    }
    base.update(overrides)
    return config.Config(**base)


def _save(cfg: config.Config, temp_media: str, media_for_transcription: Optional[str]) -> Mock:
    """Run the cache-write helper and return the mock of the underlying cache writer."""
    job = Mock()
    job.idx = 1
    with (
        patch("podcast_scraper.cache.transcript_cache.save_transcript_to_cache") as mock_save,
        patch("podcast_scraper.cache.transcript_cache.get_audio_hash", return_value="hash-abc"),
        patch(
            "podcast_scraper.workflow.episode_processor._get_provider_model_name",
            return_value="base",
        ),
    ):
        episode_processor._save_transcript_to_cache_if_needed(
            job,
            cfg,
            temp_media,
            "The transcript text.",
            Mock(),
            media_for_transcription=media_for_transcription,
        )
    return mock_save


def test_fallback_to_raw_audio_is_not_cached(media):
    """THE defect: preprocessing was asked for, produced nothing, so the key would lie.

    ``media_for_transcription is temp_media`` means the preprocessor returned the original file —
    the #18 fallback. Caching here is what makes the step-6 repair run a false green.
    """
    mock_save = _save(_cfg(preprocessing_enabled=True), media, media)
    mock_save.assert_not_called()


def test_successful_preprocessing_is_cached(media, preprocessed):
    """The normal path must be untouched: a real preprocessed file means the key is honest."""
    mock_save = _save(_cfg(preprocessing_enabled=True), media, preprocessed)
    mock_save.assert_called_once()
    assert mock_save.call_args.kwargs["preprocessing"].startswith("pp=on")


def test_preprocessing_disabled_is_cached(media):
    """``pp=off`` with raw audio is an HONEST key, not a lie — it must still cache.

    This is the case that stops the fix from being "never cache when the paths match", which
    would disable the transcript cache for every user who has preprocessing turned off.
    """
    mock_save = _save(_cfg(preprocessing_enabled=False), media, media)
    mock_save.assert_called_once()
    assert mock_save.call_args.kwargs["preprocessing"] == "pp=off"


def test_the_skip_is_logged_loudly_enough_to_explain_a_cache_miss(media, caplog):
    """A silently absent cache entry is the kind of thing that gets debugged for an hour.

    The run still succeeds and still produces a transcript; only the cache write is skipped. That
    must be visible, or the next person sees an unexplained cache miss on a re-run.
    """
    with caplog.at_level("WARNING"):
        _save(_cfg(preprocessing_enabled=True), media, media)
    seen = [record.message for record in caplog.records]
    assert any(
        "preprocessing" in message.lower() and "cache" in message.lower() for message in seen
    ), f"expected a warning naming preprocessing and the cache; got {seen}"


def test_caller_must_state_what_was_transcribed(media):
    """``media_for_transcription`` is keyword-only and REQUIRED, by design.

    A default would let a future call site silently reintroduce the poisoning — which is precisely
    how this bug existed: the helper was handed ``temp_media`` and had no way to know whether that
    was what the provider actually received.
    """
    with pytest.raises(TypeError):
        episode_processor._save_transcript_to_cache_if_needed(Mock(), _cfg(), media, "text", Mock())


def test_missing_media_file_still_short_circuits(tmp_path):
    """Pre-existing guard must survive: no media file, no cache write, no crash."""
    missing = str(tmp_path / "not_there.mp3")
    mock_save = _save(_cfg(), missing, missing)
    mock_save.assert_not_called()


def test_cache_disabled_still_short_circuits(media, preprocessed):
    """Pre-existing guard must survive: the config switch still wins over everything."""
    mock_save = _save(_cfg(transcript_cache_enabled=False), media, preprocessed)
    mock_save.assert_not_called()


def test_realpath_equality_counts_as_fallback(tmp_path):
    """The preprocessor returning the same file under a different spelling is still a fallback.

    Guards against a symlinked or non-normalised temp dir making the paths compare unequal while
    pointing at one file — the fingerprint would lie just the same.
    """
    raw = tmp_path / "episode.mp3"
    raw.write_bytes(b"raw")
    alias = tmp_path / "sub" / ".." / "episode.mp3"
    os.makedirs(tmp_path / "sub", exist_ok=True)
    mock_save = _save(_cfg(preprocessing_enabled=True), str(raw), str(alias))
    mock_save.assert_not_called()
