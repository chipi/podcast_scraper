#!/usr/bin/env python3
"""Groq provider E2E tests (ADR-147).

These tests verify that the first-class Groq provider works correctly in complete pipeline
workflows, through the E2E server's mock endpoints (``configure_groq_mock_server`` in
tests/e2e/conftest.py, autouse) — real HTTP requests, no network (``--disable-socket``).

Unlike deepseek/grok/qwen, Groq is DUAL-USE: the SAME ``GroqProvider`` class serves both the LLM
stages (speaker detection / summarization, via ``/v1/chat/completions``) AND whisper-large-v3-
turbo transcription (via ``/v1/audio/transcriptions``). This file covers both halves plus one
combined test that exercises them together in a single pipeline run — the case that can only be
observed at the E2E layer, since integration tests exercise each half in isolation.

Do NOT confuse ``groq`` (this provider) with ``grok`` (xAI). One letter apart.

Real API Mode:
    When USE_REAL_GROQ_API=1, tests use the real Groq API and (optionally) a real RSS feed.
    This is for manual testing only, incurs API costs, and must never run in CI.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.llm]

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import workflow

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import from parent conftest explicitly to avoid pytest resolution issues
from tests.conftest import (  # noqa: E402
    create_test_config,
)

# Check if we should use real Groq API (for manual testing only) — mirrors
# configure_groq_mock_server's own USE_REAL_GROQ_API gate in tests/e2e/conftest.py.
USE_REAL_GROQ_API = os.getenv("USE_REAL_GROQ_API", "0") == "1"

# Feed selection — shared convention across LLM provider E2E suites (see
# test_deepseek_provider_e2e.py / test_openai_provider_e2e.py). Default "multi" works in both
# fast and multi_episode E2E_TEST_MODE.
LLM_TEST_FEED = os.getenv("LLM_TEST_FEED", "multi")
REAL_TEST_RSS_FEED = os.getenv("LLM_TEST_RSS_FEED", None)

_FEED_MAPPING = {
    "multi": "podcast1_multi_episode",
    "fast": "podcast1",
    "p01": "podcast1",
    "p02": "podcast2",
    "p03": "podcast3",
    "p04": "podcast4",
    "p05": "podcast5",
}

# Groq's own non-empty model ids (unlike openai/deepseek, groq_speaker_model has NO built-in
# default — Config defaults it to "" — and an empty-string model has no pricing row, so the
# #650/#651 cost assertions below would silently see calls>0/cost==0 without this).
_SPEAKER_MODEL = "llama-3.3-70b-versatile"
_SUMMARY_MODEL = "llama-3.3-70b-versatile"

# Unlike test_openai_provider_e2e.py / test_deepseek_provider_e2e.py's conftest-level dummy keys
# (OPENAI_API_KEY / DEEPSEEK_API_KEY, set in tests/e2e/conftest.py), there is no dummy
# GROQ_API_KEY fallback registered there yet. GroqProvider's auth is warn-not-raise so a missing
# key doesn't fail Config construction, but it does emit a loud per-provider-init warning and
# sends "EMPTY" as the bearer. The mock server doesn't validate auth, so functionally this would
# still work — but pinning an explicit dummy key here keeps this file's mock-mode runs quiet and
# self-contained instead of relying on a conftest.py addition (out of scope for this task).
_DUMMY_GROQ_API_KEY = "gsk_e2e-key"


def _get_test_feed_url(
    e2e_server: Optional[Any] = None,
) -> tuple[str, Optional[str], Optional[str]]:
    """Get RSS feed URL and Groq config based on LLM_TEST_FEED.

    Returns:
        Tuple of (rss_url, groq_api_base, groq_api_key).
    """
    feed_type = (LLM_TEST_FEED or "multi").lower()

    if USE_REAL_GROQ_API:
        if REAL_TEST_RSS_FEED is not None:
            return REAL_TEST_RSS_FEED, None, None
        if e2e_server is None:
            raise ValueError(
                "E2E server is required when using fixture feeds with real API. "
                "Set LLM_TEST_RSS_FEED=<url> to use a real RSS feed instead."
            )
        podcast_name = _FEED_MAPPING.get(feed_type, "podcast1_multi_episode")
        rss_url = e2e_server.urls.feed(podcast_name)
        return rss_url, None, None

    if e2e_server is None:
        raise ValueError("E2E server is required for mocked API tests")

    podcast_name = _FEED_MAPPING.get(feed_type, "podcast1_multi_episode")
    rss_url = e2e_server.urls.feed(podcast_name)
    groq_api_base = e2e_server.urls.groq_api_base()
    return rss_url, groq_api_base, _DUMMY_GROQ_API_KEY


def _provider_info(metadata_content: dict, stage: str) -> dict:
    processing = metadata_content.get("processing", {})
    config_snapshot = processing.get("config_snapshot", {})
    ml_providers = config_snapshot.get("ml_providers", {})
    info: dict = ml_providers.get(stage, {})
    return info


# The mock server's /v1/audio/transcriptions handler always answers with this literal marker
# (tests/e2e/fixtures/e2e_http_server.py:_handle_audio_transcriptions), regardless of provider —
# finding it in a saved transcript proves the HTTP call actually reached the mock, as opposed to
# the episode's transcript having been direct-downloaded from a <podcast:transcript> RSS URL
# (metadata's ``ml_providers.transcription.provider`` reflects cfg.transcription_provider
# unconditionally, so it stays "groq" either way and can't distinguish the two on its own).
_MOCK_TRANSCRIPTION_MARKER = "This is a test transcription of"


def _any_transcript_hit_the_mock(temp_dir: Path) -> bool:
    for txt in Path(temp_dir).rglob("*.txt"):
        if "cleaned" in txt.name:
            continue
        try:
            if _MOCK_TRANSCRIPTION_MARKER in txt.read_text(encoding="utf-8"):
                return True
        except OSError:
            continue
    return False


@pytest.mark.slow
@pytest.mark.groq
class TestGroqProviderE2E:
    """Groq LLM stages (speaker detection / summarization) through the mock server.

    Transcription stays on local Whisper here so these tests isolate the LLM half — the dual-use
    (transcription + LLM together) case is exercised separately below.
    """

    def test_groq_speaker_detection_in_pipeline(self, e2e_server: Optional[Any]):
        """Groq speaker detection provider in the full pipeline."""
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            rss_url, groq_api_base, groq_api_key = _get_test_feed_url(e2e_server)
            config_kwargs = {
                "rss_url": rss_url,
                "output_dir": temp_dir,
                "transcription_provider": "whisper",
                "speaker_detector_provider": "groq",
                "summary_provider": "groq",
                "groq_speaker_model": _SPEAKER_MODEL,
                "groq_summary_model": _SUMMARY_MODEL,
                "generate_summaries": True,
            }
            config_kwargs["groq_api_key"] = groq_api_key
            if groq_api_base is not None:
                config_kwargs["groq_api_base"] = groq_api_base
            config_kwargs["auto_speakers"] = True
            config_kwargs["generate_metadata"] = True
            config_kwargs["max_episodes"] = int(os.getenv("LLM_TEST_MAX_EPISODES", "1"))
            config_kwargs["transcribe_missing"] = True

            cfg = create_test_config(**config_kwargs)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            assert transcripts_saved > 0, "Should have saved at least one transcript"

            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "Should have created at least one metadata file"

            with open(metadata_files[0]) as f:
                metadata_content = json.load(f)
            speaker_info = _provider_info(metadata_content, "speaker_detection")
            assert speaker_info.get("provider") == "groq"
        finally:
            if not USE_REAL_GROQ_API:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_groq_summarization_in_pipeline(self, e2e_server: Optional[Any]):
        """Groq summarization provider in the full pipeline."""
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            rss_url, groq_api_base, groq_api_key = _get_test_feed_url(e2e_server)
            config_kwargs = {
                "rss_url": rss_url,
                "output_dir": temp_dir,
                "transcription_provider": "whisper",
                "speaker_detector_provider": "spacy",
                "summary_provider": "groq",
                "groq_summary_model": _SUMMARY_MODEL,
                "generate_summaries": True,
                "generate_metadata": True,
            }
            if groq_api_key is not None:
                config_kwargs["groq_api_key"] = groq_api_key
            if groq_api_base is not None:
                config_kwargs["groq_api_base"] = groq_api_base
            config_kwargs["max_episodes"] = int(os.getenv("LLM_TEST_MAX_EPISODES", "1"))

            cfg = create_test_config(**config_kwargs)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            assert transcripts_saved > 0, "Should have saved at least one transcript"

            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "Should have created at least one metadata file"

            with open(metadata_files[0]) as f:
                metadata_content = json.load(f)
            summary_info = _provider_info(metadata_content, "summarization")
            assert summary_info.get("provider") == "groq"
            assert "summary" in metadata_content, "Summary should exist in metadata"
        finally:
            if not USE_REAL_GROQ_API:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_groq_full_pipeline(self, e2e_server: Optional[Any]):
        """Both Groq LLM providers (speaker + summary) together; transcription local (Whisper)."""
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            rss_url, groq_api_base, groq_api_key = _get_test_feed_url(e2e_server)
            config_kwargs = {
                "rss_url": rss_url,
                "output_dir": temp_dir,
                "transcription_provider": "whisper",
                "speaker_detector_provider": "groq",
                "summary_provider": "groq",
                "groq_speaker_model": _SPEAKER_MODEL,
                "groq_summary_model": _SUMMARY_MODEL,
                "auto_speakers": True,
                "generate_metadata": True,
                "generate_summaries": True,
                "preload_models": False,
                "transcribe_missing": True,
                "max_episodes": int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            }
            if groq_api_key is not None:
                config_kwargs["groq_api_key"] = groq_api_key
            if groq_api_base is not None:
                config_kwargs["groq_api_base"] = groq_api_base

            cfg = create_test_config(**config_kwargs)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            assert transcripts_saved > 0, "Should have saved at least one transcript"

            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "Should have created at least one metadata file"

            with open(metadata_files[0]) as f:
                metadata_content = json.load(f)
            assert _provider_info(metadata_content, "speaker_detection").get("provider") == "groq"
            assert _provider_info(metadata_content, "summarization").get("provider") == "groq"

            # #650/#651 cost assertions: Groq powers speaker + summary (billable).
            # Transcription is local Whisper (free) in this test.
            from tests.e2e.conftest import assert_cost_fields_populated

            assert_cost_fields_populated(
                Path(temp_dir),
                billable_stages=["speaker_detection", "summarization"],
                local_stages=["transcription"],
            )
        finally:
            if not USE_REAL_GROQ_API:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_groq_mega_bundled_pipeline_records_cost(self, e2e_server: Optional[Any]):
        """#650/#651 mega_bundled parity for Groq (LLM stages only; transcription local)."""
        temp_dir = tempfile.mkdtemp()
        try:
            rss_url, groq_api_base, groq_api_key = _get_test_feed_url(e2e_server)
            config_kwargs = {
                "rss_url": rss_url,
                "output_dir": temp_dir,
                "transcription_provider": "whisper",
                "speaker_detector_provider": "groq",
                "summary_provider": "groq",
                "groq_speaker_model": _SPEAKER_MODEL,
                "groq_summary_model": _SUMMARY_MODEL,
                "auto_speakers": True,
                "generate_metadata": True,
                "generate_summaries": True,
                "preload_models": False,
                "transcribe_missing": True,
                "llm_pipeline_mode": "mega_bundled",
                "max_episodes": int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            }
            if groq_api_key is not None:
                config_kwargs["groq_api_key"] = groq_api_key
            if groq_api_base is not None:
                config_kwargs["groq_api_base"] = groq_api_base
            cfg = create_test_config(**config_kwargs)

            transcripts_saved, summary = workflow.run_pipeline(cfg)
            assert transcripts_saved > 0

            from tests.e2e.conftest import assert_cost_fields_populated

            assert_cost_fields_populated(
                Path(temp_dir),
                billable_stages=["speaker_detection", "summarization"],
                local_stages=["transcription"],
            )
        finally:
            if not USE_REAL_GROQ_API:
                shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.groq
class TestGroqProviderTranscriptionE2E:
    """Groq transcription (whisper-large-v3-turbo) through the mock server.

    Mirrors test_openai_provider_e2e.py's ``test_openai_transcription_in_pipeline`` — LLM stages
    stay off/local here so this isolates the transcription half.
    """

    def test_groq_transcription_in_pipeline(self, e2e_server: Optional[Any]):
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            rss_url, groq_api_base, groq_api_key = _get_test_feed_url(e2e_server)
            config_kwargs = {
                "rss_url": rss_url,
                "output_dir": temp_dir,
                "transcription_provider": "groq",
                "speaker_detector_provider": "spacy",
                "summary_provider": "transformers",
                "transcribe_missing": True,
                "generate_metadata": True,
                "generate_summaries": False,
                "auto_speakers": False,
                "preload_models": False,
                # The default "multi" feed's episodes 1-2 ship a <podcast:transcript> URL (direct
                # download wins over transcribe_missing), episodes 3-5 do not — max_episodes must
                # cover at least one of those or this test would pass even with a broken
                # provider.transcribe() (no mock /v1/audio/transcriptions hit, just an RSS
                # download). "5" (whole feed) guarantees at least one real transcription call.
                "max_episodes": int(os.getenv("LLM_TEST_MAX_EPISODES", "5")),
            }
            if groq_api_key is not None:
                config_kwargs["groq_api_key"] = groq_api_key
            if groq_api_base is not None:
                config_kwargs["groq_api_base"] = groq_api_base

            cfg = create_test_config(**config_kwargs)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            assert transcripts_saved > 0, "Should have saved at least one transcript"

            transcript_files = list(Path(temp_dir).rglob("*.txt")) + list(
                Path(temp_dir).rglob("*.vtt")
            )
            assert len(transcript_files) >= 1, "Should have created at least one transcript file"

            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "Should have created at least one metadata file"
            with open(metadata_files[0]) as f:
                metadata_content = json.load(f)
            transcription_info = _provider_info(metadata_content, "transcription")
            assert transcription_info.get("provider") == "groq"

            # Proves the mock /v1/audio/transcriptions was actually hit (not just that a
            # transcript file exists — episodes 1-2 of the "multi" feed ship a
            # <podcast:transcript> URL and would produce one via direct download even if
            # provider.transcribe() were completely broken; the metadata "provider" field above
            # reflects cfg.transcription_provider unconditionally and can't distinguish the two).
            # NOTE: NOT asserting llm_transcription_calls/cost via assert_cost_fields_populated
            # here — empirically (verified with a throwaway debug harness against BOTH groq and
            # openai transcription_provider, same multi-episode setup, not committed) that stays
            # 0 even though the mock IS hit and the transcript content proves it. This reproduces
            # for openai too, so it isn't a groq-specific regression; likely an artifact of the
            # async transcription worker queue not propagating episode_duration_seconds /
            # pipeline_metrics the same way the synchronous LLM stages do. See this file's final
            # report for the full writeup — flagged, not fixed (non-test code, out of scope here).
            assert _any_transcript_hit_the_mock(Path(temp_dir)), (
                "No saved transcript contains the mock server's marker text — groq's "
                "transcribe() path was not exercised (all episodes may have used direct "
                "RSS download instead)."
            )
        finally:
            if not USE_REAL_GROQ_API:
                shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.groq
class TestGroqDualUseE2E:
    """The headline dual-use case (ADR-147): ONE pipeline run where transcription_provider=groq
    AND summary_provider=groq (+ speaker_detector_provider=groq) — both halves of the same
    provider class, end to end, through the mock server in a single run. Unlike deepseek/grok
    (chat-only) and unlike a unit/integration test of either half in isolation, this is the only
    layer that proves the two capabilities don't stomp on each other's client/state when driven by
    one GroqProvider instance in one pipeline execution.
    """

    def test_groq_transcription_and_summarization_together(self, e2e_server: Optional[Any]):
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            rss_url, groq_api_base, groq_api_key = _get_test_feed_url(e2e_server)
            config_kwargs = {
                "rss_url": rss_url,
                "output_dir": temp_dir,
                "transcription_provider": "groq",
                "speaker_detector_provider": "groq",
                "summary_provider": "groq",
                "groq_speaker_model": _SPEAKER_MODEL,
                "groq_summary_model": _SUMMARY_MODEL,
                "transcribe_missing": True,
                "auto_speakers": True,
                "generate_metadata": True,
                "generate_summaries": True,
                "preload_models": False,
                # See test_groq_transcription_in_pipeline: episodes 1-2 of the default "multi" feed
                # ship a transcript URL (direct download bypasses transcribe()), so max_episodes
                # must reach episode 3+ or the marker-text assertion below would pass without ever
                # hitting the mock's /v1/audio/transcriptions.
                "max_episodes": int(os.getenv("LLM_TEST_MAX_EPISODES", "5")),
            }
            if groq_api_key is not None:
                config_kwargs["groq_api_key"] = groq_api_key
            if groq_api_base is not None:
                config_kwargs["groq_api_base"] = groq_api_base

            cfg = create_test_config(**config_kwargs)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            assert transcripts_saved > 0, "Should have saved at least one transcript"

            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "Should have created at least one metadata file"

            with open(metadata_files[0]) as f:
                metadata_content = json.load(f)

            # Both halves of the dual-use provider routed through groq in the SAME run.
            assert _provider_info(metadata_content, "transcription").get("provider") == "groq"
            assert _provider_info(metadata_content, "speaker_detection").get("provider") == "groq"
            assert _provider_info(metadata_content, "summarization").get("provider") == "groq"
            assert "summary" in metadata_content, "Summary should exist in metadata"

            # The transcription half actually reached the mock (see
            # test_groq_transcription_in_pipeline for why this isn't a cost/calls assertion).
            assert _any_transcript_hit_the_mock(Path(temp_dir)), (
                "No saved transcript contains the mock server's marker text — groq's "
                "transcribe() path was not exercised in this dual-use run."
            )

            # #650/#651: the LLM half (speaker + summary) is billable and IS reliably recorded
            # (synchronous call path, unlike transcription's async worker queue — see above).
            from tests.e2e.conftest import assert_cost_fields_populated

            assert_cost_fields_populated(
                Path(temp_dir),
                billable_stages=["speaker_detection", "summarization"],
                local_stages=[],
            )
        finally:
            if not USE_REAL_GROQ_API:
                shutil.rmtree(temp_dir, ignore_errors=True)
