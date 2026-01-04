#!/usr/bin/env python3
"""Basic E2E tests (happy paths).

These tests verify basic functionality using real HTTP client and E2E server:
- CLI command: basic transcript download
- Library API: run_pipeline() with basic config
- Service API: service.run() with basic config

All tests use real HTTP client (no mocking) and E2E server fixture.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

import podcast_scraper
import podcast_scraper.cli as cli
from podcast_scraper import config

# Import cache helpers from integration tests
integration_dir = Path(__file__).parent.parent / "integration"
if str(integration_dir) not in sys.path:
    sys.path.insert(0, str(integration_dir))


@pytest.mark.e2e
@pytest.mark.critical_path
class TestBasicCLIE2E:
    """Basic CLI E2E tests using real HTTP client."""

    @pytest.mark.critical_path
    def test_cli_basic_transcript_download_path1(self, e2e_server):
        """Test complete CLI critical path (Path 1: when transcript URL exists).

        This test validates the COMPLETE Path 1 of the critical path:
        RSS → Parse → Download Transcript → NER → Summarization → Metadata → Files

        Uses podcast1_with_transcript which has a transcript URL, so Whisper is NOT needed.
        Uses real ML providers (local spaCy for NER, local transformers for summarization).
        Requires models to be pre-cached (skip if not available).
        Enables NER, summarization, and metadata to cover the full critical path.
        """
        # Require ML models to be cached (skip if not available)
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
        )

        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        rss_url = e2e_server.urls.feed("podcast1_with_transcript")

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = cli.main(
                [
                    rss_url,
                    "--output-dir",
                    tmpdir,
                    "--max-episodes",
                    "1",
                    "--auto-speakers",  # Enable NER (speaker detection) - uses local ML
                    "--generate-summaries",  # Enable summarization - uses local ML (transformers)
                    "--summary-model",
                    config.TEST_DEFAULT_SUMMARY_MODEL,  # Use test default (small, fast)
                    "--summary-reduce-model",
                    config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,  # Use cached test model
                    "--generate-metadata",  # Enable metadata generation
                ]
            )

            assert exit_code == 0, f"CLI should succeed, got exit code {exit_code}"

            # Verify transcript file was downloaded (not transcribed)
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) > 0, "At least one transcript file should be downloaded"

            # Verify metadata file was created (indicates NER and summarization ran)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata file should be created (indicates NER and summarization ran)"

            # Verify metadata contains NER and summarization results
            if metadata_files:
                import json

                with open(metadata_files[0], "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    # Should have speaker detection results (NER)
                    assert (
                        "detected_hosts" in metadata["content"]
                        or "detected_guests" in metadata["content"]
                        or "speakers" in metadata["content"]
                    ), "Metadata should contain speaker detection results (NER)"
                    # Should have summary (summarization) - summary is top-level
                    assert "summary" in metadata, "Metadata should contain summary"
                    assert metadata["summary"] is not None, "Summary should not be None"

    @pytest.mark.critical_path
    def test_cli_basic_transcript_download_path2(self, e2e_server):
        """Test complete CLI critical path (Path 2: when transcript URL missing).

        This test validates the COMPLETE Path 2 of the critical path:
        RSS → Parse → Download Audio → Whisper Transcription → NER →
        Summarization → Metadata → Files

        Uses podcast1 which has NO transcript URL, so Whisper transcription is needed.
        Uses real ML providers (local Whisper for transcription, local spaCy for NER,
        local transformers for summarization).
        Requires models to be pre-cached (skip if not available).
        Enables NER, summarization, and metadata to cover the full critical path.
        """
        # Require ML models to be cached (skip if not available)
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
            require_whisper_model_cached,
        )

        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        rss_url = e2e_server.urls.feed("podcast1")

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = cli.main(
                [
                    rss_url,
                    "--output-dir",
                    tmpdir,
                    "--max-episodes",
                    "1",
                    "--transcribe-missing",  # Enable Whisper transcription
                    "--whisper-model",
                    config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
                    "--auto-speakers",  # Enable NER (speaker detection) - uses local ML (spaCy)
                    "--generate-summaries",  # Enable summarization - uses local ML (transformers)
                    "--summary-model",
                    config.TEST_DEFAULT_SUMMARY_MODEL,  # Use test default (small, fast)
                    "--summary-reduce-model",
                    config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,  # Use cached test model
                    "--generate-metadata",  # Enable metadata generation
                ]
            )

            assert exit_code == 0, f"CLI should succeed, got exit code {exit_code}"

            # Verify transcript file was transcribed
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) > 0, "At least one transcript file should be transcribed"

            # Verify metadata file was created (indicates NER and summarization ran)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata file should be created (indicates NER and summarization ran)"

            # Verify metadata contains NER and summarization results
            if metadata_files:
                import json

                with open(metadata_files[0], "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    # Should have speaker detection results (NER)
                    assert (
                        "detected_hosts" in metadata["content"]
                        or "detected_guests" in metadata["content"]
                        or "speakers" in metadata["content"]
                    ), "Metadata should contain speaker detection results (NER)"
                    # Should have summary (summarization) - summary is top-level
                    assert "summary" in metadata, "Metadata should contain summary"
                    assert metadata["summary"] is not None, "Summary should not be None"


@pytest.mark.e2e
@pytest.mark.critical_path
class TestBasicLibraryAPIE2E:
    """Basic library API E2E tests using real HTTP client."""

    @pytest.mark.critical_path
    @pytest.mark.flaky
    def test_library_api_basic_pipeline_path1(self, e2e_server):
        """Test complete library API critical path (Path 1: transcript download).

        This test validates the COMPLETE Path 1 of the critical path using Library API:
        RSS → Parse → Download Transcript → NER → Summarization → Metadata → Files

        Uses podcast1_with_transcript which has a transcript URL, so Whisper is NOT needed.
        Uses real ML providers (local spaCy for NER, local transformers for summarization).
        Requires models to be pre-cached (skip if not available).
        Enables NER, summarization, and metadata to cover the full critical path.

        This test also validates MAP-REDUCE summarization workflow with REDUCE model
        (addresses issue #144). By setting summary_chunk_size=350, we force chunking
        even though the fast transcript (~345-388 tokens) would normally fit in a single
        chunk. This ensures REDUCE model (allenai/led-base-16384) is tested in fast tests.
        """
        # Require ML models to be cached (skip if not available)
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
        )

        # Require both MAP and REDUCE models to be cached
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL, None)

        rss_url = e2e_server.urls.feed("podcast1_with_transcript")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = config.Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=False,  # No transcription needed when transcript URL exists
                auto_speakers=True,  # Enable NER (speaker detection) - uses local ML (spaCy)
                generate_summaries=True,  # Enable summarization - uses local ML (transformers)
                summary_provider="transformers",  # Use transformers ML provider - default
                summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,  # MAP: bart-base
                summary_reduce_model=config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,  # REDUCE: led
                summary_chunk_size=350,  # Force chunking: fast transcript > chunk_size
                generate_metadata=True,  # Enable metadata generation
                metadata_format="json",
            )

            # Run pipeline
            podcast_scraper.run_pipeline(cfg)

            # Verify transcript file was downloaded (not transcribed)
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) > 0, "At least one transcript file should be downloaded"

            # Verify metadata file was created (indicates NER and summarization ran)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata file should be created (indicates NER and summarization ran)"

            # Verify metadata contains NER and summarization results
            if metadata_files:
                import json

                with open(metadata_files[0], "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    # Should have speaker detection results (NER)
                    assert (
                        "detected_hosts" in metadata["content"]
                        or "detected_guests" in metadata["content"]
                        or "speakers" in metadata["content"]
                    ), "Metadata should contain speaker detection results (NER)"
                    # Should have summary (summarization) - summary is top-level
                    assert "summary" in metadata, "Metadata should contain summary"
                    assert metadata["summary"] is not None, "Summary should not be None"

    @pytest.mark.critical_path
    def test_library_api_basic_pipeline_path2(self, e2e_server):
        """Test complete library API critical path (Path 2: transcription).

        This test validates the COMPLETE Path 2 of the critical path using Library API:
        RSS → Parse → Download Audio → Whisper Transcription → NER →
        Summarization → Metadata → Files

        Uses podcast1 which has NO transcript URL, so Whisper transcription is needed.
        Uses real ML providers (local Whisper for transcription, local spaCy for NER,
        local transformers for summarization).
        Requires models to be pre-cached (skip if not available).
        Enables NER, summarization, and metadata to cover the full critical path.
        """
        # Require ML models to be cached (skip if not available)
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
            require_whisper_model_cached,
        )

        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        rss_url = e2e_server.urls.feed("podcast1")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = config.Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,  # Enable Whisper transcription (mocked)
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                auto_speakers=True,  # Enable NER (speaker detection) - uses local ML (spaCy)
                generate_summaries=True,  # Enable summarization - uses local ML (transformers)
                summary_provider="transformers",  # Use transformers ML provider - default
                summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,  # Use test default (small, fast)
                summary_reduce_model=config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,  # Cached
                generate_metadata=True,  # Enable metadata generation
                metadata_format="json",
            )

            # Run pipeline
            podcast_scraper.run_pipeline(cfg)

            # Verify transcript file was transcribed
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) > 0, "At least one transcript file should be transcribed"

            # Verify metadata file was created (indicates NER and summarization ran)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata file should be created (indicates NER and summarization ran)"

            # Verify metadata contains NER and summarization results
            if metadata_files:
                import json

                with open(metadata_files[0], "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    # Should have speaker detection results (NER)
                    assert (
                        "detected_hosts" in metadata["content"]
                        or "detected_guests" in metadata["content"]
                        or "speakers" in metadata["content"]
                    ), "Metadata should contain speaker detection results (NER)"
                    # Should have summary (summarization) - summary is top-level
                    assert "summary" in metadata, "Metadata should contain summary"
                    assert metadata["summary"] is not None, "Summary should not be None"


@pytest.mark.e2e
@pytest.mark.critical_path
class TestBasicServiceAPIE2E:
    """Basic service API E2E tests using real HTTP client."""

    @pytest.mark.critical_path
    @pytest.mark.flaky
    def test_service_api_basic_run_path1(self, e2e_server):
        """Test complete service API critical path (Path 1: transcript download).

        This test validates the COMPLETE Path 1 of the critical path using Service API:
        RSS → Parse → Download Transcript → NER → Summarization → Metadata → Files

        Uses podcast1_with_transcript which has a transcript URL, so Whisper is NOT needed.
        Uses real ML providers (local spaCy for NER, local transformers for summarization).
        Requires models to be pre-cached (skip if not available).
        Enables NER, summarization, and metadata to cover the full critical path.
        """
        # Require ML models to be cached (skip if not available)
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
        )

        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        from podcast_scraper import service

        rss_url = e2e_server.urls.feed("podcast1_with_transcript")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = config.Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=False,  # No transcription needed when transcript URL exists
                auto_speakers=True,  # Enable NER (speaker detection) - uses local ML (spaCy)
                generate_summaries=True,  # Enable summarization - uses local ML (transformers)
                summary_provider="transformers",  # Use transformers ML provider - default
                summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,  # Use test default (small, fast)
                summary_reduce_model=config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,  # Cached
                generate_metadata=True,  # Enable metadata generation
                metadata_format="json",
            )

            # Run service
            result = service.run(cfg)

            # Verify service returned success
            assert result.success is True, f"Service should succeed, got: {result}"
            assert result.error is None, f"Service should not have errors, got: {result.error}"
            assert result.episodes_processed > 0, "At least one episode should be processed"

            # Verify transcript file was downloaded (not transcribed)
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) > 0, "At least one transcript file should be downloaded"

            # Verify metadata file was created (indicates NER and summarization ran)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata file should be created (indicates NER and summarization ran)"

            # Verify metadata contains NER and summarization results
            if metadata_files:
                import json

                with open(metadata_files[0], "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    # Should have speaker detection results (NER)
                    assert (
                        "detected_hosts" in metadata["content"]
                        or "detected_guests" in metadata["content"]
                        or "speakers" in metadata["content"]
                    ), "Metadata should contain speaker detection results (NER)"
                    # Should have summary (summarization) - summary is top-level
                    assert "summary" in metadata, "Metadata should contain summary"
                    assert metadata["summary"] is not None, "Summary should not be None"

    @pytest.mark.critical_path
    @pytest.mark.flaky
    def test_service_api_basic_run_path2(self, e2e_server):
        """Test complete service API critical path (Path 2: transcription).

        This test validates the COMPLETE Path 2 of the critical path using Service API:
        RSS → Parse → Download Audio → Whisper Transcription → NER →
        Summarization → Metadata → Files

        Uses podcast1 which has NO transcript URL, so Whisper transcription is needed.
        Uses real ML providers (local Whisper for transcription, local spaCy for NER,
        local transformers for summarization).
        Requires models to be pre-cached (skip if not available).
        Enables NER, summarization, and metadata to cover the full critical path.
        """
        # Require ML models to be cached (skip if not available)
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
        )

        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        from podcast_scraper import service

        rss_url = e2e_server.urls.feed("podcast1")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = config.Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,  # Enable Whisper transcription (mocked)
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                auto_speakers=True,  # Enable NER (speaker detection) - uses local ML (spaCy)
                generate_summaries=True,  # Enable summarization - uses local ML (transformers)
                summary_provider="transformers",  # Use transformers ML provider - default
                summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,  # Use test default (small, fast)
                summary_reduce_model=config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,  # Cached
                generate_metadata=True,  # Enable metadata generation
                metadata_format="json",
            )

            # Run service
            result = service.run(cfg)

            # Verify service returned success
            assert result.success is True, f"Service should succeed, got: {result}"
            assert result.error is None, f"Service should not have errors, got: {result.error}"
            assert result.episodes_processed > 0, "At least one episode should be processed"

            # Verify transcript file was transcribed
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) > 0, "At least one transcript file should be transcribed"

            # Verify metadata file was created (indicates NER and summarization ran)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata file should be created (indicates NER and summarization ran)"

            # Verify metadata contains NER and summarization results
            if metadata_files:
                import json

                with open(metadata_files[0], "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    # Should have speaker detection results (NER)
                    assert (
                        "detected_hosts" in metadata["content"]
                        or "detected_guests" in metadata["content"]
                        or "speakers" in metadata["content"]
                    ), "Metadata should contain speaker detection results (NER)"
                    # Should have summary (summarization) - summary is top-level
                    assert "summary" in metadata, "Metadata should contain summary"
                    assert metadata["summary"] is not None, "Summary should not be None"


# ============================================================================
# OpenAI Provider Tests (Skipped for now)
# ============================================================================


@pytest.mark.e2e
@pytest.mark.critical_path
@pytest.mark.openai
@pytest.mark.llm
class TestBasicCLIE2E_OpenAI:
    """OpenAI provider E2E tests using mock OpenAI endpoints on E2E server."""

    @pytest.mark.flaky
    def test_cli_basic_transcript_download_path1_openai(self, e2e_server, monkeypatch):
        """Test complete CLI critical path (Path 1) with OpenAI providers.

        This test validates Path 1 with OpenAI providers:
        RSS → Parse → Download Transcript → OpenAI NER → OpenAI Summarization → Metadata → Files

        Uses podcast1_with_transcript which has a transcript URL,
        so OpenAI transcription is NOT needed.
        Uses OpenAI providers for speaker detection and summarization
        (via E2E server mock endpoints).
        """
        rss_url = e2e_server.urls.feed("podcast1_with_transcript")

        # Set environment variable for API key (required for Config validation)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = cli.main(
                [
                    rss_url,
                    "--output-dir",
                    tmpdir,
                    "--max-episodes",
                    "1",
                    "--auto-speakers",  # Enable speaker detection - uses OpenAI provider
                    "--speaker-detector-provider",
                    "openai",  # Use OpenAI for speaker detection
                    "--generate-summaries",  # Enable summarization - uses OpenAI provider
                    "--summary-provider",
                    "openai",  # Use OpenAI for summarization
                    "--generate-metadata",  # Enable metadata generation
                    "--metadata-format",
                    "json",
                    "--openai-api-base",
                    e2e_server.urls.openai_api_base(),  # Use E2E server
                ]
            )

            assert exit_code == 0, "CLI should exit with code 0"

            # Verify transcript file was downloaded (not transcribed)
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) > 0, "At least one transcript file should be downloaded"

            # Verify metadata file was created (indicates OpenAI NER and summarization ran)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata file should be created (indicates OpenAI NER and summarization ran)"

            # Verify metadata contains OpenAI NER and summarization results
            if metadata_files:
                import json

                with open(metadata_files[0], "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    # Should have speaker detection results (OpenAI)
                    assert (
                        "detected_hosts" in metadata["content"]
                        or "detected_guests" in metadata["content"]
                        or "speakers" in metadata["content"]
                    ), "Metadata should contain speaker detection results (OpenAI)"
                    # Should have summary (OpenAI summarization) - summary is top-level
                    assert (
                        "summary" in metadata
                    ), "Metadata should contain summary (OpenAI summarization)"
                    assert metadata["summary"] is not None, "Summary should not be None"

    @pytest.mark.critical_path
    @pytest.mark.flaky
    def test_cli_basic_transcript_download_path2_openai(self, e2e_server, monkeypatch):
        """Test complete CLI critical path (Path 2) with OpenAI providers.

        This test validates Path 2 with OpenAI providers:
        RSS → Parse → Download Audio → OpenAI Transcription → OpenAI NER →
        OpenAI Summarization → Metadata → Files

        Uses podcast1 which has NO transcript URL, so OpenAI transcription is needed.
        Uses OpenAI providers for transcription, speaker detection, and summarization
        (via E2E server mock endpoints).
        """
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        # Set environment variable for API key (required for Config validation)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = cli.main(
                [
                    rss_url,
                    "--output-dir",
                    tmpdir,
                    "--max-episodes",
                    "1",
                    "--transcribe-missing",  # Enable OpenAI transcription
                    "--transcription-provider",
                    "openai",  # Use OpenAI for transcription
                    "--auto-speakers",  # Enable speaker detection - uses OpenAI provider
                    "--speaker-detector-provider",
                    "openai",  # Use OpenAI for speaker detection
                    "--generate-summaries",  # Enable summarization - uses OpenAI provider
                    "--summary-provider",
                    "openai",  # Use OpenAI for summarization
                    "--generate-metadata",  # Enable metadata generation
                    "--metadata-format",
                    "json",
                    "--openai-api-base",
                    e2e_server.urls.openai_api_base(),  # Use E2E server
                ]
            )

            assert exit_code == 0, "CLI should exit with code 0"

            # Verify transcript file was transcribed
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) > 0, "At least one transcript file should be transcribed"

            # Verify metadata file was created (indicates OpenAI NER and summarization ran)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata file should be created (indicates OpenAI NER and summarization ran)"

            # Verify metadata contains OpenAI NER and summarization results
            if metadata_files:
                import json

                with open(metadata_files[0], "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    # Should have speaker detection results (OpenAI)
                    assert (
                        "detected_hosts" in metadata["content"]
                        or "detected_guests" in metadata["content"]
                        or "speakers" in metadata["content"]
                    ), "Metadata should contain speaker detection results (OpenAI)"
                    # Should have summary (OpenAI summarization) - summary is top-level
                    assert (
                        "summary" in metadata
                    ), "Metadata should contain summary (OpenAI summarization)"
                    assert metadata["summary"] is not None, "Summary should not be None"


@pytest.mark.e2e
@pytest.mark.critical_path
@pytest.mark.openai
@pytest.mark.llm
class TestBasicLibraryAPIE2E_OpenAI:
    """OpenAI provider Library API E2E tests using mock OpenAI endpoints on E2E server."""

    def test_library_api_basic_pipeline_path1_openai(self, e2e_server):
        """Test complete Library API critical path (Path 1) with OpenAI providers.

        This test validates Path 1 with OpenAI providers using Library API:
        RSS → Parse → Download Transcript → OpenAI NER → OpenAI Summarization → Metadata → Files

        Uses podcast1_with_transcript which has a transcript URL,
        so OpenAI transcription is NOT needed.
        Uses OpenAI providers for speaker detection and summarization
        (via E2E server mock endpoints).
        """
        rss_url = e2e_server.urls.feed("podcast1_with_transcript")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = config.Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=False,  # No transcription needed when transcript URL exists
                auto_speakers=True,  # Enable speaker detection - uses OpenAI provider
                speaker_detector_provider="openai",  # Use OpenAI for speaker detection
                generate_summaries=True,  # Enable summarization - uses OpenAI provider
                summary_provider="openai",  # Use OpenAI for summarization
                generate_metadata=True,  # Enable metadata generation
                metadata_format="json",
                openai_api_key="sk-test123",  # Dummy key for E2E server
                openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
            )

            # Run pipeline
            podcast_scraper.run_pipeline(cfg)

            # Verify transcript file was downloaded (not transcribed)
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) > 0, "At least one transcript file should be downloaded"

            # Verify metadata file was created (indicates OpenAI NER and summarization ran)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata file should be created (indicates OpenAI NER and summarization ran)"

            # Verify metadata contains OpenAI NER and summarization results
            if metadata_files:
                import json

                with open(metadata_files[0], "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    # Should have speaker detection results (OpenAI)
                    assert (
                        "detected_hosts" in metadata["content"]
                        or "detected_guests" in metadata["content"]
                        or "speakers" in metadata["content"]
                    ), "Metadata should contain speaker detection results (OpenAI)"
                    # Should have summary (OpenAI summarization) - summary is top-level
                    assert (
                        "summary" in metadata
                    ), "Metadata should contain summary (OpenAI summarization)"
                    assert metadata["summary"] is not None, "Summary should not be None"

    @pytest.mark.critical_path
    @pytest.mark.flaky
    def test_library_api_basic_pipeline_path2_openai(self, e2e_server):
        """Test complete Library API critical path (Path 2) with OpenAI providers.

        This test validates Path 2 with OpenAI providers using Library API:
        RSS → Parse → Download Audio → OpenAI Transcription → OpenAI NER →
        OpenAI Summarization → Metadata → Files

        Uses podcast1 which has NO transcript URL, so OpenAI transcription is needed.
        Uses OpenAI providers for transcription, speaker detection, and summarization
        (via E2E server mock endpoints).
        """
        rss_url = e2e_server.urls.feed("podcast1")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = config.Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,  # Enable OpenAI transcription
                transcription_provider="openai",  # Use OpenAI for transcription
                auto_speakers=True,  # Enable speaker detection - uses OpenAI provider
                speaker_detector_provider="openai",  # Use OpenAI for speaker detection
                generate_summaries=True,  # Enable summarization - uses OpenAI provider
                summary_provider="openai",  # Use OpenAI for summarization
                generate_metadata=True,  # Enable metadata generation
                metadata_format="json",
                openai_api_key="sk-test123",  # Dummy key for E2E server
                openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
            )

            # Run pipeline
            podcast_scraper.run_pipeline(cfg)

            # Verify transcript file was transcribed
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) > 0, "At least one transcript file should be transcribed"

            # Verify metadata file was created (indicates OpenAI NER and summarization ran)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata file should be created (indicates OpenAI NER and summarization ran)"

            # Verify metadata contains OpenAI NER and summarization results
            if metadata_files:
                import json

                with open(metadata_files[0], "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    # Should have speaker detection results (OpenAI)
                    assert (
                        "detected_hosts" in metadata["content"]
                        or "detected_guests" in metadata["content"]
                        or "speakers" in metadata["content"]
                    ), "Metadata should contain speaker detection results (OpenAI)"
                    # Should have summary (OpenAI summarization) - summary is top-level
                    assert (
                        "summary" in metadata
                    ), "Metadata should contain summary (OpenAI summarization)"
                    assert metadata["summary"] is not None, "Summary should not be None"


@pytest.mark.e2e
@pytest.mark.critical_path
@pytest.mark.openai
@pytest.mark.llm
class TestBasicServiceAPIE2E_OpenAI:
    """OpenAI provider Service API E2E tests using mock OpenAI endpoints on E2E server."""

    @pytest.mark.flaky
    def test_service_api_basic_run_path1_openai(self, e2e_server):
        """Test complete Service API critical path (Path 1) with OpenAI providers.

        This test validates Path 1 with OpenAI providers using Service API:
        RSS → Parse → Download Transcript → OpenAI NER → OpenAI Summarization → Metadata → Files

        Uses podcast1_with_transcript which has a transcript URL,
        so OpenAI transcription is NOT needed.
        Uses OpenAI providers for speaker detection and summarization
        (via E2E server mock endpoints).
        """
        from podcast_scraper import service

        rss_url = e2e_server.urls.feed("podcast1_with_transcript")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = config.Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=False,  # No transcription needed when transcript URL exists
                auto_speakers=True,  # Enable speaker detection - uses OpenAI provider
                speaker_detector_provider="openai",  # Use OpenAI for speaker detection
                generate_summaries=True,  # Enable summarization - uses OpenAI provider
                summary_provider="openai",  # Use OpenAI for summarization
                generate_metadata=True,  # Enable metadata generation
                metadata_format="json",
                openai_api_key="sk-test123",  # Dummy key for E2E server
                openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
            )

            # Run pipeline via Service API
            service.run(cfg)

            # Verify transcript file was downloaded (not transcribed)
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) > 0, "At least one transcript file should be downloaded"

            # Verify metadata file was created (indicates OpenAI NER and summarization ran)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata file should be created (indicates OpenAI NER and summarization ran)"

            # Verify metadata contains OpenAI NER and summarization results
            if metadata_files:
                import json

                with open(metadata_files[0], "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    # Should have speaker detection results (OpenAI)
                    assert (
                        "detected_hosts" in metadata["content"]
                        or "detected_guests" in metadata["content"]
                        or "speakers" in metadata["content"]
                    ), "Metadata should contain speaker detection results (OpenAI)"
                    # Should have summary (OpenAI summarization) - summary is top-level
                    assert (
                        "summary" in metadata
                    ), "Metadata should contain summary (OpenAI summarization)"
                    assert metadata["summary"] is not None, "Summary should not be None"

    @pytest.mark.critical_path
    def test_service_api_basic_run_path2_openai(self, e2e_server):
        """Test complete Service API critical path (Path 2) with OpenAI providers.

        This test validates Path 2 with OpenAI providers using Service API:
        RSS → Parse → Download Audio → OpenAI Transcription → OpenAI NER →
        OpenAI Summarization → Metadata → Files

        Uses podcast1 which has NO transcript URL, so OpenAI transcription is needed.
        Uses OpenAI providers for transcription, speaker detection, and summarization
        (via E2E server mock endpoints).
        """
        from podcast_scraper import service

        rss_url = e2e_server.urls.feed("podcast1")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = config.Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,  # Enable OpenAI transcription
                transcription_provider="openai",  # Use OpenAI for transcription
                auto_speakers=True,  # Enable speaker detection - uses OpenAI provider
                speaker_detector_provider="openai",  # Use OpenAI for speaker detection
                generate_summaries=True,  # Enable summarization - uses OpenAI provider
                summary_provider="openai",  # Use OpenAI for summarization
                generate_metadata=True,  # Enable metadata generation
                metadata_format="json",
                openai_api_key="sk-test123",  # Dummy key for E2E server
                openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
            )

            # Run pipeline via Service API
            service.run(cfg)

            # Verify transcript file was transcribed
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) > 0, "At least one transcript file should be transcribed"

            # Verify metadata file was created (indicates OpenAI NER and summarization ran)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata file should be created (indicates OpenAI NER and summarization ran)"

            # Verify metadata contains OpenAI NER and summarization results
            if metadata_files:
                import json

                with open(metadata_files[0], "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    # Should have speaker detection results (OpenAI)
                    assert (
                        "detected_hosts" in metadata["content"]
                        or "detected_guests" in metadata["content"]
                        or "speakers" in metadata["content"]
                    ), "Metadata should contain speaker detection results (OpenAI)"
                    # Should have summary (OpenAI summarization) - summary is top-level
                    assert (
                        "summary" in metadata
                    ), "Metadata should contain summary (OpenAI summarization)"
                    assert metadata["summary"] is not None, "Summary should not be None"
