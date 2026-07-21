"""Unit tests for podcast_scraper.workflow.stages.processing module.

Tests for processing stage helper functions.
"""

import json
import os
import queue
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from parent conftest explicitly to avoid conflicts
parent_tests_dir = Path(__file__).parent.parent.parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

import pytest

from podcast_scraper import Config, models
from podcast_scraper.workflow import metrics as workflow_metrics
from podcast_scraper.workflow.stages import processing
from podcast_scraper.workflow.types import HostDetectionResult, TranscriptionResources

# Import directly from tests.conftest (works with pytest-xdist)
from tests.conftest import create_test_config, create_test_episode  # noqa: E402


@pytest.mark.unit
class TestHandleDryRunHostDetection(unittest.TestCase):
    """Tests for _handle_dry_run_host_detection helper function."""

    def setUp(self):
        """Set up test fixtures."""
        # Realistic "First Last" person names — these pass the network/org author filter
        # (#876); names with digits like "Host 1" are correctly treated as non-person.
        self.feed = models.RssFeed(
            title="Test Feed",
            authors=["Jane Doe", "John Smith"],
            items=[],
            base_url="https://example.com",
        )

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_handle_dry_run_host_detection_with_authors(self, mock_logger):
        """Test dry-run host detection with RSS authors."""
        result = processing._handle_dry_run_host_detection(self.feed)

        self.assertIsInstance(result, HostDetectionResult)
        self.assertEqual(result.cached_hosts, {"Jane Doe", "John Smith"})
        self.assertIsNone(result.heuristics)
        self.assertIsNone(result.speaker_detector)
        mock_logger.info.assert_called()

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_handle_dry_run_host_detection_without_authors(self, mock_logger):
        """Test dry-run host detection without RSS authors."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=None,
            items=[],
            base_url="https://example.com",
        )

        result = processing._handle_dry_run_host_detection(feed)

        self.assertIsInstance(result, HostDetectionResult)
        self.assertEqual(result.cached_hosts, set())
        self.assertIsNone(result.heuristics)
        self.assertIsNone(result.speaker_detector)

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_handle_dry_run_host_detection_with_empty_authors(self, mock_logger):
        """Test dry-run host detection with empty authors list."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=[],
            items=[],
            base_url="https://example.com",
        )

        result = processing._handle_dry_run_host_detection(feed)

        self.assertIsInstance(result, HostDetectionResult)
        self.assertEqual(result.cached_hosts, set())


@pytest.mark.unit
class TestCreateSpeakerDetectorIfNeeded(unittest.TestCase):
    """Tests for _create_speaker_detector_if_needed helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(auto_speakers=True)

    def test_create_speaker_detector_if_needed_with_existing(self):
        """Test that existing speaker detector is returned."""
        mock_detector = Mock()
        result = processing._create_speaker_detector_if_needed(self.cfg, mock_detector)

        self.assertEqual(result, mock_detector)

    @patch("podcast_scraper.workflow.stages.processing.logger")
    @patch("podcast_scraper.speaker_detectors.factory.create_speaker_detector")
    def test_create_speaker_detector_if_needed_creates_new(self, mock_create_detector, mock_logger):
        """Test that new speaker detector is created when None provided."""
        mock_detector = Mock()
        mock_detector.initialize = Mock()
        mock_create_detector.return_value = mock_detector

        # Mock sys.modules to not have create_speaker_detector
        with patch("sys.modules") as mock_modules:
            mock_workflow = Mock()
            del mock_workflow.create_speaker_detector  # Attribute doesn't exist
            mock_modules.get.return_value = mock_workflow

            result = processing._create_speaker_detector_if_needed(self.cfg, None)

            self.assertEqual(result, mock_detector)
            mock_detector.initialize.assert_called_once()
            mock_logger.warning.assert_called()

    @patch("podcast_scraper.workflow.stages.processing.logger")
    @patch("podcast_scraper.speaker_detectors.factory.create_speaker_detector")
    def test_create_speaker_detector_if_needed_handles_exception(
        self, mock_create_detector, mock_logger
    ):
        """Test that exception during creation is handled gracefully."""
        mock_create_detector.side_effect = RuntimeError("Init failed")

        result = processing._create_speaker_detector_if_needed(self.cfg, None)

        self.assertIsNone(result)
        mock_logger.error.assert_called()


@pytest.mark.unit
class TestDetectHostsFromFeed(unittest.TestCase):
    """Tests for _detect_hosts_from_feed helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.feed = models.RssFeed(
            title="Test Feed",
            authors=["Host 1"],
            items=[],
            base_url="https://example.com",
        )
        self.speaker_detector = Mock()
        self.speaker_detector.detect_hosts = Mock(return_value={"Host 1", "Host 2"})

    def test_detect_hosts_from_feed_success(self):
        """Test successful host detection from feed."""
        result = processing._detect_hosts_from_feed(self.feed, self.speaker_detector)

        self.assertEqual(result, {"Host 1", "Host 2"})
        self.speaker_detector.detect_hosts.assert_called_once_with(
            feed_title="Test Feed",
            feed_description=None,
            feed_authors=["Host 1"],
        )

    def test_detect_hosts_from_feed_without_authors(self):
        """Test host detection when feed has no authors."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=None,
            items=[],
            base_url="https://example.com",
        )

        result = processing._detect_hosts_from_feed(feed, self.speaker_detector)

        self.assertEqual(result, {"Host 1", "Host 2"})
        self.speaker_detector.detect_hosts.assert_called_once_with(
            feed_title="Test Feed",
            feed_description=None,
            feed_authors=None,
        )


@pytest.mark.unit
class TestValidateHostsWithFirstEpisode(unittest.TestCase):
    """Tests for _validate_hosts_with_first_episode helper function."""

    def setUp(self):
        """Set up test fixtures."""
        import xml.etree.ElementTree as ET

        self.item = ET.Element("item")
        ET.SubElement(self.item, "title").text = "Episode 1"
        ET.SubElement(self.item, "description").text = "Episode description"

        self.episode = create_test_episode(idx=1, title="Episode 1")
        self.episode.item = self.item

        self.feed = models.RssFeed(
            title="Test Feed",
            authors=None,  # No authors, so validation will run
            items=[self.item],
            base_url="https://example.com",
        )

        self.speaker_detector = Mock()
        self.speaker_detector.detect_speakers = Mock(
            return_value=(["Host 1", "Host 2"], set(), True, False)
        )

    def test_validate_hosts_with_first_episode_skips_when_authors_exist(self):
        """Test that validation is skipped when feed has authors."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=["Host 1"],
            items=[self.item],
            base_url="https://example.com",
        )

        feed_hosts = {"Host 1", "Host 2"}
        result = processing._validate_hosts_with_first_episode(
            feed_hosts, feed, [self.episode], self.speaker_detector, None
        )

        self.assertEqual(result, feed_hosts)
        self.speaker_detector.detect_speakers.assert_not_called()

    def test_validate_hosts_with_first_episode_skips_when_no_hosts(self):
        """Test that validation is skipped when no hosts detected."""
        feed_hosts = set()
        result = processing._validate_hosts_with_first_episode(
            feed_hosts, self.feed, [self.episode], self.speaker_detector, None
        )

        self.assertEqual(result, feed_hosts)
        self.speaker_detector.detect_speakers.assert_not_called()

    def test_validate_hosts_with_first_episode_skips_when_no_episodes(self):
        """Test that validation is skipped when no episodes."""
        feed_hosts = {"Host 1", "Host 2"}
        result = processing._validate_hosts_with_first_episode(
            feed_hosts, self.feed, [], self.speaker_detector, None
        )

        self.assertEqual(result, feed_hosts)
        self.speaker_detector.detect_speakers.assert_not_called()

    def test_validate_hosts_with_first_episode_validates_successfully(self):
        """Test successful host validation."""
        feed_hosts = {"Host 1", "Host 2"}
        result = processing._validate_hosts_with_first_episode(
            feed_hosts, self.feed, [self.episode], self.speaker_detector, None
        )

        # Both hosts appear in first episode, so both should be validated
        self.assertEqual(result, {"Host 1", "Host 2"})
        self.speaker_detector.detect_speakers.assert_called_once()

    def test_validate_hosts_with_first_episode_filters_invalid_hosts(self):
        """Test that hosts not in first episode are filtered out."""
        feed_hosts = {"Host 1", "Host 2", "Host 3"}
        # Only Host 1 and Host 2 appear in first episode
        self.speaker_detector.detect_speakers = Mock(
            return_value=(["Host 1", "Host 2"], set(), True, False)
        )

        result = processing._validate_hosts_with_first_episode(
            feed_hosts, self.feed, [self.episode], self.speaker_detector, None
        )

        # Host 3 should be filtered out
        self.assertEqual(result, {"Host 1", "Host 2"})

    def test_validate_hosts_with_first_episode_with_pipeline_metrics(self):
        """Test validation with pipeline_metrics parameter."""
        from podcast_scraper.workflow import metrics

        pipeline_metrics = metrics.Metrics()
        feed_hosts = {"Host 1", "Host 2"}

        # Mock signature to include pipeline_metrics
        import inspect

        def mock_detect_speakers(
            episode_title, episode_description, known_hosts, pipeline_metrics=None
        ):
            return (["Host 1", "Host 2"], set(), True, False)

        self.speaker_detector.detect_speakers = Mock(side_effect=mock_detect_speakers)
        self.speaker_detector.detect_speakers.__signature__ = inspect.Signature(
            parameters=[
                inspect.Parameter("episode_title", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("episode_description", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("known_hosts", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("pipeline_metrics", inspect.Parameter.KEYWORD_ONLY),
            ]
        )

        result = processing._validate_hosts_with_first_episode(
            feed_hosts, self.feed, [self.episode], self.speaker_detector, pipeline_metrics
        )

        self.assertEqual(result, {"Host 1", "Host 2"})
        # Verify pipeline_metrics was passed
        call_kwargs = self.speaker_detector.detect_speakers.call_args[1]
        self.assertEqual(call_kwargs.get("pipeline_metrics"), pipeline_metrics)


@pytest.mark.unit
class TestFallbackToEpisodeAuthors(unittest.TestCase):
    """Tests for _fallback_to_episode_authors helper function."""

    def setUp(self):
        """Set up test fixtures."""
        import xml.etree.ElementTree as ET

        self.item1 = ET.Element("item")
        ET.SubElement(self.item1, "title").text = "Episode 1"
        author1 = ET.SubElement(self.item1, "{http://www.itunes.com/dtds/podcast-1.0.dtd}author")
        author1.text = "Author 1"

        self.item2 = ET.Element("item")
        ET.SubElement(self.item2, "title").text = "Episode 2"
        author2 = ET.SubElement(self.item2, "{http://www.itunes.com/dtds/podcast-1.0.dtd}author")
        author2.text = "Author 2"

        self.episode1 = create_test_episode(idx=1, title="Episode 1")
        self.episode1.item = self.item1

        self.episode2 = create_test_episode(idx=2, title="Episode 2")
        self.episode2.item = self.item2

    def test_fallback_to_episode_authors_disabled(self):
        """Test that empty set is returned when auto_speakers is disabled."""
        cfg = create_test_config(auto_speakers=False)

        result = processing._fallback_to_episode_authors(cfg, [self.episode1, self.episode2])

        self.assertEqual(result, set())

    def test_fallback_to_episode_authors_no_episodes(self):
        """Test that empty set is returned when no episodes."""
        cfg = create_test_config(auto_speakers=True)

        result = processing._fallback_to_episode_authors(cfg, [])

        self.assertEqual(result, set())

    @patch("podcast_scraper.rss.parser.extract_episode_authors")
    def test_fallback_to_episode_authors_success(self, mock_extract_authors):
        """Test successful extraction of episode authors."""
        mock_extract_authors.side_effect = [["Author 1"], ["Author 2"]]

        cfg = create_test_config(auto_speakers=True)

        result = processing._fallback_to_episode_authors(cfg, [self.episode1, self.episode2])

        self.assertEqual(result, {"Author 1", "Author 2"})
        self.assertEqual(mock_extract_authors.call_count, 2)

    @patch("podcast_scraper.rss.parser.extract_episode_authors")
    def test_fallback_to_episode_authors_filters_organizations(self, mock_extract_authors):
        """Test that organization names are filtered out."""
        # Organization: all caps, short, no spaces
        mock_extract_authors.side_effect = [["NPR"], ["John Doe"]]

        cfg = create_test_config(auto_speakers=True)

        result = processing._fallback_to_episode_authors(cfg, [self.episode1, self.episode2])

        # NPR should be filtered out (organization)
        self.assertEqual(result, {"John Doe"})

    @patch("podcast_scraper.rss.parser.extract_episode_authors")
    def test_fallback_to_episode_authors_limits_to_first_three(self, mock_extract_authors):
        """Test that only first 3 episodes are checked."""
        episodes = [create_test_episode(idx=i, title=f"Episode {i}") for i in range(1, 6)]
        mock_extract_authors.return_value = ["Author"]

        cfg = create_test_config(auto_speakers=True)

        processing._fallback_to_episode_authors(cfg, episodes)

        # Should only call extract_episode_authors 3 times (first 3 episodes)
        self.assertEqual(mock_extract_authors.call_count, 3)


@pytest.mark.unit
class TestLogDetectedHosts(unittest.TestCase):
    """Tests for _log_detected_hosts helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.feed = models.RssFeed(
            title="Test Feed",
            authors=["Host 1"],
            items=[],
            base_url="https://example.com",
        )

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_log_detected_hosts_from_feed_authors(self, mock_logger):
        """Test logging hosts detected from feed authors."""
        cached_hosts = {"Host 1", "Host 2"}
        episode_authors = set()

        processing._log_detected_hosts(cached_hosts, self.feed, episode_authors, None)

        # Verify log was called with correct source
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        feed_log_found = any("RSS author tags" in str(call) for call in log_calls)
        self.assertTrue(feed_log_found, "Should log hosts from RSS author tags")

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_log_detected_hosts_from_episode_authors(self, mock_logger):
        """Test logging hosts detected from episode authors."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=None,
            items=[],
            base_url="https://example.com",
        )
        cached_hosts = {"Host 1"}
        episode_authors = {"Host 1"}

        processing._log_detected_hosts(cached_hosts, feed, episode_authors, None)

        # Verify log was called with episode-level authors source
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        episode_log_found = any("episode-level authors" in str(call) for call in log_calls)
        self.assertTrue(episode_log_found, "Should log hosts from episode-level authors")

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_log_detected_hosts_from_config(self, mock_logger):
        """Test logging hosts detected from config known_hosts."""
        cfg = create_test_config(known_hosts=["Host 1", "Host 2"])
        feed = models.RssFeed(
            title="Test Feed",
            authors=None,
            items=[],
            base_url="https://example.com",
        )
        cached_hosts = {"Host 1", "Host 2"}
        episode_authors = set()

        processing._log_detected_hosts(cached_hosts, feed, episode_authors, cfg)

        # Verify log was called with config source
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        config_log_found = any("config known_hosts" in str(call) for call in log_calls)
        self.assertTrue(config_log_found, "Should log hosts from config known_hosts")

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_log_detected_hosts_from_ner(self, mock_logger):
        """Test logging hosts detected from NER (default source)."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=None,
            items=[],
            base_url="https://example.com",
        )
        cached_hosts = {"Host 1"}
        episode_authors = set()
        cfg = create_test_config(auto_speakers=True)

        processing._log_detected_hosts(cached_hosts, feed, episode_authors, cfg)

        # Verify log was called with feed metadata (NER) source
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        ner_log_found = any("feed metadata (NER)" in str(call) for call in log_calls)
        self.assertTrue(ner_log_found, "Should log hosts from feed metadata (NER)")

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_log_detected_hosts_empty(self, mock_logger):
        """Test logging when no hosts detected."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=None,
            items=[],
            base_url="https://example.com",
        )
        cached_hosts = set()
        episode_authors = set()
        cfg = create_test_config(auto_speakers=True)

        processing._log_detected_hosts(cached_hosts, feed, episode_authors, cfg)

        # Should log debug message about no hosts
        log_calls = [str(call) for call in mock_logger.debug.call_args_list]
        no_hosts_log_found = any("No hosts detected" in str(call) for call in log_calls)
        self.assertTrue(no_hosts_log_found, "Should log debug message when no hosts")


@pytest.mark.unit
class TestSetupProcessingResources(unittest.TestCase):
    """Tests for setup_processing_resources function."""

    def test_setup_processing_resources_single_worker(self):
        """Test setup with single worker (no locks needed)."""
        cfg = create_test_config(workers=1, transcription_parallelism=1, processing_parallelism=1)

        result = processing.setup_processing_resources(cfg)

        self.assertIsNotNone(result)
        self.assertIsNone(result.processing_jobs_lock)
        self.assertIsNotNone(result.processing_complete_event)

    def test_setup_processing_resources_multiple_workers(self):
        """Test setup with multiple workers (locks needed)."""
        cfg = create_test_config(workers=4)

        result = processing.setup_processing_resources(cfg)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.processing_jobs_lock)
        self.assertIsNotNone(result.processing_complete_event)

    def test_setup_processing_resources_with_parallelism(self):
        """Test setup with transcription parallelism enabled."""
        cfg = create_test_config(workers=1, transcription_parallelism=2)

        result = processing.setup_processing_resources(cfg)

        # Should have lock even with workers=1 if parallelism > 1
        self.assertIsNotNone(result.processing_jobs_lock)


@pytest.mark.unit
class TestPrepareEpisodeDownloadArgsAppendResume(unittest.TestCase):
    """Append mode filtering in prepare_episode_download_args (GitHub #444)."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.feed_url = "https://example.com/podcast.xml"
        self.run_suffix = "append_abc12345"
        self.transcripts = os.path.join(self.tmp, "transcripts")
        self.metadata = os.path.join(self.tmp, "metadata")
        os.makedirs(self.transcripts, exist_ok=True)
        os.makedirs(self.metadata, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _episode(self, idx: int, guid: str, title_safe: str) -> models.Episode:
        item = ET.Element("item")
        ET.SubElement(item, "title").text = f"Title{idx}"
        ET.SubElement(item, "guid").text = guid
        return models.Episode(
            idx=idx,
            title=f"Title{idx}",
            title_safe=title_safe,
            item=item,
            transcript_urls=[],
        )

    def _write_complete_artifact(self, idx: int, title_safe: str, episode_id: str) -> None:
        trel = f"transcripts/{idx:04d} - {title_safe}_{self.run_suffix}.txt"
        os.makedirs(os.path.dirname(os.path.join(self.tmp, trel)), exist_ok=True)
        with open(os.path.join(self.tmp, trel), "w", encoding="utf-8") as handle:
            handle.write("body")
        meta_name = f"{idx:04d} - {title_safe}_{self.run_suffix}.metadata.json"
        doc = {
            "episode": {"episode_id": episode_id},
            "content": {"transcript_file_path": trel},
        }
        with open(os.path.join(self.metadata, meta_name), "w", encoding="utf-8") as handle:
            json.dump(doc, handle)

    def _call_prepare(
        self,
        episodes: list,
        *,
        append: bool,
        pipeline_metrics: workflow_metrics.Metrics,
    ) -> list:
        cfg = Config(
            rss=self.feed_url,
            output_dir=self.tmp,
            append=append,
            generate_metadata=True,
            generate_summaries=False,
            transcribe_missing=False,
            auto_speakers=False,
        )
        tres = TranscriptionResources(
            transcription_provider=None,
            temp_dir=self.tmp,
            transcription_jobs=queue.Queue(),
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        host = HostDetectionResult(set(), None, None)
        return processing.prepare_episode_download_args(
            episodes,
            cfg,
            self.tmp,
            self.run_suffix,
            tres,
            host,
            pipeline_metrics,
        )

    def test_download_args_tuple_arity_contract(self) -> None:
        """Producer/consumer contract for the download_args tuple.

        prepare_episode_download_args emits a 9-tuple; summarization unpacks all 9 (index 7 =
        detected guests, index 8 = stated hosts) and transcription reads args[7]. When #1169 grew
        the producer to 9 elements the 8-element unpack in summarization crashed EVERY real
        summarization run, and only e2e caught it. This pins the arity cheaply at unit level.
        """
        ep = self._episode(1, "g1", "ep")
        m = workflow_metrics.Metrics()
        args_list = self._call_prepare([ep], append=False, pipeline_metrics=m)

        self.assertEqual(len(args_list), 1)
        args = args_list[0]
        self.assertEqual(len(args), 9, f"download_args arity drifted: {len(args)} != 9")
        # the exact unpack summarization._collect_episodes_for_summarization performs
        episode_obj, _, _, _, _, _, _, detected_names, stated = args
        self.assertIs(episode_obj, ep)
        # transcription reads args[7]; it must be addressable (guests, or None)
        self.assertEqual(args[7], detected_names)

    def test_append_skips_complete_episode_and_keeps_incomplete(self) -> None:
        """Complete on-disk episode is omitted from download args; incomplete is kept."""
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep1 = self._episode(1, "g1", "ep")
        ep2 = self._episode(2, "g2", "ep2")
        eid1, _ = get_episode_id_from_episode(ep1, self.feed_url)
        self._write_complete_artifact(1, "ep", eid1)

        m = workflow_metrics.Metrics()
        args_list = self._call_prepare([ep1, ep2], append=True, pipeline_metrics=m)

        self.assertEqual(len(args_list), 1)
        self.assertEqual(args_list[0][0].idx, 2)
        append_stages = [s for s in m.episode_statuses if s.stage == "append_skipped_complete"]
        self.assertEqual(len(append_stages), 1)
        self.assertEqual(append_stages[0].episode_id, eid1)

    def test_append_false_includes_all_episodes(self) -> None:
        """Without append, both episodes are prepared even if artifacts exist."""
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode

        ep1 = self._episode(1, "g1", "ep")
        ep2 = self._episode(2, "g2", "ep2")
        eid1, _ = get_episode_id_from_episode(ep1, self.feed_url)
        self._write_complete_artifact(1, "ep", eid1)

        m = workflow_metrics.Metrics()
        args_list = self._call_prepare([ep1, ep2], append=False, pipeline_metrics=m)

        self.assertEqual(len(args_list), 2)
        self.assertEqual({args_list[0][0].idx, args_list[1][0].idx}, {1, 2})


@pytest.mark.unit
class TestFlattenSpeakerNameEntries(unittest.TestCase):
    """Tests for speaker name normalization used in host filtering."""

    def test_flatten_nested_strings(self) -> None:
        """Nested lists are flattened to strings."""
        out = processing._flatten_speaker_name_entries(["a", ["b", "c"]])
        self.assertEqual(out, ["a", "b", "c"])

    def test_empty_and_whitespace(self) -> None:
        """Empty strings skipped."""
        self.assertEqual(processing._flatten_speaker_name_entries(["  ", "x"]), ["x"])

    def test_speaker_names_to_str_set(self) -> None:
        """Set membership works with mixed nesting."""
        s = processing._speaker_names_to_str_set([["Host"], "Guest"])
        self.assertEqual(s, {"Host", "Guest"})


@pytest.mark.unit
def test_enforce_cost_soft_cap_after_episode_abort() -> None:
    """Processing stage delegates to cost_monitoring soft cap (#804)."""
    from podcast_scraper.workflow.cost_monitoring import CostCapExceeded

    cfg = create_test_config(
        openai_api_key="sk-test",
        cost_soft_cap_usd_per_run=0.01,
        cost_soft_cap_action="abort",
    )
    pm = workflow_metrics.Metrics()
    pm.record_llm_transcription_call(1.0, cost_usd=0.05)
    with pytest.raises(CostCapExceeded):
        processing._enforce_cost_soft_cap_after_episode(cfg, pm)


def test_reprocess_fuse_open_halts_the_whole_batch() -> None:
    """ADR-119 item 3: a ResilienceFuseOpenError from one episode HALTS the batch (propagates
    like CostCapExceeded) instead of being swallowed by the broad except and grinding every
    remaining episode through a dead endpoint in reprocess mode."""
    from podcast_scraper.providers.resilience import ResilienceFuseOpenError

    ep = Mock()
    ep.idx = 3
    args = (ep,) + tuple([None] * 7)  # the loop reads args[0]=episode and args[7]=detected_names
    with patch.object(
        processing,
        "_process_episode_with_retry",
        side_effect=ResilienceFuseOpenError("moss: endpoint down"),
    ):
        with pytest.raises(ResilienceFuseOpenError):
            processing._process_episodes_sequential([args], Mock(), Mock(), Mock(), Mock())


def test_reprocess_fuse_open_halts_concurrent_drain() -> None:
    """ADR-119 item 3 (concurrent path): a future that raises ResilienceFuseOpenError halts the
    executor drain loop too — not just the sequential loop. The whole batch stops rather than
    reaping every remaining future against a dead endpoint."""
    from concurrent.futures import Future

    from podcast_scraper.providers.resilience import ResilienceFuseOpenError

    down: Future = Future()
    down.set_exception(ResilienceFuseOpenError("moss: endpoint down"))
    futures = {down: 7}
    with pytest.raises(ResilienceFuseOpenError):
        processing._drain_completed_processing_futures(futures, Mock(), Mock())


def test_reprocess_batch_ordinary_failure_continues_but_fuse_open_halts() -> None:
    """COMBINED/system: the isolated mechanisms compose, and the batch distinguishes an ORDINARY
    failure from a genuine OUTAGE. Episode 1 fails normally (returned success=False) — the batch
    moves on. Episode 2 opens the fuse (endpoint genuinely down, no fallover in hold) — the batch
    HALTS and episode 3 is never processed. This is 'a bad episode is not a dead endpoint' end to
    end at the loop level."""
    from podcast_scraper.providers.resilience import ResilienceFuseOpenError

    def _episode(idx: int):
        ep = Mock()
        ep.idx = idx
        return (ep,) + tuple([None] * 7)

    batch = [_episode(1), _episode(2), _episode(3)]
    # ep1 -> ordinary failure (valid 4-tuple, success=False); ep2 -> fuse open; ep3 -> unreached.
    outcomes = [
        (False, None, None, 0),
        ResilienceFuseOpenError("dgx-whisper: endpoint down"),
    ]

    def _proc_side_effect(*args, **kwargs):
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    with (
        patch.object(
            processing, "_process_episode_with_retry", side_effect=_proc_side_effect
        ) as proc,
        patch.object(processing, "_handle_episode_download_result", return_value=0) as handle,
    ):
        with pytest.raises(ResilienceFuseOpenError):
            processing._process_episodes_sequential(batch, Mock(), Mock(), Mock(), Mock())
    # ep1 (ordinary fail, did NOT halt) + ep2 (fuse open, halts) were attempted; ep3 never reached.
    assert proc.call_count == 2
    # ep1's ordinary failure still ran the normal result handler; ep2 raised before handling.
    assert handle.call_count == 1
