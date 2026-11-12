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
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest

from podcast_scraper import Config, models
from podcast_scraper.workflow import metrics as workflow_metrics
from podcast_scraper.workflow.stages import processing
from podcast_scraper.workflow.types import HostDetectionResult, TranscriptionResources

# Import directly from tests.conftest (works with pytest-xdist)
from tests.conftest import create_test_config


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

        prepare_episode_download_args emits a 10-tuple: index 7 = detected guests, index 8 =
        stated names, index 9 = the feed's hosts. When #1169 grew the producer, the fixed-arity
        unpack then in summarization crashed EVERY real summarization run and only e2e caught it.
        This pins the arity cheaply at unit level.

        Grown 9 -> 10 deliberately: a publisher transcript that names its own turns is rostered on
        the DOWNLOAD path, where the feed's hosts anchor seat-matching and canonicalise a stated
        mononym to the feed's spelling — and that is the only point before the transcription stage
        where they exist. Safe because every consumer reads by INDEX (`args[0]`, `args[7]`) rather
        than unpacking; summarization says so in a comment at its read site. The unpack below is
        the test's own and must track the producer.
        """
        ep = self._episode(1, "g1", "ep")
        m = workflow_metrics.Metrics()
        args_list = self._call_prepare([ep], append=False, pipeline_metrics=m)

        self.assertEqual(len(args_list), 1)
        args = args_list[0]
        self.assertEqual(len(args), 10, f"download_args arity drifted: {len(args)} != 10")
        episode_obj, _, _, _, _, _, _, detected_names, stated, feed_hosts = args
        self.assertIs(episode_obj, ep)
        # transcription reads args[7]; it must be addressable (guests, or None)
        self.assertEqual(args[7], detected_names)
        self.assertEqual(args[8], stated)
        # Index 9 must be a LIST, never None: a publisher-named transcript is rostered without it,
        # but a None here would mean the feed's hosts silently stopped reaching the download path,
        # so a stated mononym would never canonicalise to the feed's spelling.
        self.assertIsInstance(feed_hosts, list, f"feed_hosts must be a list, got {feed_hosts!r}")

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
    """ADR-122 item 3: a ResilienceFuseOpenError from one episode HALTS the batch (propagates
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
    """ADR-122 item 3 (concurrent path): a future that raises ResilienceFuseOpenError halts the
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
