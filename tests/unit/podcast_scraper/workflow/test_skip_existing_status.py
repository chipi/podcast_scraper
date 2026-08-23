"""Unit tests for F1/C1 — skip-existing must tally as ``skipped``, not ``failed``.

Before this fix the skip-existing branch in ``download_media_for_transcription`` returned
``None`` without touching ``pipeline_metrics`` at all. Only the policy-skip and exception
paths ever recorded ``status="skipped"``. The consequence: a clean, correct, $0 all-skip
run reported ``{failed: 1}`` and failed the Step-0/Step-1 EXIT criteria while having done
precisely the right thing.

This matters operationally as well as cosmetically — an EXIT gate that fails on success
trains operators to ignore it.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper.workflow import episode_processor


class _Cfg:
    def __init__(self, rss_url="https://example.com/feed.xml"):
        self.rss_url = rss_url


class TestMarkEpisodeSkippedExisting(unittest.TestCase):
    def setUp(self):
        self.episode = Mock()
        self.episode.idx = 3
        self.cfg = _Cfg()

    def test_records_skipped_status_with_stable_episode_id(self):
        metrics = Mock()
        with patch(
            "podcast_scraper.workflow.helpers.get_episode_id_from_episode",
            return_value=("ep-abc123", None),
        ):
            episode_processor._mark_episode_skipped_existing(
                self.episode, self.cfg, metrics, "transcript already exists: /x/y.txt"
            )

        metrics.update_episode_status.assert_called_once()
        kwargs = metrics.update_episode_status.call_args.kwargs
        self.assertEqual(kwargs["status"], "skipped")
        self.assertEqual(kwargs["episode_id"], "ep-abc123")
        self.assertEqual(kwargs["error_type"], "SkipExisting")
        self.assertEqual(kwargs["stage"], "transcription")

    def test_status_is_not_failed(self):
        """The regression this whole fix exists to prevent."""
        metrics = Mock()
        with patch(
            "podcast_scraper.workflow.helpers.get_episode_id_from_episode",
            return_value=("ep-1", None),
        ):
            episode_processor._mark_episode_skipped_existing(
                self.episode, self.cfg, metrics, "already there"
            )
        self.assertNotEqual(metrics.update_episode_status.call_args.kwargs["status"], "failed")

    def test_increments_the_skipped_counter(self):
        metrics = Mock()
        with (
            patch(
                "podcast_scraper.workflow.helpers.get_episode_id_from_episode",
                return_value=("ep-1", None),
            ),
            patch("podcast_scraper.workflow.helpers.update_metric_safely") as bump,
        ):
            episode_processor._mark_episode_skipped_existing(
                self.episode, self.cfg, metrics, "already there"
            )
        bump.assert_called_once_with(metrics, "episodes_skipped_total", 1)

    def test_stage_is_overridable(self):
        metrics = Mock()
        with patch(
            "podcast_scraper.workflow.helpers.get_episode_id_from_episode",
            return_value=("ep-1", None),
        ):
            episode_processor._mark_episode_skipped_existing(
                self.episode, self.cfg, metrics, "reason", stage="processing"
            )
        self.assertEqual(metrics.update_episode_status.call_args.kwargs["stage"], "processing")

    def test_no_metrics_object_is_a_silent_no_op(self):
        """Must not raise when metrics are disabled."""
        episode_processor._mark_episode_skipped_existing(self.episode, self.cfg, None, "reason")

    def test_no_episode_is_a_silent_no_op(self):
        episode_processor._mark_episode_skipped_existing(None, self.cfg, Mock(), "reason")

    def test_metrics_failure_never_breaks_the_skip(self):
        """A telemetry error must not convert a correct skip into a failed run.

        This is the whole point of the fix: recording that we skipped must be strictly
        less important than the skip itself succeeding.
        """
        metrics = Mock()
        metrics.update_episode_status.side_effect = RuntimeError("metrics backend down")
        with patch(
            "podcast_scraper.workflow.helpers.get_episode_id_from_episode",
            return_value=("ep-1", None),
        ):
            episode_processor._mark_episode_skipped_existing(
                self.episode, self.cfg, metrics, "reason"
            )

    def test_episode_id_resolution_failure_never_breaks_the_skip(self):
        metrics = Mock()
        with patch(
            "podcast_scraper.workflow.helpers.get_episode_id_from_episode",
            side_effect=ValueError("bad episode"),
        ):
            episode_processor._mark_episode_skipped_existing(
                self.episode, self.cfg, metrics, "reason"
            )
        metrics.update_episode_status.assert_not_called()

    def test_reason_is_redacted(self):
        """Reasons embed filesystem paths; they go through the redactor like other logs."""
        metrics = Mock()
        with (
            patch(
                "podcast_scraper.workflow.helpers.get_episode_id_from_episode",
                return_value=("ep-1", None),
            ),
            patch(
                "podcast_scraper.workflow.episode_processor.redact_for_log",
                return_value="<redacted>",
            ) as redactor,
        ):
            episode_processor._mark_episode_skipped_existing(
                self.episode, self.cfg, metrics, "secret/path.txt"
            )
        redactor.assert_called_once()
        self.assertEqual(
            metrics.update_episode_status.call_args.kwargs["error_message"], "<redacted>"
        )


if __name__ == "__main__":
    unittest.main()
