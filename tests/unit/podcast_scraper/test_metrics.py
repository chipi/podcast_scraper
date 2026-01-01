#!/usr/bin/env python3
"""Tests for metrics collection functionality.

These tests verify the Metrics class that tracks pipeline performance
and execution statistics.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper import metrics


class TestMetricsInitialization(unittest.TestCase):
    """Test Metrics class initialization."""

    def test_default_values(self):
        """Test that all fields are initialized with correct default values."""
        m = metrics.Metrics()
        self.assertEqual(m.run_duration_seconds, 0.0)
        self.assertEqual(m.episodes_scraped_total, 0)
        self.assertEqual(m.episodes_skipped_total, 0)
        self.assertEqual(m.errors_total, 0)
        self.assertEqual(m.bytes_downloaded_total, 0)
        self.assertEqual(m.transcripts_downloaded, 0)
        self.assertEqual(m.transcripts_transcribed, 0)
        self.assertEqual(m.episodes_summarized, 0)
        self.assertEqual(m.metadata_files_generated, 0)
        self.assertEqual(m.time_scraping, 0.0)
        self.assertEqual(m.time_parsing, 0.0)
        self.assertEqual(m.time_normalizing, 0.0)
        self.assertEqual(m.time_writing_storage, 0.0)
        self.assertEqual(m.download_media_times, [])
        self.assertEqual(m.transcribe_times, [])
        self.assertEqual(m.extract_names_times, [])
        self.assertEqual(m.summarize_times, [])

    def test_start_time_initialized(self):
        """Test that _start_time is initialized."""
        m = metrics.Metrics()
        self.assertIsNotNone(m._start_time)
        self.assertIsInstance(m._start_time, float)
        self.assertGreater(m._start_time, 0)


class TestRecordStage(unittest.TestCase):
    """Test record_stage method."""

    def test_record_scraping_stage(self):
        """Test recording scraping stage time."""
        m = metrics.Metrics()
        m.record_stage("scraping", 1.5)
        self.assertEqual(m.time_scraping, 1.5)
        m.record_stage("scraping", 2.0)
        self.assertEqual(m.time_scraping, 3.5)

    def test_record_parsing_stage(self):
        """Test recording parsing stage time."""
        m = metrics.Metrics()
        m.record_stage("parsing", 0.5)
        self.assertEqual(m.time_parsing, 0.5)
        m.record_stage("parsing", 1.0)
        self.assertEqual(m.time_parsing, 1.5)

    def test_record_normalizing_stage(self):
        """Test recording normalizing stage time."""
        m = metrics.Metrics()
        m.record_stage("normalizing", 2.0)
        self.assertEqual(m.time_normalizing, 2.0)
        m.record_stage("normalizing", 1.5)
        self.assertEqual(m.time_normalizing, 3.5)

    def test_record_writing_storage_stage(self):
        """Test recording writing_storage stage time."""
        m = metrics.Metrics()
        m.record_stage("writing_storage", 0.8)
        self.assertEqual(m.time_writing_storage, 0.8)
        m.record_stage("writing_storage", 0.2)
        self.assertEqual(m.time_writing_storage, 1.0)

    def test_record_invalid_stage(self):
        """Test that invalid stage name is ignored (no error)."""
        m = metrics.Metrics()
        # Should not raise an error
        m.record_stage("invalid_stage", 1.0)
        # All stage times should remain 0
        self.assertEqual(m.time_scraping, 0.0)
        self.assertEqual(m.time_parsing, 0.0)
        self.assertEqual(m.time_normalizing, 0.0)
        self.assertEqual(m.time_writing_storage, 0.0)

    def test_record_stage_accumulates(self):
        """Test that multiple calls to same stage accumulate."""
        m = metrics.Metrics()
        m.record_stage("scraping", 1.0)
        m.record_stage("scraping", 2.0)
        m.record_stage("scraping", 0.5)
        self.assertEqual(m.time_scraping, 3.5)


class TestRecordDownloadMediaTime(unittest.TestCase):
    """Test record_download_media_time method."""

    def test_record_single_download_time(self):
        """Test recording a single download time."""
        m = metrics.Metrics()
        m.record_download_media_time(1.5)
        self.assertEqual(m.download_media_times, [1.5])

    def test_record_multiple_download_times(self):
        """Test recording multiple download times."""
        m = metrics.Metrics()
        m.record_download_media_time(1.0)
        m.record_download_media_time(2.0)
        m.record_download_media_time(1.5)
        self.assertEqual(m.download_media_times, [1.0, 2.0, 1.5])

    def test_record_zero_duration(self):
        """Test recording zero duration."""
        m = metrics.Metrics()
        m.record_download_media_time(0.0)
        self.assertEqual(m.download_media_times, [0.0])


class TestRecordTranscribeTime(unittest.TestCase):
    """Test record_transcribe_time method."""

    def test_record_single_transcribe_time(self):
        """Test recording a single transcription time."""
        m = metrics.Metrics()
        m.record_transcribe_time(5.0)
        self.assertEqual(m.transcribe_times, [5.0])

    def test_record_multiple_transcribe_times(self):
        """Test recording multiple transcription times."""
        m = metrics.Metrics()
        m.record_transcribe_time(5.0)
        m.record_transcribe_time(10.0)
        m.record_transcribe_time(7.5)
        self.assertEqual(m.transcribe_times, [5.0, 10.0, 7.5])


class TestRecordExtractNamesTime(unittest.TestCase):
    """Test record_extract_names_time method."""

    def test_record_single_extract_names_time(self):
        """Test recording a single name extraction time."""
        m = metrics.Metrics()
        m.record_extract_names_time(0.5)
        self.assertEqual(m.extract_names_times, [0.5])

    def test_record_multiple_extract_names_times(self):
        """Test recording multiple name extraction times."""
        m = metrics.Metrics()
        m.record_extract_names_time(0.5)
        m.record_extract_names_time(1.0)
        m.record_extract_names_time(0.75)
        self.assertEqual(m.extract_names_times, [0.5, 1.0, 0.75])


class TestRecordSummarizeTime(unittest.TestCase):
    """Test record_summarize_time method."""

    def test_record_single_summarize_time(self):
        """Test recording a single summarization time."""
        m = metrics.Metrics()
        m.record_summarize_time(3.0)
        self.assertEqual(m.summarize_times, [3.0])

    def test_record_multiple_summarize_times(self):
        """Test recording multiple summarization times."""
        m = metrics.Metrics()
        m.record_summarize_time(3.0)
        m.record_summarize_time(5.0)
        m.record_summarize_time(4.0)
        self.assertEqual(m.summarize_times, [3.0, 5.0, 4.0])


class TestFinish(unittest.TestCase):
    """Test finish method."""

    @patch("podcast_scraper.metrics.time.time")
    def test_finish_calculates_run_duration(self, mock_time):
        """Test that finish calculates run duration."""
        # Mock start time and end time
        mock_time.return_value = 105.5  # End time
        m = metrics.Metrics()
        # Re-initialize _start_time with start time
        m._start_time = 100.0
        result = m.finish()
        self.assertEqual(result["run_duration_seconds"], 5.5)
        self.assertEqual(m.run_duration_seconds, 5.5)

    @patch("podcast_scraper.metrics.time.time")
    def test_finish_rounds_duration(self, mock_time):
        """Test that finish rounds duration to 2 decimal places."""
        mock_time.return_value = 105.123456
        m = metrics.Metrics()
        m._start_time = 100.0
        result = m.finish()
        self.assertEqual(result["run_duration_seconds"], 5.12)

    def test_finish_calculates_averages_with_data(self):
        """Test that finish calculates averages when data exists."""
        m = metrics.Metrics()
        m.download_media_times = [1.0, 2.0, 3.0]
        m.transcribe_times = [5.0, 10.0]
        m.extract_names_times = [0.5, 1.0, 1.5]
        m.summarize_times = [3.0, 5.0, 4.0, 6.0]

        with patch("podcast_scraper.metrics.time.time", return_value=100.0):
            m._start_time = 100.0
            result = m.finish()

        self.assertEqual(result["avg_download_media_seconds"], 2.0)  # (1+2+3)/3
        self.assertEqual(result["avg_transcribe_seconds"], 7.5)  # (5+10)/2
        self.assertEqual(result["avg_extract_names_seconds"], 1.0)  # (0.5+1+1.5)/3
        self.assertEqual(result["avg_summarize_seconds"], 4.5)  # (3+5+4+6)/4

    def test_finish_handles_empty_lists(self):
        """Test that finish handles empty time lists."""
        m = metrics.Metrics()
        with patch("podcast_scraper.metrics.time.time", return_value=100.0):
            m._start_time = 100.0
            result = m.finish()

        self.assertEqual(result["avg_download_media_seconds"], 0.0)
        self.assertEqual(result["avg_transcribe_seconds"], 0.0)
        self.assertEqual(result["avg_extract_names_seconds"], 0.0)
        self.assertEqual(result["avg_summarize_seconds"], 0.0)

    def test_finish_rounds_averages(self):
        """Test that finish rounds averages to 2 decimal places."""
        m = metrics.Metrics()
        m.download_media_times = [1.0, 2.0, 3.0, 4.0]  # Average = 2.5
        m.transcribe_times = [1.0, 2.0, 3.0]  # Average = 2.0
        m.extract_names_times = [0.333, 0.666]  # Average = 0.4995 -> 0.5
        m.summarize_times = [1.111, 2.222]  # Average = 1.6665 -> 1.67

        with patch("podcast_scraper.metrics.time.time", return_value=100.0):
            m._start_time = 100.0
            result = m.finish()

        self.assertEqual(result["avg_download_media_seconds"], 2.5)
        self.assertEqual(result["avg_transcribe_seconds"], 2.0)
        self.assertEqual(result["avg_extract_names_seconds"], 0.5)
        self.assertEqual(result["avg_summarize_seconds"], 1.67)

    def test_finish_returns_all_metrics(self):
        """Test that finish returns all metrics in the result dict."""
        m = metrics.Metrics()
        m.episodes_scraped_total = 10
        m.episodes_skipped_total = 2
        m.errors_total = 1
        m.bytes_downloaded_total = 1000
        m.transcripts_downloaded = 5
        m.transcripts_transcribed = 3
        m.episodes_summarized = 2
        m.metadata_files_generated = 1
        m.record_stage("scraping", 1.0)
        m.record_stage("parsing", 0.5)
        m.record_stage("normalizing", 2.0)
        m.record_stage("writing_storage", 0.5)

        with patch("podcast_scraper.metrics.time.time", return_value=100.0):
            m._start_time = 100.0
            result = m.finish()

        # Check all expected keys are present
        expected_keys = {
            "run_duration_seconds",
            "episodes_scraped_total",
            "episodes_skipped_total",
            "errors_total",
            "bytes_downloaded_total",
            "transcripts_downloaded",
            "transcripts_transcribed",
            "episodes_summarized",
            "metadata_files_generated",
            "time_scraping",
            "time_parsing",
            "time_normalizing",
            "time_writing_storage",
            "avg_download_media_seconds",
            "avg_transcribe_seconds",
            "avg_extract_names_seconds",
            "avg_summarize_seconds",
            "download_media_count",
            "transcribe_count",
            "extract_names_count",
            "summarize_count",
        }
        self.assertEqual(set(result.keys()), expected_keys)

        # Check values
        self.assertEqual(result["episodes_scraped_total"], 10)
        self.assertEqual(result["episodes_skipped_total"], 2)
        self.assertEqual(result["errors_total"], 1)
        self.assertEqual(result["bytes_downloaded_total"], 1000)
        self.assertEqual(result["transcripts_downloaded"], 5)
        self.assertEqual(result["transcripts_transcribed"], 3)
        self.assertEqual(result["episodes_summarized"], 2)
        self.assertEqual(result["metadata_files_generated"], 1)
        self.assertEqual(result["time_scraping"], 1.0)
        self.assertEqual(result["time_parsing"], 0.5)
        self.assertEqual(result["time_normalizing"], 2.0)
        self.assertEqual(result["time_writing_storage"], 0.5)

    def test_finish_includes_operation_counts(self):
        """Test that finish includes operation counts."""
        m = metrics.Metrics()
        m.download_media_times = [1.0, 2.0, 3.0]
        m.transcribe_times = [5.0, 10.0]
        m.extract_names_times = [0.5]
        m.summarize_times = []

        with patch("podcast_scraper.metrics.time.time", return_value=100.0):
            m._start_time = 100.0
            result = m.finish()

        self.assertEqual(result["download_media_count"], 3)
        self.assertEqual(result["transcribe_count"], 2)
        self.assertEqual(result["extract_names_count"], 1)
        self.assertEqual(result["summarize_count"], 0)


class TestLogMetrics(unittest.TestCase):
    """Test log_metrics method."""

    @patch("podcast_scraper.metrics.logger.info")
    @patch("podcast_scraper.metrics.time.time")
    def test_log_metrics_calls_finish(self, mock_time, mock_log):
        """Test that log_metrics calls finish and logs the result."""
        mock_time.return_value = 100.0
        m = metrics.Metrics()
        m._start_time = 100.0
        m.episodes_scraped_total = 10

        m.log_metrics()

        # Should call finish (which calculates duration)
        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][0]
        self.assertIn("Pipeline finished:", call_args)
        self.assertIn("Episodes Scraped Total: 10", call_args)

    @patch("podcast_scraper.metrics.logger.info")
    @patch("podcast_scraper.metrics.time.time")
    def test_log_metrics_format(self, mock_time, mock_log):
        """Test that log_metrics formats output correctly."""
        mock_time.return_value = 100.0
        m = metrics.Metrics()
        m._start_time = 100.0
        m.episodes_scraped_total = 5
        m.transcripts_downloaded = 3

        m.log_metrics()

        call_args = mock_log.call_args[0][0]
        # Should have header
        self.assertIn("Pipeline finished:", call_args)
        # Should have formatted keys (underscores replaced, title case)
        self.assertIn("Episodes Scraped Total", call_args)
        self.assertIn("Transcripts Downloaded", call_args)
        # Should have values
        self.assertIn(": 5", call_args)
        self.assertIn(": 3", call_args)

    @patch("podcast_scraper.metrics.logger.info")
    @patch("podcast_scraper.metrics.time.time")
    def test_log_metrics_includes_all_metrics(self, mock_time, mock_log):
        """Test that log_metrics includes all metrics from finish()."""
        mock_time.return_value = 100.0
        m = metrics.Metrics()
        m._start_time = 100.0

        m.log_metrics()

        call_args = mock_log.call_args[0][0]
        # Should include all metrics from finish()
        self.assertIn("Run Duration Seconds", call_args)
        self.assertIn("Episodes Scraped Total", call_args)
        self.assertIn("Time Scraping", call_args)
        self.assertIn("Avg Download Media Seconds", call_args)


if __name__ == "__main__":
    unittest.main()
