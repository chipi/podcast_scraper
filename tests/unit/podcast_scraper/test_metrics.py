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
            # LLM metrics
            "llm_transcription_calls",
            "llm_transcription_audio_minutes",
            "llm_speaker_detection_calls",
            "llm_speaker_detection_input_tokens",
            "llm_speaker_detection_output_tokens",
            "llm_summarization_calls",
            "llm_summarization_input_tokens",
            "llm_summarization_output_tokens",
            # Preprocessing metrics
            "avg_preprocessing_seconds",
            "preprocessing_count",
            "avg_preprocessing_size_reduction_percent",
            "preprocessing_cache_hits",
            "preprocessing_cache_misses",
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


class TestPreprocessingMetrics(unittest.TestCase):
    """Test preprocessing metrics recording."""

    def test_record_preprocessing_time(self):
        """Test recording preprocessing time."""
        m = metrics.Metrics()
        m.record_preprocessing_time(1.5)
        self.assertEqual(m.preprocessing_times, [1.5])
        m.record_preprocessing_time(2.0)
        self.assertEqual(m.preprocessing_times, [1.5, 2.0])

    def test_record_preprocessing_size_reduction(self):
        """Test recording preprocessing size reduction."""
        m = metrics.Metrics()
        m.record_preprocessing_size_reduction(1000000, 500000)  # 1MB -> 0.5MB
        self.assertEqual(m.preprocessing_original_sizes, [1000000])
        self.assertEqual(m.preprocessing_preprocessed_sizes, [500000])
        m.record_preprocessing_size_reduction(2000000, 800000)  # 2MB -> 0.8MB
        self.assertEqual(m.preprocessing_original_sizes, [1000000, 2000000])
        self.assertEqual(m.preprocessing_preprocessed_sizes, [500000, 800000])

    def test_record_preprocessing_cache_hit(self):
        """Test recording preprocessing cache hit."""
        m = metrics.Metrics()
        m.record_preprocessing_cache_hit()
        self.assertEqual(m.preprocessing_cache_hits, 1)
        m.record_preprocessing_cache_hit()
        self.assertEqual(m.preprocessing_cache_hits, 2)

    def test_record_preprocessing_cache_miss(self):
        """Test recording preprocessing cache miss."""
        m = metrics.Metrics()
        m.record_preprocessing_cache_miss()
        self.assertEqual(m.preprocessing_cache_misses, 1)
        m.record_preprocessing_cache_miss()
        self.assertEqual(m.preprocessing_cache_misses, 2)

    def test_finish_includes_preprocessing_metrics(self):
        """Test that finish includes preprocessing metrics."""
        m = metrics.Metrics()
        m.record_preprocessing_time(1.5)
        m.record_preprocessing_time(2.5)
        m.record_preprocessing_size_reduction(1000000, 500000)  # 50% reduction
        m.record_preprocessing_size_reduction(2000000, 1000000)  # 50% reduction
        m.record_preprocessing_cache_hit()
        m.record_preprocessing_cache_miss()

        with patch("podcast_scraper.metrics.time.time", return_value=100.0):
            m._start_time = 100.0
            result = m.finish()

        # Check preprocessing metrics are included
        self.assertEqual(result["avg_preprocessing_seconds"], 2.0)  # (1.5 + 2.5) / 2
        self.assertEqual(result["preprocessing_count"], 2)
        self.assertEqual(result["avg_preprocessing_size_reduction_percent"], 50.0)  # Average 50%
        self.assertEqual(result["preprocessing_cache_hits"], 1)
        self.assertEqual(result["preprocessing_cache_misses"], 1)

    def test_finish_preprocessing_metrics_with_no_data(self):
        """Test that finish handles preprocessing metrics with no data."""
        m = metrics.Metrics()

        with patch("podcast_scraper.metrics.time.time", return_value=100.0):
            m._start_time = 100.0
            result = m.finish()

        # Check preprocessing metrics default to 0
        self.assertEqual(result["avg_preprocessing_seconds"], 0.0)
        self.assertEqual(result["preprocessing_count"], 0)
        self.assertEqual(result["avg_preprocessing_size_reduction_percent"], 0.0)
        self.assertEqual(result["preprocessing_cache_hits"], 0)
        self.assertEqual(result["preprocessing_cache_misses"], 0)

    def test_finish_updates_expected_keys_with_preprocessing(self):
        """Test that finish includes preprocessing keys in expected keys."""
        m = metrics.Metrics()
        m.record_preprocessing_time(1.0)

        with patch("podcast_scraper.metrics.time.time", return_value=100.0):
            m._start_time = 100.0
            result = m.finish()

        # Check preprocessing keys are present
        self.assertIn("avg_preprocessing_seconds", result)
        self.assertIn("preprocessing_count", result)
        self.assertIn("avg_preprocessing_size_reduction_percent", result)
        self.assertIn("preprocessing_cache_hits", result)
        self.assertIn("preprocessing_cache_misses", result)


class TestLogMetrics(unittest.TestCase):
    """Test log_metrics method."""

    @patch("podcast_scraper.metrics.logger.debug")
    @patch("podcast_scraper.metrics.time.time")
    def test_log_metrics_calls_finish(self, mock_time, mock_log):
        """Test that log_metrics calls finish and logs the result."""
        mock_time.return_value = 100.0
        m = metrics.Metrics()
        m._start_time = 100.0
        m.episodes_scraped_total = 10

        m.log_metrics()

        # Should call finish (which calculates duration) and log at DEBUG level (per RFC-027)
        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][0]
        self.assertIn("Pipeline finished", call_args)
        self.assertIn("Episodes Scraped Total: 10", call_args)

    @patch("podcast_scraper.metrics.logger.debug")
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
        self.assertIn("Pipeline finished", call_args)
        # Should have formatted keys (underscores replaced, title case)
        self.assertIn("Episodes Scraped Total", call_args)
        self.assertIn("Transcripts Downloaded", call_args)
        # Should have values
        self.assertIn(": 5", call_args)
        self.assertIn(": 3", call_args)

    @patch("podcast_scraper.metrics.logger.debug")
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


class TestRecordLLMTranscriptionCall(unittest.TestCase):
    """Test record_llm_transcription_call method."""

    def test_record_single_transcription_call(self):
        """Test recording a single transcription call."""
        m = metrics.Metrics()
        m.record_llm_transcription_call(5.5)  # 5.5 minutes
        self.assertEqual(m.llm_transcription_calls, 1)
        self.assertEqual(m.llm_transcription_audio_minutes, 5.5)

    def test_record_multiple_transcription_calls(self):
        """Test recording multiple transcription calls."""
        m = metrics.Metrics()
        m.record_llm_transcription_call(3.0)
        m.record_llm_transcription_call(5.5)
        m.record_llm_transcription_call(2.5)
        self.assertEqual(m.llm_transcription_calls, 3)
        self.assertEqual(m.llm_transcription_audio_minutes, 11.0)  # 3.0 + 5.5 + 2.5

    def test_record_zero_minutes(self):
        """Test recording zero minutes."""
        m = metrics.Metrics()
        m.record_llm_transcription_call(0.0)
        self.assertEqual(m.llm_transcription_calls, 1)
        self.assertEqual(m.llm_transcription_audio_minutes, 0.0)


class TestRecordLLMSpeakerDetectionCall(unittest.TestCase):
    """Test record_llm_speaker_detection_call method."""

    def test_record_single_speaker_detection_call(self):
        """Test recording a single speaker detection call."""
        m = metrics.Metrics()
        m.record_llm_speaker_detection_call(input_tokens=100, output_tokens=50)
        self.assertEqual(m.llm_speaker_detection_calls, 1)
        self.assertEqual(m.llm_speaker_detection_input_tokens, 100)
        self.assertEqual(m.llm_speaker_detection_output_tokens, 50)

    def test_record_multiple_speaker_detection_calls(self):
        """Test recording multiple speaker detection calls."""
        m = metrics.Metrics()
        m.record_llm_speaker_detection_call(input_tokens=100, output_tokens=50)
        m.record_llm_speaker_detection_call(input_tokens=200, output_tokens=75)
        m.record_llm_speaker_detection_call(input_tokens=150, output_tokens=60)
        self.assertEqual(m.llm_speaker_detection_calls, 3)
        self.assertEqual(m.llm_speaker_detection_input_tokens, 450)  # 100 + 200 + 150
        self.assertEqual(m.llm_speaker_detection_output_tokens, 185)  # 50 + 75 + 60

    def test_record_zero_tokens(self):
        """Test recording zero tokens."""
        m = metrics.Metrics()
        m.record_llm_speaker_detection_call(input_tokens=0, output_tokens=0)
        self.assertEqual(m.llm_speaker_detection_calls, 1)
        self.assertEqual(m.llm_speaker_detection_input_tokens, 0)
        self.assertEqual(m.llm_speaker_detection_output_tokens, 0)


class TestRecordLLMSummarizationCall(unittest.TestCase):
    """Test record_llm_summarization_call method."""

    def test_record_single_summarization_call(self):
        """Test recording a single summarization call."""
        m = metrics.Metrics()
        m.record_llm_summarization_call(input_tokens=1000, output_tokens=500)
        self.assertEqual(m.llm_summarization_calls, 1)
        self.assertEqual(m.llm_summarization_input_tokens, 1000)
        self.assertEqual(m.llm_summarization_output_tokens, 500)

    def test_record_multiple_summarization_calls(self):
        """Test recording multiple summarization calls."""
        m = metrics.Metrics()
        m.record_llm_summarization_call(input_tokens=1000, output_tokens=500)
        m.record_llm_summarization_call(input_tokens=2000, output_tokens=800)
        m.record_llm_summarization_call(input_tokens=1500, output_tokens=600)
        self.assertEqual(m.llm_summarization_calls, 3)
        self.assertEqual(m.llm_summarization_input_tokens, 4500)  # 1000 + 2000 + 1500
        self.assertEqual(m.llm_summarization_output_tokens, 1900)  # 500 + 800 + 600

    def test_record_zero_tokens(self):
        """Test recording zero tokens."""
        m = metrics.Metrics()
        m.record_llm_summarization_call(input_tokens=0, output_tokens=0)
        self.assertEqual(m.llm_summarization_calls, 1)
        self.assertEqual(m.llm_summarization_input_tokens, 0)
        self.assertEqual(m.llm_summarization_output_tokens, 0)


class TestFinishIncludesLLMMetrics(unittest.TestCase):
    """Test that finish() includes LLM metrics in the output."""

    def test_finish_includes_llm_transcription_metrics(self):
        """Test that finish includes LLM transcription metrics."""
        m = metrics.Metrics()
        m.record_llm_transcription_call(10.5)
        m.record_llm_transcription_call(5.0)

        with patch("podcast_scraper.metrics.time.time", return_value=100.0):
            m._start_time = 100.0
            result = m.finish()

        self.assertEqual(result["llm_transcription_calls"], 2)
        self.assertEqual(result["llm_transcription_audio_minutes"], 15.5)

    def test_finish_includes_llm_speaker_detection_metrics(self):
        """Test that finish includes LLM speaker detection metrics."""
        m = metrics.Metrics()
        m.record_llm_speaker_detection_call(input_tokens=100, output_tokens=50)
        m.record_llm_speaker_detection_call(input_tokens=200, output_tokens=75)

        with patch("podcast_scraper.metrics.time.time", return_value=100.0):
            m._start_time = 100.0
            result = m.finish()

        self.assertEqual(result["llm_speaker_detection_calls"], 2)
        self.assertEqual(result["llm_speaker_detection_input_tokens"], 300)
        self.assertEqual(result["llm_speaker_detection_output_tokens"], 125)

    def test_finish_includes_llm_summarization_metrics(self):
        """Test that finish includes LLM summarization metrics."""
        m = metrics.Metrics()
        m.record_llm_summarization_call(input_tokens=1000, output_tokens=500)
        m.record_llm_summarization_call(input_tokens=2000, output_tokens=800)

        with patch("podcast_scraper.metrics.time.time", return_value=100.0):
            m._start_time = 100.0
            result = m.finish()

        self.assertEqual(result["llm_summarization_calls"], 2)
        self.assertEqual(result["llm_summarization_input_tokens"], 3000)
        self.assertEqual(result["llm_summarization_output_tokens"], 1300)

    def test_finish_rounds_audio_minutes(self):
        """Test that finish rounds audio minutes to 2 decimal places."""
        m = metrics.Metrics()
        m.record_llm_transcription_call(10.123456)

        with patch("podcast_scraper.metrics.time.time", return_value=100.0):
            m._start_time = 100.0
            result = m.finish()

        self.assertEqual(result["llm_transcription_audio_minutes"], 10.12)

    def test_finish_includes_all_llm_metrics_when_empty(self):
        """Test that finish includes all LLM metrics even when zero."""
        m = metrics.Metrics()

        with patch("podcast_scraper.metrics.time.time", return_value=100.0):
            m._start_time = 100.0
            result = m.finish()

        # Should include all LLM metrics with zero values
        self.assertEqual(result["llm_transcription_calls"], 0)
        self.assertEqual(result["llm_transcription_audio_minutes"], 0.0)
        self.assertEqual(result["llm_speaker_detection_calls"], 0)
        self.assertEqual(result["llm_speaker_detection_input_tokens"], 0)
        self.assertEqual(result["llm_speaker_detection_output_tokens"], 0)
        self.assertEqual(result["llm_summarization_calls"], 0)
        self.assertEqual(result["llm_summarization_input_tokens"], 0)
        self.assertEqual(result["llm_summarization_output_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
