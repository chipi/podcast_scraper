#!/usr/bin/env python3
"""Unit tests for JSONL metrics emitter."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import pytest

from podcast_scraper import config
from podcast_scraper.workflow import jsonl_emitter, metrics

pytestmark = [pytest.mark.unit]


class TestJSONLEmitter(unittest.TestCase):
    """Tests for JSONL metrics emitter."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.jsonl_path = os.path.join(self.temp_dir, "test.jsonl")
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            jsonl_metrics_enabled=True,
            jsonl_metrics_path=self.jsonl_path,
        )
        self.metrics = metrics.Metrics()
        self.run_id = "test-run-123"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_emitter_initialization_enabled(self):
        """Test emitter initialization when enabled."""
        emitter = jsonl_emitter.JSONLEmitter(self.metrics, self.jsonl_path)
        self.assertEqual(emitter.jsonl_path, Path(self.jsonl_path))
        # Cleanup
        if emitter._file_handle:
            emitter.close()

    def test_emitter_initialization_disabled(self):
        """Test emitter initialization when disabled."""
        # Even when disabled, emitter can be created with a path
        emitter = jsonl_emitter.JSONLEmitter(self.metrics, self.jsonl_path)
        self.assertEqual(emitter.jsonl_path, Path(self.jsonl_path))

    def test_emit_run_started(self):
        """Test emitting run_started event."""
        emitter = jsonl_emitter.JSONLEmitter(self.metrics, self.jsonl_path)
        with emitter:
            emitter.emit_run_started(self.cfg, self.run_id)

        # Verify file was created
        self.assertTrue(os.path.exists(self.jsonl_path))

        # Verify content
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            line = f.readline()
            event = json.loads(line)

        self.assertEqual(event["event_type"], "run_started")
        self.assertEqual(event["run_id"], self.run_id)
        self.assertIn("timestamp", event)
        self.assertIn("config", event)

    def test_emit_episode_finished(self):
        """Test emitting episode_finished event."""
        emitter = jsonl_emitter.JSONLEmitter(self.metrics, self.jsonl_path)

        # Add episode metrics using the correct API
        episode_id = "episode-1"
        episode_number = 1
        self.metrics.get_or_create_episode_metrics(episode_id, episode_number)
        self.metrics.update_episode_metrics(
            episode_id=episode_id,
            audio_sec=100.0,
            transcribe_sec=5.0,
            summary_sec=2.0,
        )

        with emitter:
            emitter.emit_episode_finished(episode_id)

        # Verify content
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            event = json.loads(lines[-1])

        self.assertEqual(event["event_type"], "episode_finished")
        self.assertEqual(event["episode_id"], episode_id)
        self.assertEqual(event["audio_sec"], 100.0)
        self.assertIn("timestamp", event)

    def test_emit_run_finished(self):
        """Test emitting run_finished event."""
        emitter = jsonl_emitter.JSONLEmitter(self.metrics, self.jsonl_path)
        with emitter:
            emitter.emit_run_finished()

        # Verify content
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            event = json.loads(lines[-1])

        self.assertEqual(event["event_type"], "run_finished")
        self.assertIn("timestamp", event)
        self.assertIn("run_duration_seconds", event)

    def test_emitter_appends_to_existing_file(self):
        """Test that emitter appends to existing file."""
        emitter = jsonl_emitter.JSONLEmitter(self.metrics, self.jsonl_path)

        # Emit first event
        with emitter:
            emitter.emit_run_started(self.cfg, self.run_id)

        # Emit second event (should append)
        with emitter:
            emitter.emit_run_finished()

        # Verify both events are in file
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)

            first_event = json.loads(lines[0])
            second_event = json.loads(lines[1])

            self.assertEqual(first_event["event_type"], "run_started")
            self.assertEqual(second_event["event_type"], "run_finished")

    def test_emitter_handles_write_failure_gracefully(self):
        """Test that emitter handles write failures gracefully."""
        emitter = jsonl_emitter.JSONLEmitter(self.metrics, self.jsonl_path)

        # Try to write without opening context manager - should raise RuntimeError
        with self.assertRaises(RuntimeError):
            emitter.emit_run_started(self.cfg, self.run_id)
