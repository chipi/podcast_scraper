"""Unit tests for KG artifact read/write helpers."""

import json
import unittest
from pathlib import Path

import pytest

from podcast_scraper.kg.io import read_artifact, write_artifact

pytestmark = [pytest.mark.unit]

_FIXTURE_MINIMAL = Path(__file__).resolve().parents[3] / "fixtures" / "kg" / "minimal.kg.json"


def _minimal_valid_payload() -> dict:
    return {
        "schema_version": "1.0",
        "episode_id": "e:test",
        "extraction": {
            "model_version": "stub",
            "extracted_at": "2026-01-01T00:00:00Z",
            "transcript_ref": "t.txt",
        },
        "nodes": [],
        "edges": [],
    }


class TestKgIoWriteRead(unittest.TestCase):
    """write_artifact / read_artifact."""

    def test_read_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_artifact(Path("/nonexistent/nope.kg.json"))

    def test_write_read_round_trip(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nested" / "episode.kg.json"
            payload = _minimal_valid_payload()
            write_artifact(path, payload, validate=True)
            self.assertTrue(path.is_file())
            loaded = read_artifact(path)
            self.assertEqual(loaded["episode_id"], "e:test")
            self.assertEqual(loaded["nodes"], [])
            self.assertEqual(loaded["edges"], [])

    def test_write_from_fixture_round_trip(self) -> None:
        import tempfile

        raw = json.loads(_FIXTURE_MINIMAL.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.kg.json"
            write_artifact(path, raw, validate=True)
            loaded = read_artifact(path)
            self.assertEqual(loaded["episode_id"], raw["episode_id"])
            self.assertEqual(len(loaded["nodes"]), 1)

    def test_write_validate_true_rejects_invalid_payload(self) -> None:
        import tempfile

        bad = {"schema_version": "1.0"}  # missing required keys
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.kg.json"
            with self.assertRaises(ValueError):
                write_artifact(path, bad, validate=True)
            self.assertFalse(path.exists())

    def test_write_validate_false_writes_without_validation(self) -> None:
        import tempfile

        arbitrary = {"foo": 1}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "raw.kg.json"
            write_artifact(path, arbitrary, validate=False)
            loaded = read_artifact(path)
            self.assertEqual(loaded, arbitrary)
