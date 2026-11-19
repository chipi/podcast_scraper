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
        "schema_version": "2.0",
        "episode_id": "e:test",
        "extraction": {
            "model_version": "metadata_only",  # "stub" until #1657; KG 3.0 has no stub
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
            # R3b: every KG file is written as KG 3.0; the root names its item.
            self.assertEqual(loaded["schema_version"], "3.0")
            self.assertEqual(loaded["item_id"], "e:test")
            self.assertNotIn("episode_id", loaded)
            self.assertEqual(loaded["nodes"], [])
            self.assertEqual(loaded["edges"], [])

    def test_write_from_fixture_round_trip(self) -> None:
        import tempfile

        raw = json.loads(_FIXTURE_MINIMAL.read_text(encoding="utf-8"))
        # The fixture keeps the legacy "stub"; KG 3.0 names that mode metadata_only (#1657).
        raw["extraction"]["model_version"] = "metadata_only"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.kg.json"
            write_artifact(path, raw, validate=True)
            loaded = read_artifact(path)
            self.assertEqual(loaded["item_id"], raw["episode_id"])
            self.assertEqual(len(loaded["nodes"]), 1)

    def test_write_validate_true_rejects_invalid_payload(self) -> None:
        import tempfile

        bad = {"schema_version": "2.0"}  # missing required keys
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.kg.json"
            with self.assertRaises(ValueError):
                write_artifact(path, bad, validate=True)
            self.assertFalse(path.exists())

    def test_validate_false_never_writes_an_invalid_kg_3_0_file(self) -> None:
        import tempfile

        # R3b: the written file is validated strictly against KG 3.0 whatever ``validate`` says;
        # an invalid file is never written (never warn and write).
        arbitrary = {"foo": 1}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "raw.kg.json"
            with self.assertRaises(Exception):
                write_artifact(path, arbitrary, validate=False)
            self.assertFalse(path.exists())
