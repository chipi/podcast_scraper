"""Each artifact layer must be validated by ITS OWN schema (#1862 follow-up).

THE BUG. ``_write_scoped_artifacts_both_or_neither`` validated BOTH payloads with the GI
validator. The KG schema has no ``model_version`` at the top level and should not — its required
set is ``(schema_version, episode_id, extraction, nodes, edges)`` — so every bare-name scoping
pass that had a KG layer to write failed with:

    GIL artifact missing required key: 'model_version'

WHY IT WAS INVISIBLE. The function's guard 1 deliberately validates both payloads BEFORE
writing either, so that a schema failure aborts before any file is touched. That guard did its
job perfectly on a false failure: the write was refused, the caller swallowed it as
"Bare-name scoping failed (non-fatal, ids left as minted)", and the episode silently kept
UNSCOPED person ids. Ten episodes in the 2026-08-31 production batch, with an error naming the
GI artifact for a defect entirely about the KG one — so reading the message led away from the
cause.

That misdirection is the reason this gets its own test rather than a one-line fix: the shape
(right guard, wrong input, message blames the innocent layer) is not something a reader of the
log would ever untangle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.workflow.metadata_generation import (
    _write_scoped_artifacts_both_or_neither as write_pair,
)


def _gi() -> dict:
    return {
        "schema_version": "3.1",
        "model_version": "vllm:qwen3-30b",
        "prompt_version": "v2",
        "episode_id": "ep-1",
        "nodes": [],
        "edges": [],
    }


def _kg() -> dict:
    """A VALID KG artifact — note it has no top-level ``model_version`` by design."""
    return {
        "schema_version": "2.0",
        "episode_id": "ep-1",
        "extraction": {
            "model_version": "vllm:qwen3-30b",
            "extracted_at": "2026-09-01T00:00:00Z",
            "transcript_ref": "transcripts/0001 - Ep.txt",
        },
        "nodes": [],
        "edges": [],
    }


class TestBothLayersWrite:
    def test_a_valid_pair_is_written(self, tmp_path: Path):
        """THE REGRESSION. This raised 'GIL artifact missing required key: model_version'."""
        gi_p, kg_p = tmp_path / "a.gi.json", tmp_path / "a.kg.json"

        write_pair(gi_p, _gi(), kg_p, _kg())

        assert gi_p.exists() and kg_p.exists()
        assert json.loads(kg_p.read_text())["episode_id"] == "ep-1"

    def test_a_valid_kg_is_not_judged_by_the_gi_schema(self):
        """Pins WHY the pair used to fail, by showing the wrong validator rejects a good KG."""
        from podcast_scraper.gi.schema import validate_artifact as gi_validate
        from podcast_scraper.kg.schema import validate_artifact as kg_validate

        kg_validate(_kg(), strict=False)  # correct validator: fine
        with pytest.raises(ValueError, match="model_version"):
            gi_validate(_kg(), strict=False)  # the bug, verbatim

    def test_gi_only_still_works(self, tmp_path: Path):
        gi_p = tmp_path / "a.gi.json"
        write_pair(gi_p, _gi(), None, _kg())
        assert gi_p.exists()

    def test_kg_only_still_works(self, tmp_path: Path):
        kg_p = tmp_path / "a.kg.json"
        write_pair(None, _gi(), kg_p, _kg())
        assert kg_p.exists()


class TestValidationStillGuardsBothOrNeither:
    """The fix must not weaken guard 1 — validate both, then write."""

    def test_an_invalid_KG_aborts_before_the_GI_file_is_touched(self, tmp_path: Path):
        gi_p, kg_p = tmp_path / "a.gi.json", tmp_path / "a.kg.json"
        bad_kg = _kg()
        del bad_kg["extraction"]

        with pytest.raises(ValueError):
            write_pair(gi_p, _gi(), kg_p, bad_kg)

        assert not gi_p.exists(), "GI must not be written when the KG payload is invalid"
        assert not kg_p.exists()

    def test_an_invalid_GI_aborts_before_anything_is_written(self, tmp_path: Path):
        gi_p, kg_p = tmp_path / "a.gi.json", tmp_path / "a.kg.json"
        bad_gi = _gi()
        del bad_gi["model_version"]

        with pytest.raises(ValueError, match="model_version"):
            write_pair(gi_p, bad_gi, kg_p, _kg())

        assert not gi_p.exists() and not kg_p.exists()

    def test_an_existing_gi_is_left_intact_when_the_kg_payload_is_invalid(self, tmp_path: Path):
        """A refused pass must be a no-op, not a half-apply over a good prior artifact."""
        gi_p, kg_p = tmp_path / "a.gi.json", tmp_path / "a.kg.json"
        gi_p.write_text(json.dumps({"prior": "bytes"}), encoding="utf-8")
        bad_kg = _kg()
        del bad_kg["episode_id"]

        with pytest.raises(ValueError):
            write_pair(gi_p, _gi(), kg_p, bad_kg)

        assert json.loads(gi_p.read_text()) == {"prior": "bytes"}
