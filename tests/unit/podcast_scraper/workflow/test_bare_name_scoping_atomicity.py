"""#1862 defect 1: the GI+KG bare-name scoping writes must be both-or-neither.

Before the fix, GI was written first and a failure of the KG write (schema validate=True, filesystem
error) was swallowed, leaving GI scoped and KG not — the two graph layers permanently disagreeing
about who a person is (prod: 6/6 placeholders GI-only). These tests pin the all-or-nothing contract
on the extracted helper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import podcast_scraper.gi.io as gi_io
import podcast_scraper.gi.schema as gi_schema
from podcast_scraper.workflow.metadata_generation import (
    _write_scoped_artifacts_both_or_neither as write_pair,
)

pytestmark = pytest.mark.unit


def _no_validate(monkeypatch) -> None:
    monkeypatch.setattr(gi_schema, "validate_artifact", lambda *_a, **_k: None)


def test_kg_write_failure_restores_the_prior_gi(tmp_path, monkeypatch) -> None:
    gi, kg = tmp_path / "ep.gi.json", tmp_path / "ep.kg.json"
    gi.write_text("OLD-GI", encoding="utf-8")  # GI already on disk
    _no_validate(monkeypatch)

    def fake_write(path, payload, validate=True):
        if str(path).endswith(".kg.json"):
            raise OSError("disk full")
        Path(path).write_text("NEW-" + payload["id"], encoding="utf-8")

    monkeypatch.setattr(gi_io, "write_artifact", fake_write)

    with pytest.raises(OSError):
        write_pair(gi, {"id": "GI"}, kg, {"id": "KG"})

    assert gi.read_text() == "OLD-GI", "GI was left scoped after the KG write failed (#1862)"
    assert not kg.exists()


def test_kg_write_failure_removes_a_newly_created_gi(tmp_path, monkeypatch) -> None:
    gi, kg = tmp_path / "ep.gi.json", tmp_path / "ep.kg.json"  # GI does NOT pre-exist
    _no_validate(monkeypatch)

    def fake_write(path, payload, validate=True):
        if str(path).endswith(".kg.json"):
            raise OSError("io error")
        Path(path).write_text("NEW", encoding="utf-8")

    monkeypatch.setattr(gi_io, "write_artifact", fake_write)

    with pytest.raises(OSError):
        write_pair(gi, {"id": "GI"}, kg, {"id": "KG"})

    assert not gi.exists(), "a GI file created this pass survived a failed KG write (#1862)"
    assert not kg.exists()


def test_prevalidation_failure_writes_neither_file(tmp_path, monkeypatch) -> None:
    gi, kg = tmp_path / "ep.gi.json", tmp_path / "ep.kg.json"
    written: list[str] = []
    monkeypatch.setattr(
        gi_io, "write_artifact", lambda p, _pay, validate=True: written.append(str(p))
    )

    def bad_validate(payload, strict=False):
        if payload.get("id") == "KG":
            raise ValueError("bad KG schema")

    monkeypatch.setattr(gi_schema, "validate_artifact", bad_validate)

    with pytest.raises(ValueError):
        write_pair(gi, {"id": "GI"}, kg, {"id": "KG"})

    assert written == [], "a payload was written despite the other failing pre-validation (#1862)"
    assert not gi.exists() and not kg.exists()


def test_happy_path_writes_both(tmp_path, monkeypatch) -> None:
    gi, kg = tmp_path / "ep.gi.json", tmp_path / "ep.kg.json"
    _no_validate(monkeypatch)
    monkeypatch.setattr(
        gi_io,
        "write_artifact",
        lambda path, payload, validate=True: Path(path).write_text(
            "W-" + payload["id"], encoding="utf-8"
        ),
    )

    write_pair(gi, {"id": "GI"}, kg, {"id": "KG"})

    assert gi.read_text() == "W-GI"
    assert kg.read_text() == "W-KG"


def test_none_target_skips_that_layer(tmp_path, monkeypatch) -> None:
    """A None target means 'no change for that layer' — the other still writes, no rollback."""
    kg = tmp_path / "ep.kg.json"
    _no_validate(monkeypatch)
    monkeypatch.setattr(
        gi_io,
        "write_artifact",
        lambda path, payload, validate=True: Path(path).write_text("KG", encoding="utf-8"),
    )
    write_pair(None, {"id": "GI"}, kg, {"id": "KG"})
    assert kg.read_text() == "KG"
