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
    _resolve_scope_write_targets as resolve_targets,
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


# --- #1862: the write-target refusal that forbids a GI-only scope -------------------------------
# `_resolve_scope_write_targets` is the guard that says the pass may NEVER scope one layer and
# leave the other bare on disk. It is what stops a future "just widen the gate so GI is scoped
# whenever GI exists, regardless of KG" change from reopening the desync #1862 exists to close.
# These pin it directly rather than through the whole metadata-generation harness.


def test_planned_kg_change_with_no_kg_target_refuses() -> None:
    """A KG id change with no bound KG target must raise, not scope GI alone (#1862 defect 1)."""
    with pytest.raises(RuntimeError, match="no KG write target is bound"):
        resolve_targets("ep.gi.json", True, None, True)


def test_gi_only_change_writes_gi_and_leaves_kg_untouched() -> None:
    """GI changed, KG did not: write GI, no KG target — no desync because KG is unchanged."""
    gi_target, kg_target = resolve_targets("ep.gi.json", True, "ep.kg.json", False)
    assert gi_target == Path("ep.gi.json")
    assert kg_target is None, "an unchanged KG layer must not be rewritten"


def test_both_changed_writes_both() -> None:
    gi_target, kg_target = resolve_targets("ep.gi.json", True, "ep.kg.json", True)
    assert gi_target == Path("ep.gi.json")
    assert kg_target == Path("ep.kg.json")


def test_an_unchanged_gi_layer_is_not_rewritten() -> None:
    """A no-op layer keeps its file — this is what makes re-running the pass a true no-op."""
    gi_target, kg_target = resolve_targets("ep.gi.json", False, "ep.kg.json", False)
    assert gi_target is None and kg_target is None


def test_kg_change_with_a_bound_target_is_allowed() -> None:
    """The refusal fires only when the target is MISSING — a bound target scopes normally."""
    gi_target, kg_target = resolve_targets(None, False, "ep.kg.json", True)
    assert gi_target is None
    assert kg_target == Path("ep.kg.json")
