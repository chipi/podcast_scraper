"""Indexer coverage for episode-level metadata rows (2026-07-22).

The indexer emits three additional searchable documents per episode
alongside the pre-existing summary bullets and transcript chunks:

- ``doc_type: episode_title`` — from ``doc.episode.title``.
- ``doc_type: episode_description`` — from ``doc.episode.description``.
- ``doc_type: summary_short`` — from ``doc.summary.short_summary``.

Each row carries a ``matched_field`` metadata marker so the viewer can
render "matched: Title / Description / Summary" chips next to the hit.
Existing ``summary`` (bullet) rows also gain ``matched_field:
summary_bullet`` for the same reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search.indexer import _collect_docs_for_episode

pytestmark = pytest.mark.unit


def _write_meta(root: Path, doc: dict) -> Path:
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    p = meta / "ep.metadata.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _base_doc() -> dict:
    return {
        "feed": {"feed_id": "showx", "title": "Show X"},
        "episode": {
            "episode_id": "ep1",
            "title": "How do we make AI safer?",
            "description": "A long description about AI safety, alignment, and policy.",
            "published_date": "2026-06-15T00:00:00Z",
        },
        "summary": {
            "title": "AI safety discussion",
            "bullets": [
                "AI safety is important.",
                "Alignment is hard.",
            ],
            "short_summary": (
                "Guests debate the shape of AI safety work and whether current "
                "alignment techniques scale to frontier models."
            ),
        },
    }


def _rows_by_doc_type(rows: list) -> dict[str, list[tuple[str, dict]]]:
    out: dict[str, list[tuple[str, dict]]] = {}
    for _, text, meta in rows:
        out.setdefault(meta.get("doc_type", ""), []).append((text, meta))
    return out


def test_indexer_emits_episode_title_document(tmp_path: Path) -> None:
    """One ``episode_title`` row per episode carrying the raw title text
    and a ``matched_field: title`` marker."""
    doc = _base_doc()
    p = _write_meta(tmp_path, doc)
    rows = _collect_docs_for_episode(
        tmp_path,
        p,
        doc,
        target_tokens=200,
        overlap_tokens=20,
        metadata_relative_path="metadata/ep.metadata.json",
    )
    by_type = _rows_by_doc_type(rows)
    title_rows = by_type.get("episode_title", [])
    assert len(title_rows) == 1
    text, meta = title_rows[0]
    assert text == "How do we make AI safer?"
    assert meta["matched_field"] == "title"
    assert meta["episode_id"] == "ep1"
    assert meta["source_id"] == "ep1"


def test_indexer_emits_episode_description_document(tmp_path: Path) -> None:
    doc = _base_doc()
    p = _write_meta(tmp_path, doc)
    rows = _collect_docs_for_episode(
        tmp_path,
        p,
        doc,
        target_tokens=200,
        overlap_tokens=20,
        metadata_relative_path="metadata/ep.metadata.json",
    )
    by_type = _rows_by_doc_type(rows)
    desc_rows = by_type.get("episode_description", [])
    assert len(desc_rows) == 1
    text, meta = desc_rows[0]
    assert "AI safety" in text
    assert meta["matched_field"] == "description"


def test_indexer_emits_summary_short_document(tmp_path: Path) -> None:
    """``summary.short_summary`` — the LLM-generated whole-episode paragraph
    — becomes its own ``summary_short`` row (distinct from the per-bullet
    ``summary`` rows so the viewer can label them differently)."""
    doc = _base_doc()
    p = _write_meta(tmp_path, doc)
    rows = _collect_docs_for_episode(
        tmp_path,
        p,
        doc,
        target_tokens=200,
        overlap_tokens=20,
        metadata_relative_path="metadata/ep.metadata.json",
    )
    by_type = _rows_by_doc_type(rows)
    short_rows = by_type.get("summary_short", [])
    assert len(short_rows) == 1
    text, meta = short_rows[0]
    assert "alignment techniques" in text
    assert meta["matched_field"] == "summary"
    assert meta["source_id"] == "short_summary"


def test_indexer_still_emits_summary_bullets_with_matched_field_marker(tmp_path: Path) -> None:
    """Existing per-bullet ``summary`` rows now carry
    ``matched_field: summary_bullet`` so the client can distinguish them
    from ``summary_short`` in the hit list. Row count is preserved."""
    doc = _base_doc()
    p = _write_meta(tmp_path, doc)
    rows = _collect_docs_for_episode(
        tmp_path,
        p,
        doc,
        target_tokens=200,
        overlap_tokens=20,
        metadata_relative_path="metadata/ep.metadata.json",
    )
    by_type = _rows_by_doc_type(rows)
    summary_rows = by_type.get("summary", [])
    assert len(summary_rows) == 2  # matches the two bullets in _base_doc
    for _, meta in summary_rows:
        assert meta["matched_field"] == "summary_bullet"


def test_indexer_omits_episode_metadata_rows_when_source_field_absent(
    tmp_path: Path,
) -> None:
    """Missing / empty title / description / short_summary → no row for
    that surface. Never a placeholder empty-text document."""
    doc = _base_doc()
    doc["episode"].pop("description")
    doc["summary"].pop("short_summary")
    p = _write_meta(tmp_path, doc)
    rows = _collect_docs_for_episode(
        tmp_path,
        p,
        doc,
        target_tokens=200,
        overlap_tokens=20,
        metadata_relative_path="metadata/ep.metadata.json",
    )
    by_type = _rows_by_doc_type(rows)
    assert "episode_description" not in by_type
    assert "summary_short" not in by_type
    # Title stays because we did not drop it.
    assert "episode_title" in by_type


def test_indexer_omits_episode_title_when_title_is_blank(tmp_path: Path) -> None:
    doc = _base_doc()
    doc["episode"]["title"] = "   "
    p = _write_meta(tmp_path, doc)
    rows = _collect_docs_for_episode(
        tmp_path,
        p,
        doc,
        target_tokens=200,
        overlap_tokens=20,
        metadata_relative_path="metadata/ep.metadata.json",
    )
    by_type = _rows_by_doc_type(rows)
    assert "episode_title" not in by_type
