"""Integration: ``gi.explore`` UC5 / UC1 paths (semantic viewer + CLI backend).

Drives real artifact scan and insight resolution on disk fixtures (no HTTP mocks).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.gi.explore import (
    run_uc1_topic_research,
    run_uc5_insight_explorer,
    scan_artifact_paths,
)

pytestmark = pytest.mark.integration


def _minimal_gi(episode_id: str, insight_text: str) -> dict:
    return {
        "schema_version": "3.0",
        "model_version": "stub",
        "prompt_version": "v1",
        "episode_id": episode_id,
        "nodes": [
            {
                "id": "insight-1",
                "type": "Insight",
                "properties": {
                    "text": insight_text,
                    "grounded": True,
                    "confidence": 0.88,
                },
            },
            {
                "id": "quote-1",
                "type": "Quote",
                "properties": {
                    "text": "Supporting line.",
                    "char_start": 0,
                    "char_end": 16,
                    "timestamp_start_ms": 0,
                    "timestamp_end_ms": 1000,
                    "transcript_ref": "t.txt",
                },
            },
        ],
        "edges": [
            {"type": "SUPPORTED_BY", "from": "insight-1", "to": "quote-1"},
        ],
    }


def test_scan_artifact_paths_finds_metadata_gi(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    p = meta / "a.gi.json"
    p.write_text(json.dumps(_minimal_gi("ep-a", "hello")), encoding="utf-8")
    paths = scan_artifact_paths(tmp_path)
    assert p.resolve() in [x.resolve() for x in paths]


def test_run_uc5_empty_corpus_returns_zero_episodes(tmp_path: Path) -> None:
    out = run_uc5_insight_explorer(tmp_path)
    assert out.episodes_searched == 0
    assert out.insights == []


def test_run_uc5_loads_insights_from_gi_fixture(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "one.gi.json").write_text(
        json.dumps(_minimal_gi("ep-int", "Climate policy and cities")),
        encoding="utf-8",
    )
    out = run_uc5_insight_explorer(tmp_path, topic=None, limit=20)
    assert out.episodes_searched >= 1
    assert len(out.insights) >= 1
    assert "Climate" in (out.insights[0].text or "")


def test_run_uc1_filters_by_topic_substring(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "a.gi.json").write_text(
        json.dumps(_minimal_gi("e1", "Unrelated economics")),
        encoding="utf-8",
    )
    (meta / "b.gi.json").write_text(
        json.dumps(_minimal_gi("e2", "Renewable energy outlook")),
        encoding="utf-8",
    )
    out = run_uc1_topic_research(tmp_path, topic="energy", limit=10)
    texts = [i.text or "" for i in out.insights]
    assert any("energy" in t.lower() for t in texts)
    assert not any("economics" in t.lower() for t in texts)


def _spoken_gi(episode_id: str, person: str, name: str) -> dict:
    gi = _minimal_gi(episode_id, f"Said in {episode_id}")
    gi["nodes"].append({"id": person, "type": "Person", "properties": {"name": name}})
    gi["edges"].append({"type": "SPOKEN_BY", "from": "quote-1", "to": person})
    gi["nodes"][1]["properties"]["speaker_id"] = person
    return gi


def test_run_uc5_treats_a_joined_person_as_one_speaker(tmp_path: Path, monkeypatch) -> None:
    """Prod 2026-11-08 (tier L): `explore_insights` matched the speaker as a raw id substring and
    showed each file's own spelling, so one joined person was two speakers."""
    import podcast_scraper.identity.display as D
    import podcast_scraper.kg.entity_clusters as EC

    meta = tmp_path / "metadata"
    meta.mkdir()
    for ep, pid, name in (
        ("ep-a", "person:tracy", "Tracy Aloway"),
        ("ep-b", "person:tracey-alloway", "Tracey Alloway"),
    ):
        (meta / f"{ep}.gi.json").write_text(json.dumps(_spoken_gi(ep, pid, name)), encoding="utf-8")
    monkeypatch.setattr(
        EC, "cached_entity_id_map", lambda root, **kw: {"person:tracy": "person:tracey-alloway"}
    )
    monkeypatch.setattr(
        D,
        "person_display",
        lambda root: D.PersonDisplay(
            {"person:tracy": "Tracey Alloway", "person:tracey-alloway": "Tracey Alloway"}
        ),
    )

    out = run_uc5_insight_explorer(tmp_path, speaker="person:tracy", limit=20)
    assert {i.episode_id for i in out.insights} == {"ep-a", "ep-b"}
    quotes = [q for i in out.insights for q in i.supporting_quotes]
    assert {q.speaker_id for q in quotes} == {"person:tracey-alloway"}
    assert {q.speaker_name for q in quotes} == {"Tracey Alloway"}
