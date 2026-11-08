"""Search shows only what the episode view shows, under one spelling per person (RFC-131 S6c,
D2 item 7). Invented names only."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.identity.display import PersonDisplay
from podcast_scraper.insight_stage.transcript_chunk_lift import try_lift_transcript_chunk_from_gi
from podcast_scraper.search import corpus_search as CS
from podcast_scraper.search.cli_handlers import _enrich_hit
from podcast_scraper.search.protocol import SearchResult

pytestmark = pytest.mark.unit


def _gi(person: str, name: str, *, surfaceable: bool = True) -> Dict[str, Any]:
    insight_props: Dict[str, Any] = {"text": "A point.", "tier": 3, "grounded": True}
    if not surfaceable:
        insight_props.update(surfaceable=False, speaker_voice_type="unknown")
    return {
        "nodes": [
            {"id": person, "type": "Person", "properties": {"name": name}},
            {"id": "insight:a", "type": "Insight", "properties": insight_props},
            {
                "id": "quote:a",
                "type": "Quote",
                "properties": {
                    "text": "Said so.",
                    "speaker_id": person,
                    "char_start": 0,
                    "char_end": 40,
                },
            },
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": "insight:a", "to": "quote:a"}],
    }


def test_the_scope_marks_what_the_episode_view_hides() -> None:
    (shown,) = CS._insight_scopes_from_gi("e1", _gi("person:korin-vaal", "Korin Vaal"))
    (hidden,) = CS._insight_scopes_from_gi(
        "e2", _gi("person:korin-vaal", "Korin Vaal", surfaceable=False)
    )
    assert shown.visible is True and hidden.visible is False


def test_the_scope_carries_the_speakers_canonical_id() -> None:
    (scope,) = CS._insight_scopes_from_gi(
        "e1",
        _gi("person:korin-vaal", "Korin Vaal"),
        id_map={"person:korin-vaal": "person:korin-vael"},
    )
    assert set(scope.speakers) == {"person:korin-vaal", "person:korin-vael"}


def test_a_hidden_insight_hit_is_dropped(tmp_path: Path, monkeypatch) -> None:
    gis = {
        "e1": _gi("person:a", "Korin Vaal"),
        "e2": _gi("person:b", "Odile Brand", surfaceable=False),
    }
    paths = {}
    for ep, gi in gis.items():
        p = tmp_path / f"{ep}.gi.json"
        p.write_text(__import__("json").dumps(gi), encoding="utf-8")
        paths[ep] = p
    monkeypatch.setattr(CS, "cached_episode_gi_paths", lambda _root: paths)
    monkeypatch.setattr(CS, "cached_metadata_relpath_by_scope", lambda _root: {})
    CS._SCOPES_BY_PATH_MAP.clear()
    hits = [
        SearchResult(
            f"insight:{ep}",
            1.0,
            {"doc_type": "insight", "episode_id": ep, "source_id": "insight:a"},
        )
        for ep in ("e1", "e2")
    ]
    out = CS._filter_and_enrich(
        hits,
        tmp_path,
        types_norm=None,
        feed=None,
        since=None,
        speaker=None,
        topic=None,
        episode_id=None,
        grounded_only=False,
        top_k=10,
        dedupe_kg_surfaces=False,
        collect_cap=100,
    )
    assert [r["metadata"]["episode_id"] for r in out.results] == ["e1"]


def test_supporting_quotes_carry_the_group_spelling(tmp_path: Path) -> None:
    gi = _gi("person:korin-vaal", "Korin Vaal")
    p = tmp_path / "e1.gi.json"
    p.write_text(__import__("json").dumps(gi), encoding="utf-8")
    hit = SearchResult(
        "i", 1.0, {"doc_type": "insight", "episode_id": "e1", "source_id": "insight:a"}
    )
    people = PersonDisplay({"person:korin-vaal": "Korin Vael"})
    row = _enrich_hit(hit, {"e1": p}, people=people)
    (q,) = row["supporting_quotes"]
    assert q["speaker_id"] == "person:korin-vaal" and q["speaker_name"] == "Korin Vael"
    row = _enrich_hit(hit, {"e1": p})
    assert row["supporting_quotes"][0]["speaker_name"] == "Korin Vaal"


def test_a_transcript_chunk_does_not_lift_to_a_hidden_insight(tmp_path: Path) -> None:
    gi = _gi("person:korin-vaal", "Korin Vaal", surfaceable=False)
    gi_path = tmp_path / "e.gi.json"
    lifted = try_lift_transcript_chunk_from_gi(gi, tmp_path, gi_path, char_start=0, char_end=40)
    assert lifted is not None
    lifted = try_lift_transcript_chunk_from_gi(
        gi, tmp_path, gi_path, char_start=0, char_end=40, visible=lambda iid: False
    )
    assert lifted is None
