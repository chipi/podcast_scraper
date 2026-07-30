"""ADR-136: the per-episode context digest builder.

The digest is a pure, deterministic denormalization of gi/kg/metadata — the reprocess-free
"what is this episode about?" surface. These tests pin the contract that matters:

- resolved humans vs unresolved diarization voices are separated (bare ``SPEAKER_NN`` never
  lands in ``people``);
- ``voices`` counts only what reached the CLEAN graph — cameo/commercial noise is stripped upstream
  and so is absent by construction (there is nothing in the digest to exclude it *from*);
- entities dedup across the gi + kg layers.
"""

from __future__ import annotations

import pytest

from podcast_scraper.builders.context_digest_builder import build_context_digest

pytestmark = pytest.mark.unit


def _person(nid: str, name: str) -> dict:
    return {"id": nid, "type": "Person", "properties": {"name": name}}


def _org(nid: str, name: str) -> dict:
    return {"id": nid, "type": "Organization", "properties": {"name": name}}


def _topic(nid: str, label: str) -> dict:
    return {"id": nid, "type": "Topic", "properties": {"label": label}}


_METADATA = {
    "feed": {"title": "Planet Money", "language": "en"},
    "episode": {
        "title": "A pro-worker experiment",
        "published_date": "2026-04-08T07:00:00+00:00",
        "duration_seconds": 1541,
        "episode_id": "ep-1",
    },
    "content": {"detected_hosts": ["Mary Childs"], "detected_guests": ["Pete Stavros"]},
    "summary": {"long_summary": "A worker-ownership experiment in private equity."},
}


def test_people_exclude_bare_speaker_voices() -> None:
    gi = {
        "schema_version": "3.0",
        "nodes": [
            _person("person:mary-childs", "Mary Childs"),
            _person("person:pete-stavros", "Pete Stavros"),
            # bare diarization voices — must NOT be counted as people
            _person("person:speaker-abc-03", "SPEAKER_03"),
            _person("person:speaker-abc-04", "SPEAKER_04"),
        ],
    }
    d = build_context_digest("ep-1", gi_artifact=gi, kg_artifact=None, metadata=_METADATA)
    assert d["people"] == ["Mary Childs", "Pete Stavros"]
    assert d["voices"]["total"] == 2
    assert d["voices"]["labels"] == ["SPEAKER_03", "SPEAKER_04"]
    # No per-voice classification supplied → the split is unavailable, not fabricated.
    assert d["voices"]["unknown"] is None and d["voices"]["unidentified"] is None


def test_basic_and_summary_come_from_metadata() -> None:
    d = build_context_digest(
        "ep-1", gi_artifact={"nodes": []}, kg_artifact=None, metadata=_METADATA
    )
    assert d["basic"] == {
        "title": "A pro-worker experiment",
        "show": "Planet Money",
        "published_date": "2026-04-08T07:00:00+00:00",
        "duration_seconds": 1541,
        "language": "en",
        "hosts": ["Mary Childs"],
        "guests": ["Pete Stavros"],
    }
    assert d["summary"] == "A worker-ownership experiment in private equity."


def test_entities_dedup_across_gi_and_kg() -> None:
    gi = {
        "schema_version": "3.0",
        "nodes": [_org("org:kkr", "KKR"), _topic("topic:pe", "private equity")],
    }
    kg = {
        "schema_version": "2.0",
        "nodes": [
            _org("org:kkr", "kkr"),
            _org("org:3m", "3M"),
            _topic("topic:pe", "Private Equity"),
        ],
    }
    d = build_context_digest("ep-1", gi_artifact=gi, kg_artifact=kg, metadata=_METADATA)
    # case-insensitive dedup, first-seen display kept, sorted
    assert d["companies"] == ["3M", "KKR"]
    assert d["topics"] == ["private equity"]
    assert d["source"] == {"gi_schema_version": "3.0", "kg_schema_version": "2.0"}


def test_voices_split_filled_when_classification_supplied() -> None:
    gi = {
        "nodes": [
            _person("person:speaker-x-01", "SPEAKER_01"),
            _person("person:speaker-x-02", "SPEAKER_02"),
        ]
    }
    classification = {"SPEAKER_01": "unknown", "SPEAKER_02": "unidentified"}
    d = build_context_digest(
        "ep-1",
        gi_artifact=gi,
        kg_artifact=None,
        metadata=_METADATA,
        voice_classification=classification,
    )
    assert d["voices"] == {
        "total": 2,
        "labels": ["SPEAKER_01", "SPEAKER_02"],
        "unknown": ["SPEAKER_01"],
        "unidentified": ["SPEAKER_02"],
    }


def test_deterministic_same_inputs_same_output() -> None:
    gi = {"nodes": [_person("person:a", "Zoe"), _person("person:b", "Ada")]}
    a = build_context_digest("ep-1", gi_artifact=gi, kg_artifact=None, metadata=_METADATA)
    b = build_context_digest("ep-1", gi_artifact=gi, kg_artifact=None, metadata=_METADATA)
    assert a == b
    assert a["people"] == ["Ada", "Zoe"]  # sorted, not insertion order
