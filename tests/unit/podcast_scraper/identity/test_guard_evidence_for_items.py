"""The identity guards see another type's items (core release R3a; advisor B4).

The guards compare the episodes two people share, keyed by the KG's ``episode_id``. A KG 3.0 item
has no ``episode_id``, so its people shared nothing and the guards had no evidence for them: the
evidence builder skipped the artifact. Both sides now key a KG 3.0 item by its ``item_id``, and
the evidence builder reads the same types as the entity candidates (podcast episodes unless asked).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.identity.guards import (
    collect_episode_evidence,
    guard_fires,
    GUARD_MENTION_COOCCURRENCE,
)
from podcast_scraper.kg.entity_clusters import collect_entity_candidates

pytestmark = pytest.mark.unit

ITEM = "item:test_text_item:one"
BOTH = {"podcast_episode", "test_text_item"}


def _person(pid: str, name: str) -> dict:
    return {"id": pid, "type": "Person", "properties": {"name": name}}


def _write(meta_dir: Path, stem: str, metadata: dict, kg: dict) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{stem}.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (meta_dir / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    _write(
        tmp_path / "feeds" / "pod" / "run_20261101-000000" / "metadata",
        "e1",
        {"feed": {"feed_id": "pod"}, "episode": {"episode_id": "e1"}},
        {"schema_version": "2.1", "episode_id": "e1", "nodes": [_person("person:ann", "Ann")]},
    )
    _write(
        tmp_path / "sources" / "test_text_item" / "texts" / "run_20261101-000000" / "metadata",
        "one",
        {"content_type": "test_text_item", "episode": {"guid": ITEM}},
        {
            "schema_version": "3.0",
            "item_id": ITEM,
            "content_type": "test_text_item",
            "nodes": [_person("person:bo", "Bo Smith"), _person("person:cy", "Cy Jones")],
        },
    )
    return tmp_path


def test_podcast_only_by_default(corpus: Path) -> None:
    assert set(collect_episode_evidence(corpus)) == {"e1"}
    assert set(collect_entity_candidates(corpus)) == {"person:ann"}


def test_an_items_people_share_it_and_the_guard_sees_them(corpus: Path) -> None:
    evidence = collect_episode_evidence(corpus, content_types=BOTH)
    assert set(evidence) == {"e1", ITEM}
    assert evidence[ITEM].present >= {"person:bo", "person:cy"}

    candidates = collect_entity_candidates(corpus, content_types=BOTH)
    assert candidates["person:bo"].episodes == {ITEM}
    shared = candidates["person:bo"].episodes & candidates["person:cy"].episodes
    assert guard_fires("person:bo", "person:cy", shared, evidence) == (
        GUARD_MENTION_COOCCURRENCE,
        ITEM,
    )
