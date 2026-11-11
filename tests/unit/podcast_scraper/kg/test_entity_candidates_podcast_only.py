"""Another content type's people do not become identity-ledger candidates by default.

collect_entity_candidates feeds the identity ledger, and it read every KG artifact: a KG 3.0 item
of another type (news, the test-only type) put its Person and Organization nodes among the
candidates, where they could join podcast people. The KG says its own type (KG 3.0 carries
content_type; a 2.x artifact is a podcast's), so the default is podcast items only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.kg.entity_clusters import collect_entity_candidates

pytestmark = pytest.mark.unit


def _write(root: Path, feed: str, item: str, kg: dict) -> None:
    run = root / "feeds" / feed / "run_20261101-000000" / "metadata"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{item}.metadata.json").write_text(
        json.dumps({"feed": {"feed_id": feed}, "episode": {"episode_id": item}}), encoding="utf-8"
    )
    (run / f"{item}.kg.json").write_text(json.dumps(kg), encoding="utf-8")


def _person(pid: str, name: str) -> dict:
    return {"id": pid, "type": "Person", "properties": {"name": name}}


@pytest.fixture
def mixed(tmp_path: Path) -> Path:
    _write(
        tmp_path,
        "podcast-a",
        "e1",
        {"schema_version": "2.1", "episode_id": "e1", "nodes": [_person("person:ann", "Ann")]},
    )
    _write(
        tmp_path,
        "texts",
        "t1",
        {
            "schema_version": "3.0",
            "item_id": "item:test_text_item:one",
            "content_type": "test_text_item",
            "nodes": [_person("person:bo", "Bo"), _person("person:ann", "Ann")],
        },
    )
    return tmp_path


def test_only_podcast_people_are_candidates_by_default(mixed: Path) -> None:
    candidates = collect_entity_candidates(mixed)
    assert set(candidates) == {"person:ann"}
    assert candidates["person:ann"].episodes == {"e1"}


def test_another_type_contributes_when_asked(mixed: Path) -> None:
    candidates = collect_entity_candidates(
        mixed, content_types={"podcast_episode", "test_text_item"}
    )
    assert set(candidates) == {"person:ann", "person:bo"}
