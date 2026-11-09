"""The KG index holds one object per distinct entity (memory, 2026-11-08), otherwise unchanged.

On the prod-shaped corpus the index held 90.3 MB, most of it entity models repeated across
episodes; sharing equal ones brought it to 72.6 MB. These tests pin that the shared index equals
the one built without sharing, and that equal entities are one object.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.server import kg_index

FIXTURE = Path("tests/fixtures/app-validation-corpus/v3")


@pytest.fixture()
def root() -> Path:
    if not FIXTURE.is_dir():
        pytest.fail(f"fixture corpus missing: {FIXTURE}")
    perf_cache.clear()
    return Path(os.path.abspath(FIXTURE))


def test_the_shared_index_equals_the_unshared_one(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = kg_index.build_kg_index(root)
    monkeypatch.setattr(kg_index, "_pooled", lambda _pool, items: items)
    perf_cache.clear()
    unshared = kg_index.build_kg_index(root)
    assert shared.episodes, "the fixture has no KG episodes"
    assert shared == unshared


def test_equal_entities_are_one_object(root: Path) -> None:
    index = kg_index.build_kg_index(root)
    first: dict[tuple[type, str], object] = {}
    recurring = 0
    for ep in index.episodes:
        for m in [*ep.persons, *ep.topics, *ep.orgs]:
            key = (type(m), repr(m))
            if key in first:
                recurring += 1
                assert first[key] is m, f"two objects for one entity: {m!r}"
            first[key] = m
    assert recurring, "no entity recurs, so sharing is not exercised"


def test_same_id_with_other_fields_stays_two_objects() -> None:
    """A person who hosts one episode and guests on another is two different entities."""
    from podcast_scraper.server.schemas import AppEntity

    pool: dict[tuple[type, str], object] = {}
    host = AppEntity(id="person:a", name="A", kind="person", role="host")
    guest = AppEntity(id="person:a", name="A", kind="person", role="guest")
    again = AppEntity(id="person:a", name="A", kind="person", role="host")
    got = kg_index._pooled(pool, [host]) + kg_index._pooled(pool, [guest, again])
    assert [m.role for m in got] == ["host", "guest", "host"]
    assert got[0] is got[2] and got[1] is not got[0]
