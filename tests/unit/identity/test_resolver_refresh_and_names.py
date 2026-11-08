"""The MCP's entity registry: a refresh that cannot be silently blocked, and person names in the
ledger's spelling.

Prod 2026-11-08: after tier L's respell at 01:56Z the corpus token moved (02:01:33Z) and the
newest-run files carried only the new spellings, yet `resolve_entity` kept answering
person:dara-obrey "Dara Obrey" hours later from the one process that serves it. A refresh logs
nothing, and while one is in flight every later one is skipped, so nothing showed whether a
refresh had ever started. And even a fresh registry takes a person's name from the files by a
longest-name rule, never the ledger's group spelling that every app surface shows.

Invented names only.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import pytest

import podcast_scraper.identity.display as D
import podcast_scraper.identity.resolver as res
from tests.unit.identity.test_resolver import FakeEmbedder

pytestmark = pytest.mark.unit

PID = "person:dana-obry"


def _episode(root: Path, name: str) -> None:
    meta = root / "feeds" / "show_a" / "run_ep1_20261001-000000" / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "ep1.kg.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "episode_id": "ep1",
                "nodes": [{"id": PID, "type": "Person", "properties": {"name": name}}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    (meta / "ep1.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": "ep1", "title": "ep1"},
                "knowledge_graph": {"artifact_path": "ep1.kg.json"},
            }
        ),
        encoding="utf-8",
    )


def _bump(root: Path, when: float) -> None:
    stamp = root / "corpus_run_summary.json"
    stamp.write_text("{}", encoding="utf-8")
    os.utime(stamp, (when, when))


def _name(resolver: res.EntityResolver) -> str:
    return str(resolver.registry.records.get(PID, {}).get("display_name"))


@pytest.fixture(autouse=True)
def _clean():
    res.clear_entity_resolver_cache()
    with res._entity_registries_lock:
        res._refreshing.clear()
    yield
    res.clear_entity_resolver_cache()
    with res._entity_registries_lock:
        res._refreshing.clear()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    _episode(root, "Dana Obry")
    _bump(root, 1_000_000.0)
    return root


def test_a_refresh_logs_when_it_starts_and_when_it_finishes(corpus, caplog) -> None:
    res.get_entity_resolver(corpus, embedder=FakeEmbedder())
    _episode(corpus, "Dana O Brien")
    _bump(corpus, 2_000_000.0)
    with caplog.at_level(logging.INFO, logger=res.logger.name):
        res.get_entity_resolver(corpus, embedder=FakeEmbedder())
        assert res._last_refresh is not None
        res._last_refresh.join(timeout=30)
    assert "entity registry refresh started" in caplog.text
    assert "entity registry refreshed in" in caplog.text
    assert _name(res.get_entity_resolver(corpus, embedder=FakeEmbedder())) == "Dana O Brien"


def test_a_stalled_refresh_does_not_block_the_next_one(corpus, monkeypatch, caplog) -> None:
    """A refresh in flight past the bound, still holding the build lock, is not waited for."""
    first = res.get_entity_resolver(corpus, embedder=FakeEmbedder())
    monkeypatch.setattr(res, "REFRESH_STALL_SECONDS", 0.2)
    key = res._registry_key(corpus, res.DEFAULT_FUZZY_THRESHOLD, res.DEFAULT_EMBED_MODEL_ID)
    release, held = threading.Event(), threading.Event()

    def stuck() -> None:
        with res._entity_build_lock:
            held.set()
            release.wait(30)

    holder = threading.Thread(target=stuck, daemon=True)
    holder.start()
    assert held.wait(5)
    with res._entity_registries_lock:
        res._refreshing[key] = 0.0  # a refresh that started long ago and never finished
    try:
        _episode(corpus, "Dana O Brien")
        _bump(corpus, 2_000_000.0)
        with caplog.at_level(logging.WARNING, logger=res.logger.name):
            res.get_entity_resolver(corpus, embedder=FakeEmbedder())
            assert res._last_refresh is not None
            res._last_refresh.join(timeout=30)
        fresh = res.get_entity_resolver(corpus, embedder=FakeEmbedder())
        assert fresh.registry is not first.registry
        assert _name(fresh) == "Dana O Brien"
        assert "has not finished" in caplog.text
    finally:
        release.set()
        holder.join(5)


def test_person_names_take_the_ledgers_spelling(corpus, monkeypatch) -> None:
    monkeypatch.setattr(D, "person_display", lambda root: D.PersonDisplay({PID: "Dana Ó Brien"}))
    resolver = res.EntityResolver(res.build_entity_registry(corpus, embedder=FakeEmbedder()))
    assert _name(resolver) == "Dana Ó Brien"
    assert "Dana Obry" in resolver.registry.records[PID]["aliases"]
    assert resolver.resolve("Dana Ó Brien") == PID
    assert resolver.resolve("Dana Obry") == PID


def test_with_no_ledger_spelling_a_person_keeps_its_own_name(corpus, monkeypatch) -> None:
    monkeypatch.setattr(D, "person_display", lambda root: D.PersonDisplay({}))
    resolver = res.EntityResolver(res.build_entity_registry(corpus, embedder=FakeEmbedder()))
    assert _name(resolver) == "Dana Obry"


MEMBER = "person:dana-obrien"


def _second_episode(root: Path, pid: str, name: str) -> None:
    meta = root / "feeds" / "show_a" / "run_ep2_20261002-000000" / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "ep2.kg.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "episode_id": "ep2",
                "nodes": [{"id": pid, "type": "Person", "properties": {"name": name}}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    (meta / "ep2.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": "ep2", "title": "ep2"},
                "knowledge_graph": {"artifact_path": "ep2.kg.json"},
            }
        ),
        encoding="utf-8",
    )


def test_a_joined_member_resolves_to_its_groups_canonical(corpus, monkeypatch) -> None:
    """The graph tools hold only canonical ids, so `resolve_entity` must hand out the canonical.

    Prod 2026-11-08: "Erica Barris" resolved to person:erica-barris and "Erika Beras" to
    person:erika-beras, one person under two ids, and the member id found nothing in the graph.
    """
    import podcast_scraper.kg.entity_clusters as EC

    _second_episode(corpus, MEMBER, "Dana O'Brien")
    monkeypatch.setattr(EC, "cached_entity_id_map", lambda root, **kw: {MEMBER: PID})
    resolver = res.EntityResolver(res.build_entity_registry(corpus, embedder=FakeEmbedder()))
    assert MEMBER not in resolver.registry.records
    assert resolver.resolve("Dana O'Brien") == PID
    assert resolver.resolve("Dana Obry") == PID
    assert resolver.resolve(MEMBER) == PID
    assert resolver.registry.records[PID]["members"] == [MEMBER]
    assert resolver.registry.records[PID]["freq"] == 2


def test_without_joins_every_id_keeps_its_own_record(corpus, monkeypatch) -> None:
    import podcast_scraper.kg.entity_clusters as EC

    _second_episode(corpus, MEMBER, "Dana O'Brien")
    monkeypatch.setattr(EC, "cached_entity_id_map", lambda root, **kw: {})
    resolver = res.EntityResolver(res.build_entity_registry(corpus, embedder=FakeEmbedder()))
    assert resolver.resolve("Dana O'Brien") == MEMBER
    assert resolver.registry.records[PID]["members"] == []
