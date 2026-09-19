"""Unit tests for the inverted KG entity index (relational-card perf remediation).

Guards the perf contract that makes person/topic/entity-search O(matches): the index is built once
per ingest (not per request), invalidates when the corpus changes, and its inverted maps + label
refs are correct. Card-output parity itself is covered by ``test_app_relational_view`` (which now
runs through this index on the default path).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.server import app_kg_index

pytestmark = [pytest.mark.unit]


def _write_episode(
    root: Path,
    *,
    stem: str,
    episode_id: str,
    persons: list[tuple[str, str]],
    topics: list[tuple[str, str]],
) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "f1", "title": "Show"},
        "episode": {
            "episode_id": episode_id,
            "title": episode_id,
            "published_date": "2024-01-01T00:00:00",
        },
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    nodes = [{"id": pid, "type": "Person", "properties": {"name": n}} for pid, n in persons]
    nodes += [{"id": tid, "type": "Topic", "properties": {"label": la}} for tid, la in topics]
    (root / "metadata" / f"{stem}.kg.json").write_text(
        json.dumps({"episode_id": episode_id, "nodes": nodes}), encoding="utf-8"
    )


def _stamp(root: Path, mtime: float) -> None:
    stamp = root / "corpus_run_summary.json"
    stamp.write_text("{}", encoding="utf-8")
    import os

    os.utime(stamp, (mtime, mtime))


@pytest.fixture(autouse=True)
def _fresh_cache():
    perf_cache.clear()
    yield
    perf_cache.clear()


def _corpus(root: Path) -> None:
    _write_episode(
        root,
        stem="0001",
        episode_id="e1",
        persons=[("person:jane", "Jane Doe"), ("person:bob", "Bob")],
        topics=[("topic:ai", "AI")],
    )
    _write_episode(
        root,
        stem="0002",
        episode_id="e2",
        persons=[("person:jane", "Jane Doe")],
        topics=[("topic:ai", "AI"), ("topic:ml", "Machine Learning")],
    )


def test_inverted_maps_and_label_refs(tmp_path: Path) -> None:
    _corpus(tmp_path)
    _stamp(tmp_path, 1_000_000.0)
    idx = app_kg_index.get_kg_index(tmp_path)

    # Jane is in both episodes; Bob in one; ai in both; ml in one.
    assert len(idx.person_episodes("person:jane")) == 2
    assert len(idx.person_episodes("person:bob")) == 1
    assert len(idx.topic_episodes("topic:ai")) == 2
    assert len(idx.topic_episodes("topic:ml")) == 1
    assert idx.person_episodes("person:nobody") == []

    # Normalized-label refs resolve case/punctuation-insensitively.
    assert idx.person_ref_by_norm["jane doe"].id == "person:jane"
    assert idx.topic_ref_by_norm["machine learning"].id == "topic:ml"


def test_index_built_once_then_cached(tmp_path: Path, monkeypatch) -> None:
    _corpus(tmp_path)
    _stamp(tmp_path, 1_000_000.0)
    calls = [0]
    real = app_kg_index.build_kg_index

    def _counting(root: Path):
        calls[0] += 1
        return real(root)

    monkeypatch.setattr(app_kg_index, "build_kg_index", _counting)
    for _ in range(6):
        app_kg_index.get_kg_index(tmp_path)
    assert calls[0] == 1, "the KG index was rebuilt on a cache hit (per-request KG parse is back)"


def test_index_invalidates_on_ingest(tmp_path: Path) -> None:
    _corpus(tmp_path)
    _stamp(tmp_path, 1_000_000.0)
    assert len(app_kg_index.get_kg_index(tmp_path).topic_episodes("topic:ml")) == 1

    # A new episode about ml lands and the ingest stamp advances.
    _write_episode(
        tmp_path,
        stem="0003",
        episode_id="e3",
        persons=[("person:carol", "Carol")],
        topics=[("topic:ml", "Machine Learning")],
    )
    _stamp(tmp_path, 2_000_000.0)
    idx = app_kg_index.get_kg_index(tmp_path)
    assert len(idx.topic_episodes("topic:ml")) == 2, "a stale KG index hid the new episode"
    assert idx.person_ref_by_norm["carol"].id == "person:carol"


class TestSpellingVariantsCollapseInTheAppIndex:
    """``Theo Jaffee`` and ``Theo Jaffe`` are one person on the cards (#2056).

    THE REPORTED DEFECT, read off production. Both spellings exist as separate ids on
    *The a16z Show*::

        person:theo-jaffee   episodes 34e47cf8, 530a274b, e4ebaa91
        person:theo-jaffe    episodes 1f41bb8a, 916c5e0a, 0efb210b, 67fac3b8, 8fe5cec4, 59027622

    Same show, different episodes. So this is NOT blocked on matcher precision: the resolver
    already matches the pair and ``same_show_required=True`` already permits it, because the two
    ids share a show. Nothing was wrong with the matcher — it was never consulted.

    ``build_kg_index`` reads entities straight out of each ``*.kg.json`` via ``iter_kg_entities``
    and canonicalises NOTHING. Only ``search/corpus_graph.py`` and ``server/cil_queries.py`` apply
    ``build_entity_id_map``; the consumer relational cards ("Related people") bypass both. That is
    candidate 2 in #2056 — "the surface may not go through CorpusGraph" — confirmed.

    The index is the right place to fix it: it is already a once-per-ingest cached full corpus
    pass, so folding the canonical map in costs nothing per request.
    """

    @staticmethod
    def _episode_on_show(root: Path, *, stem: str, episode_id: str, persons, show: str) -> None:
        """Like ``_write_episode`` but with an Episode NODE carrying ``podcast_id``.

        ``collect_entity_candidates`` reads an entity's show from that node, and
        ``same_show_required=True`` compares only entities that share one. Every one of the 287
        production artifacts checked has it; a fixture without it silently disables the merge and
        would make this test pass or fail for the wrong reason.
        """
        _write_episode(
            root, stem=stem, episode_id=episode_id, persons=persons, topics=[("topic:ai", "AI")]
        )
        kg_path = root / "metadata" / f"{stem}.kg.json"
        doc = json.loads(kg_path.read_text(encoding="utf-8"))
        doc["nodes"].append(
            {
                "id": f"episode:{episode_id}",
                "type": "Episode",
                "properties": {"podcast_id": show, "title": episode_id},
            }
        )
        kg_path.write_text(json.dumps(doc), encoding="utf-8")

    def _two_spellings(self, root: Path) -> None:
        self._episode_on_show(
            root,
            stem="0001",
            episode_id="e1",
            persons=[("person:theo-jaffee", "Theo Jaffee")],
            show="show:a16z",
        )
        self._episode_on_show(
            root,
            stem="0002",
            episode_id="e2",
            persons=[("person:theo-jaffe", "Theo Jaffe")],
            show="show:a16z",
        )
        _stamp(root, 1000.0)

    def test_both_spellings_resolve_to_one_person(self, tmp_path: Path) -> None:
        self._two_spellings(tmp_path)
        index = app_kg_index.build_kg_index(tmp_path)
        jaffe_ids = {pid for pid in index.person_to_eps if "jaffe" in pid}
        assert len(jaffe_ids) == 1, f"one human must have one id on the cards, got {jaffe_ids}"

    def test_the_surviving_person_carries_both_episodes(self, tmp_path: Path) -> None:
        self._two_spellings(tmp_path)
        index = app_kg_index.build_kg_index(tmp_path)
        pid = next(p for p in index.person_to_eps if "jaffe" in p)
        assert len(index.person_to_eps[pid]) == 2, "the merged person appears in both episodes"

    def test_searching_either_spelling_finds_the_same_person(self, tmp_path: Path) -> None:
        self._two_spellings(tmp_path)
        index = app_kg_index.build_kg_index(tmp_path)
        refs = {
            app_kg_index.normalize_label("Theo Jaffee"),
            app_kg_index.normalize_label("Theo Jaffe"),
        }
        found = {index.person_ref_by_norm[r].id for r in refs if r in index.person_ref_by_norm}
        assert len(found) == 1, f"both spellings must lead to one id, got {found}"

    def test_two_different_people_are_still_two(self, tmp_path: Path) -> None:
        # The canonical map must not become a blunt instrument on this surface either.
        self._episode_on_show(
            tmp_path,
            stem="0001",
            episode_id="e1",
            persons=[("person:albert-einstein", "Albert Einstein")],
            show="show:a16z",
        )
        self._episode_on_show(
            tmp_path,
            stem="0002",
            episode_id="e2",
            persons=[("person:robert-jensen", "Robert Jensen")],
            show="show:a16z",
        )
        _stamp(tmp_path, 1000.0)
        index = app_kg_index.build_kg_index(tmp_path)
        assert "person:albert-einstein" in index.person_to_eps
        assert "person:robert-jensen" in index.person_to_eps
