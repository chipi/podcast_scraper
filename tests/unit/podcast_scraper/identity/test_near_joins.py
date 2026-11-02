"""Unreviewed near-spelling joins are listed for a read (RFC-131 D2). Synthetic names only."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.identity import ledger as L
from podcast_scraper.identity.guards import collect_episode_evidence
from podcast_scraper.identity.near_joins import pairs, unreviewed_near_joins
from podcast_scraper.kg.entity_clusters import collect_entity_candidates


def _kg_episode(root: Path, feed: str, run: str, ep: str, persons) -> None:
    """One English episode's metadata and KG with ``persons`` ``{id: name}``."""
    meta = root / "feeds" / feed / run / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / f"{ep}.metadata.json").write_text(
        json.dumps({"episode": {"episode_id": ep}, "feed": {"title": feed, "language": "en"}}),
        encoding="utf-8",
    )
    nodes = [{"id": f"episode:{ep}", "type": "Episode", "properties": {"podcast_id": feed}}]
    nodes += [{"id": i, "type": "Person", "properties": {"name": n}} for i, n in persons.items()]
    (meta / f"{ep}.kg.json").write_text(
        json.dumps({"episode_id": ep, "nodes": nodes, "edges": []}), encoding="utf-8"
    )


def _corpus(root: Path, second: str = "Korin Vaal") -> Path:
    for n, (ep, pid, name) in enumerate(
        [
            ("e1", "person:korin-vael", "Korin Vael"),
            ("e2", "person:korin-vael", "Korin Vael"),
            ("e3", f"person:{second.lower().replace(' ', '-')}", second),
            ("e4", f"person:{second.lower().replace(' ', '-')}", second),
        ]
    ):
        _kg_episode(root, "show_a", f"run_{ep}_2026100{n + 1}-000000", ep, {pid: name})
    return root


def _rows(root: Path, reviewed):
    ledger = L.build_ledger(root)
    return unreviewed_near_joins(
        ledger, collect_entity_candidates(root), collect_episode_evidence(root), reviewed
    )


def test_a_near_spelling_join_is_listed(tmp_path: Path) -> None:
    rows = _rows(_corpus(tmp_path / "c"), {"groups": [], "apart": []})
    assert pairs(rows) == [("person:korin-vaal", "person:korin-vael")]


def test_a_reviewed_group_is_not_listed(tmp_path: Path) -> None:
    reviewed = {"groups": [["person:korin-vaal", "person:korin-vael"]], "apart": []}
    assert _rows(_corpus(tmp_path / "c"), reviewed) == []


def test_a_pair_reviewed_apart_is_not_listed(tmp_path: Path) -> None:
    reviewed = {"groups": [], "apart": [["person:korin-vaal", "person:korin-vael", "r"]]}
    assert _rows(_corpus(tmp_path / "c"), reviewed) == []


def test_an_equal_spelling_is_not_listed(tmp_path: Path) -> None:
    rows = _rows(_corpus(tmp_path / "c", second="korin  vael"), {"groups": [], "apart": []})
    assert rows == []
