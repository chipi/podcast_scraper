"""The search page is enriched in rank order until it is full, not over every candidate.

Episode-related enriched ~2,350 hits per call to return at most 100 (65% of its time on a
prod-sized index, 2026-11-09). Without the KG surface dedupe every step works row by row and only
drops rows, so stopping once the page is full gives the same page; with the dedupe, a winner's
counts include every later duplicate, so the whole list is still enriched. These pin both, on rows
built to exercise what the prod corpus copy does not: dropped rows and duplicates past the page.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.search import corpus_search

DROPPED = {1, 2, 3, 5, 8, 13, 21, 34}


@pytest.fixture()
def patched(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    enriched: list[int] = []

    def enrich(h: int, *_a: Any, **_k: Any) -> dict:
        enriched.append(h)
        return {"id": h, "text": f"surface {h % 7}", "metadata": {"doc_type": "kg_topic"}}

    def drop_storylines(rows: list[dict], _root: Path) -> list[dict]:
        return [r for r in rows if r["id"] not in DROPPED]

    def dedupe(rows: list[dict]) -> list[dict]:
        out: dict[str, dict] = {}
        for r in rows:
            if r["text"] in out:
                out[r["text"]]["metadata"]["dups"] = out[r["text"]]["metadata"].get("dups", 0) + 1
            else:
                out[r["text"]] = r
        return list(out.values())

    monkeypatch.setattr(corpus_search, "_enrich_hit", enrich)
    monkeypatch.setattr(corpus_search.read_hooks, "lift_transcript_hits", lambda *a, **k: None)
    monkeypatch.setattr(corpus_search, "_attach_topic_cluster_metadata", lambda rows, root: None)
    monkeypatch.setattr(corpus_search, "_attach_storyline_metadata", drop_storylines)
    monkeypatch.setattr(corpus_search, "dedupe_kg_surface_rows", dedupe)
    monkeypatch.setattr("podcast_scraper.identity.display.person_display", lambda root: None)
    return enriched


def _page(top_k: int, dedupe: bool, n: int = 200) -> list[dict]:
    hits: list[Any] = list(range(n))
    page, _stats = corpus_search._enrich_lift_and_slice(
        hits, Path("/c"), {}, {}, top_k=top_k, dedupe_kg_surfaces=dedupe
    )
    return page


def test_dropped_rows_are_refilled_in_rank_order(patched: list[int]) -> None:
    page = _page(10, dedupe=False)
    survivors = [i for i in range(200) if i not in DROPPED]
    assert [r["id"] for r in page] == survivors[:10]
    assert len(patched) < 200, "every candidate was enriched"


def test_a_short_list_returns_what_survives(patched: list[int]) -> None:
    page = _page(10, dedupe=False, n=12)
    assert [r["id"] for r in page] == [i for i in range(12) if i not in DROPPED]


def test_the_dedupe_still_sees_every_candidate(patched: list[int]) -> None:
    page = _page(3, dedupe=True)
    assert len(patched) == 200
    # 200 candidates over 7 surfaces, minus the dropped ones: the winners' counts cover the list.
    assert sum(r["metadata"].get("dups", 0) for r in page) > 3
