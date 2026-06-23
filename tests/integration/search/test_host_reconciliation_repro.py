"""End-to-end reproduction of the network-feed host-naming gap (#1056).

ADR-095 institutional rule: a real-corpus bug lands a higher-tier reproduction before
the fix. This builds a corpus *on disk* in the shape that surfaced the bug on prod-v2 —
a network-authored show (feed author is the org, so the per-episode roster can't name the
host) where one episode happens to name the host "Katie Martin" but a sibling episode
leaves the same recurring voice as a bare ``SPEAKER_03`` — and exercises the **real**
``CorpusGraph`` artifact-load path (not a hand-built graph), asserting both the bug
(without reconciliation the voice is anonymous and disconnected) and the fix (with it the
voice is named across the show). It also walks the relational ``co_speakers`` surface an
agent/viewer hits, so the regression is guarded where humans see it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search import relational_queries as rq
from podcast_scraper.search.corpus_graph import CorpusGraph

pytestmark = pytest.mark.integration


def _kg(episode_id: str, host_id: str, host_name: str, *, guest_id: str, guest_name: str) -> dict:
    """One episode of ``podcast:unhedged`` — a host + a guest, both ABOUT topic:markets."""
    return {
        "schema_version": "1.2",
        "episode_id": episode_id,
        "nodes": [
            {"id": "podcast:unhedged", "type": "Podcast", "properties": {"title": "Unhedged"}},
            {"id": f"episode:{episode_id}", "type": "Episode", "properties": {}},
            {"id": host_id, "type": "Person", "properties": {"name": host_name, "role": "host"}},
            {"id": guest_id, "type": "Person", "properties": {"name": guest_name, "role": "guest"}},
            {"id": "topic:markets", "type": "Topic", "properties": {"label": "Markets"}},
            {
                "id": f"insight:{episode_id}h",
                "type": "Insight",
                "properties": {"text": f"{host_name} on markets", "episode_id": episode_id},
            },
            {
                "id": f"insight:{episode_id}g",
                "type": "Insight",
                "properties": {"text": f"{guest_name} on markets", "episode_id": episode_id},
            },
        ],
        "edges": [
            {"type": "HAS_EPISODE", "from": "podcast:unhedged", "to": f"episode:{episode_id}"},
            {"type": "MENTIONS", "from": host_id, "to": f"episode:{episode_id}"},
            {"type": "MENTIONS", "from": guest_id, "to": f"episode:{episode_id}"},
            {
                "type": "HAS_INSIGHT",
                "from": f"episode:{episode_id}",
                "to": f"insight:{episode_id}h",
            },
            {
                "type": "HAS_INSIGHT",
                "from": f"episode:{episode_id}",
                "to": f"insight:{episode_id}g",
            },
            {"type": "STATES", "from": host_id, "to": f"insight:{episode_id}h"},
            {"type": "STATES", "from": guest_id, "to": f"insight:{episode_id}g"},
            {"type": "ABOUT", "from": f"insight:{episode_id}h", "to": "topic:markets"},
            {"type": "ABOUT", "from": f"insight:{episode_id}g", "to": "topic:markets"},
        ],
    }


def _network_feed_corpus(tmp_path: Path) -> Path:
    # Katie hosts all three episodes but is only *named* in two; the third diarized her
    # voice as a bare SPEAKER_03 (network feed → no name to anchor). Distinct guests.
    episodes = [
        _kg("u1", "person:katie-martin", "Katie Martin", guest_id="person:rob", guest_name="Rob"),
        _kg("u2", "person:katie-martin", "Katie Martin", guest_id="person:ana", guest_name="Ana"),
        _kg("u3", "person:speaker-03", "SPEAKER_03", guest_id="person:sam", guest_name="Sam"),
    ]
    for ep in episodes:
        (tmp_path / f"{ep['episode_id']}.kg.json").write_text(json.dumps(ep))
    return tmp_path


def test_network_feed_host_is_anonymous_without_reconciliation(tmp_path: Path) -> None:
    # The bug: the third episode's host is a disconnected SPEAKER_03; the named host
    # never reaches that episode, so connectivity over it is anonymous.
    graph = CorpusGraph.build(_network_feed_corpus(tmp_path), derive_speaker_links=True)
    assert graph.get_node("person:speaker-03") is not None
    assert "episode:u3" not in graph.neighbors("person:katie-martin")


def test_network_feed_host_named_across_show_with_reconciliation(tmp_path: Path) -> None:
    # The fix: feed-anchored reconciliation folds the SPEAKER_03 voice into Katie, so she
    # is named across all three episodes and the relational surface an agent/viewer hits
    # (co_speakers) reaches the guest from the previously-anonymous episode.
    graph = CorpusGraph.build(
        _network_feed_corpus(tmp_path), derive_speaker_links=True, reconcile_hosts=True
    )
    assert graph.get_node("person:speaker-03") is None  # merged away
    assert "episode:u3" in graph.neighbors("person:katie-martin")
    # The relational layer now connects Katie to Sam (the u3 guest) via shared topic.
    co = {n.id for n in rq.co_speakers(graph, "person:katie-martin", k=20)}
    assert "person:sam" in co
