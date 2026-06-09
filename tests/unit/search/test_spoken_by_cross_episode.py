"""End-to-end fixture proof: corpus-wide SPOKEN_BY + cross-episode person identity.

Covers the #876 -> #909 chain on fixtures only (no real audio, no models, no real
corpus): a multi-episode corpus whose episodes carry *named* diarized screenplay
transcripts -- exactly what re-diarized ``whisper_transcription`` episodes produce
(#875) -- flows through

    enrich-edges  ->  SPOKEN_BY emission (#876)  ->  CorpusGraph person->insight (#909)

and a recurring guest, attributed in every episode, resolves to ONE canonical
``person:{slug}`` that reaches the insights of ALL their episodes.

This is the Tier-2 matrix row that de-risks the real-corpus re-diarize run: it
proves the code path before any 90-episode reprocess (#876 runbook), per the
"matrix row before real corpus" rule. It deliberately starts from a named
diarized transcript string (the diarizer's output) rather than running pyannote,
so it's fast and deterministic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from podcast_scraper.gi.speakers import person_id
from podcast_scraper.search.cli_handlers import parse_enrich_edges_argv, run_enrich_edges_cli
from podcast_scraper.search.corpus_graph import CorpusGraph

pytestmark = pytest.mark.unit

_LOG = logging.getLogger("test-spoken-by-cross-episode")

GUEST = "Priya Sharma"


def _write_episode(
    corpus: Path,
    stem: str,
    episode_id: str,
    *,
    host: str,
    quote_text: str,
    insight_text: str,
) -> None:
    """Write one episode's metadata + named-diarized transcript + gi/kg artifacts.

    Mirrors the on-disk layout enrich-edges walks (see test_enrich_edges_cli):
    a ``metadata/<stem>.metadata.json`` pointing at a root-relative transcript and
    gi artifact, plus sibling ``<stem>.gi.json`` / ``<stem>.kg.json``.
    """
    transcript = f"{host}: Welcome to the show.\n{GUEST}: {quote_text}\n"
    char_start = transcript.index(quote_text)  # falls inside the guest's named turn

    (corpus / "metadata").mkdir(exist_ok=True)
    (corpus / "metadata" / f"{stem}.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"title": "Reliability Pod"},
                "episode": {"episode_id": episode_id},
                "content": {
                    "transcript_file_path": f"{stem}.txt",
                    "detected_hosts": [host],
                    "detected_guests": [GUEST],
                },
                "grounded_insights": {"artifact_path": f"{stem}.gi.json"},
            }
        ),
        encoding="utf-8",
    )
    (corpus / f"{stem}.txt").write_text(transcript, encoding="utf-8")
    (corpus / f"{stem}.gi.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "model_version": "t",
                "prompt_version": "t",
                "episode_id": episode_id,
                "nodes": [
                    {"id": f"episode:{episode_id}", "type": "Episode", "properties": {}},
                    {
                        "id": f"insight:{episode_id}",
                        "type": "Insight",
                        "properties": {"text": insight_text},
                    },
                    {
                        "id": f"quote:{episode_id}",
                        "type": "Quote",
                        "properties": {"char_start": char_start, "text": quote_text},
                    },
                ],
                # Insight is SUPPORTED_BY the quote; SPOKEN_BY (quote->person) is what
                # enrich-edges adds, completing person -> quote -> insight.
                "edges": [
                    {
                        "type": "SUPPORTED_BY",
                        "from": f"insight:{episode_id}",
                        "to": f"quote:{episode_id}",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (corpus / f"{stem}.kg.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": person_id(GUEST),
                        "type": "Entity",
                        "properties": {"name": GUEST, "kind": "person"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_recurring_guest_aggregates_across_rediarized_whisper_episodes(tmp_path):
    corpus = tmp_path
    _write_episode(
        corpus,
        "ep1",
        "ep1",
        host="Maya",
        quote_text="Reliability is the real challenge in production.",
        insight_text="Reliability is the core production challenge.",
    )
    _write_episode(
        corpus,
        "ep2",
        "ep2",
        host="Ethan",
        quote_text="We measure reliability with error budgets.",
        insight_text="Error budgets quantify reliability.",
    )

    guest_id = person_id(GUEST)

    # 1) #876 -- enrich-edges emits SPOKEN_BY corpus-wide (every episode), not just one.
    rc = run_enrich_edges_cli(parse_enrich_edges_argv(["--output-dir", str(corpus)]), _LOG)
    assert rc == 0
    for episode_id in ("ep1", "ep2"):
        gi = json.loads((corpus / f"{episode_id}.gi.json").read_text())
        spoken = [e for e in gi["edges"] if e.get("type") == "SPOKEN_BY"]
        assert any(
            e.get("from") == f"quote:{episode_id}" and e.get("to") == guest_id for e in spoken
        ), f"{episode_id}: expected SPOKEN_BY quote:{episode_id} -> {guest_id}, got {spoken}"

    # 2) #909 -- the recurring guest resolves to ONE canonical person id whose derived
    #    person->insight links span BOTH episodes (cross-episode aggregation).
    graph = CorpusGraph.build(corpus, derive_speaker_links=True)
    assert graph.get_node(guest_id) is not None, f"{guest_id} not in corpus graph"
    insight_neighbors = {
        n
        for n in graph.neighbors(guest_id)
        if (node := graph.get_node(n)) is not None and node.type == "insight"
    }
    assert {"insight:ep1", "insight:ep2"} <= insight_neighbors, (
        "recurring guest should reach insights from BOTH episodes; "
        f"got {sorted(insight_neighbors)}"
    )


def test_no_detected_people_means_no_attribution(tmp_path):
    """Negative control: with no detected host/guest, the named markers match
    nobody, so the chain does not fabricate SPOKEN_BY / cross-episode person links
    (it doesn't guess identities the diarizer never named)."""
    corpus = tmp_path
    _write_episode(
        corpus,
        "ep1",
        "ep1",
        host="Maya",
        quote_text="Reliability is the real challenge in production.",
        insight_text="Reliability is the core production challenge.",
    )
    # Drop BOTH detected host and guest: no known names -> no named turns -> the
    # named markers ("Maya:", "Priya Sharma:") match no detected person, and the
    # generic "Speaker N" fallback finds no "Speaker N" markers either.
    meta_path = corpus / "metadata" / "ep1.metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["content"]["detected_hosts"] = []
    meta["content"]["detected_guests"] = []
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    rc = run_enrich_edges_cli(parse_enrich_edges_argv(["--output-dir", str(corpus)]), _LOG)
    assert rc == 0
    gi = json.loads((corpus / "ep1.gi.json").read_text())
    assert not [
        e for e in gi["edges"] if e.get("type") == "SPOKEN_BY"
    ], "no detected guest should mean no SPOKEN_BY attribution"
