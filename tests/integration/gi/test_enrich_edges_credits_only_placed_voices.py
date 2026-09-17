"""enrich-edges — which rewrites quote attribution at every finalize — credits placed voices only (#2075).

Two production failures, driven through the real `enrich-edges` CLI over files on disk:

* **The guess on an anonymous transcript.** A transcript with only `Speaker N` markers used to go
  through a role heuristic that gave the first voice `hosts[0]`. On the #2075 validation run that
  credited `Tracy Alloway` on an Odd Lots episode no voice was ever matched to; in production it
  credited 81 quotes across 10 Odd Lots episodes.
* **A person the record says was not placed.** The speaker record keeps people a source named but
  no voice was matched to. If enrich-edges treated them as speakers, a stale `Name:` marker would
  hand them quotes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.search.cli_handlers import parse_enrich_edges_argv, run_enrich_edges_cli

pytestmark = [pytest.mark.integration]


_LOG = logging.getLogger("test-enrich-edges-placed")


def _corpus(tmp: Path, transcript: str, quote_text: str, content: Dict[str, Any]) -> Path:
    (tmp / "metadata").mkdir()
    (tmp / "transcript.txt").write_text(transcript, encoding="utf-8")
    meta = {
        "feed": {"title": "Odd Lots"},
        "episode": {"episode_id": "ep1"},
        "content": {"transcript_file_path": "transcript.txt", **content},
        "grounded_insights": {"artifact_path": "ep1.gi.json"},
    }
    (tmp / "metadata" / "ep1.metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    gi = {
        "schema_version": "3.0",
        "model_version": "t",
        "prompt_version": "t",
        "episode_id": "ep1",
        "nodes": [
            {"id": "episode:ep1", "type": "Episode", "properties": {}},
            {"id": "insight:1", "type": "Insight", "properties": {"text": "x"}},
            {
                "id": "quote:1",
                "type": "Quote",
                "properties": {"text": quote_text, "char_start": transcript.index(quote_text)},
            },
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"}],
    }
    (tmp / "ep1.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    return tmp


def _spoken_by(tmp: Path) -> List[Dict[str, Any]]:
    rc = run_enrich_edges_cli(parse_enrich_edges_argv(["--output-dir", str(tmp)]), _LOG)
    assert rc == 0
    gi = json.loads((tmp / "ep1.gi.json").read_text(encoding="utf-8"))
    return [e for e in gi["edges"] if e.get("type") == "SPOKEN_BY"]


def test_an_anonymous_transcript_credits_nobody_even_with_a_guessed_host(tmp_path: Path) -> None:
    transcript = (
        "Speaker 1: Welcome to Odd Lots, today we talk about the economy.\n"
        "Speaker 2: Thanks for having me, the labour market is still tight.\n"
    )
    corpus = _corpus(
        tmp_path,
        transcript,
        "the labour market is still tight",
        {
            # The record for a never-diarized episode: the guess, kept as NOT placed.
            "speakers": [
                {"id": "unplaced_1", "name": "Tracy Alloway", "role": "host", "placed": False},
                {"id": "unplaced_2", "name": "Austan Goolsbee", "role": "guest", "placed": False},
            ],
            # A pre-1.2.0 field left on the artifact must not be read as a fallback either.
            "detected_hosts": ["Tracy Alloway"],
            "detected_guests": ["Austan Goolsbee"],
        },
    )
    assert _spoken_by(corpus) == []


def test_a_named_marker_for_someone_not_placed_is_not_credited(tmp_path: Path) -> None:
    transcript = (
        "Michael Barbaro: From The New York Times, this is The Daily.\n"
        "Natalie Kitroeff: Canada walked away from the trade talks.\n"
    )
    corpus = _corpus(
        tmp_path,
        transcript,
        "Canada walked away from the trade talks",
        {
            "speakers": [
                {
                    "id": "host",
                    "name": "Michael Barbaro",
                    "role": "host",
                    "placed": True,
                    "voices": ["SPEAKER_02"],
                },
                {"id": "unplaced_1", "name": "Natalie Kitroeff", "role": "host", "placed": False},
            ],
        },
    )
    assert _spoken_by(corpus) == []


def test_a_placed_voice_is_still_credited(tmp_path: Path) -> None:
    transcript = (
        "Michael Barbaro: From The New York Times, this is The Daily.\n"
        "Matina Stevis-Gridneff: Canada walked away from the trade talks.\n"
    )
    corpus = _corpus(
        tmp_path,
        transcript,
        "Canada walked away from the trade talks",
        {
            "speakers": [
                {"id": "host", "name": "Michael Barbaro", "role": "host", "placed": True},
                {"id": "guest", "name": "Matina Stevis-Gridneff", "role": "guest", "placed": True},
            ],
        },
    )
    edges = _spoken_by(corpus)
    assert [(e["from"], e["to"]) for e in edges] == [("quote:1", "person:matina-stevis-gridneff")]


def test_a_placed_host_is_not_guessed_onto_an_anonymous_voice(tmp_path: Path) -> None:
    """The heuristic itself: a real host exists, but nothing says WHICH voice is theirs.

    The removed role heuristic gave the first `Speaker N` to speak `hosts[0]`. Tracy Alloway is a
    genuine placed host here — and still no marker names her, so no quote may be credited to her.
    """
    transcript = (
        "Speaker 1: Welcome to Odd Lots, today we talk about the economy.\n"
        "Speaker 2: Thanks for having me, the labour market is still tight.\n"
    )
    corpus = _corpus(
        tmp_path,
        transcript,
        "today we talk about the economy",
        {
            "speakers": [
                {
                    "id": "host",
                    "name": "Tracy Alloway",
                    "role": "host",
                    "placed": True,
                    "voices": ["SPEAKER_00"],
                },
            ],
        },
    )
    assert _spoken_by(corpus) == []
