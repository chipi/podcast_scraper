"""Unit tests for the enrich-edges CLI (parse + walker) — #874 activation."""

from __future__ import annotations

import json
import logging
from argparse import Namespace

import pytest

from podcast_scraper.search.cli_handlers import parse_enrich_edges_argv, run_enrich_edges_cli

pytestmark = pytest.mark.unit

_LOG = logging.getLogger("test-enrich-edges")


def test_parse_sets_command_and_output_dir():
    args = parse_enrich_edges_argv(["--output-dir", "/tmp/corpus"])
    assert args.command == "enrich-edges"
    assert args.output_dir == "/tmp/corpus"
    assert args.no_speaker is False


def test_parse_no_speaker_flag():
    args = parse_enrich_edges_argv(["--output-dir", "/tmp/corpus", "--no-speaker"])
    assert args.no_speaker is True


def test_parse_requires_output_dir():
    with pytest.raises(SystemExit):
        parse_enrich_edges_argv([])


def test_run_requires_output_dir():
    assert run_enrich_edges_cli(Namespace(), _LOG) == 2  # EXIT_INVALID_ARGS


def _build_corpus(tmp_path):
    """Minimal one-episode corpus: metadata + gi.json + kg.json + diarized transcript."""
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "ep1.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"title": "Test Show"},
                "episode": {"episode_id": "ep1"},
                "content": {
                    "transcript_file_path": "transcript.txt",
                    "detected_hosts": [],
                    "detected_guests": ["Elon Musk"],
                },
                "grounded_insights": {"artifact_path": "ep1.gi.json"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ep1.gi.json").write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "model_version": "t",
                "prompt_version": "t",
                "episode_id": "ep1",
                "nodes": [
                    {"id": "episode:ep1", "type": "Episode", "properties": {}},
                    {
                        "id": "insight:1",
                        "type": "Insight",
                        "properties": {"text": "Elon Musk plans to list SpaceX."},
                    },
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ep1.kg.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "person:elon-musk",
                        "type": "Entity",
                        "properties": {"name": "Elon Musk", "kind": "person"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "transcript.txt").write_text(
        "Speaker 1: Hello. Speaker 2: Elon Musk.", encoding="utf-8"
    )


def test_run_persists_relational_edges(tmp_path):
    _build_corpus(tmp_path)
    rc = run_enrich_edges_cli(parse_enrich_edges_argv(["--output-dir", str(tmp_path)]), _LOG)
    assert rc == 0
    art = json.loads((tmp_path / "ep1.gi.json").read_text())
    edge_types = {e["type"] for e in art["edges"]}
    assert "HAS_EPISODE" in edge_types  # Podcast→Episode (from feed title)
    # RFC-097 v3.0: typed MENTIONS_PERSON / MENTIONS_ORG replace generic MENTIONS.
    assert edge_types & {"MENTIONS_PERSON", "MENTIONS_ORG"}  # Insight→Person/Org
    assert any(n["id"] == "podcast:test-show" and n["type"] == "Podcast" for n in art["nodes"])
    # idempotent: a second pass adds nothing new
    rc2 = run_enrich_edges_cli(parse_enrich_edges_argv(["--output-dir", str(tmp_path)]), _LOG)
    art2 = json.loads((tmp_path / "ep1.gi.json").read_text())
    assert rc2 == 0 and len(art2["edges"]) == len(art["edges"])


def test_run_no_speaker_skips_transcript(tmp_path):
    _build_corpus(tmp_path)
    rc = run_enrich_edges_cli(
        parse_enrich_edges_argv(["--output-dir", str(tmp_path), "--no-speaker"]), _LOG
    )
    assert rc == 0
    art = json.loads((tmp_path / "ep1.gi.json").read_text())
    # show + entity edges still emitted; SPOKEN_BY path skipped
    edge_types = {e["type"] for e in art["edges"]}
    assert "HAS_EPISODE" in edge_types
    # RFC-097 v3.0: typed MENTIONS_PERSON / MENTIONS_ORG replace generic MENTIONS.
    assert edge_types & {"MENTIONS_PERSON", "MENTIONS_ORG"}


def test_run_skips_unreadable_gi(tmp_path):
    _build_corpus(tmp_path)
    (tmp_path / "ep1.gi.json").write_text("{ not valid json", encoding="utf-8")
    # malformed gi.json is skipped gracefully, run still succeeds
    assert run_enrich_edges_cli(parse_enrich_edges_argv(["--output-dir", str(tmp_path)]), _LOG) == 0


def test_run_handles_unreadable_kg(tmp_path):
    _build_corpus(tmp_path)
    (tmp_path / "ep1.kg.json").write_text("{ bad", encoding="utf-8")
    rc = run_enrich_edges_cli(parse_enrich_edges_argv(["--output-dir", str(tmp_path)]), _LOG)
    assert rc == 0
    # MENTIONS skipped (kg unreadable) but HAS_EPISODE still added
    art = json.loads((tmp_path / "ep1.gi.json").read_text())
    assert "HAS_EPISODE" in {e["type"] for e in art["edges"]}


def test_run_without_kg_or_transcript(tmp_path):
    _build_corpus(tmp_path)
    (tmp_path / "ep1.kg.json").unlink()  # no kg → MENTIONS branch skipped
    (tmp_path / "transcript.txt").unlink()  # no transcript → SPOKEN_BY branch skipped
    rc = run_enrich_edges_cli(parse_enrich_edges_argv(["--output-dir", str(tmp_path)]), _LOG)
    assert rc == 0
    edge_types = {e["type"] for e in json.loads((tmp_path / "ep1.gi.json").read_text())["edges"]}
    # RFC-097 v3.0: typed MENTIONS_PERSON / MENTIONS_ORG also absent (kg.json gone).
    assert "HAS_EPISODE" in edge_types
    assert not (edge_types & {"MENTIONS", "MENTIONS_PERSON", "MENTIONS_ORG"})


# === #876: corpus-wide SPOKEN_BY for re-diarized whisper episodes (named transcript) ===

_DIARIZED_TRANSCRIPT = "Maya: Welcome to the show.\nLiam: SpaceX will list soon.\n"


def _build_diarized_corpus(tmp_path):
    """One-episode corpus whose transcript is the NEW diarization's *named* screenplay
    (``Maya:`` / ``Liam:``) — the shape a re-diarized whisper episode (#876) has — with a
    Quote in the guest's turn."""
    (tmp_path / "metadata").mkdir()
    quote_char = _DIARIZED_TRANSCRIPT.index("SpaceX will list soon")
    (tmp_path / "metadata" / "ep1.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"title": "Test Show"},
                "episode": {"episode_id": "ep1"},
                "content": {
                    "transcript_file_path": "transcript.txt",
                    "detected_hosts": ["Maya"],
                    "detected_guests": ["Liam"],
                },
                "grounded_insights": {"artifact_path": "ep1.gi.json"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ep1.gi.json").write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "model_version": "t",
                "prompt_version": "t",
                "episode_id": "ep1",
                "nodes": [
                    {"id": "episode:ep1", "type": "Episode", "properties": {}},
                    {"id": "insight:1", "type": "Insight", "properties": {"text": "SpaceX IPO."}},
                    {
                        "id": "quote:1",
                        "type": "Quote",
                        "properties": {
                            "char_start": quote_char,
                            "text": "SpaceX will list soon.",
                        },
                    },
                ],
                "edges": [{"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "transcript.txt").write_text(_DIARIZED_TRANSCRIPT, encoding="utf-8")


def test_run_emits_spoken_by_for_named_diarized_transcript(tmp_path):
    """#876: enrich-edges emits SPOKEN_BY for the new diarization's NAMED transcript
    (re-diarized whisper episodes), attributing the guest's quote to person:liam via the
    #875 named path — this is the corpus-wide coverage the reprocess unlocks."""
    _build_diarized_corpus(tmp_path)
    rc = run_enrich_edges_cli(parse_enrich_edges_argv(["--output-dir", str(tmp_path)]), _LOG)
    assert rc == 0
    art = json.loads((tmp_path / "ep1.gi.json").read_text())
    spoken = {(e["from"], e["to"]) for e in art["edges"] if e["type"] == "SPOKEN_BY"}
    assert ("quote:1", "person:liam") in spoken
    assert any(n["id"] == "person:liam" and n["type"] == "Person" for n in art["nodes"])
