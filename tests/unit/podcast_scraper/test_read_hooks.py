"""The server's read side reaches the speaker, ad and identity stages through read_hooks
(E04 S3), kept out of the registry corpus migrations reach."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper import builtin_read_hooks, read_hooks
from podcast_scraper.extensions import Extension, use_extensions
from podcast_scraper.read_hooks import ReadHooks

pytestmark = pytest.mark.unit

_NONE = ReadHooks(name="builtin")


def test_without_any_stage_each_hook_answers_its_plain_default() -> None:
    with patch.object(builtin_read_hooks, "BUILTIN", _NONE):
        assert read_hooks.segment_speaker({"speaker": "SPEAKER_00"}) == (None, None)
        assert read_hooks.unidentified_label() == "Unidentified speaker"
        assert read_hooks.ad_reads([{"start": 0.0, "end": 1.0}]) == []
        assert read_hooks.looks_like_publisher("NPR") is False
        assert read_hooks.entity_id_map(Path("/c")) == {}
        assert read_hooks.gi_integrity(Path("/c")) is None
        assert (
            read_hooks.topic_perspectives(
                Path("/c"), ["topic:ai"], keep_episode_ids=None, slug_by_episode={}
            )
            == []
        )
        assert read_hooks.group_borrows(Path("/c"), []) == ({}, {})
        assert read_hooks.served_id_map(Path("/c")) == {}
        assert read_hooks.bridge_node_episodes("/c", "/c", "g:topic:ai") == (
            "g:topic:ai",
            [],
            False,
            None,
        )


def test_without_the_built_in_an_extension_owns_the_hook() -> None:
    ext = Extension(name="x", read_hooks=ReadHooks(name="x", unidentified_label=lambda: "Voice"))
    with patch.object(builtin_read_hooks, "BUILTIN", _NONE), use_extensions([ext]):
        assert read_hooks.unidentified_label() == "Voice"


def test_the_built_in_answers_before_an_extension() -> None:
    ext = Extension(name="x", read_hooks=ReadHooks(name="x", unidentified_label=lambda: "Voice"))
    with use_extensions([ext]):
        assert read_hooks.unidentified_label() == "Unidentified speaker"


def test_the_publisher_reader_gets_the_language() -> None:
    with patch(
        "podcast_scraper.speaker_detectors.hosts.looks_like_publisher", return_value=True
    ) as reader:
        assert read_hooks.looks_like_publisher("Rádio Senado", "pt") is True
    reader.assert_called_once_with("Rádio Senado", "pt")


def test_without_an_insight_stage_a_topic_has_no_perspectives(tmp_path: Path) -> None:
    from podcast_scraper.server.relational_view import build_topic_perspectives

    with patch.object(builtin_read_hooks, "BUILTIN", _NONE):
        assert build_topic_perspectives(tmp_path, "topic:ai") is None


def test_the_built_in_routers_are_the_stage_routes_on_the_public_operator_surface() -> None:
    from podcast_scraper.server.routes import cil, corpus_persons, explore

    mounts = read_hooks.builtin_routers()
    assert [m.router for m in mounts] == [explore.router, corpus_persons.router, cil.router]
    assert {(m.plane, m.operator_public) for m in mounts} == {("operator", True)}


def test_the_bridge_hook_answers_the_canonical_id() -> None:
    with patch(
        "podcast_scraper.server.cil_queries.episodes_for_bridge_node_id",
        return_value=([{"gi_relative_path": "a"}], False, None),
    ) as listing:
        canon, rows, truncated, total = read_hooks.bridge_node_episodes(
            "/c", "/a", "g:k:topic:ai", max_episodes=5
        )
    assert (canon, rows, truncated, total) == ("topic:ai", [{"gi_relative_path": "a"}], False, None)
    listing.assert_called_once_with("/c", "/a", "g:k:topic:ai", max_episodes=5)


def test_without_the_stages_search_reads_kg_alone(tmp_path: Path) -> None:
    rows = [{"metadata": {"doc_type": "transcript", "episode_id": "e1"}}]
    with patch.object(builtin_read_hooks, "BUILTIN", _NONE):
        read_hooks.lift_transcript_hits(rows, tmp_path, {"e1": tmp_path / "e1.gi.json"})
        assert rows == [{"metadata": {"doc_type": "transcript", "episode_id": "e1"}}]
        assert list(read_hooks.graph_artifacts(tmp_path)) == []
        assert read_hooks.host_variants("Erica Barris", "Erika Barris") is False


def test_the_graph_hook_streams_the_gi_layer(tmp_path: Path) -> None:
    with (
        patch("podcast_scraper.gi.explore.scan_artifact_paths", return_value=[]),
        patch(
            "podcast_scraper.gi.corpus.iter_gi_artifacts",
            return_value=iter([(tmp_path / "a.gi.json", {"nodes": []})]),
        ),
    ):
        layers = read_hooks.graph_artifacts(tmp_path)
        assert not isinstance(layers, list)
        assert list(layers) == [("gi", {"nodes": []})]
