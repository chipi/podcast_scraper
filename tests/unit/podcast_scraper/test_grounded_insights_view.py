"""GI 4.0 files read with the same answers as their GI 3.1 form (R3b's parity on served surfaces).

``_as_v4_file`` writes a 3.1 artifact in GI 4.0's stored shape the way Common's
``tests/_gi_v4.py`` ``as_v4_file`` does (R3b's output): every 3.1 name removed, so a reader that
still reads one finds nothing. Each reader below gets both forms and must give one answer.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper import read_hooks
from podcast_scraper.corpus_completeness import _collect_edge_types
from podcast_scraper.enrichment.enrichers._loaders import load_gi, publish_date
from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle
from podcast_scraper.grounded_insights_view import gi_read_view, is_gi_v4
from podcast_scraper.identity.bare_name_scope import person_ids_in, rewrite_ids
from podcast_scraper.search import relational_queries
from podcast_scraper.search.cli_handlers import _quotes_for_insight
from podcast_scraper.search.corpus_graph import CorpusGraph
from podcast_scraper.search.corpus_search import _insight_scopes_from_gi
from podcast_scraper.server.corpus_access import load_json_artifact

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures/app-validation-corpus/v3/feeds/p05/run_20260101-000000/metadata/p05_e04.gi.json"
)
_OLD_SHOW = "podcast:the-risk-panel"
_NEW_SHOW = "podcast:p05"


def _gi_3_1() -> Dict[str, Any]:
    """The fixture plus the Podcast node and HAS_EPISODE edge the edge pass adds in 3.1."""
    gi = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    ep = next(n for n in gi["nodes"] if n["type"] == "Episode")
    gi["nodes"].append({"id": _OLD_SHOW, "type": "Podcast", "properties": {"title": "Risk"}})
    gi["edges"].append({"type": "HAS_EPISODE", "from": _OLD_SHOW, "to": ep["id"]})
    return gi


def _nest(props: Dict[str, Any], names: tuple[str, ...]) -> None:
    nested = {k: props.pop(k) for k in names if k in props}
    if nested:
        props["podcast_episode"] = nested


def _as_v4_file(gi: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(gi)
    out["item_id"] = out.pop("episode_id")
    out["content_type"] = "podcast_episode"
    renamed: Dict[str, str] = {}
    for node in out["nodes"]:
        props = node.setdefault("properties", {})
        kind = node["type"]
        if kind == "Episode":
            node["type"] = "ContentItem"
            props["published_at"] = props.pop("publish_date")
            props["source_id"] = _NEW_SHOW
            _nest(props, ("podcast_id",))
        elif kind == "Podcast":
            node["type"] = "Source"
            renamed[node["id"]] = _NEW_SHOW
            node["id"] = _NEW_SHOW
        elif kind == "Insight":
            props["item_id"] = props.pop("episode_id")
            node.pop("confidence", None)
            _nest(props, ("speaker", "speaker_voice_type"))
        elif kind == "Quote":
            props["item_id"] = props.pop("episode_id")
            props["document_ref"] = props.pop("transcript_ref")
            start, end = props.pop("timestamp_start_ms"), props.pop("timestamp_end_ms")
            if end > start:
                props["start_ms"], props["end_ms"] = start, end
            _nest(props, ("speaker_id", "speaker_name", "speaker_voice_type"))
    for edge in out["edges"]:
        edge["from"] = renamed.get(edge["from"], edge["from"])
        edge["to"] = renamed.get(edge["to"], edge["to"])
        if edge["type"] == "HAS_EPISODE":
            edge["type"] = "HAS_ITEM"
    return out


@pytest.fixture()
def v3() -> Dict[str, Any]:
    return _gi_3_1()


@pytest.fixture()
def v4(v3: Dict[str, Any]) -> Dict[str, Any]:
    out = _as_v4_file(v3)
    blob = json.dumps(out)
    for name in ("HAS_EPISODE", '"Episode"', "timestamp_start_ms", "transcript_ref"):
        assert name not in blob, f"the 4.0 sample still carries {name}"
    for node in out["nodes"]:
        assert "speaker_id" not in node["properties"], "4.0 nests it under podcast_episode"
    return out


def _write(tmp_path: Path, gi: Dict[str, Any]) -> Path:
    path = tmp_path / "metadata" / "p05_e04.gi.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gi), encoding="utf-8")
    return path


def test_a_3_1_file_is_returned_as_is(v3: Dict[str, Any]) -> None:
    assert not is_gi_v4(v3)
    assert gi_read_view(v3) is v3


def test_the_view_never_mutates_the_file(v4: Dict[str, Any]) -> None:
    before = copy.deepcopy(v4)
    gi_read_view(v4)
    assert v4 == before


def test_a_4_0_file_reads_every_3_1_name(v3: Dict[str, Any], v4: Dict[str, Any]) -> None:
    view = gi_read_view(v4)
    assert view["episode_id"] == v3["episode_id"]
    assert sorted(n["type"] for n in view["nodes"]) == sorted(n["type"] for n in v3["nodes"])
    assert sorted(e["type"] for e in view["edges"]) == sorted(e["type"] for e in v3["edges"])
    old = {n["id"]: n["properties"] for n in v3["nodes"]}
    for node in view["nodes"]:
        if node["type"] in ("Quote", "Insight", "Episode"):
            for key, value in old[node["id"]].items():
                assert node["properties"].get(key) == value, (node["id"], key)


def test_insight_quotes_match(v3: Dict[str, Any], v4: Dict[str, Any]) -> None:
    insights = [n["id"] for n in v3["nodes"] if n["type"] == "Insight"]
    for iid in insights:
        assert _quotes_for_insight(gi_read_view(v4), iid) == _quotes_for_insight(v3, iid)
    assert any(_quotes_for_insight(v3, iid) for iid in insights)


def test_search_scopes_match(v3: Dict[str, Any], v4: Dict[str, Any]) -> None:
    ep = v3["episode_id"]
    new = _insight_scopes_from_gi(ep, gi_read_view(v4))
    assert new == _insight_scopes_from_gi(ep, v3)
    assert any(s.speakers for s in new)


def test_enricher_publish_date_matches(
    tmp_path: Path, v3: Dict[str, Any], v4: Dict[str, Any]
) -> None:
    def date(gi: Dict[str, Any]) -> Any:
        bundle = EpisodeArtifactBundle(
            metadata_path=tmp_path / "m.json",
            gi_path=_write(tmp_path, gi),
            kg_path=None,
            bridge_path=None,
            episode_id="e",
            stem="p05_e04",
        )
        return publish_date(load_gi(bundle))

    assert date(v4) == date(v3) == "2025-07-03T00:00:00"


def test_completeness_sees_has_episode(tmp_path: Path, v4: Dict[str, Any]) -> None:
    _write(tmp_path, v4)
    edge_types, seen = _collect_edge_types(tmp_path)
    assert seen == 1 and "HAS_EPISODE" in edge_types


def test_server_loader_views_gi_files_only(
    tmp_path: Path, v3: Dict[str, Any], v4: Dict[str, Any]
) -> None:
    _write(tmp_path, v4)
    loaded = load_json_artifact(tmp_path, "metadata/p05_e04.gi.json")
    assert loaded is not None and loaded["episode_id"] == v3["episode_id"]
    kg = tmp_path / "metadata" / "p05_e04.kg.json"
    kg.write_text(json.dumps({"item_id": "x", "nodes": [], "edges": []}), encoding="utf-8")
    assert "episode_id" not in (load_json_artifact(tmp_path, "metadata/p05_e04.kg.json") or {})


def test_graph_feed_views_the_gi_layer(monkeypatch: pytest.MonkeyPatch, v4: Dict[str, Any]) -> None:
    monkeypatch.setattr(
        read_hooks, "_owner", lambda name: (lambda corpus_dir, validate: iter([("gi", v4)]))
    )
    ((source, data),) = list(read_hooks.graph_artifacts(Path(".")))
    assert source == "gi" and any(e["type"] == "HAS_EPISODE" for e in data["edges"])


def test_show_episodes_match(v3: Dict[str, Any], v4: Dict[str, Any]) -> None:
    def episodes(gi: Dict[str, Any], show: str) -> list:
        graph = CorpusGraph()
        graph._ingest(gi, "gi")
        return [(r.id, r.type) for r in relational_queries.episodes_of(graph, show)]

    old = episodes(v3, _OLD_SHOW)
    assert old and episodes(gi_read_view(v4), _NEW_SHOW) == old


def test_identity_rewrite_reaches_the_4_0_speaker_id(
    v3: Dict[str, Any], v4: Dict[str, Any]
) -> None:
    sid = next(n["properties"]["speaker_id"] for n in v3["nodes"] if n["type"] == "Quote")
    assert sid in person_ids_in(v4) and sid in person_ids_in(v3)
    id_map = {sid: "person:someone-else"}
    new3, changes3 = rewrite_ids(v3, id_map)
    new4, changes4 = rewrite_ids(v4, id_map)
    assert changes4 == changes3 > 0
    assert sid not in person_ids_in(new4)
    quotes4 = [n["properties"] for n in new4["nodes"] if n["type"] == "Quote"]
    assert all("speaker_id" not in q for q in quotes4), "a 4.0 file stays 4.0"
    assert _quotes_for_insight(gi_read_view(new4), "x") == _quotes_for_insight(new3, "x")
    old_ids = sorted(q["podcast_episode"]["speaker_id"] for q in quotes4)
    new_ids = sorted(n["properties"]["speaker_id"] for n in new3["nodes"] if n["type"] == "Quote")
    assert old_ids == new_ids
