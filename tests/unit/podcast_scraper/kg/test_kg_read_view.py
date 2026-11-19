"""KG 3.0 files read with the same answers as their KG 2.x form (R3b's parity on served surfaces).

``_as_v3_file`` writes a 2.x artifact as R3b's migration stores it (database's
``kg/v3_file.to_v3_file``: the ``kg_as_v3`` view without ``adapted_from`` and the root
``episode_id``), so a reader that still reads a 2.x name finds nothing. Each reader gets both
forms and must give one answer.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.enrichment.enrichers._loaders import load_kg
from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle
from podcast_scraper.kg.as_v3 import kg_as_v3
from podcast_scraper.kg.corpus import load_kg_artifacts
from podcast_scraper.kg.quality_metrics import compute_kg_quality_metrics
from podcast_scraper.kg.read_view import is_kg_v3, kg_read_view
from podcast_scraper.search import relational_queries
from podcast_scraper.search.corpus_graph import CorpusGraph
from podcast_scraper.server.corpus_access import load_json_artifact

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "fixtures/app-validation-corpus/v3/feeds/p14/run_20260101-000000/metadata/p14_e03.kg.json"
)
_SHOW = "podcast:p14"


def _kg_2() -> Dict[str, Any]:
    """The fixture plus the Podcast node, HAS_EPISODE and HOSTS edges KG's writer adds."""
    kg: Dict[str, Any] = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    ep = next(n for n in kg["nodes"] if n["type"] == "Episode")
    host = next(n for n in kg["nodes"] if n["type"] == "Person")
    kg["nodes"].append({"id": _SHOW, "type": "Podcast", "properties": {"title": "P14"}})
    kg["edges"] += [
        {"type": "HAS_EPISODE", "from": _SHOW, "to": ep["id"]},
        {"type": "HOSTS", "from": host["id"], "to": _SHOW},
    ]
    return kg


def _as_v3_file(kg: Dict[str, Any]) -> Dict[str, Any]:
    out = kg_as_v3(kg)
    out.pop("adapted_from", None)
    out.pop("episode_id", None)
    return out


@pytest.fixture()
def v2() -> Dict[str, Any]:
    return _kg_2()


@pytest.fixture()
def v3(v2: Dict[str, Any]) -> Dict[str, Any]:
    out = _as_v3_file(v2)
    blob = json.dumps(out)
    for name in (
        '"Episode"',
        '"Podcast"',
        "HAS_EPISODE",
        "HOSTS",
        "transcript_ref",
        "publish_date",
    ):
        assert name not in blob, f"the 3.0 sample still carries {name}"
    assert "episode_id" not in out
    return out


def _write(tmp_path: Path, kg: Dict[str, Any]) -> Path:
    path = tmp_path / "metadata" / "p14_e03.kg.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kg), encoding="utf-8")
    return path


def test_a_2x_file_is_returned_as_is(v2: Dict[str, Any]) -> None:
    assert not is_kg_v3(v2) and kg_read_view(v2) is v2


def test_the_view_never_mutates_the_file(v3: Dict[str, Any]) -> None:
    before = copy.deepcopy(v3)
    kg_read_view(v3)
    assert v3 == before


def test_a_3_0_file_reads_every_2x_name(v2: Dict[str, Any], v3: Dict[str, Any]) -> None:
    view = kg_read_view(v3)
    assert view["episode_id"] == v2["episode_id"]
    assert sorted(n["type"] for n in view["nodes"]) == sorted(n["type"] for n in v2["nodes"])
    assert sorted(e["type"] for e in view["edges"]) == sorted(e["type"] for e in v2["edges"])
    assert view["extraction"]["transcript_ref"] == v2["extraction"]["transcript_ref"]
    old_ep = next(n for n in v2["nodes"] if n["type"] == "Episode")["properties"]
    new_ep = next(n for n in view["nodes"] if n["type"] == "Episode")["properties"]
    for key in ("podcast_id", "publish_date"):
        assert new_ep.get(key) == old_ep.get(key)


def test_the_corpus_loader_matches(tmp_path: Path, v2: Dict[str, Any], v3: Dict[str, Any]) -> None:
    ((_, a),) = load_kg_artifacts([_write(tmp_path / "a", v2)], validate=False)
    ((_, b),) = load_kg_artifacts([_write(tmp_path / "b", v3)], validate=False)
    assert b["episode_id"] == a["episode_id"]
    assert {n["type"] for n in b["nodes"]} == {n["type"] for n in a["nodes"]}


def test_quality_metrics_match(tmp_path: Path, v2: Dict[str, Any], v3: Dict[str, Any]) -> None:
    def metrics(kg: Dict[str, Any], sub: str) -> Dict[str, Any]:
        found = asdict(compute_kg_quality_metrics([_write(tmp_path / sub, kg)]))
        assert not found.pop("errors")
        return found

    a, b = metrics(v2, "a"), metrics(v3, "b")
    assert a["artifact_paths"] == 1 and b == a


def test_the_enricher_loader_matches(
    tmp_path: Path, v2: Dict[str, Any], v3: Dict[str, Any]
) -> None:
    def loaded(kg: Dict[str, Any], sub: str) -> Dict[str, Any]:
        bundle = EpisodeArtifactBundle(
            metadata_path=tmp_path / "m.json",
            gi_path=None,
            kg_path=_write(tmp_path / sub, kg),
            bridge_path=None,
            episode_id="e",
            stem="p14_e03",
        )
        return load_kg(bundle)

    assert loaded(v3, "b")["episode_id"] == loaded(v2, "a")["episode_id"]


def test_the_server_loader_views_kg_files(
    tmp_path: Path, v2: Dict[str, Any], v3: Dict[str, Any]
) -> None:
    _write(tmp_path, v3)
    got = load_json_artifact(tmp_path, "metadata/p14_e03.kg.json")
    assert got is not None and got["episode_id"] == v2["episode_id"]


def test_show_episodes_match(v2: Dict[str, Any], v3: Dict[str, Any]) -> None:
    def episodes(kg: Dict[str, Any]) -> list:
        graph = CorpusGraph()
        graph._ingest(kg, "kg")
        return [(r.id, r.type) for r in relational_queries.episodes_of(graph, _SHOW)]

    old = episodes(v2)
    assert old and episodes(kg_read_view(v3)) == old
