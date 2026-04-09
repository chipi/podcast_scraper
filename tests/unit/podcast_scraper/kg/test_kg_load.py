"""Unit tests for ``kg.load`` (artifact lookup by episode id)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from podcast_scraper.kg.load import episode_node, find_kg_artifact_by_episode_id


def _write_episode_metadata(meta_dir: Path, base: str, *, feed_id: str, episode_id: str) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    doc = {"feed": {"feed_id": feed_id}, "episode": {"episode_id": episode_id}}
    (meta_dir / f"{base}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture
def minimal_kg_path() -> Path:
    return Path(__file__).resolve().parents[3] / "fixtures" / "kg" / "minimal.kg.json"


@pytest.mark.unit
class TestFindKgArtifactByEpisodeId:
    """find_kg_artifact_by_episode_id."""

    def test_no_metadata_dir_returns_none(self, tmp_path: Path) -> None:
        assert find_kg_artifact_by_episode_id(tmp_path, "any") is None

    def test_finds_matching_episode(self, tmp_path: Path, minimal_kg_path: Path) -> None:
        meta = tmp_path / "metadata"
        meta.mkdir(parents=True)
        target = meta / "a.kg.json"
        shutil.copy(minimal_kg_path, target)
        found = find_kg_artifact_by_episode_id(tmp_path, "fixture:minimal-kg")
        assert found == target

    def test_no_match_returns_none(self, tmp_path: Path, minimal_kg_path: Path) -> None:
        meta = tmp_path / "metadata"
        meta.mkdir(parents=True)
        shutil.copy(minimal_kg_path, meta / "a.kg.json")
        assert find_kg_artifact_by_episode_id(tmp_path, "other-episode") is None

    def test_skips_corrupt_json_continues_scan(self, tmp_path: Path, minimal_kg_path: Path) -> None:
        meta = tmp_path / "metadata"
        meta.mkdir(parents=True)
        (meta / "bad.kg.json").write_text("{not json", encoding="utf-8")
        shutil.copy(minimal_kg_path, meta / "good.kg.json")
        found = find_kg_artifact_by_episode_id(tmp_path, "fixture:minimal-kg")
        assert found == meta / "good.kg.json"

    def test_multi_feed_parent_resolves_with_feed_id(
        self, tmp_path: Path, minimal_kg_path: Path
    ) -> None:
        corpus = tmp_path / "corpus"
        for fid, slug in (("feed_a", "rss_a"), ("feed_b", "rss_b")):
            mdir = corpus / "feeds" / slug / "run" / "metadata"
            _write_episode_metadata(mdir, "x", feed_id=fid, episode_id="dup")
            shutil.copy(minimal_kg_path, mdir / "x.kg.json")
            data = json.loads((mdir / "x.kg.json").read_text(encoding="utf-8"))
            data["episode_id"] = "dup"
            (mdir / "x.kg.json").write_text(json.dumps(data), encoding="utf-8")

        assert find_kg_artifact_by_episode_id(corpus, "dup") is None
        got = find_kg_artifact_by_episode_id(corpus, "dup", feed_id="feed_a")
        assert got is not None
        assert "rss_a" in str(got)


@pytest.mark.unit
class TestEpisodeNode:
    """episode_node helper."""

    def test_returns_episode_dict(self, minimal_kg_path: Path) -> None:
        data = json.loads(minimal_kg_path.read_text(encoding="utf-8"))
        n = episode_node(data)
        assert n is not None
        assert n.get("type") == "Episode"
        assert "properties" in n

    def test_returns_none_when_no_episode(self) -> None:
        assert episode_node({"nodes": [{"type": "Topic", "id": "t1"}], "edges": []}) is None

    def test_returns_none_for_empty_nodes(self) -> None:
        assert episode_node({"nodes": [], "edges": []}) is None
