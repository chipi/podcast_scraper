# mypy: disable-error-code="call-arg"
# Deliberate: Config(rss_url=...) — alias="rss"; populate-by-name accepts either at runtime.
"""The corpus's extraction votes reach the speaker-candidate filter of a real run config (#2220).

The seams: a corpus on disk -> ``votes_for_cfg`` resolving its root from a per-feed run config
(built the way the corpus loops build it) -> ``HostDetectionResult.kind_votes`` -> the
per-episode host filter. Measured shape: "The Brazilian Report" seated as a host on 40 episodes
while extraction calls it an Organization 23 times and a Person never.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.entity_kind_votes import corpus_kind_votes, votes_for_cfg
from podcast_scraper.utils import filesystem
from podcast_scraper.workflow.stages.processing import hosts_for_episode
from podcast_scraper.workflow.types import HostDetectionResult

pytestmark = [pytest.mark.integration]

FEED_URL = "https://feed-brazil.example/rss"
ITUNES = "http://www.itunes.com/dtds/podcast-1.0.dtd"


def _corpus_with_votes(root: Path, name: str, org_votes: int) -> None:
    for i in range(org_votes):
        meta = root / "feeds" / "other" / f"run_{i}" / "metadata"
        meta.mkdir(parents=True)
        (meta / f"ep{i}.metadata.json").write_text(
            json.dumps({"episode": {"episode_id": f"e{i}"}, "feed": {"feed_id": "other"}})
        )
        kg = {
            "nodes": [
                {
                    "id": "org:x",
                    "type": "Organization",
                    "properties": {"name": name, "role": "mentioned"},
                }
            ]
        }
        (meta / f"ep{i}.kg.json").write_text(json.dumps(kg))


def _episode_with_author(author: str) -> object:
    item = ET.Element("item")
    ET.SubElement(item, f"{{{ITUNES}}}author").text = author

    class _Ep:
        pass

    ep = _Ep()
    ep.item = item  # type: ignore[attr-defined]
    return ep


def test_an_organisation_by_vote_is_not_seated_as_this_episodes_host(tmp_path: Path) -> None:
    corpus_kind_votes.cache_clear()
    _corpus_with_votes(tmp_path, "The Brazilian Report", 23)
    cfg = config.Config(
        rss_url=FEED_URL, output_dir=filesystem.corpus_feed_output_dir(str(tmp_path), FEED_URL)
    )
    votes = votes_for_cfg(cfg)
    assert votes is not None and votes.calls_organisation("The Brazilian Report")

    result = HostDetectionResult(
        cached_hosts={"The Brazilian Report"},
        heuristics=None,
        feed_title="Explaining Brazil",
        episode_author_hosts=frozenset({"The Brazilian Report"}),
        kind_votes=votes,
    )
    assert hosts_for_episode(result, _episode_with_author("The Brazilian Report")) == set()


def test_without_a_corpus_the_same_author_is_kept_as_before(tmp_path: Path) -> None:
    """No evidence, no change: outside a corpus the filter behaves exactly as it did."""
    corpus_kind_votes.cache_clear()
    cfg = config.Config(rss_url=FEED_URL, output_dir=str(tmp_path / "plain"))
    assert votes_for_cfg(cfg) is None
    result = HostDetectionResult(
        cached_hosts={"The Brazilian Report"},
        heuristics=None,
        feed_title="Explaining Brazil",
        episode_author_hosts=frozenset({"The Brazilian Report"}),
        kind_votes=None,
    )
    assert hosts_for_episode(result, _episode_with_author("The Brazilian Report")) == {
        "The Brazilian Report"
    }
