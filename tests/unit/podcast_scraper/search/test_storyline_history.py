"""RFC-128 (#2349): seed the storyline registry from history, keeping every published storyline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search import storyline_history as h

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
CORPUS = ROOT / "tests" / "fixtures" / "app-validation-corpus" / "v3"
DAY = 86400.0


def _cluster(cid: str, name: str, topics: dict[str, list[str]], anchor: str) -> dict:
    return {
        "graph_compound_parent_id": cid,
        "canonical_label": name,
        "members": [
            {"topic_id": t, "episode_ids": eps, "lift_to_cluster": 9.0 if t == anchor else 1.0}
            for t, eps in topics.items()
        ],
    }


def _entry(name: str, topics: dict[str, float], created: float, last: float, **kw: object) -> dict:
    return {
        "name": name,
        "aliases": [],
        "topics": topics,
        "anchor": None,
        "episodes": ["e1"],
        "created_at": created,
        "last_active_at": last,
        "rename": None,
        **kw,
    }


def test_a_current_storyline_takes_history_and_keeps_its_id_name_and_anchor() -> None:
    history = {
        "as_of": 50 * DAY,
        "entries": {
            "thc:old-name": _entry("old name", {"topic:a": 3.0, "topic:b": 2.0}, 1 * DAY, 9 * DAY),
        },
        "candidates": [{"topics": ["topic:q"]}],
    }
    current = [_cluster("thc:today", "today", {"topic:a": ["e1"], "topic:z": ["e2"]}, "topic:z")]
    registry, report = h.merge(history, current, episode_dates={}, now=60 * DAY)
    (sid,) = registry["entries"]
    e = registry["entries"][sid]
    assert sid == "thc:today" and e["name"] == "today" and e["anchor"] == "topic:z"
    assert e["aliases"] == ["old name"]
    assert (e["created_at"], e["last_active_at"]) == (1 * DAY, 9 * DAY)
    assert e["topics"] == {"topic:a": 3.0, "topic:b": 2.0, "topic:z": 1.0}
    assert registry["as_of"] == 50 * DAY and registry["candidates"] == history["candidates"]
    assert report["with_history"] == 1 and report["added_from_history"] == []


def test_replayed_storylines_inside_a_wider_current_one_fold_into_it() -> None:
    """A replayed thread can be a small part of today's storyline (the share is measured both
    ways); two of them fold into it instead of coming back as same-named duplicates."""
    history = {
        "entries": {
            "thc:x": _entry("x", {"topic:a": 1.0, "topic:b": 1.0}, 5 * DAY, 6 * DAY),
            "thc:y": _entry("y", {"topic:c": 1.0, "topic:d": 1.0}, 2 * DAY, 8 * DAY),
        },
    }
    wide = {t: ["e1"] for t in ("topic:a", "topic:b", "topic:c", "topic:d", "topic:e", "topic:f")}
    current = [_cluster("thc:wide", "wide", wide, "topic:a")]
    registry, report = h.merge(history, current, episode_dates={}, now=60 * DAY)
    assert list(registry["entries"]) == ["thc:wide"]
    e = registry["entries"]["thc:wide"]
    assert (e["created_at"], e["last_active_at"]) == (2 * DAY, 8 * DAY)
    assert sorted(e["aliases"]) == ["x", "y"] and report["folded"] == 2


def test_a_storyline_without_history_is_dated_by_its_episodes() -> None:
    current = [_cluster("thc:new", "new", {"topic:a": ["e1"], "topic:b": ["e2"]}, "topic:a")]
    registry, _ = h.merge({}, current, episode_dates={"e1": 3 * DAY, "e2": 7 * DAY}, now=60 * DAY)
    e = registry["entries"]["thc:new"]
    assert (e["created_at"], e["last_active_at"], e["anchor"]) == (3 * DAY, 7 * DAY, "topic:a")


def test_a_replayed_storyline_with_no_current_match_is_added_without_taking_an_id() -> None:
    history = {"entries": {"thc:same": _entry("other", {"topic:q": 1.0}, 0.0, 0.0)}}
    current = [_cluster("thc:same", "same", {"topic:a": ["e1"]}, "topic:a")]
    registry, report = h.merge(history, current, episode_dates={}, now=0.0)
    assert sorted(registry["entries"]) == ["thc:same", "thc:same-1"]
    assert registry["entries"]["thc:same"]["name"] == "same"
    assert report["added_from_history"] == ["thc:same-1"]


def _corpus(tmp_path: Path) -> Path:
    """A copy of the fixture corpus's episodes (discovery does not follow a symlink) and a
    storyline artifact."""
    import shutil

    from podcast_scraper.enrichment.paths import discover_episode_bundles

    root = tmp_path / "corpus"
    shutil.copytree(CORPUS / "feeds", root / "feeds")
    h.run_month(discover_episode_bundles(root), root, 0.0)
    return root


def test_dry_run_writes_nothing_and_apply_writes_only_the_registry_with_a_backup(
    tmp_path: Path,
) -> None:
    root = _corpus(tmp_path)
    art = root / "enrichments" / "topic_theme_clusters.json"
    before = art.read_text(encoding="utf-8")
    assert json.loads(before)["data"]["clusters"], "the fixture corpus has storylines"

    report = h.seed(root, apply=False, now=100 * DAY)
    assert report["applied"] is False and art.read_text(encoding="utf-8") == before
    assert list(art.parent.iterdir()) == [art]

    report = h.seed(root, apply=True, now=100 * DAY)
    backup = Path(report["backup"])
    assert backup.read_text(encoding="utf-8") == before
    old, new = json.loads(before)["data"], json.loads(art.read_text(encoding="utf-8"))["data"]
    assert {k: v for k, v in new.items() if k != "registry"} == {
        k: v for k, v in old.items() if k != "registry"
    }
    published = {c["graph_compound_parent_id"] for c in old["clusters"]}
    assert published <= set(new["registry"]["entries"]), "every published storyline is kept"
