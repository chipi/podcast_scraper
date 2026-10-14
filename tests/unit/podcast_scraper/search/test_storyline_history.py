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


def _data(root: Path) -> dict:
    raw = json.loads((root / "enrichments" / "topic_theme_clusters.json").read_text("utf-8"))
    return dict(raw["data"])


def _run(root: Path, now: float) -> dict:
    from podcast_scraper.enrichment.paths import discover_episode_bundles

    data: dict = h.run_month(discover_episode_bundles(root), root, now)
    return data


def test_the_seed_never_writes_the_artifact_and_the_next_run_takes_it_once(
    tmp_path: Path,
) -> None:
    root = _corpus(tmp_path)
    art = root / "enrichments" / "topic_theme_clusters.json"
    before = art.read_bytes()
    published = {c["graph_compound_parent_id"] for c in _data(root)["clusters"]}
    assert published, "the fixture corpus has storylines"
    files = sorted(p.name for p in art.parent.iterdir())

    dry = h.seed(root, apply=False, now=100 * DAY)
    assert dry["applied"] is False and art.read_bytes() == before
    assert sorted(p.name for p in art.parent.iterdir()) == files

    report = h.seed(root, apply=True, now=100 * DAY)
    assert report["applied"] and art.read_bytes() == before, "only the seed file is written"

    data = _run(root, 100 * DAY)
    assert data["registry"]["seeded_from"] == report["seed_id"]
    assert data["registry_health"]["source"] == "seed" and data["registry_health"]["ok"]
    assert published <= {c["graph_compound_parent_id"] for c in data["clusters"]}

    again = h.seed(root, apply=True, now=100 * DAY)
    assert again["already_taken"] and not again["applied"], "the same seed twice is a no-op"
    assert _run(root, 101 * DAY)["registry_health"]["source"] == "artifact"


def test_undo_withdraws_a_pending_seed_and_reverses_a_taken_one(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    seed_file = root / "enrichments" / "storyline_registry.seed.json"
    h.seed(root, apply=True, now=100 * DAY)
    assert h.undo(root)["undone"] == "withdrawn" and not seed_file.exists()

    replaced = _data(root)["registry"]["entries"]
    taken = h.seed(root, apply=True, now=100 * DAY)["seed_id"]
    _run(root, 100 * DAY)
    result = h.undo(root)
    assert result["undone"] == "restore seeded" and result["undoes"] == taken
    data = _run(root, 101 * DAY)
    assert data["registry"]["seeded_from"] == result["seed_id"]
    assert set(replaced) <= set(data["registry"]["entries"])
