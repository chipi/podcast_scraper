"""Retired-tree exclusion and unrecognised-layout defence in depth (#2161 / #2162).

The bug these guard: corpus cleanup moves superseded artifacts to
``.trash/<ts>/feeds/<feed>/run_*/metadata/``. That path parses as NEITHER corpus layout, so it was
kept unconditionally and never compared against the live copy — the episode was collected twice,
its id-keyed rows collided, and the index prune deleted the LIVE copy's rows as "superseded".

Purely filesystem-driven: nothing here touches the network or runs enrichment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from podcast_scraper.search.corpus_scope import (
    classify_metadata_relpath,
    corpus_relpath_is_excluded,
    dedupe_metadata_paths_newest_run_per_episode,
    discover_all_metadata_files,
    discover_metadata_files,
    LAYOUT_FEED_RUN,
    LAYOUT_FLAT_RUN,
    LAYOUT_ROOT_FLAT,
)

LIVE_RUN = "run_499d721a_20260910-142323"
TRASHED_RUN = "run_69842bef_20260905-122652"
EPISODE_ID = "Buzzsprout-19721665"


def _write_metadata(root: Path, relpath: str, episode_id: str = EPISODE_ID) -> Path:
    """Write a metadata doc at *relpath* under *root*, carrying *episode_id*."""
    doc = {
        "feed": {"feed_id": "f1", "title": "S"},
        "episode": {
            "episode_id": episode_id,
            "title": episode_id,
            "published_date": "2026-01-01",
        },
    }
    p = root / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# corpus_relpath_is_excluded
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rel",
    [
        ".trash/20260907T024613Z/feeds/pod/run_a/metadata/ep.metadata.json",
        ".viewer/jobs/some.log",
        "feeds/pod/.trash/run_a/metadata/ep.metadata.json",
        ".cache/whatever.json",
    ],
)
def test_excluded_paths_are_recognised_as_retired(rel: str) -> None:
    assert corpus_relpath_is_excluded(rel) is True


@pytest.mark.parametrize(
    "rel",
    [
        "feeds/pod/run_a/metadata/ep.metadata.json",
        "metadata/ep.metadata.json",
        "run_a/metadata/ep.metadata.json",
        # A dotted FILE is not a hidden directory, and must stay included.
        "feeds/pod/run_a/metadata/0001 - ep.metadata.json",
    ],
)
def test_live_paths_are_not_excluded(rel: str) -> None:
    assert corpus_relpath_is_excluded(rel) is False


def test_dot_and_dotdot_segments_are_not_treated_as_hidden() -> None:
    """``os.path.relpath`` yields ``.`` for the root itself — that must not exclude the corpus."""
    assert corpus_relpath_is_excluded(".") is False
    assert corpus_relpath_is_excluded("./metadata/ep.metadata.json") is False
    assert corpus_relpath_is_excluded("../metadata/ep.metadata.json") is False


# --------------------------------------------------------------------------- #
# classify_metadata_relpath
# --------------------------------------------------------------------------- #


def test_classify_recognises_the_three_known_layouts() -> None:
    assert classify_metadata_relpath("feeds/pod/run_a/metadata/ep.metadata.json") == LAYOUT_FEED_RUN
    assert classify_metadata_relpath("run_a/metadata/ep.metadata.json") == LAYOUT_FLAT_RUN
    assert classify_metadata_relpath("metadata/ep.metadata.json") == LAYOUT_ROOT_FLAT


def test_classify_returns_none_for_an_unrecognised_shape() -> None:
    """The exact shape that let ``.trash/`` through: neither parser claims it."""
    rel = ".trash/20260907T024613Z/feeds/pod/run_a/metadata/ep.metadata.json"
    assert classify_metadata_relpath(rel) is None
    assert classify_metadata_relpath("weird/metadata/ep.metadata.json") is None


# --------------------------------------------------------------------------- #
# #2161 — the row-loss reproduction
# --------------------------------------------------------------------------- #


def test_discover_skips_trashed_copy_of_a_live_episode(tmp_path: Path) -> None:
    """THE BUG: a trashed copy of a live episode must not be collected at all.

    Before the fix this returned TWO paths for one ``(feed_id, episode_id)`` — the live copy and
    the ``.trash`` copy — which is what produced ``duplicate row id`` on every build and let the
    prune delete the live rows.
    """
    live = _write_metadata(tmp_path, f"feeds/pod/{LIVE_RUN}/metadata/ep.metadata.json")
    _write_metadata(
        tmp_path,
        f".trash/20260907T024613Z/feeds/pod/{TRASHED_RUN}/metadata/ep.metadata.json",
    )

    found = discover_metadata_files(tmp_path)

    assert found == [live], f"expected only the live copy, got {[p.as_posix() for p in found]}"
    assert not any(".trash" in p.as_posix() for p in found)


def test_discover_all_metadata_files_also_skips_trash(tmp_path: Path) -> None:
    """The cumulative-count path does NOT go through the dedupe, so it needs its own exclusion.

    Without this, corpus library / stats routes would count retired episodes as live.
    """
    live = _write_metadata(tmp_path, f"feeds/pod/{LIVE_RUN}/metadata/ep.metadata.json")
    _write_metadata(
        tmp_path,
        f".trash/20260907T024613Z/feeds/pod/{TRASHED_RUN}/metadata/ep.metadata.json",
    )

    found = discover_all_metadata_files(tmp_path)

    assert found == [live]


def test_trashed_copy_is_excluded_even_when_it_is_the_only_copy(tmp_path: Path) -> None:
    """A retired episode with no live copy is retired, not a corpus member."""
    _write_metadata(
        tmp_path,
        f".trash/20260907T024613Z/feeds/pod/{TRASHED_RUN}/metadata/ep.metadata.json",
    )
    # ``feeds/`` must exist for the walk branch to be the one under test.
    (tmp_path / "feeds" / "pod").mkdir(parents=True, exist_ok=True)

    assert discover_metadata_files(tmp_path) == []


def test_live_multi_run_supersession_still_works(tmp_path: Path) -> None:
    """Guard against over-reach: the exclusion must not disturb normal newest-run-wins."""
    _write_metadata(tmp_path, f"feeds/pod/{TRASHED_RUN}/metadata/ep.metadata.json")
    newer = _write_metadata(tmp_path, f"feeds/pod/{LIVE_RUN}/metadata/ep.metadata.json")

    found = discover_metadata_files(tmp_path)

    assert found == [newer]


# --------------------------------------------------------------------------- #
# #2162 — defence in depth for the branch that allowed it
# --------------------------------------------------------------------------- #


def test_unrecognised_path_colliding_with_a_live_copy_is_dropped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unclassifiable path must NOT become a second canonical copy of a live episode.

    This is the generic backstop: even if a future retirement directory is not covered by the
    exclusion rule, it cannot duplicate a row id.
    """
    live = _write_metadata(tmp_path, f"feeds/pod/{LIVE_RUN}/metadata/ep.metadata.json")
    surprise = _write_metadata(tmp_path, "weird/metadata/ep.metadata.json")

    with caplog.at_level(logging.WARNING):
        kept = dedupe_metadata_paths_newest_run_per_episode(tmp_path, [live, surprise])

    assert kept == [live]
    assert "REFUSING to treat unrecognised path" in caplog.text
    assert "weird/metadata/ep.metadata.json" in caplog.text


def test_unrecognised_path_without_a_collision_is_kept_but_warned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Never silently DROP data either — an unowned surprise is kept, but loudly.

    The original failure was silence: no log said "I could not classify this path", so the only
    symptom was a ``duplicate row id`` warning three layers downstream in another module.
    """
    live = _write_metadata(tmp_path, f"feeds/pod/{LIVE_RUN}/metadata/ep.metadata.json")
    surprise = _write_metadata(
        tmp_path, "weird/metadata/other.metadata.json", episode_id="some-other-episode"
    )

    with caplog.at_level(logging.WARNING):
        kept = dedupe_metadata_paths_newest_run_per_episode(tmp_path, [live, surprise])

    assert sorted(kept) == sorted([live, surprise])
    assert "matches no known corpus layout" in caplog.text


def test_root_flat_layout_is_silent(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The flat fixture shape is legitimate and common — it must not warn.

    Without an explicit ``metadata/`` classification this branch would warn once per file on every
    flat corpus, which is the kind of noise that trains people to ignore the log.
    """
    flat = _write_metadata(tmp_path, "metadata/ep.metadata.json")

    with caplog.at_level(logging.WARNING):
        kept = dedupe_metadata_paths_newest_run_per_episode(tmp_path, [flat])

    assert kept == [flat]
    assert caplog.text == ""


def test_trash_is_dropped_by_the_dedupe_itself(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The central membership rule drops retired paths even when a caller supplies them directly."""
    live = _write_metadata(tmp_path, f"feeds/pod/{LIVE_RUN}/metadata/ep.metadata.json")
    trashed = _write_metadata(
        tmp_path,
        f".trash/20260907T024613Z/feeds/pod/{TRASHED_RUN}/metadata/ep.metadata.json",
    )

    with caplog.at_level(logging.WARNING):
        kept = dedupe_metadata_paths_newest_run_per_episode(tmp_path, [live, trashed])

    assert kept == [live]
    # Excluded outright, so it is not even reported as an unrecognised surprise.
    assert "matches no known corpus layout" not in caplog.text
