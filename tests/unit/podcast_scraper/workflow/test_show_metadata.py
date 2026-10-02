"""Show metadata (``feeds/<feed>/show.json``): what we concluded about a show, one case per test.

Operator, 2026-10-02: instead of a report, keep our own show-level metadata that can be queried —
who hosts the show and from which feed field, which transcripts the feed offers and whether we
used them, and whether each episode actually put a host on a voice. Every fixture is synthetic
(never-commit-real-episodes); the shapes mirror the prod cases of that day.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

from podcast_scraper.workflow import show_metadata as sm

pytestmark = pytest.mark.unit


def _feed(
    title: str = "Shape and Signal",
    description: str = "A weekly show about software and design.",
    authors: Any = None,
) -> Dict[str, Any]:
    return {
        "title": title,
        "description": description,
        "authors": authors if authors is not None else [],
        "url": "https://feed.example/rss",
        "feed_id": "sha256:feed",
    }


def _episode(
    feed_dir: Path,
    n: int,
    *,
    feed: Dict[str, Any],
    speakers: Optional[List[Dict[str, Any]]] = None,
    voices: Optional[int] = 2,
    transcript_types: Optional[List[str]] = None,
    source: str = "whisper_transcription",
    known_hosts: Optional[List[str]] = None,
) -> None:
    run = feed_dir / f"run_{n:04d}"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    (run / "transcripts").mkdir(parents=True, exist_ok=True)
    rel = f"transcripts/ep{n}.txt"
    doc = {
        "feed": feed,
        "episode": {"episode_id": f"ep-{n}", "title": f"Episode {n}"},
        "content": {
            "transcript_urls": [{"url": "u", "type": t} for t in transcript_types or []],
            "transcript_source": source,
            "transcript_file_path": rel,
            "speakers": speakers if speakers is not None else [],
            "diarization_num_speakers": voices,
        },
    }
    (run / "metadata" / f"ep{n}.metadata.json").write_text(json.dumps(doc))
    if known_hosts is not None:
        (run / rel.replace(".txt", ".speakers.diagnostics.json")).write_text(
            json.dumps({"tried": {"known_hosts": known_hosts}})
        )


def _host(name: str, placed: bool = True) -> Dict[str, Any]:
    return {"name": name, "role": "host", "placed": placed}


def _build(feed_dir: Path) -> Dict[str, Any]:
    doc = sm.build_show_metadata(feed_dir)
    assert doc is not None
    return doc


@pytest.fixture
def feed_dir(tmp_path: Path) -> Path:
    d = tmp_path / "corpus" / "feeds" / "rss_feed.example_abc"
    d.mkdir(parents=True)
    return d


# --- where a run's feed directory is -------------------------------------------------------


def test_a_run_directory_under_feeds_belongs_to_its_feed(tmp_path: Path) -> None:
    run = tmp_path / "feeds" / "rss_x" / "run_123"
    assert sm.feed_dir_for_run(run) == tmp_path / "feeds" / "rss_x"


def test_a_flat_output_directory_has_no_feed(tmp_path: Path) -> None:
    assert sm.feed_dir_for_run(tmp_path / "output") is None
    assert sm.feed_dir_for_run(tmp_path / "runs" / "run_1") is None


# --- hosts and where each one comes from ---------------------------------------------------


def test_a_host_stated_twice_is_one_host_with_both_sources(feed_dir: Path) -> None:
    # The Every Podcast shape: "Co-hosts A and B talk…" plus an author tag naming A.
    feed = _feed(
        description="Co-hosts Dana Shipley and Nora Quint talk with founders every week.",
        authors=["Dana Shipley"],
    )
    _episode(feed_dir, 1, feed=feed, speakers=[_host("Dana Shipley")])
    hosts = {h["name"]: h["sources"] for h in _build(feed_dir)["hosts"]}
    assert hosts == {
        "Dana Shipley": ["feed_statement", "author_tag"],
        "Nora Quint": ["feed_statement"],
    }


def test_a_junk_statement_and_the_real_author_tags_are_both_visible(feed_dir: Path) -> None:
    # The AI and Design shape: the statement yields a non-person, the author tag the real hosts.
    # The show metadata does not decide between them; it makes the disagreement queryable.
    feed = _feed(
        description="Two Carnegie Mellon faculty explore how AI is reshaping design.",
        authors=["Ben Carter and Ana Ortiz"],
    )
    _episode(feed_dir, 1, feed=feed)
    hosts = {h["name"]: h["sources"] for h in _build(feed_dir)["hosts"]}
    assert hosts["Ben Carter"] == ["author_tag"]
    assert hosts["Ana Ortiz"] == ["author_tag"]
    assert hosts["Two Carnegie Mellon"] == ["feed_statement"]


def test_an_organisation_author_tag_is_not_a_host(feed_dir: Path) -> None:
    _episode(feed_dir, 1, feed=_feed(authors=["Vox Media Podcast Network"]))
    assert _build(feed_dir)["hosts"] == []


def test_authors_stored_as_the_string_of_a_list_are_read(feed_dir: Path) -> None:
    # Prod writes feed.authors as "['Dan Shipper']" on some episodes.
    _episode(feed_dir, 1, feed=_feed(authors="['Dana Shipley']"))
    names = [h["name"] for h in _build(feed_dir)["hosts"]]
    assert names == ["Dana Shipley"]


def test_the_hosts_the_pipeline_used_are_recorded_and_counted(feed_dir: Path) -> None:
    feed = _feed(authors=["Dana Shipley"])
    _episode(feed_dir, 1, feed=feed, known_hosts=["Dana Shipley"], speakers=[_host("Dana Shipley")])
    _episode(
        feed_dir,
        2,
        feed=feed,
        known_hosts=["Dana Shipley"],
        speakers=[_host("Dana Shipley", False)],
    )
    _episode(feed_dir, 3, feed=feed, known_hosts=["Mercer Institute Online"])
    hosts = {h["name"]: h for h in _build(feed_dir)["hosts"]}
    assert hosts["Dana Shipley"]["sources"] == ["author_tag", "used_by_pipeline"]
    assert hosts["Dana Shipley"]["episodes_used_by_pipeline"] == 2
    assert hosts["Dana Shipley"]["episodes_on_a_voice"] == 1
    # A name only the pipeline used (the Conversations with Tyler shape) is listed, not hidden.
    assert hosts["Mercer Institute Online"]["sources"] == ["used_by_pipeline"]


# --- the per-episode host check ------------------------------------------------------------


def _reasons(feed_dir: Path) -> Dict[str, Any]:
    checks: Dict[str, Any] = _build(feed_dir)["checks"]["host_on_a_voice"]
    return checks


def test_a_host_on_a_voice_passes(feed_dir: Path) -> None:
    _episode(feed_dir, 1, feed=_feed(), speakers=[_host("Dana Shipley")])
    c = _reasons(feed_dir)
    assert (c["ok"], c["by_reason"], c["failing"]) == (1, {}, [])


def test_a_known_host_on_no_voice_fails_with_that_reason(feed_dir: Path) -> None:
    # How I Write shape: the host name is known, the roster put it on no voice.
    _episode(
        feed_dir,
        1,
        feed=_feed(),
        speakers=[
            {"name": "Guest Person", "role": "guest", "placed": True},
            _host("Dave P", False),
        ],
    )
    c = _reasons(feed_dir)
    assert c["by_reason"] == {sm.NO_HOST_NOT_PLACED: 1}
    assert c["failing"] == [
        {"episode_id": "ep-1", "title": "Episode 1", "reason": sm.NO_HOST_NOT_PLACED}
    ]


def test_no_host_name_at_all_fails_with_that_reason(feed_dir: Path) -> None:
    # Curiosity Shop shape: voices exist, nobody is named host.
    _episode(
        feed_dir,
        1,
        feed=_feed(),
        speakers=[{"name": "Promo Voice", "role": "guest", "placed": True}],
    )
    assert _reasons(feed_dir)["by_reason"] == {sm.NO_HOST_NO_NAME: 1}


def test_a_one_voice_transcript_fails_with_that_reason(feed_dir: Path) -> None:
    # Every shape: publisher captions with no turns -> one voice -> every name unplaced.
    _episode(
        feed_dir,
        1,
        feed=_feed(),
        speakers=[_host("Dana Shipley", False)],
        voices=None,
        transcript_types=["text/vtt"],
        source="direct_download",
    )
    assert _reasons(feed_dir)["by_reason"] == {sm.NO_HOST_ONE_VOICE: 1}


def test_an_episode_without_a_placement_record_is_counted_not_failed(feed_dir: Path) -> None:
    # Episodes written before the placed field existed cannot be judged; they are not failures.
    _episode(feed_dir, 1, feed=_feed(), speakers=[{"name": "Dana Shipley", "role": "host"}])
    c = _reasons(feed_dir)
    assert (c["without_record"], c["episodes_with_record"], c["failing"]) == (1, 0, [])


# --- transcripts -----------------------------------------------------------------------------


def test_transcripts_offered_and_used_are_counted(feed_dir: Path) -> None:
    _episode(feed_dir, 1, feed=_feed(), transcript_types=["text/vtt"], source="direct_download")
    _episode(feed_dir, 2, feed=_feed(), transcript_types=["text/plain"])
    _episode(feed_dir, 3, feed=_feed())
    t = _build(feed_dir)["transcripts"]
    assert t["offered_types"] == {"text/vtt": 1, "text/plain": 1}
    assert t["transcript_source"] == {"direct_download": 1, "whisper_transcription": 2}
    assert t["publisher_offered_but_transcribed"] == 1


# --- writing ---------------------------------------------------------------------------------


def test_the_file_is_written_beside_the_runs(feed_dir: Path) -> None:
    _episode(feed_dir, 1, feed=_feed(), speakers=[_host("Dana Shipley")])
    out = sm.write_show_metadata(feed_dir)
    assert out == feed_dir / "show.json"
    assert json.loads(out.read_text())["schema_version"] == sm.SCHEMA_VERSION


def test_a_feed_with_no_episodes_writes_nothing(feed_dir: Path) -> None:
    assert sm.write_show_metadata(feed_dir) is None
    assert not (feed_dir / "show.json").exists()


def test_a_failure_never_escapes(feed_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_d: Path) -> None:
        raise RuntimeError("corrupt corpus")

    monkeypatch.setattr(sm, "build_show_metadata", boom)
    assert sm.write_show_metadata(feed_dir) is None


def test_the_run_finalize_writes_it(feed_dir: Path) -> None:
    """DRIVE IT: the real finalize step, a real run directory under feeds/."""
    from podcast_scraper import config
    from podcast_scraper.workflow import metrics, orchestration

    _episode(feed_dir, 1, feed=_feed(), speakers=[_host("Dana Shipley")])
    kw: Dict[str, Any] = {"rss_url": "https://feed.example/rss", "vector_search": False}
    cfg = config.Config(**kw, run_id="r")
    resources = Mock()
    resources.temp_dir = None
    with patch.object(orchestration.wf_helpers, "generate_pipeline_summary", return_value=(1, "")):
        orchestration._finalize_pipeline(
            cfg,
            1,
            resources,
            str(feed_dir / "run_0001"),
            "r",
            metrics.Metrics(),
            [],
            None,
            None,
            None,
            None,
        )
    assert (feed_dir / "show.json").is_file()
