"""Show sidecar — the live half (run events) and the backfill, one behaviour per test.

The disk-derived half is in ``test_show_metadata.py``. Here: a run binds its show, the pipeline
appends what happened (hosts detected, transcript refused, KG extraction failed, stage failed,
any ERROR logged), and ``show.json`` folds those events beside per-episode artifact problems read
from disk. The backfill writes ``show.json`` for every existing show. All fixtures synthetic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

from podcast_scraper.utils import correlation
from podcast_scraper.workflow import show_events as se, show_metadata as sm

pytestmark = pytest.mark.unit


@pytest.fixture
def feed_dir(tmp_path: Path) -> Path:
    d = tmp_path / "corpus" / "feeds" / "rss_feed.example_abc"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def bound(feed_dir: Path) -> Iterator[Path]:
    path = se.bind_show(feed_dir / "run_0002")
    assert path is not None
    yield path
    se.unbind_show()


def _events(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _episode(
    feed_dir: Path,
    n: int,
    *,
    kg: Any = "ok",
    gi: Any = "ok",
    summary: Any = "ok",
    transcript: bool = True,
) -> None:
    run = feed_dir / f"run_{n:04d}"
    meta = run / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (run / "transcripts").mkdir(parents=True, exist_ok=True)
    rel = f"transcripts/ep{n}.txt"
    if transcript:
        (run / rel).write_text("Host: hello. Guest: hi.")
    doc: Dict[str, Any] = {
        "feed": {"title": "Shape and Signal", "authors": []},
        "episode": {"episode_id": f"ep-{n}", "title": f"Episode {n}"},
        "content": {"transcript_file_path": rel, "speakers": []},
    }
    if summary == "ok":
        doc["summary"] = {"short_summary": "A summary.", "schema_status": "valid"}
    elif summary == "invalid":
        doc["summary"] = {"short_summary": "A summary.", "schema_status": "invalid"}
    (meta / f"ep{n}.metadata.json").write_text(json.dumps(doc))
    if kg == "ok":
        (meta / f"ep{n}.kg.json").write_text(
            json.dumps({"extraction": {"model_version": "provider:qwen"}, "nodes": []})
        )
    elif kg == "failed":
        (meta / f"ep{n}.kg.json").write_text(
            json.dumps({"extraction": {"model_version": "provider:extraction_failed"}, "nodes": []})
        )
    if gi == "ok":
        (meta / f"ep{n}.gi.json").write_text(json.dumps({"nodes": [{"type": "Insight"}]}))
    elif gi == "empty":
        (meta / f"ep{n}.gi.json").write_text(json.dumps({"nodes": [{"type": "Episode"}]}))


# --- binding -----------------------------------------------------------------------------------


def test_nothing_is_recorded_when_no_show_is_bound(feed_dir: Path) -> None:
    se.unbind_show()
    se.record_show_event("hosts_detected", hosts=["Dana Shipley"])
    assert not (feed_dir / se.EVENTS_SUBDIR).exists()


def test_a_run_outside_the_feed_layout_binds_nothing(tmp_path: Path) -> None:
    assert se.bind_show(tmp_path / "output") is None
    assert se.events_path() is None


def test_a_bound_run_appends_to_its_own_file(feed_dir: Path, bound: Path) -> None:
    assert bound == feed_dir / "show_events" / "run_0002.jsonl"
    se.record_show_event("hosts_detected", hosts=["Dana Shipley"], source="RSS author tags")
    se.record_show_event("transcript_refused", reason="no_speaker_turns")
    kinds = [e["event_type"] for e in _events(bound)]
    assert kinds == ["hosts_detected", "transcript_refused"]


def test_binding_the_next_feed_replaces_the_previous_one(feed_dir: Path, tmp_path: Path) -> None:
    se.bind_show(feed_dir / "run_0001")
    other = tmp_path / "corpus" / "feeds" / "rss_other"
    second = se.bind_show(other / "run_0009")
    se.record_show_event("hosts_detected", hosts=[])
    assert second is not None and second.is_file()
    assert not (feed_dir / "show_events").exists()
    se.unbind_show()


# --- errors logged during the run --------------------------------------------------------------


def test_an_error_logged_during_the_run_lands_in_the_sidecar_with_its_episode(bound: Path) -> None:
    with correlation.episode_scope("ep-7"):
        logging.getLogger("podcast_scraper.kg.llm_extract").error("kg reply was not JSON: %s", "x")
    (event,) = _events(bound)
    assert event["event_type"] == "error"
    assert event["episode_id"] == "ep-7"
    assert event["logger_name"] == "podcast_scraper.kg.llm_extract"
    assert event["message"] == "kg reply was not JSON: x"


def test_a_warning_is_not_an_error(bound: Path) -> None:
    logging.getLogger("podcast_scraper.workflow").warning("just a warning")
    assert not bound.exists()


def test_after_unbinding_errors_are_no_longer_captured(feed_dir: Path) -> None:
    path = se.bind_show(feed_dir / "run_0003")
    se.unbind_show()
    logging.getLogger("podcast_scraper.workflow").error("after the run")
    assert path is not None and not path.exists()


def test_a_huge_message_is_bounded(bound: Path) -> None:
    logging.getLogger("podcast_scraper.workflow").error("x" * 10_000)
    assert len(_events(bound)[0]["message"]) == 600


# --- the emit sites ----------------------------------------------------------------------------


def test_feed_host_detection_records_the_hosts_and_their_source(bound: Path) -> None:
    from podcast_scraper.workflow.stages.processing import _record_hosts_detected

    _record_hosts_detected({"Dana Shipley"}, "RSS author tags", {"Dana Shipley"}, {"Mercer Inc"})
    (event,) = _events(bound)
    assert event["event_type"] == "hosts_detected"
    assert event["hosts"] == ["Dana Shipley"]
    assert event["source"] == "RSS author tags"
    assert event["dropped_non_person"] == ["Mercer Inc"]


def test_a_failed_kg_extraction_is_recorded(bound: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Huberman case: a provider is configured and returns no graph."""
    from podcast_scraper.kg import pipeline

    class _Failing:
        summary_model = "test-model"

        def extract_kg_graph(self, *_a: Any, **_k: Any) -> None:
            return None

    monkeypatch.setattr(pipeline, "_resolve_source", lambda _cfg: "provider")
    pipeline.build_artifact(
        "ep:x", "x", podcast_id="podcast:p1", episode_title="T", kg_extraction_provider=_Failing()
    )
    (event,) = [e for e in _events(bound) if e["event_type"] == "kg_extraction_failed"]
    assert event["episode_id"] == "ep:x"


def test_a_failed_transcription_is_recorded(bound: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from podcast_scraper.workflow.stages import transcription

    monkeypatch.setattr(
        transcription, "factory_transcribe_media_to_text", lambda *a, **k: (False, None, 0)
    )
    transcription.transcribe_media_to_text(object(), object())
    (event,) = [e for e in _events(bound) if e["event_type"] == "stage_failed"]
    assert (event["stage"], event["reason"]) == ("transcribe", "success=False")


def test_a_refused_publisher_transcript_is_recorded(
    bound: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Every shape: captions with no speaker, refused, the episode goes to transcription."""
    import queue

    from podcast_scraper import config
    from podcast_scraper.models.entities import Episode
    from podcast_scraper.workflow import episode_processor as epx

    turnless = (
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nDo you prompt on the fast model?\n\n"
        "00:00:02.000 --> 00:00:03.000\nI do.\n"
    )
    monkeypatch.setattr(
        epx, "_fetch_transcript_content", lambda url, cfg: (turnless.encode(), "text/vtt")
    )
    monkeypatch.setattr(epx, "download_media_for_transcription", lambda *a, **k: None)
    cfg = config.Config(
        output_dir=str(tmp_path), require_transcript_speakers=True, transcribe_missing=True
    )
    import xml.etree.ElementTree as ET

    episode = Episode(
        idx=1,
        title="Dots",
        title_safe="Dots",
        item=ET.Element("item"),
        transcript_urls=[("http://feed.example/t.vtt", "text/vtt")],
    )
    epx.process_episode_download(
        episode, cfg, str(tmp_path), str(tmp_path), None, queue.Queue(), None
    )
    (event,) = [e for e in _events(bound) if e["event_type"] == "transcript_refused"]
    assert event["reason"] == "no_speaker_turns"
    assert event["offered_types"] == ["text/vtt"]
    assert event["episode_title"] == "Dots"


# --- folding into show.json --------------------------------------------------------------------


def test_show_json_folds_runs_host_detections_and_recent_errors(feed_dir: Path) -> None:
    _episode(feed_dir, 1)
    ev = feed_dir / "show_events"
    ev.mkdir()
    (ev / "run_0001.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-10-01T03:00:00+00:00",
                "event_type": "hosts_detected",
                "hosts": ["Dana Shipley"],
                "source": "RSS author tags",
            }
        )
        + "\n"
    )
    (ev / "run_0002.jsonl").write_text(
        "\n".join(
            json.dumps(e)
            for e in (
                {
                    "ts": "2026-10-02T03:00:00+00:00",
                    "event_type": "hosts_detected",
                    "hosts": ["Dana Shipley", "Nora Quint"],
                    "source": "feed statement",
                },
                {
                    "ts": "2026-10-02T03:10:00+00:00",
                    "event_type": "kg_extraction_failed",
                    "episode_id": "ep-1",
                },
                {"ts": "2026-10-02T03:11:00+00:00", "event_type": "transcript_refused"},
                "not json",
            )
        )
        + "\n"
    )
    doc = sm.build_show_metadata(feed_dir)
    assert doc is not None
    assert {r["run"]: r["events"] for r in doc["runs"]} == {
        "run_0001": {"hosts_detected": 1},
        "run_0002": {"hosts_detected": 1, "kg_extraction_failed": 1, "transcript_refused": 1},
    }
    assert doc["host_detections"][0]["hosts"] == ["Dana Shipley", "Nora Quint"]
    assert doc["host_set_changed_between_runs"] is True
    assert doc["recent_errors"] == [
        {
            "run": "run_0002",
            "ts": "2026-10-02T03:10:00+00:00",
            "kind": "kg_extraction_failed",
            "episode_id": "ep-1",
        }
    ]
    assert doc["transcripts"]["refused_in_recent_runs"] == 1


# --- artifact problems read from disk (what the backfill sees with no events) -----------------


def _issues(feed_dir: Path) -> Dict[str, Any]:
    doc = sm.build_show_metadata(feed_dir)
    assert doc is not None
    artifacts: Dict[str, Any] = doc["checks"]["artifacts"]
    return artifacts


def test_a_healthy_episode_has_no_issues(feed_dir: Path) -> None:
    _episode(feed_dir, 1)
    assert _issues(feed_dir) == {"episodes_with_issues": 0, "by_issue": {}, "episodes": []}


@pytest.mark.parametrize(
    "kwargs,issue",
    [
        ({"kg": "failed"}, sm.ISSUE_KG_FAILED),  # the Huberman case
        ({"kg": None}, sm.ISSUE_KG_MISSING),
        ({"gi": None}, sm.ISSUE_GI_MISSING),
        ({"gi": "empty"}, sm.ISSUE_GI_EMPTY),
        ({"summary": None}, sm.ISSUE_SUMMARY_MISSING),
        ({"summary": "invalid"}, sm.ISSUE_SUMMARY_INVALID),
        ({"transcript": False}, sm.ISSUE_TRANSCRIPT_MISSING),
    ],
)
def test_each_artifact_problem_is_named(feed_dir: Path, kwargs: Dict[str, Any], issue: str) -> None:
    _episode(feed_dir, 1, **kwargs)
    found = _issues(feed_dir)
    assert found["by_issue"] == {issue: 1}
    assert found["episodes"] == [{"episode_id": "ep-1", "title": "Episode 1", "issues": [issue]}]


# --- the backfill ------------------------------------------------------------------------------


def test_the_backfill_writes_every_show(feed_dir: Path, tmp_path: Path) -> None:
    _episode(feed_dir, 1)
    other = tmp_path / "corpus" / "feeds" / "rss_other"
    _episode(other, 1)
    (tmp_path / "corpus" / "feeds" / "rss_empty").mkdir()
    assert sm.main(["--corpus", str(tmp_path / "corpus")]) == 0
    assert (feed_dir / "show.json").is_file()
    assert (other / "show.json").is_file()
    assert not (tmp_path / "corpus" / "feeds" / "rss_empty" / "show.json").exists()


def test_the_backfill_dry_run_writes_nothing(feed_dir: Path, tmp_path: Path, capsys) -> None:
    _episode(feed_dir, 1)
    assert sm.main(["--corpus", str(tmp_path / "corpus"), "--dry-run"]) == 0
    assert not (feed_dir / "show.json").exists()
    assert "would write show.json for 1 of 1 feeds" in capsys.readouterr().out


def test_the_backfill_can_target_one_feed(feed_dir: Path, tmp_path: Path) -> None:
    _episode(feed_dir, 1)
    other = tmp_path / "corpus" / "feeds" / "rss_other"
    _episode(other, 1)
    sm.main(["--corpus", str(tmp_path / "corpus"), "--feed", "rss_other"])
    assert (other / "show.json").is_file()
    assert not (feed_dir / "show.json").exists()


def test_the_backfill_refuses_a_root_without_feeds(tmp_path: Path) -> None:
    assert sm.main(["--corpus", str(tmp_path)]) == 2
