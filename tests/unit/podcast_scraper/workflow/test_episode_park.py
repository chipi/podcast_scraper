# mypy: disable-error-code="call-arg"
"""An episode whose transcription keeps failing is parked, so the nightly stops retrying it.

Prod 2026-11-11 and 11-12: episode 26d7aade hung dgx-whisper on two consecutive nightlies
(ResilienceFuseOpenError after 900 s of pause-and-probe), about 15 minutes of each nightly, while
the other 22 episodes transcribed. Nothing read the incident log back, so it would hang every
night. Behaviour through the real failure handler, the batch finalize, the download stage and the
CLI.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator, List
from unittest.mock import MagicMock

import pytest

from podcast_scraper import cli, config
from podcast_scraper.models import Episode
from podcast_scraper.providers.resilience import ResilienceFuseOpenError
from podcast_scraper.workflow import episode_park, episode_processor as ep
from podcast_scraper.workflow.helpers import get_episode_id_from_episode

pytestmark = [pytest.mark.unit]

FEED = "https://example.com/feed.xml"


def _episode(idx: int, guid: str) -> Episode:
    item = ET.Element("item")
    ET.SubElement(item, "guid").text = guid
    return Episode(
        idx=idx,
        title=f"Episode {idx}",
        title_safe=f"ep-{idx}",
        item=item,
        transcript_urls=[],
        media_url=f"https://cdn/{guid}.mp3",
        media_type="audio/mpeg",
    )


@pytest.fixture()
def alerts() -> List[Any]:
    return []


@pytest.fixture()
def corpus(tmp_path: Path, monkeypatch, alerts: List[Any]) -> Iterator[Path]:
    episode_park._reset_for_tests()
    monkeypatch.setattr(
        "podcast_scraper.utils.sentry_init.capture_enrichment_message",
        lambda message, **k: alerts.append((message, k)),
    )
    tmp_path.joinpath("tmp").mkdir()
    yield tmp_path
    episode_park._reset_for_tests()


def _cfg(corpus: Path) -> config.Config:
    return config.Config(
        rss_url=FEED,
        output_dir=str(corpus),
        incident_log_path=str(corpus / "corpus_incidents.jsonl"),
    ).model_copy(update={"feed_declared_language": "en"})


def _fail_transcription(monkeypatch, cfg: config.Config, episode: Episode) -> None:
    """The real handler path for a fuse-open failure, with the real incident append."""
    monkeypatch.setattr(ep, "_bind_episode_correlation", lambda job, cfg: None)
    monkeypatch.setattr(ep, "_maybe_dispatch_reprocess_stage", lambda *a, **k: None)
    monkeypatch.setattr(ep, "_check_and_reuse_existing_transcript", lambda *a, **k: None)
    monkeypatch.setattr(ep, "_check_transcript_cache", lambda *a, **k: None)
    monkeypatch.setattr(ep, "_preprocess_audio_if_needed", lambda job, cfg, m, pm: m)
    monkeypatch.setattr(ep, "_cleanup_temp_media", lambda *a, **k: None)
    monkeypatch.setattr(ep, "_capture_stage_exception", lambda *a, **k: None)
    monkeypatch.setattr(ep, "_record_unresolved_transcript", lambda *a, **k: None)

    def _raise(*a: Any, **k: Any) -> Any:
        raise ResilienceFuseOpenError("dgx-whisper: endpoint did not recover within 900s")

    monkeypatch.setattr(ep, "_transcribe_with_segments_maybe_chunked", _raise)
    job = MagicMock()
    job.idx = episode.idx
    job.temp_media = None
    job.episode = episode
    ok, _, _ = ep.transcribe_media_to_text(job, cfg, None, None, "/tmp/out", MagicMock(), None)
    assert ok is False


def _eid(episode: Episode) -> str:
    return get_episode_id_from_episode(episode, FEED)[0]


def _incidents(corpus: Path) -> List[dict]:
    path = corpus / "corpus_incidents.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _schedule(cfg: config.Config, corpus: Path, episode: Episode):
    return ep.download_media_for_transcription(
        episode,
        cfg,
        temp_dir=str(corpus / "tmp"),
        effective_output_dir=str(corpus),
        run_suffix=None,
    )


def _batch(monkeypatch, cfg, hanging: Episode, others_transcribed: bool) -> dict:
    _fail_transcription(monkeypatch, cfg, hanging)
    if others_transcribed:
        episode_park.note_transcribed(cfg.incident_log_path, "some-other-episode")
    return episode_park.commit_batch(cfg.incident_log_path)


def test_a_repeatedly_hanging_episode_is_parked_alerted_and_skipped(
    monkeypatch, corpus: Path, caplog, alerts: List[Any]
) -> None:
    cfg = _cfg(corpus)
    hanging = _episode(1, "26d7aade")
    eid = _eid(hanging)

    first = _batch(monkeypatch, cfg, hanging, others_transcribed=True)
    assert first == {"counted": True, "parked": [], "cleared": []}
    assert episode_park.parked_entry(cfg.incident_log_path, eid) is None

    with caplog.at_level(logging.ERROR):
        second = _batch(monkeypatch, cfg, hanging, others_transcribed=True)
    assert second["parked"] == [eid]
    entry = episode_park.parked_entry(cfg.incident_log_path, eid)
    assert entry is not None and len(entry["failures"]) == episode_park.PARK_AFTER_FAILURES
    assert entry["failures"][-1]["exception_type"] == "ResilienceFuseOpenError"
    assert entry["feed_url"] == FEED
    # The three alert routes: an error log line with a fixed marker, a GlitchTip event, an incident.
    assert any(
        episode_park.LOG_MARKER in r.getMessage() and eid in r.getMessage() for r in caplog.records
    )
    assert [m for m, _ in alerts] == [f"Episode parked after 2 transcription failures: {eid}"]
    assert [
        i["episode_id"] for i in _incidents(corpus) if i["exception_type"] == "EpisodeParked"
    ] == [eid]

    # The next batch never downloads it, and says so in the incident log.
    monkeypatch.setattr(
        ep, "_download_or_reuse_media", lambda *a, **k: pytest.fail("a parked episode downloaded")
    )
    assert _schedule(cfg, corpus, hanging) is None
    skips = [i for i in _incidents(corpus) if i["exception_type"] == "EpisodeParked"]
    assert len(skips) == 2 and skips[-1]["category"] == "policy"


def test_a_batch_where_nothing_transcribed_counts_nothing(monkeypatch, corpus: Path) -> None:
    """A DGX outage fails every episode; counting it would park the whole batch."""
    cfg = _cfg(corpus)
    hanging = _episode(1, "26d7aade")
    for _ in range(3):
        assert _batch(monkeypatch, cfg, hanging, others_transcribed=False)["counted"] is False
    assert not (corpus / episode_park.STATE_FILE).exists()


def test_a_transcription_clears_the_count(monkeypatch, corpus: Path) -> None:
    cfg = _cfg(corpus)
    flaky = _episode(1, "flaky")
    eid = _eid(flaky)
    _batch(monkeypatch, cfg, flaky, others_transcribed=True)
    episode_park.note_transcribed(cfg.incident_log_path, eid)
    assert episode_park.commit_batch(cfg.incident_log_path)["cleared"] == [eid]
    assert _batch(monkeypatch, cfg, flaky, others_transcribed=True)["parked"] == []


def test_the_cli_lists_and_releases_parked_episodes(monkeypatch, corpus: Path, capsys) -> None:
    cfg = _cfg(corpus)
    hanging = _episode(1, "26d7aade")
    eid = _eid(hanging)
    for _ in range(2):
        _batch(monkeypatch, cfg, hanging, others_transcribed=True)

    assert cli.main(["parked-episodes", "--output-dir", str(corpus)]) == 0
    out = capsys.readouterr().out
    assert "parked: 1  counting: 0" in out and f"PARKED {eid} failures=2" in out

    assert cli.main(["parked-episodes", "--output-dir", str(corpus), "--release", "nope"]) == 1
    assert episode_park.parked_entry(cfg.incident_log_path, eid) is not None

    assert cli.main(["parked-episodes", "--output-dir", str(corpus), "--release", eid]) == 0
    assert episode_park.parked_entry(cfg.incident_log_path, eid) is None
    monkeypatch.setattr(ep, "_download_or_reuse_media", lambda *a, **k: (False, 0, 0.0))
    _schedule(cfg, corpus, hanging)
    assert sum(i["exception_type"] == "EpisodeParked" for i in _incidents(corpus)) == 1


def test_the_batch_finalize_commits_the_count(monkeypatch, corpus: Path) -> None:
    """Both batch loops (cli and service) end in finalize_multi_feed_batch; it does the counting."""
    from podcast_scraper.workflow.corpus_operations import finalize_multi_feed_batch

    cfg = _cfg(corpus)
    hanging = _episode(1, "26d7aade")
    for _ in range(2):
        _fail_transcription(monkeypatch, cfg, hanging)
        episode_park.note_transcribed(cfg.incident_log_path, "some-other-episode")
        summary = finalize_multi_feed_batch(
            str(corpus), cfg, [], incident_log_path=cfg.incident_log_path
        )
    assert episode_park.parked_entry(cfg.incident_log_path, _eid(hanging)) is not None
    assert summary["batch_incidents"]["episode_incidents_unique"]["policy"] == 1


@pytest.mark.parametrize(
    "park_after, batches, parked", [(1, 1, True), (0, 3, False)], ids=["one", "off"]
)
def test_the_threshold_comes_from_config(
    monkeypatch, corpus: Path, park_after: int, batches: int, parked: bool
) -> None:
    from podcast_scraper.workflow.corpus_operations import finalize_multi_feed_batch

    cfg = _cfg(corpus).model_copy(update={"transcription_park_after_failures": park_after})
    hanging = _episode(1, "26d7aade")
    for _ in range(batches):
        _fail_transcription(monkeypatch, cfg, hanging)
        episode_park.note_transcribed(cfg.incident_log_path, "some-other-episode")
        finalize_multi_feed_batch(str(corpus), cfg, [], incident_log_path=cfg.incident_log_path)
    assert (episode_park.parked_entry(cfg.incident_log_path, _eid(hanging)) is not None) is parked


def test_an_operator_parks_by_hand_and_the_next_batch_skips_it(
    monkeypatch, corpus: Path, capsys
) -> None:
    cfg = _cfg(corpus)
    hanging = _episode(1, "26d7aade")
    eid = _eid(hanging)
    assert cli.main(["parked-episodes", "--output-dir", str(corpus), "--park", eid]) == 1
    assert episode_park.parked_entry(cfg.incident_log_path, eid) is None

    argv = ["parked-episodes", "--output-dir", str(corpus), "--park", eid]
    assert cli.main([*argv, "--reason", "hangs dgx-whisper two nights"]) == 0
    assert f"parked 1: {eid}" in capsys.readouterr().out
    entry = episode_park.parked_entry(cfg.incident_log_path, eid)
    assert entry is not None and entry["failures"][-1]["exception_type"] == "ParkedByHand"

    monkeypatch.setattr(
        ep, "_download_or_reuse_media", lambda *a, **k: pytest.fail("a parked episode downloaded")
    )
    assert _schedule(cfg, corpus, hanging) is None
