"""Seat logic v4's feed history in the pipeline: the roster is told how often THIS feed's other
episodes had two (three…) hosts named by evidence. Synthetic feed workspaces only.

Gold-gate measurement 2026-10-02: one-presenter feeds that state two hosts (The Journal 2 of 94
episodes with two evidence-named hosts, The Daily 7 of 100) got a phantom second seat; real
co-host shows sit far above (Unhedged 43 of 79, Odd Lots 41 of 50).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from podcast_scraper.providers.ml.diarization import pipeline as P
from podcast_scraper.providers.ml.diarization.roster import host_copresence_from_diagnostics

pytestmark = pytest.mark.unit

HOSTS = ["Ana Ortiz", "Ben Carter"]


def _episode(feed: Path, n: int, named_hosts: List[str], source: str = "self_intro") -> None:
    run = feed / f"run_{n:04d}"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir()
    rel = f"transcripts/ep{n}.txt"
    (run / "metadata" / f"ep{n}.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"feed_id": "f"},
                "episode": {"episode_id": f"ep-{n}"},
                "content": {"transcript_file_path": rel},
            }
        )
    )
    voices: List[Dict[str, Any]] = [
        {
            "voice": f"SPEAKER_{i:02d}",
            "resolved_name": h,
            "named": True,
            "role": "host",
            "source": source,
        }
        for i, h in enumerate(named_hosts)
    ]
    (run / rel.replace(".txt", ".speakers.diagnostics.json")).write_text(
        json.dumps({"voices": voices})
    )


def _cfg(feed: Path) -> Any:
    P._copresence_cache.clear()
    return SimpleNamespace(output_dir=str(feed))


@pytest.fixture
def feed(tmp_path: Path) -> Path:
    d = tmp_path / "corpus" / "feeds" / "rss_feed.example_abc"
    d.mkdir(parents=True)
    return d


def test_a_co_host_feed_allows_the_second_seat(feed: Path) -> None:
    for n in range(1, 5):
        _episode(feed, n, HOSTS)
    prior = host_copresence_from_diagnostics(P._feed_sibling_diagnostics(_cfg(feed)), HOSTS)
    assert prior is not None and prior[2] == 1.0


def test_a_one_presenter_feed_does_not(feed: Path) -> None:
    for n in range(1, 5):
        _episode(feed, n, [HOSTS[n % 2]])
    prior = host_copresence_from_diagnostics(P._feed_sibling_diagnostics(_cfg(feed)), HOSTS)
    assert prior is not None and prior[2] == 0.0


def test_forced_names_are_not_evidence(feed: Path) -> None:
    # The prior must not feed on its own output: a name the seat rule forced is not a host heard.
    for n in range(1, 5):
        _episode(feed, n, HOSTS, source="known_hosts")
    prior = host_copresence_from_diagnostics(P._feed_sibling_diagnostics(_cfg(feed)), HOSTS)
    assert prior is not None and prior[2] == 0.0


def test_too_little_history_is_cold_start(feed: Path) -> None:
    for n in range(1, 4):
        _episode(feed, n, HOSTS)
    assert host_copresence_from_diagnostics(P._feed_sibling_diagnostics(_cfg(feed)), HOSTS) is None


def test_no_output_dir_reads_nothing() -> None:
    P._copresence_cache.clear()
    no_dir: Any = SimpleNamespace(output_dir="")
    assert P._feed_sibling_diagnostics(no_dir) == []


def test_new_sidecars_are_picked_up(feed: Path) -> None:
    for n in range(1, 4):
        _episode(feed, n, HOSTS)
    cfg = _cfg(feed)
    assert len(P._feed_sibling_diagnostics(cfg)) == 3
    _episode(feed, 4, HOSTS)
    assert len(P._feed_sibling_diagnostics(cfg)) == 4


def test_the_pipeline_passes_the_prior_to_the_roster(
    feed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DRIVE IT: the real apply path hands the roster the feed's history."""
    from podcast_scraper.config import Config
    from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment

    for n in range(1, 5):
        _episode(feed, n, HOSTS)
    P._copresence_cache.clear()
    seen: Dict[str, Any] = {}
    real = P.resolve_speaker_roster

    def spy(*a: Any, **k: Any) -> Any:
        seen["host_copresence"] = k.get("host_copresence")
        return real(*a, **k)

    monkeypatch.setattr(P, "resolve_speaker_roster", spy)
    kw: Dict[str, Any] = {"output_dir": str(feed), "speaker_resolution_llm": False}
    cfg = Config(**kw)
    diar = DiarizationResult(
        segments=[
            DiarizationSegment(0, 30, "SPEAKER_00"),
            DiarizationSegment(30, 60, "SPEAKER_01"),
        ],
        num_speakers=2,
    )
    result = {
        "text": "Welcome. Thanks.",
        "segments": [
            {"start": 0, "end": 30, "text": "Welcome to the show."},
            {"start": 30, "end": 60, "text": "Thanks for having me."},
        ],
    }
    P.apply_diarization_to_result(
        result, "", cfg, [], precomputed_diarization=diar, feed_hosts=HOSTS
    )
    assert seen["host_copresence"] is not None and seen["host_copresence"][2] == 1.0
