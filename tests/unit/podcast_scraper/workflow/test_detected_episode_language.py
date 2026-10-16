"""The language the ASR identified for an episode is what metadata and translation read (#2369)."""

from __future__ import annotations

import json

import pytest

from podcast_scraper.workflow.metadata_generation import detected_episode_language

pytestmark = pytest.mark.unit


def _write(tmp_path, record):
    (tmp_path / "transcripts").mkdir(exist_ok=True)
    (tmp_path / "transcripts" / "ep.asr.json").write_text(json.dumps(record), encoding="utf-8")
    return "transcripts/ep.txt"


def test_the_detected_language_is_read(tmp_path) -> None:
    rel = _write(tmp_path, {"language": {"requested": "en", "reported": "en", "detected": "ES"}})
    assert detected_episode_language(str(tmp_path), rel) == "es"


@pytest.mark.parametrize(
    "record",
    [
        {"language": {"requested": "en", "reported": "en", "mismatch": False}},
        {"language": "en"},
        {"model": "x"},
        [],
    ],
)
def test_an_episode_without_a_detection_keeps_its_shows_language(tmp_path, record) -> None:
    assert detected_episode_language(str(tmp_path), _write(tmp_path, record)) is None


def test_no_record_or_no_transcript_is_none(tmp_path) -> None:
    assert detected_episode_language(str(tmp_path), "transcripts/missing.txt") is None
    assert detected_episode_language(str(tmp_path), None) is None


def test_a_broken_record_is_none(tmp_path) -> None:
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "ep.asr.json").write_text("{not json", encoding="utf-8")
    assert detected_episode_language(str(tmp_path), "transcripts/ep.txt") is None


def _prepare(tmp_path, transcript_relpath):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from podcast_scraper import config as config_mod
    from podcast_scraper.workflow import metadata_generation as mg

    feed = SimpleNamespace(title="Show", language="en", authors=[], licence=None)
    episode = MagicMock()
    episode.title, episode.idx, episode.title_safe = "Episode", 1, "Episode"
    cfg = config_mod.Config(rss="https://example.com/f.xml")
    return mg._prepare_base_metadata_objects(
        feed,
        episode,
        "https://example.com/f.xml",
        "feed",
        "ep",
        cfg,
        str(tmp_path),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        [],
        [],
        transcript_file_path=transcript_relpath,
    )[1]


def test_the_episode_block_carries_the_detected_language(tmp_path) -> None:
    rel = _write(tmp_path, {"language": {"requested": "en", "reported": "en", "detected": "es"}})
    ep = _prepare(tmp_path, rel)
    assert (ep.language, ep.language_source) == ("es", "detected")


def test_an_episode_without_a_detection_keeps_its_shows_language_block(tmp_path) -> None:
    rel = _write(tmp_path, {"language": {"requested": "en", "reported": "en", "mismatch": False}})
    ep = _prepare(tmp_path, rel)
    assert ep.language == "en" and ep.language_source != "detected"


def test_translation_reads_the_episodes_language(tmp_path) -> None:
    from podcast_scraper import config as config_mod
    from podcast_scraper.workflow.metadata_generation import episode_language_cfg
    from podcast_scraper.workflow.translation_stage import decide_translation

    cfg = config_mod.Config(rss="https://example.com/f.xml").model_copy(
        update={"feed_declared_language": "en", "language_override": "en"}
    )
    rel = _write(tmp_path, {"language": {"requested": "en", "reported": "en", "detected": "es"}})
    episode_cfg = episode_language_cfg(cfg, str(tmp_path), rel)
    assert decide_translation(episode_cfg, feed_language="en").source_language == "es"
    plain = _write(tmp_path, {"language": {"requested": "en", "reported": "en"}})
    assert episode_language_cfg(cfg, str(tmp_path), plain) is cfg


def test_translation_and_the_gate_are_handed_the_episodes_cfg() -> None:
    """Structural: the two calls are deep in the metadata path; this pins what they receive."""
    import inspect

    from podcast_scraper.workflow import metadata_generation as mg

    source = inspect.getsource(mg)
    assert "episode_cfg = episode_language_cfg(cfg, output_dir, transcript_file_path)" in source
    assert "run_translation_stage(\n        episode_cfg," in source
    assert "analysis_blocked_reason(\n        episode_cfg," in source


def test_the_asr_record_carries_the_detected_language() -> None:
    from podcast_scraper import config as config_mod
    from podcast_scraper.workflow.episode_processor import _asr_language_record

    cfg = config_mod.Config(rss="https://example.com/f.xml")
    result = {"language_requested": "en", "language_reported": "en", "language_detected": "es"}
    assert _asr_language_record(result, cfg)["detected"] == "es"
    assert "detected" not in _asr_language_record({"language_requested": "en"}, cfg)
