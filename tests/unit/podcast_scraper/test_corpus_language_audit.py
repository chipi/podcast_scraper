"""The corpus language audit (#2175 / slice S0.4).

The point of these tests is that the audit CAN FAIL, and fails for the right reason. An audit
that reports a confident 100% ``en`` whatever the corpus contains is worse than no audit: it
closes the question it was built to open.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from podcast_scraper.corpus_language_audit import (
    assess_languages,
    check_corpus,
    format_report,
)

pytestmark = pytest.mark.unit


def _episode(
    root: Path,
    feed_id: str,
    episode_id: str,
    *,
    language: Optional[str] = "en",
    source: Optional[str] = "rss",
    language_raw: Optional[str] = "en-US",
    episode_pair: bool = True,
) -> Path:
    """One served episode. ``episode_pair=False`` mimics a pre-#2172 artifact (feed block only)."""
    meta_dir = root / "feeds" / feed_id / "run_20260101-000000" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / f"{episode_id}.metadata.json"
    feed: Dict[str, Any] = {"feed_id": feed_id, "title": feed_id, "url": "https://e.com/f.xml"}
    episode: Dict[str, Any] = {"episode_id": episode_id, "title": episode_id}
    if language is not None:
        feed["language"] = language
        if episode_pair:
            episode["language"] = language
    if source is not None:
        feed["language_source"] = source
        if episode_pair:
            episode["language_source"] = source
    if language_raw is not None:
        feed["language_raw"] = language_raw
    path.write_text(
        json.dumps({"feed": feed, "episode": episode, "schema_version": "1.0"}), encoding="utf-8"
    )
    return path


class TestItCanFail:
    def test_a_corpus_that_only_echoes_the_config_FAILS(self, tmp_path: Path) -> None:
        """The whole reason this audit exists.

        Every episode says ``en`` from ``profile_default`` — which is what the corpus looked like
        before m0021. A report that called this a pass would have measured our own configuration
        and closed the question.
        """
        for i in range(3):
            _episode(tmp_path, "p01", f"p01_e0{i}", source="profile_default", language_raw=None)

        ok, report = check_corpus(tmp_path)

        assert ok is False
        assert "measured the CONFIGURATION" in report
        assert "VERDICT: FAIL" in report

    def test_an_empty_corpus_FAILS(self, tmp_path: Path) -> None:
        """Nothing measured is not a pass. "Found no problems" and "looked at nothing" must not
        produce the same verdict."""
        ok, report = check_corpus(tmp_path)
        assert ok is False
        assert "NO EPISODES FOUND" in report

    def test_an_unrecognised_source_FAILS(self, tmp_path: Path) -> None:
        """Something wrote a provenance this audit does not understand; reporting it as a clean
        pass would hide whatever produced it."""
        _episode(tmp_path, "p01", "p01_e01", source="guessed_by_something")

        ok, report = check_corpus(tmp_path)

        assert ok is False
        assert "UNKNOWN SOURCE" in report


class TestItPassesOnlyOnRealEvidence:
    def test_one_publisher_declared_language_is_enough(self, tmp_path: Path) -> None:
        _episode(tmp_path, "p01", "p01_e01", source="rss")
        ok, report = check_corpus(tmp_path)
        assert ok is True
        assert "VERDICT: PASS" in report
        assert "at least one publisher said so itself" in report

    def test_a_non_english_episode_is_a_RESULT_not_a_failure(self, tmp_path: Path) -> None:
        """Finding Spanish in the corpus is what the audit is FOR. It must report it loudly and
        still pass, or the gate would punish the discovery it exists to make."""
        _episode(tmp_path, "p01", "p01_e01", source="rss")
        _episode(tmp_path, "p02", "p02_e01", language="es", source="rss", language_raw="es-ES")

        ok, report = check_corpus(tmp_path)

        assert ok is True, "a discovery is not a defect"
        assert "NON-ENGLISH — 1 episode(s)" in report
        assert "p02/p02_e01" in report
        assert "raw='es-ES'" in report


class TestWhatItReports:
    def test_the_resolution_source_is_reported_per_episode(self, tmp_path: Path) -> None:
        """Distribution alone cannot distinguish a measured corpus from a defaulted one."""
        _episode(tmp_path, "p01", "p01_e01", source="rss")
        _episode(tmp_path, "p01", "p01_e02", source="profile_default")
        _episode(tmp_path, "p02", "p02_e01", source="override")

        report = assess_languages(tmp_path)

        assert report.by_source == {"rss": 1, "profile_default": 1, "override": 1}
        assert report.measured is True

    def test_a_language_the_registry_does_not_enable_is_flagged(self, tmp_path: Path) -> None:
        """What S0.8 would skip — worth knowing before S0.8 makes it happen."""
        _episode(tmp_path, "p01", "p01_e01", source="rss")
        _episode(tmp_path, "p02", "p02_e01", language="ja", source="rss", language_raw="ja-JP")

        report = assess_languages(tmp_path)
        text = format_report(report)

        assert [e.episode_id for e in report.not_enabled] == ["p02_e01"]
        assert "NOT ENABLED" in text
        assert "S0.8 would skip these" in text

    def test_a_regional_tag_is_normalized_on_read(self, tmp_path: Path) -> None:
        """A pre-#2174 artifact carrying "en-us" must not appear as its own language."""
        _episode(tmp_path, "p01", "p01_e01", language="en-us", source="rss")

        report = assess_languages(tmp_path)

        assert report.by_language == {"en": 1}

    def test_the_episode_language_wins_over_the_feed(self, tmp_path: Path) -> None:
        """The episode is the answer; the feed block is the fallback for older artifacts."""
        path = _episode(tmp_path, "p01", "p01_e01", language="en", source="rss")
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["episode"]["language"] = "es"
        doc["episode"]["language_source"] = "override"
        path.write_text(json.dumps(doc), encoding="utf-8")

        report = assess_languages(tmp_path)

        assert report.by_language == {"es": 1}
        assert report.by_source == {"override": 1}

    def test_a_pre_2172_artifact_falls_back_to_the_feed_block(self, tmp_path: Path) -> None:
        _episode(tmp_path, "p01", "p01_e01", source="rss", episode_pair=False)

        report = assess_languages(tmp_path)

        assert report.by_language == {"en": 1}
        assert report.by_source == {"rss": 1}

    def test_unparsable_metadata_is_counted_not_fatal(self, tmp_path: Path) -> None:
        _episode(tmp_path, "p01", "p01_e01", source="rss")
        broken = (
            tmp_path / "feeds" / "p01" / "run_20260101-000000" / "metadata" / "bad.metadata.json"
        )
        broken.write_text("{not json", encoding="utf-8")

        report = assess_languages(tmp_path)

        assert len(report.episodes) == 1
        assert len(report.unparsable) == 1
        assert "UNPARSABLE" in format_report(report)


class TestItIsReadOnly:
    def test_nothing_is_written(self, tmp_path: Path) -> None:
        """An audit that mutates the thing it measures is not an audit."""
        paths = [_episode(tmp_path, "p01", f"p01_e0{i}", source="rss") for i in range(3)]
        before = {p: p.read_bytes() for p in paths}
        listing_before = sorted(str(p) for p in tmp_path.rglob("*"))

        check_corpus(tmp_path)

        assert {p: p.read_bytes() for p in paths} == before
        assert sorted(str(p) for p in tmp_path.rglob("*")) == listing_before

    def test_it_is_re_runnable_with_stable_output(self, tmp_path: Path) -> None:
        _episode(tmp_path, "p01", "p01_e01", source="rss")
        first = check_corpus(tmp_path)
        second = check_corpus(tmp_path)
        assert first == second
