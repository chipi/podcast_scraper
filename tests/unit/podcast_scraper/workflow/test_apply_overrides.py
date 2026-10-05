"""Unit rules for applying operator overrides in the pipeline (#2283).

End to end (fetch -> gate -> files) is covered by
tests/integration/workflow/test_operator_overrides_pipeline.py; this pins each rule alone.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 - builds elements, parses nothing
from datetime import date
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from podcast_scraper import config, overrides as ov
from podcast_scraper.workflow import apply_overrides as ap
from podcast_scraper.workflow.episode_processor import _unsupported_language_skip_reason

pytestmark = pytest.mark.unit


def _cfg(feed_lang: Optional[str] = None) -> config.Config:
    kw: dict = {"rss": "https://example.com/f.xml"}
    if feed_lang is not None:
        kw["feed_declared_language"] = feed_lang
    return config.Config(**kw)


def _episode(guid: str = "g1", title: str = "Original") -> Any:
    # Built, not parsed: nothing here needs an XML parser.
    item = ET.Element("item")
    for tag, text in (("title", title), ("guid", guid), ("pubDate", "x")):
        ET.SubElement(item, tag).text = text
    return SimpleNamespace(item=item, title=title, title_safe=title, idx=1)


class TestFeedFields:
    def test_language_becomes_the_override_and_title_lands_on_the_feed(self) -> None:
        feed = SimpleNamespace(title="Old", description=None, authors=[])
        meta = SimpleNamespace(description=None, image_url=None)
        cfg = ap.apply_feed_fields(
            _cfg(),
            feed,
            meta,
            ov.FeedFields(
                language="English",
                title="New",
                description="D",
                image_url="https://x/i.png",
                authors=["A"],
            ),
        )
        assert cfg.language_override == "en"
        assert (feed.title, feed.description, feed.authors) == ("New", "D", ["A"])
        assert (meta.description, meta.image_url) == ("D", "https://x/i.png")


class TestEpisodeFields:
    def test_the_item_is_patched_and_the_file_name_is_not(self) -> None:
        ep = _episode()
        ap.apply_episode_fields(
            ep,
            ov.EpisodeFields(title="Fixed", description="Desc", published_date=date(2026, 1, 2)),
        )
        assert ep.item.find("title").text == "Fixed"
        assert ep.item.find("description").text == "Desc"
        assert "02 Jan 2026" in ep.item.find("pubDate").text
        assert ep.title == "Fixed" and ep.title_safe == "Original"

    def test_an_override_for_another_guid_touches_nothing(self, tmp_path) -> None:
        ov.set_episode_fields(
            tmp_path, "https://example.com/f.xml", "other", ov.EpisodeFields(title="X")
        )
        ep = _episode()
        cfg = _cfg("en").model_copy(update={"output_dir": str(tmp_path)})
        ap.apply_to_run(cfg, SimpleNamespace(title="F"), None, [ep])
        assert ep.item.find("title").text == "Original"
        assert getattr(ep, "override_fields", None) is None


class TestTheLanguageGateWithEpisodeOverrides:
    def _with_lang(self, lang: Optional[str]) -> Any:
        ep = _episode()
        ep.override_fields = ov.EpisodeFields(language=lang) if lang else None
        return ep

    def test_a_refused_feed_runs_only_the_episodes_an_override_gives_a_language(self) -> None:
        a, b = self._with_lang("en"), self._with_lang(None)
        cfg, eps, refusal = ap.gate_languages(_cfg(), [a, b], _unsupported_language_skip_reason)
        assert refusal is None and eps == [a] and cfg.language_override == "en"

    def test_mixed_episode_languages_are_refused(self) -> None:
        eps_in = [self._with_lang("en"), self._with_lang("es")]
        _cfg2, eps, refusal = ap.gate_languages(_cfg(), eps_in, _unsupported_language_skip_reason)
        assert eps == [] and refusal is not None and "more than one language" in refusal

    def test_a_disabled_episode_language_is_refused(self) -> None:
        _c, eps, refusal = ap.gate_languages(
            _cfg(), [self._with_lang("ja")], _unsupported_language_skip_reason
        )
        assert eps == [] and refusal is not None and "'ja' is not enabled" in refusal

    def test_an_accepted_feed_skips_an_episode_in_another_language(self) -> None:
        same, other = self._with_lang("en"), self._with_lang("es")
        _c, eps, refusal = ap.gate_languages(
            _cfg("en"), [same, other], _unsupported_language_skip_reason
        )
        assert refusal is None and eps == [same]

    def test_no_override_anywhere_keeps_the_feed_refusal(self) -> None:
        _c, eps, refusal = ap.gate_languages(
            _cfg(), [self._with_lang(None)], _unsupported_language_skip_reason
        )
        assert eps == [] and refusal is not None and "declares no <language>" in refusal


class TestDerivedFields:
    def test_an_episode_hosts_override_replaces_the_pool(self) -> None:
        from podcast_scraper.workflow.stages.processing import hosts_for_episode
        from podcast_scraper.workflow.types import HostDetectionResult

        ep = _episode()
        ep.override_fields = ov.EpisodeFields(hosts=["Jane Doe"])
        result = HostDetectionResult({"Detected Host"}, None, None)
        assert hosts_for_episode(result, ep) == {"Jane Doe"}

    def test_speaker_renames_apply_to_named_voices_only(self) -> None:
        import dataclasses

        from podcast_scraper.providers.ml.diarization.pipeline import _apply_speaker_renames

        @dataclasses.dataclass(frozen=True)
        class Role:
            name: str
            named: bool

        @dataclasses.dataclass(frozen=True)
        class Roster:
            by_voice: dict

        roster = Roster(
            by_voice={"SPEAKER_00": Role("Jon Smyth", True), "SPEAKER_01": Role("Jon Smyth", False)}
        )
        out = _apply_speaker_renames(roster, {"Jon Smyth": "John Smith"})
        assert out.by_voice["SPEAKER_00"].name == "John Smith"
        assert out.by_voice["SPEAKER_01"].name == "Jon Smyth", "an unnamed voice is not renamed"
        assert _apply_speaker_renames(roster, {}) is roster

    def test_an_episode_guests_override_replaces_detection(self) -> None:
        from podcast_scraper.workflow.stages.processing import _detect_speakers_for_episode
        from podcast_scraper.workflow.types import HostDetectionResult

        ep = _episode()
        ep.override_fields = ov.EpisodeFields(guests=["Ada Lovelace"])
        got = _detect_speakers_for_episode(
            ep, _cfg("en"), HostDetectionResult(set(), None, None), None  # type: ignore[arg-type]
        )
        assert got is not None and got.guests == ["Ada Lovelace"]
        assert ep.speaker_detection_report["reason"] == "operator_override"
