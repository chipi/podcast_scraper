"""Guards that the corpus feeds actually describe the corpus.

The defect these exist for: ``build_app_validation_corpus.py`` builds episodes from
``tests/fixtures/transcripts/<ver>/`` and reads RSS only for feed-level metadata, so nothing ever
checked that the feeds listed the corpus's episodes. They didn't — between them the hand-authored
fixtures advertised 26 episodes, with p07 and p08 carrying exactly one each. A real pipeline run can
only process what a feed advertises, so the corpus was unbuildable by the pipeline that is supposed
to build it, and it took pointing the real pipeline at them to notice.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "build_corpus_feeds.py"
RSS_DIR = ROOT / "tests" / "fixtures" / "rss"
_ITUNES_NS = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"

# build_app_validation_corpus.py builds 9 shows x --max-episodes-per-feed 4.
CORPUS_SHOWS = 9
MIN_EPISODES_PER_SHOW = 4


def _version() -> str:
    return (ROOT / "tests" / "fixtures" / "FIXTURES_VERSION").read_text(encoding="utf-8").strip()


def _load():
    spec = importlib.util.spec_from_file_location("_corpus_feeds", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _corpus_feed_paths() -> list[Path]:
    return sorted(RSS_DIR.glob("p*_corpus.xml"))


def _items(path: Path) -> list[ET.Element]:
    channel = ET.parse(path).getroot().find("channel")
    return [] if channel is None else channel.findall("item")


class TestCoverage:
    def test_one_feed_per_show(self) -> None:
        paths = _corpus_feed_paths()
        assert len(paths) == CORPUS_SHOWS, (
            f"expected {CORPUS_SHOWS} corpus feeds, found {[p.name for p in paths]}"
        )

    def test_every_show_covers_the_corpus_depth(self) -> None:
        """The original defect: p07 and p08 advertised ONE episode each."""
        thin = {
            path.name: len(_items(path))
            for path in _corpus_feed_paths()
            if len(_items(path)) < MIN_EPISODES_PER_SHOW
        }
        assert not thin, (
            f"these feeds advertise fewer than the corpus's {MIN_EPISODES_PER_SHOW} episodes per "
            f"show, so a pipeline run cannot rebuild it: {thin}"
        )

    def test_every_episode_has_transcript_and_audio(self) -> None:
        transcripts = ROOT / "tests" / "fixtures" / "transcripts" / _version()
        audio = ROOT / "tests" / "fixtures" / "audio" / _version()
        missing: list[str] = []
        for path in _corpus_feed_paths():
            for item in _items(path):
                guid = (item.findtext("guid") or "").strip()
                if not (transcripts / f"{guid}.txt").is_file():
                    missing.append(f"{guid}: no transcript")
                if not (audio / f"{guid}.mp3").is_file():
                    missing.append(f"{guid}: no audio")
        assert not missing, f"feeds advertise episodes with no media: {missing}"


class TestMeasuredNotAsserted:
    def test_enclosure_length_matches_the_file(self) -> None:
        """Hand-authored feeds carried invented sizes; generated ones are read off disk."""
        audio = ROOT / "tests" / "fixtures" / "audio" / _version()
        wrong: list[str] = []
        for path in _corpus_feed_paths():
            for item in _items(path):
                guid = (item.findtext("guid") or "").strip()
                enclosure = item.find("enclosure")
                mp3 = audio / f"{guid}.mp3"
                if enclosure is None or not mp3.is_file():
                    continue
                declared = int(enclosure.get("length") or 0)
                actual = mp3.stat().st_size
                if declared != actual:
                    wrong.append(f"{guid}: feed says {declared}, file is {actual}")
        assert not wrong, f"enclosure length does not match the media: {wrong}"

    def test_every_item_has_a_nonzero_duration(self) -> None:
        bad: list[str] = []
        for path in _corpus_feed_paths():
            for item in _items(path):
                guid = (item.findtext("guid") or "").strip()
                duration = item.find(f"{_ITUNES_NS}duration")
                text = "" if duration is None else (duration.text or "").strip()
                if not re.match(r"^\d\d:\d\d:\d\d$", text) or text == "00:00:00":
                    bad.append(f"{guid}: {text!r}")
        assert not bad, f"items with missing or zero duration: {bad}"

    def test_cover_art_reference_resolves(self) -> None:
        images = ROOT / "tests" / "fixtures" / "images" / _version()
        missing: list[str] = []
        for path in _corpus_feed_paths():
            channel = ET.parse(path).getroot().find("channel")
            assert channel is not None
            image = channel.find(f"{_ITUNES_NS}image")
            href = "" if image is None else str(image.get("href") or "")
            name = href.rsplit("/", 1)[-1]
            if not name or not (images / name).is_file():
                missing.append(f"{path.name} -> {href!r}")
        assert not missing, f"corpus feeds point at cover art that does not exist: {missing}"


class TestGenerator:
    def test_committed_feeds_are_current(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"], cwd=ROOT, capture_output=True, text=True
        )
        assert result.returncode == 0, (
            f"committed corpus feeds are stale:\n{result.stdout}\n{result.stderr}"
        )

    def test_shows_match_the_corpus_builder(self) -> None:
        """If APP_SHOWS gains a show, these feeds must too — else it silently has no episodes."""
        mod = _load()
        builder = (ROOT / "scripts" / "build_app_validation_corpus.py").read_text(encoding="utf-8")
        for _stem, show in mod.SHOWS:
            assert f'"{show}"' in builder, f"{show} is in build_corpus_feeds.SHOWS but not the builder"
        assert len(mod.SHOWS) == CORPUS_SHOWS
